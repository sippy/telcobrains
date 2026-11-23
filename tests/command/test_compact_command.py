import asyncio
import json
import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.session_helpers import run_session

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import ContextCompactionEvent
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import TurnCompleted
from nanobot.command.builtin import _format_compact_dry_run, cmd_compact, cmd_stop
from nanobot.command.router import CommandContext
from nanobot.events import ContextCompactionEvent as AgentCompactionEvent
from nanobot.events import EventSink
from nanobot.llm_usage.types import LLMUsage
from nanobot.providers.base import GenerationSettings, LLMResponse, ProviderConversationState
from nanobot.runtime_context import public_history_message
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT


@pytest.fixture
async def loop(tmp_path):
    bus = MessageBus()
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.can_resume_conversation_state.return_value = True
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="Portable checkpoint.",
        finish_reason="stop",
    ))
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=128_000,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.commands.exact("/summarize", cmd_compact, session_queue=True)
    try:
        yield loop
    finally:
        await loop.aclose()


async def test_compact_session_io_runs_outside_event_loop(loop, monkeypatch) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "summarize this conversation")
    session.provider_state = ProviderConversationState(
        kind="openai_responses", provider="openai:test", model="test-model",
        version=1, payload={"items": []},
    )
    loop.sessions.save(session)

    event_loop_thread = threading.get_ident()
    calls = []

    def observe(name, operation):
        def checked(*args, **kwargs):
            assert threading.get_ident() != event_loop_thread
            calls.append(name)
            return operation(*args, **kwargs)
        return checked

    for name in ("load", "save"):
        monkeypatch.setattr(loop.sessions._store, name, observe(name, getattr(loop.sessions._store, name)))

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact")
    await loop.commands.dispatch(CommandContext(
        msg=msg, session=None, key=key, raw="/compact", loop=loop,
    ))

    assert "load" in calls and "save" in calls
    monkeypatch.undo()
    assert loop.sessions.get_or_create(key).provider_state is None


async def test_empty_compact_with_floor_replies_without_summarizing(loop) -> None:
    loop.compactor_min_verbatim_tail_tokens = 90_000
    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="empty", content="/compact"),
        runtime=loop.llm_runtime(),
    )
    assert reply is not None
    assert reply.content == "No history to compact."
    assert loop.bus.outbound_size == 0
    loop.provider.chat_stream_with_retry.assert_not_awaited()


async def test_floor_skip_does_not_mark_a_replaced_session(loop, monkeypatch) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    loop.sessions.save(session)
    loop.compactor_min_verbatim_tail_tokens = 90_000
    governor = loop.consolidator._context_governor
    prepare = governor.prepare_in_background

    async def prepare_then_reset(*args):
        result = await prepare(*args)
        await loop.sessions.state.reset(key)
        return result

    monkeypatch.setattr(governor, "prepare_in_background", prepare_then_reset)
    runtime, tools, no_tool_contract, replay_config = await loop.compaction_runtime_for_session(session)
    assert await loop.consolidator.compact_idle_session(
        key, runtime=runtime, tools=tools, no_tool_contract=no_tool_contract, replay_config=replay_config,
    ) == ""
    replacement = await loop.sessions.state.get(key)
    assert replacement.generation != session.generation
    assert "_min_tail_skip" not in replacement.metadata
    loop.provider.chat_stream_with_retry.assert_not_awaited()


@pytest.mark.parametrize(("error", "phase"), [(OSError, "failed"), (asyncio.CancelledError, "cancelled")])
async def test_compaction_sizing_error_emits_paired_lifecycle(loop, monkeypatch, error, phase) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    loop.sessions.save(session)
    loop.compactor_min_verbatim_tail_tokens = 90_000
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    monkeypatch.setattr(
        loop.consolidator._context_governor, "prepare_in_background", AsyncMock(side_effect=error),
    )
    runtime, tools, no_tool_contract, replay_config = await loop.compaction_runtime_for_session(session)
    publish = AsyncMock()
    with pytest.raises(error):
        await loop.consolidator.compact_idle_session(
            key, runtime=runtime, tools=tools, no_tool_contract=no_tool_contract, replay_config=replay_config,
            events=EventSink(publish=publish), reason="manual", notify=True,
        )
    started, terminal = [call.args[0] for call in publish.await_args_list]
    assert isinstance(started, AgentCompactionEvent)
    assert isinstance(terminal, AgentCompactionEvent)
    assert (started.phase, terminal.phase) == ("started", phase)
    assert started.compaction_id == terminal.compaction_id
    assert started.reason == terminal.reason == "manual"
    assert started.notify and terminal.notify
    loop.provider.chat_stream_with_retry.assert_not_awaited()


async def test_compact_lock_contention_keeps_heartbeat_running(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "summarize this conversation")
    loop.sessions.save(session)

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact")
    blocker = sqlite3.connect(loop.sessions._store.path)
    blocker.execute("BEGIN IMMEDIATE")
    task = asyncio.create_task(loop.commands.dispatch(CommandContext(
        msg=msg, session=session, key=key, raw="/compact", loop=loop,
    )))
    try:
        for _ in range(5):
            await asyncio.sleep(0.01)
            assert not task.done(), "compaction must wait for storage without blocking the loop"
    finally:
        blocker.rollback()
        blocker.close()
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/compact", " /COMPACT@nanobot ", "/summarize"])
async def test_compact_emits_one_lifecycle_and_keeps_the_session(loop, command) -> None:
    bus = loop.bus
    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "important question")
    session.add_message("assistant", "important answer")
    session.provider_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": []},
    )
    loop.sessions.save(session)

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="test", content=command)
    response = await loop._process_message(msg, runtime=loop.llm_runtime())

    assert response is None
    assert bus.outbound_size == 2
    started = bus.outbound.get_nowait().event
    completed = bus.outbound.get_nowait().event
    assert isinstance(started, ContextCompactionEvent)
    assert isinstance(completed, ContextCompactionEvent)
    assert started.phase == "started"
    assert completed.phase == "succeeded"
    assert started.compaction_id == completed.compaction_id
    assert started.reason == completed.reason == "manual"
    assert started.notify is True
    assert completed.notify is True

    loop.sessions.invalidate("cli:test")
    reloaded = loop.sessions.get_or_create("cli:test")
    assert reloaded.provider_state is None
    assert reloaded.messages[:-1] == session.messages
    assert is_hidden_history_message(reloaded.messages[-1])
    assert reloaded.last_archived == 2
    assert reloaded.get_history() == []
    assert reloaded.metadata["_last_summary"]["text"] == "Portable checkpoint."
    assert len(loop.consolidator.store.read_unprocessed_history(0)) == 1

    response = await loop._process_message(msg, runtime=loop.llm_runtime())
    assert response is None
    assert bus.outbound_size == 0
    loop.provider.chat_stream_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "summary"),
    [
        ("manual", "The checkpoint inspection is complete."),
        ("idle", "(nothing)"),
    ],
)
async def test_compacted_session_waits_for_new_input_without_continuation(
    loop, trigger, summary,
) -> None:
    key = "cli:checkpoint-resume"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "Inspect the checkpoint")
    session.add_message("assistant", "Inspection complete.")
    if trigger == "idle":
        loop.auto_compact._ttl = 15
        session.updated_at = datetime.now() - timedelta(minutes=20)
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(content=summary)

    if trigger == "manual":
        await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="checkpoint-resume",
                           content="/compact"),
            runtime=loop.llm_runtime(),
        )
    else:
        await loop.auto_compact._archive(key, runtime=loop.llm_runtime())

    loop.sessions.invalidate(key)
    loop.auto_compact._summaries.clear()
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.metadata["_last_summary"]["text"] == summary
    assert reloaded.last_archived == 2
    assert reloaded.get_history() == []
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    assert loop.bus.inbound_size == 0

    loop.provider.chat_stream_with_retry.reset_mock()
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(content="Hello!")
    response = await loop.process_direct("hi", session_key=key)
    assert response.content == "Hello!"
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    sent = loop.provider.chat_stream_with_retry.call_args.kwargs["messages"]
    assert [message["role"] for message in sent] == ["system", "user"]
    expected_summary = reloaded.metadata["_last_summary"] if summary != "(nothing)" else None
    assert sent[0]["content"] == loop.context.build_system_prompt(
        channel="cli", session_summary=expected_summary, has_cron=False,
    )
    assert sent[1]["content"].startswith("hi\n\n")
    assert "File access: no workspace restriction." in sent[1]["content"]

    loop.sessions.invalidate(key)
    resumed = loop.sessions.get_or_create(key)
    assert resumed.get_history(include_runtime_context=False) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_commands", [False, True])
async def test_empty_compact_finishes_silently_and_does_not_schedule_idle_archive(
    loop, legacy_commands,
) -> None:
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "already archived")
    session.add_message("assistant", "old answer")
    session.last_archived = 2
    if legacy_commands:
        session.add_message("user", "/compact", _command=True)
        session.add_message("assistant", "Nothing to compact.", _command=True)
    session.updated_at = datetime.now() - timedelta(minutes=30)
    loop.sessions.save(session)
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)

    await run_session(loop, InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content="/compact",
        metadata={"webui_turn_id": "compact-turn"},
    ))

    assert loop.bus.outbound_size == 0
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "compact-turn"
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 2
    assert loop.consolidator.store.read_unprocessed_history(0) == []

    loop.auto_compact._ttl = 1
    assert loop.auto_compact._is_expired(reloaded.updated_at)
    schedule = MagicMock()
    await loop.auto_compact.check_expired(schedule, loop.runtime_for_session)
    schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/COMPACT@nanobot", "/summarize"])
async def test_compact_during_active_turn_waits_for_the_session_lock(loop, command) -> None:
    key = "websocket:test"
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content=command,
    )
    lock = loop._get_session_lock(key)
    async with lock:
        await loop._dispatch_command_inline(msg, msg.content, loop.commands.dispatch)
        tasks = list(loop._active_tasks[key])
        assert len(tasks) == 1
        await asyncio.sleep(0)
        assert not tasks[0].done()
        assert loop.bus.outbound_size == 0
        session = loop.sessions.get_or_create(key)
        session.add_message("user", "active turn question")
        session.add_message("assistant", "active turn answer")
        loop.sessions.save(session)

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert loop.bus.outbound_size == 2
    assert loop.sessions.get_or_create(key).last_archived == 2
    loop.provider.chat_stream_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/compact", "/summarize"])
async def test_stop_completes_a_compact_command_waiting_for_the_session_lock(loop, command) -> None:
    key = "websocket:test"
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content=command,
        metadata={"webui_turn_id": "queued-compact"},
    )
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    async with loop._get_session_lock(key):
        await loop._dispatch_command_inline(msg, msg.content, loop.commands.dispatch)
        reply = await cmd_stop(CommandContext(
            msg=msg, session=None, key=key, raw="/stop", loop=loop,
        ))
    assert reply.content == "Stopped 1 task(s)."
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "queued-compact"


@pytest.mark.asyncio
@pytest.mark.parametrize("command_text", ["/compact", "/summarize"])
async def test_compact_is_a_fifo_barrier_during_an_active_turn(loop, command_text) -> None:
    key = "cli:test"
    started = asyncio.Event()
    release = asyncio.Event()
    requests = []
    compacted_history = []
    compact = loop.consolidator.compact_idle_session

    async def capture_compaction(*args, **kwargs):
        compacted_history.extend(
            dict(message) for message in loop.sessions.get_or_create(key).messages
        )
        return await compact(*args, **kwargs)

    loop.consolidator.compact_idle_session = capture_compaction

    async def chat(*, messages, **kwargs):
        requests.append([dict(message) for message in messages])
        if len(requests) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="answer", finish_reason="stop")

    loop.provider.chat_stream_with_retry = chat
    task = asyncio.create_task(run_session(loop, InboundMessage(
        channel="cli", sender_id="u", chat_id="test", content="initial question",
    )))
    try:
        # Include cold context initialization and durable commits on slower CI disks.
        await asyncio.wait_for(started.wait(), timeout=30)
        (await loop._enqueue_session_message(InboundMessage(
            channel="cli", sender_id="u", chat_id="test", content="before compaction",
        )))
        command = InboundMessage(channel="cli", sender_id="u", chat_id="test", content=command_text)
        await loop._dispatch_command_inline(command, command.content, loop.commands.dispatch)
        (await loop._enqueue_session_message(InboundMessage(
            channel="cli", sender_id="u", chat_id="test", content="after compaction",
        )))
        release.set()
        await asyncio.wait_for(task, timeout=30)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert [public_history_message(message)["content"] for message in compacted_history if message["role"] == "user"] == [
        "initial question", "before compaction",
    ]
    assert all(command_text != message.get("content") for request in requests for message in request)
    assert "after compaction" in str(requests[-1])
    events = [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]
    assert [event.phase for event in events if isinstance(event, ContextCompactionEvent)] == [
        "started", "succeeded",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("command_text", ["/compact", "/summarize"])
async def test_stop_completes_compact_queued_behind_an_active_turn(loop, command_text) -> None:
    key = "websocket:test"
    started = asyncio.Event()

    async def chat(**kwargs):
        started.set()
        await asyncio.Event().wait()

    loop.provider.chat_stream_with_retry = chat
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    (await loop._enqueue_session_message(InboundMessage(
        channel="websocket", sender_id="u", chat_id="test", content="question",
    )))
    task = next(iter(loop._active_tasks[key]))
    await asyncio.wait_for(started.wait(), timeout=5)
    command = InboundMessage(
        channel="websocket", sender_id="u", chat_id="test", content=command_text,
        metadata={"webui_turn_id": "queued-compact"},
    )
    await loop._dispatch_command_inline(command, command.content, loop.commands.dispatch)
    await cmd_stop(CommandContext(msg=command, session=None, key=key, raw="/stop", loop=loop))

    assert task.cancelled()
    assert sum(
        event.context.metadata.get("webui_turn_id") == "queued-compact" for event in completions
    ) == 1
    assert key not in loop._pending_queues


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/compact", "/summarize"])
async def test_stop_finishes_inflight_compaction_as_cancelled(loop, command) -> None:
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "important question")
    session.add_message("assistant", "important answer")
    loop.sessions.save(session)
    entered = asyncio.Event()

    async def wait_for_cancel(**_kwargs):
        entered.set()
        await asyncio.Event().wait()

    loop.provider.chat_stream_with_retry.side_effect = wait_for_cancel
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    msg = InboundMessage(
        channel="websocket", sender_id="user", chat_id="test", content=command,
        metadata={"webui_turn_id": "compact-turn"},
    )
    (await loop._enqueue_session_message(msg))
    task = next(iter(loop._active_tasks[key]))
    await asyncio.wait_for(entered.wait(), timeout=5)

    reply = await cmd_stop(CommandContext(
        msg=msg, session=session, key=key, raw="/stop", loop=loop,
    ))

    assert reply.content == "Stopped 1 task(s)."
    assert task.cancelled()
    assert len(completions) == 1
    assert completions[0].context.metadata["webui_turn_id"] == "compact-turn"
    events = [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]
    assert all(isinstance(event, ContextCompactionEvent) for event in events)
    assert [event.phase for event in events] == ["started", "cancelled"]
    assert [event.reason for event in events] == ["manual", "manual"]
    assert events[0].compaction_id == events[1].compaction_id
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 0
    assert reloaded.get_history() == session.get_history()


@pytest.mark.asyncio
async def test_idle_and_manual_compact_share_persisted_checkpoint(loop) -> None:
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "large tool turn")
    for i in range(20):
        session.add_message("assistant", "", tool_calls=[{
            "id": f"tool-{i}", "type": "function",
            "function": {"name": "exec", "arguments": "{}"},
        }])
        session.add_message("tool", "x" * 10_000, tool_call_id=f"tool-{i}")
    session.add_message("assistant", "done")
    loop.sessions.save(session)
    runtime = loop.llm_runtime()
    await loop.consolidator.compact_idle_session(key, runtime=runtime)
    assert loop.sessions.get_or_create(key).get_history() == []

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact"),
        runtime=runtime,
    )
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    assert loop.bus.outbound_size == 0
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert len(reloaded.messages) == 43
    assert is_hidden_history_message(reloaded.messages[-1])
    assert reloaded.get_history() == []
    assert reloaded.metadata["_last_summary"]["text"] == "Portable checkpoint."

    reloaded = await loop.sessions.state.get(key)
    reloaded.add_message("user", "next question")
    reloaded.add_message("assistant", "next answer")
    await loop.sessions.state.finish_turn(reloaded)
    await loop.consolidator.compact_idle_session(key, runtime=runtime)
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.get_history() == []


async def test_compact_dry_run_reports_the_real_summary_and_changes_nothing(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "first question")
    session.add_message("assistant", "first answer")
    session.provider_state = ProviderConversationState(
        kind="openai_responses", provider="openai:test", model="test-model",
        version=1, payload={"items": []},
    )
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="Dry run checkpoint.",
        finish_reason="stop",
        usage=LLMUsage.reported(input_tokens=120, output_tokens=10).with_timing(
            generation_ms=400, ttft_ms=50,
        ),
    )

    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )

    assert reply is not None
    assert reply.content.startswith("## Compaction dry run")
    assert "- Outcome: summary generated by the compactor (discarded)" in reply.content
    assert "- Model: `test-model`" in reply.content
    assert "- Elapsed: " in reply.content
    assert "- Throughput (estimate): 25.0 output tokens/s over the measured streaming generation time" in reply.content
    assert (
        "- Provider usage: 120 input / 10 output tokens (provider-reported; 1 summarizer call(s))"
        in reply.content
    )
    assert "- Before (estimate): 2 replayable message(s)" in reply.content
    assert "- Full unarchived prefix: 2 message(s) summarized into one checkpoint" in reply.content
    assert "(estimate; discarded, not persisted)" in reply.content
    assert "After, if applied (estimate)" in reply.content
    loop.provider.chat_stream_with_retry.assert_awaited_once()

    # No transcript, checkpoint, journal, provider-state, or lifecycle change:
    # the reply only lands as the ordinary _command exchange.
    reloaded = loop.sessions.get_or_create(key)
    assert [m for m in reloaded.messages if not m.get("_command")] == session.messages
    assert reloaded.last_archived == 0
    assert reloaded.provider_state is not None
    assert "_last_summary" not in reloaded.metadata
    assert "_min_tail_skip" not in reloaded.metadata
    assert loop.consolidator.store.read_unprocessed_history(0) == []
    assert loop.bus.outbound_size == 0


async def test_compact_dry_run_matches_the_real_compaction_request(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "first question")
    session.add_message("assistant", "first answer")
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )
    dry_run_request = loop.provider.chat_stream_with_retry.call_args.kwargs["messages"]

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact"),
        runtime=loop.llm_runtime(),
    )
    real_request = loop.provider.chat_stream_with_retry.call_args_list[1].kwargs["messages"]

    assert real_request == dry_run_request
    assert loop.sessions.get_or_create(key).metadata["_last_summary"]["text"] == "Portable checkpoint."


def _tool_heavy_messages() -> list[dict]:
    """A long batch-log investigation followed by a short follow-up turn."""
    messages = [{"role": "user", "content": "Compare the processing logs for shards 0 through 13."}]
    for i in range(14):
        messages.append({
            "role": "assistant", "content": None, "tool_calls": [{
                "id": f"call-{i}", "type": "function",
                "function": {"name": "exec", "arguments": json.dumps({"command": f"cat shard-{i}.log"})},
            }],
        })
        log = "\n".join(
            f"2026-04-17 INFO shard={i:02} batch={batch:04} rows={batch * 17:04} status=processed"
            for batch in range(155)
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{i}", "name": "exec", "content": log})
    messages.extend([
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": "next answer"},
    ])
    return messages


async def test_compact_mid_turn_cut_for_tool_heavy_first_turn_matches_dry_run(loop, monkeypatch) -> None:
    """A tool-heavy first turn where the user-only cut cannot keep the floor.

    The next user boundary keeps ~20 tokens, far below the 90k verbatim-tail
    floor, so the legacy user-only cut would skip the compaction. The
    tool-pair-safe assistant boundary inside the tool-heavy turn retains the
    floor; the dry run and the real /compact must apply that identical cut,
    and the reloaded assistant-leading tail must stay verbatim with every
    tool pair intact.
    """
    key = "cli:tool-heavy"
    session = loop.sessions.get_or_create(key)
    session.messages = _tool_heavy_messages()
    full_history = session.get_history()
    loop.sessions.save(session)
    loop.compactor_min_verbatim_tail_tokens = 90_000
    loop.compactor_next_turn_input_reserve_tokens = 4_000

    def token_cost(message: dict) -> int:
        # Deterministic conservative byte meter, like the local fallback.
        return 4 + sum(len(str(message[key]).encode()) for key in (
            "content", "tool_calls", "tool_call_id", "name",
        ) if key in message and message[key] is not None)

    loop.provider.estimate_prompt_tokens = lambda messages, tools, model: (
        sum(token_cost(message) for message in messages), "test-counter",
    )
    monkeypatch.setattr("nanobot.agent.context_governance.estimate_message_tokens", token_cost)
    monkeypatch.setattr("nanobot.session.manager.estimate_message_tokens", token_cost)
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="Mid-turn checkpoint.", finish_reason="stop",
    )

    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="tool-heavy",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )

    assert reply is not None
    assert "- Outcome: summary generated by the compactor (discarded)" in reply.content
    assert "- Before (estimate): 32 replayable message(s)" in reply.content
    # Rounds 0-1 are summarized; the rest of the tool-heavy turn plus turn
    # two stay verbatim because their ~120k tokens meet the floor and fit the
    # replay headroom, while no user boundary does.
    assert "- Protected prefix: 5 message(s) summarized into one checkpoint" in reply.content
    assert "- Retained tail: 27 message(s) kept verbatim" in reply.content
    dry_run_request = loop.provider.chat_stream_with_retry.call_args.kwargs

    # Verify the dry run applied nothing durably before issuing real /compact.
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.last_archived == 0
    assert "_last_summary" not in reloaded.metadata
    assert reloaded.get_history() == full_history

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="tool-heavy",
                       content="/compact"),
        runtime=loop.llm_runtime(),
    )

    assert loop.provider.chat_stream_with_retry.await_count == 2
    real_request = loop.provider.chat_stream_with_retry.call_args.kwargs
    assert real_request == dry_run_request
    assert real_request["messages"][:-1] == full_history[:5]

    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.last_archived == 5
    assert is_hidden_history_message(reloaded.messages[5])
    assert reloaded.messages[5]["content"] == SUMMARY_CONTINUATION_TEXT
    replayed = reloaded.get_history(include_runtime_context=False)
    assert replayed == full_history[5:]
    assert replayed[0]["role"] == "assistant"
    assert replayed[0]["tool_calls"][0]["id"] == "call-2"
    assert replayed[1]["tool_call_id"] == "call-2"
    declared = {
        call["id"]
        for message in replayed
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    assert {m["tool_call_id"] for m in replayed if m.get("role") == "tool"} == declared


async def test_compact_dry_run_matches_real_compact_with_prior_checkpoint_and_partial_cut(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "first question")
    session.add_message("assistant", "first answer")
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="Prior checkpoint.", finish_reason="stop",
    )
    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    # Persist a prior command exchange inside the next archive range.
    reloaded = await loop.sessions.state.get(key)
    assert reloaded.last_archived == 2
    reloaded.add_message("user", "/status", _command=True)
    reloaded.add_message("assistant", "Status report.", _command=True)
    reloaded.add_message("user", "second question")
    reloaded.add_message("assistant", "second answer")
    reloaded.add_message("user", "third question")
    reloaded.add_message("assistant", "third answer")
    await loop.sessions.state.finish_turn(reloaded)

    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="Fresh checkpoint.", finish_reason="stop",
    )
    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )
    assert reply is not None
    assert "- Outcome: summary generated by the compactor (discarded)" in reply.content
    # The protected cut archives the checkpoint, the command exchange, and the
    # oldest replayable pair while keeping the newest pair verbatim.
    assert "- Protected prefix: 5 message(s) summarized into one checkpoint" in reply.content
    assert "- Retained tail: 2 message(s) kept verbatim" in reply.content

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/compact"),
        runtime=loop.llm_runtime(),
    )
    dry_run_request = loop.provider.chat_stream_with_retry.call_args_list[1].kwargs["messages"]
    real_request = loop.provider.chat_stream_with_retry.call_args_list[2].kwargs["messages"]
    assert real_request == dry_run_request

    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.last_archived == 7
    assert [message["content"] for message in reloaded.get_history()] == [
        "third question", "third answer",
    ]


async def test_compact_dry_run_cannot_replay_raw_fallback_makes_no_provider_calls(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "first question")
    session.add_message("assistant", "x" * 4000)
    loop.sessions.save(session)

    # A window that fits the answer but not both messages, so the full-prefix
    # replay cannot be reassembled and the archiver raw-dumps early.
    small_window = replace(loop.llm_runtime(), context_window_tokens=2000)
    preview = await loop.consolidator.compact_idle_session_dry_run(
        key, runtime=small_window, replay_runtime=small_window,
    )
    report = _format_compact_dry_run(preview)

    assert preview.outcome == "raw_fallback"
    assert preview.provider_calls == 0
    assert preview.summary is not None
    assert "- Outcome: raw fallback checkpoint — the summarizer could not be asked" in report
    assert "- Full unarchived prefix: 2 message(s) replaced by one raw-dump checkpoint" in report
    assert "No summarizer call was made:" in report
    loop.provider.chat_stream_with_retry.assert_not_awaited()
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == session.messages
    assert reloaded.last_archived == 0
    assert "_last_summary" not in reloaded.metadata
    assert loop.consolidator.store.read_unprocessed_history(0) == []


async def test_compact_dry_run_floor_skip_makes_no_provider_call(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    loop.sessions.save(session)
    loop.compactor_min_verbatim_tail_tokens = 90_000

    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )

    assert reply is not None
    # A single short message sits at or below the floor: a no-op by design,
    # not a budget rejection.
    assert "- Outcome: skipped — est. " in reply.content
    assert "at or below the configured 90000-token verbatim-tail floor; nothing to archive" in reply.content
    assert "- Model that would summarize: `test-model`" in reply.content
    assert "No summarizer call was made:" in reply.content
    loop.provider.chat_stream_with_retry.assert_not_awaited()
    reloaded = loop.sessions.get_or_create(key)
    assert [m for m in reloaded.messages if not m.get("_command")] == session.messages
    # A dry run must not arm the idle-scan skip marker.
    assert "_min_tail_skip" not in reloaded.metadata


async def test_compact_dry_run_no_history_makes_no_provider_call(loop) -> None:
    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="empty",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )
    assert reply is not None
    assert "- Outcome: no history to compact." in reply.content
    assert "No summarizer call was made:" in reply.content
    loop.provider.chat_stream_with_retry.assert_not_awaited()
    assert loop.bus.outbound_size == 0


@pytest.mark.parametrize(
    "content", ["/compact --bogus", "/compact now extra", "/compact --dry-run extra"],
)
async def test_compact_unknown_argument_does_not_compact(loop, content) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "question")
    loop.sessions.save(session)

    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content=content),
        runtime=loop.llm_runtime(),
    )

    assert reply is not None
    assert reply.content.startswith("Usage: /compact [--dry-run]")
    loop.provider.chat_stream_with_retry.assert_not_awaited()
    reloaded = loop.sessions.get_or_create(key)
    assert [m for m in reloaded.messages if not m.get("_command")] == session.messages
    assert reloaded.last_archived == 0


async def test_compact_dry_run_reports_raw_fallback_without_writing(loop) -> None:
    key = "cli:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "question")
    session.add_message("assistant", "answer")
    loop.sessions.save(session)
    loop.provider.estimate_prompt_tokens.return_value = (100, "test")
    loop.provider.chat_stream_with_retry.return_value = LLMResponse(
        content="unusable", finish_reason="error",
    )

    reply = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test",
                       content="/compact --dry-run"),
        runtime=loop.llm_runtime(),
    )

    assert reply is not None
    assert "- Outcome: raw fallback checkpoint — the summarizer call produced no usable output" in reply.content
    assert "(usage not reported by the provider)" in reply.content
    loop.provider.chat_stream_with_retry.assert_awaited_once()
    reloaded = loop.sessions.get_or_create(key)
    assert [m for m in reloaded.messages if not m.get("_command")] == session.messages
    assert reloaded.last_archived == 0
    assert "_last_summary" not in reloaded.metadata
    assert loop.consolidator.store.read_unprocessed_history(0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content", ["/compact --dry-run", "/COMPACT@nanobot --DRY-RUN"],
)
async def test_compact_dry_run_is_session_serialized(loop, content) -> None:
    key = "websocket:test"
    msg = InboundMessage(
        channel="websocket", sender_id="u", chat_id="test", content=content,
    )
    lock = loop._get_session_lock(key)
    async with lock:
        await loop._dispatch_command_inline(msg, msg.content, loop.commands.dispatch)
        tasks = list(loop._active_tasks[key])
        assert len(tasks) == 1
        await asyncio.sleep(0)
        assert not tasks[0].done()
        assert loop.bus.outbound_size == 0
        session = loop.sessions.get_or_create(key)
        session.add_message("user", "active turn question")
        session.add_message("assistant", "active turn answer")
        loop.sessions.save(session)

    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    # The dry run ran behind the turn's activity and applied nothing.
    assert loop.bus.outbound_size == 1
    assert "dry run" in loop.bus.outbound.get_nowait().content
    assert loop.sessions.get_or_create(key).last_archived == 0
    loop.provider.chat_stream_with_retry.assert_awaited_once()
