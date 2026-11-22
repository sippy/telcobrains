import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import Consolidator
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext, current_request_context, request_context
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.subagent import SubagentTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import TurnRunStatusChanged
from nanobot.config import Config
from nanobot.config.schema import ModelPresetConfig
from nanobot.nanobot import Nanobot
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.factory import ProviderSnapshot
from nanobot.session.manager import SESSION_CACHE_MAX_SIZE, SessionManager
from nanobot.session.webui_turns import WebuiTurnRoutePolicy
from nanobot.utils.llm_runtime import LLMRuntime


def setup(tmp_path, **kwargs):
    manager = SubagentManager(
        workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16000,
        consolidator=MagicMock(spec=Consolidator), **kwargs,
    )
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings()
    runtime = LLMRuntime.capture(provider, "test", context_window_tokens=128000)
    ctx = RequestContext("websocket", "route", session_key="owner", runtime=runtime)
    return manager, provider, ctx


async def spawn(manager, ctx, text="task"):
    with request_context(ctx):
        result = await SubagentTool(manager).execute(action="create", task=text)
    task_id = re.search(r"id: ([\w-]+)", result).group(1)
    assert str(uuid.UUID(task_id)) == task_id
    return task_id


async def control(manager, ctx, task_id, action=None, message=None):
    with request_context(ctx):
        return await SubagentTool(manager).execute(action or "check", task_id, message)


async def settled(manager):
    await asyncio.gather(*list(manager._running_tasks.values()), return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_direct_turn_waits_for_child_without_a_background_consumer(loop_factory):
    loop = loop_factory()
    loop.provider.provider_name = "test"
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate", name="subagent",
            arguments={"action": "create", "task": "Inspect the configuration"},
        )]),
        LLMResponse(content="Verified child findings"),
        LLMResponse(content="Parent conclusion from the child findings"),
    ])
    try:
        response = await loop.process_direct("Delegate this inspection")
        assert response.content == "Parent conclusion from the child findings"
        parent_followup = loop.provider.chat_stream_with_retry.call_args_list[-1].kwargs["messages"]
        assert any(m.get("role") == "tool" and m.get("content") == "Verified child findings"
                   for m in parent_followup)
        assert loop.bus.inbound_size == 0
        assert loop.subagents.get_running_count() == 0
        assert next(iter(loop.subagents.statuses_for_session("cli:direct").values())).state == "done"
    finally:
        await loop.subagents.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [False, True])
async def test_child_model_preset_preserves_parent_runtime_and_task_ownership(tmp_path, wait):
    parent = MagicMock(spec=LLMProvider, provider_name="parent")
    parent.get_default_model.return_value = "parent-model"
    parent.generation = GenerationSettings()
    child = MagicMock(spec=LLMProvider, provider_name="child")
    child.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Child findings"))
    preset = ModelPresetConfig(model="child-model", max_tokens=2048, context_window_tokens=32000)
    loop = AgentLoop(
        MessageBus(), parent, tmp_path, model_presets={"local": preset},
        preset_snapshot_loader=lambda name: ProviderSnapshot(
            provider=child, model=preset.model, context_window_tokens=preset.context_window_tokens,
            signature=(name,), generation=preset.to_generation_settings(), model_preset=name,
        ),
    )
    runtime = loop.llm_runtime()
    ctx = RequestContext("websocket", "topic", session_key="websocket:topic", runtime=runtime)
    try:
        with request_context(ctx):
            missing = await loop.tools.execute("subagent", {
                "action": "create", "task": "Inspect the configuration", "model_preset": "missing",
            })
            assert is_tool_error_result(missing)
            assert "model_preset 'missing' not found" in missing
            assert "Available: local" in missing
            blank = await loop.tools.execute("subagent", {
                "action": "create", "task": "Inspect the configuration", "model_preset": " ",
            })
            assert is_tool_error_result(blank)
            assert "model_preset must be a non-empty string" in blank
            assert not loop.subagents.statuses_for_session(ctx.session_key)
            result = await loop.tools.execute("subagent", {
                "action": "create", "task": "Inspect the configuration", "model_preset": "local", "wait": wait,
            })
        if wait:
            assert result == "Child findings"
        else:
            assert "started" in result
            await settled(loop.subagents)
        status, = loop.subagents.statuses_for_session(ctx.session_key).values()
        assert status.state == "done"
        assert status.result == "Child findings"
        if not wait:
            notice = await asyncio.wait_for(loop.bus.consume_inbound(), timeout=2)
            assert notice.session_key == ctx.session_key
            assert notice.metadata["subagent_task_id"] == status.task_id
            assert "Child findings" in notice.content
        child.chat_stream_with_retry.assert_awaited_once()
        assert child.chat_stream_with_retry.call_args.kwargs["model"] == "child-model"
        assert child.chat_stream_with_retry.call_args.kwargs["max_tokens"] == 2048
        assert loop.llm_runtime() == runtime
        assert loop.provider is parent
    finally:
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("cancelled", [False, True], ids=["completed-source", "cancelled-source"])
async def test_follow_up_replays_owned_subagent_history_after_restart(tmp_path, wait, cancelled):
    (tmp_path / "note.txt").write_text("The launch code is violet", encoding="utf-8")
    owner = "websocket:topic"
    preset = ModelPresetConfig(model="local-model", context_window_tokens=32000)

    def open_loop(responses):
        parent = MagicMock(spec=LLMProvider, provider_name="parent")
        parent.get_default_model.return_value = "parent-model"
        parent.generation = GenerationSettings()
        child = MagicMock(spec=LLMProvider, provider_name="local")
        child.chat_stream_with_retry = AsyncMock(side_effect=responses)
        loop = AgentLoop(
            MessageBus(), parent, tmp_path, model_presets={"local": preset},
            preset_snapshot_loader=lambda name: ProviderSnapshot(
                provider=child, model=preset.model, context_window_tokens=preset.context_window_tokens,
                signature=(name,), generation=preset.to_generation_settings(), model_preset=name,
            ),
        )
        return loop, child

    read_note = LLMResponse(content=None, tool_calls=[ToolCallRequest("read", "read_file", {"path": "note.txt"})])
    loop, child = open_loop([read_note, LLMResponse(content="I read the launch note.")])
    awaiting_report = asyncio.Event()

    async def stall_before_report(**kwargs):
        if child.chat_stream_with_retry.await_count == 1:
            return read_note
        awaiting_report.set()
        await asyncio.Future()

    if cancelled:
        child.chat_stream_with_retry.side_effect = stall_before_report
    try:
        with request_context(RequestContext("websocket", "topic", session_key=owner, runtime=loop.llm_runtime())):
            result = await loop.tools.execute("subagent", {
                "action": "create", "model_preset": "local", "task": "Read the launch note", "wait": not cancelled,
            })
        if cancelled:
            assert "started" in result
            await asyncio.wait_for(awaiting_report.wait(), timeout=2)
            active, = loop.subagents.statuses_for_session(owner).values()
            status = await loop.subagents.cancel(active.task_id, owner)
            assert status.state == "cancelled"
        else:
            assert result == "I read the launch note."
        original, = loop.subagents.statuses_for_session(owner).values()
        _, history = loop.subagents.read_session(original.task_id, owner)
    finally:
        await loop.aclose()

    # If the earlier tool is rerun or its result is not replayed, the follow-up cannot see this value.
    (tmp_path / "note.txt").unlink()
    loop, child = open_loop([LLMResponse(content="Violet.")])
    follow_up = {"action": "follow_up", "task_id": original.task_id, "message": "What was the code?", "wait": wait}
    try:
        with request_context(RequestContext("websocket", "other", runtime=loop.llm_runtime())):
            denied = await loop.tools.execute("subagent", follow_up)
            assert is_tool_error_result(denied) and "task unavailable" in denied
        ctx = RequestContext("websocket", "topic", session_key=owner, runtime=loop.llm_runtime(), turn_id="new-turn")
        with request_context(ctx):
            response = await loop.tools.execute("subagent", follow_up)
        if wait:
            assert response == "Violet."
        else:
            assert "started" in response
            await settled(loop.subagents)
            notice = await asyncio.wait_for(loop.bus.consume_inbound(), timeout=2)
            assert notice.session_key == owner
            assert "Violet." in notice.content
        child.chat_stream_with_retry.assert_awaited_once()
        call = child.chat_stream_with_retry.call_args.kwargs
        assert call["model"] == "local-model"
        assert call["messages"][-1]["content"] == "What was the code?"
        assert any(m["role"] == "tool" and "The launch code is violet" in m["content"] for m in call["messages"])
        assert loop.model == "parent-model"
        task, = (t for t in loop.subagents.statuses_for_session(owner).values() if t.task_id != original.task_id)
        assert task.continued_from_task_id == original.task_id
        assert task.origin_turn_id == "new-turn"
        assert task.result == "Violet."
        original_after, unchanged = loop.subagents.read_session(original.task_id, owner)
        assert original_after.state == ("cancelled" if cancelled else "done")
        assert unchanged.messages == history.messages
    finally:
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_parent", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
async def test_sdk_ephemeral_delegation_never_persists_on_later_flush(
    loop_factory, existing_parent, streamed,
):
    loop = loop_factory()
    bot = Nanobot(loop)
    key = "sdk:ephemeral-delegation"
    parent = loop.sessions.get_or_create(key)
    if existing_parent:
        parent.add_message("user", "Earlier durable question")
        loop.sessions.save(parent)
    before = loop.sessions.read_session_file(key)
    secret = "private-delegation-marker"
    source = loop.workspace / f"{secret}.txt"
    source.write_text(secret * 1000, encoding="utf-8")
    loop.max_tool_result_chars = 2048
    responses = iter([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate", name="subagent", arguments={"action": "create", "task": secret},
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="inspect", name="read_file", arguments={"path": str(source)},
        )]),
        LLMResponse(content=f"Child findings: {secret}"),
        LLMResponse(content=f"Parent conclusion: {secret}"),
    ])
    requests = []

    async def respond(**kwargs):
        requests.append(current_request_context())
        return next(responses)

    loop.provider.provider_name = "test"
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=respond)
    logs = []
    sink = logger.add(lambda message: logs.append(message.record["message"]))
    consumer = asyncio.create_task(loop.run())
    try:
        await asyncio.sleep(0)
        assert loop._running
        if streamed:
            stream = await bot.run_streamed(secret, session_key=key, ephemeral=True)
            result = await stream.wait()
        else:
            result = await bot.run(secret, session_key=key, ephemeral=True)
        assert result.content == f"Parent conclusion: {secret}"
        assert all(ctx is not None and not ctx.log_content and not ctx.persist_session for ctx in requests)
        assert secret not in "\n".join(logs)
        assert loop.sessions.read_session_file(key) == before
        assert not list(loop.sessions.sessions_dir.glob("*.checkpoint.json"))
        assert not (loop.workspace / ".nanobot" / "tool-results").exists()
        assert loop.bus.inbound_size == 0

        loop.provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Durable answer"))
        await bot.run("Durable follow-up", session_key=key)
        saved = json.dumps(loop.sessions.read_session_file(key))
        assert "Durable answer" in saved
        assert ("Earlier durable question" in saved) is existing_parent
        assert secret not in saved
        assert [row["key"] for row in loop.sessions.list_sessions()] == [key]
    finally:
        loop.stop()
        try:
            # Bound cleanup without imposing a two-second disk/thread shutdown contract.
            await asyncio.wait_for(consumer, timeout=30)
        finally:
            logger.remove(sink)
            await bot.aclose()


@pytest.mark.asyncio
async def test_child_origin_uses_the_channel_turn_identity(loop_factory):
    loop = loop_factory()
    loop.provider.provider_name = "test"
    loop.turn_delivery_factory.route_policy = WebuiTurnRoutePolicy(loop.sessions)
    loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="delegate", name="subagent", arguments={"action": "create", "task": "Inspect config"},
        )]),
        LLMResponse(content="Child findings"),
        LLMResponse(content="Parent conclusion"),
    ])
    message = InboundMessage("websocket", "user", "origin-check", "Delegate this inspection",
                             metadata={"webui_turn_id": "visible-parent-turn"})
    try:
        await loop._process_message(message)
        status = next(iter(loop.subagents.statuses_for_session(message.session_key).values()))
        assert status.origin_turn_id == "visible-parent-turn"
    finally:
        await loop.subagents.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_invalidation", ["handles", "eviction"])
async def test_default_loop_keeps_slack_thread_for_late_subagent_result(
    tmp_path, monkeypatch, cache_invalidation,
):
    monkeypatch.setattr("nanobot.agent.loop._SUBAGENT_TERMINAL_WAIT_SECONDS", 0.01)
    # This checks real default-runtime routing and persistence, not disk latency.
    # Keep bounded waits without treating a loaded Windows runner as a routing failure.
    timeout = 30
    child_started, release_child, parent_idle = (asyncio.Event() for _ in range(3))
    parent_tool_results = []

    class Provider(LLMProvider):
        def __init__(self):
            super().__init__(provider_name="test")
            self.parent_calls = 0

        async def chat(self, messages, tools=None, **kwargs):
            if not any(tool.get("function", {}).get("name") == "subagent" for tool in tools or []):
                child_started.set()
                await release_child.wait()
                return LLMResponse(content="Child findings")
            self.parent_calls += 1
            if self.parent_calls == 1:
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="delegate", name="subagent",
                    arguments={"action": "create", "task": "Inspect config", "wait": False},
                )])
            if self.parent_calls == 2:
                parent_tool_results.extend(m["content"] for m in messages if m.get("role") == "tool")
            return LLMResponse(content="Parent reply" if self.parent_calls == 2 else "Late follow-up")

        def get_default_model(self):
            return "test-model"

    workspace = tmp_path / "workspace"
    sessions_root = tmp_path / "sessions"
    sessions = SessionManager(workspace, sessions_root=sessions_root)
    config = Config.model_validate({"agents": {"defaults": {
        "workspace": str(workspace), "model": "test-model", "sessionTtlMinutes": 0,
    }}})
    bus = MessageBus()
    loop = AgentLoop.from_config(
        config, bus, provider=Provider(), session_manager=sessions, tool_registry=ToolRegistry(),
    )
    session_key = "slack:C1:1234.5678"
    slack_route = {"thread_ts": "1234.5678", "channel_type": "channel"}
    thread_metadata = {"slack": {"thread_ts": "1234.5678"}}

    def observe_idle(event):
        if event.context.session_key == session_key and event.status == "idle":
            parent_idle.set()

    unsubscribe = bus.subscribe(observe_idle, TurnRunStatusChanged)

    async def next_reply(content):
        async with asyncio.timeout(timeout):
            while True:
                message = await bus.consume_outbound()
                if message.content == content and message.event is None:
                    return message

    consumer = asyncio.create_task(loop.run())
    try:
        await bus.publish_inbound(InboundMessage(
            channel="slack", sender_id="U1", chat_id="C1", content="Delegate this inspection",
            metadata={"slack": slack_route}, session_key_override=session_key,
        ))
        initial_reply = await next_reply("Parent reply")
        assert any("Don't duplicate its work." in result for result in parent_tool_results)
        await asyncio.wait_for(child_started.wait(), timeout=timeout)
        await asyncio.wait_for(parent_idle.wait(), timeout=timeout)
        saved = SessionManager(workspace, sessions_root=sessions_root).get_existing(session_key)
        assert saved is not None
        saved_route = saved.metadata["_compaction_route"]
        if cache_invalidation == "handles":
            await sessions.state.ensure_session_handles()
        else:
            for i in range(SESSION_CACHE_MAX_SIZE + 1):
                await sessions.state.get_header(f"telegram:other-{i}")
        release_child.set()
        late_reply = await next_reply("Late follow-up")

        assert (initial_reply.channel, initial_reply.chat_id) == ("slack", "C1")
        assert initial_reply.metadata["slack"] == slack_route
        assert saved_route == {"channel": "slack", "chat_id": "C1", "metadata": thread_metadata}
        assert (late_reply.channel, late_reply.chat_id) == ("slack", "C1")
        assert late_reply.metadata == {**thread_metadata, "response_input_ids": []}
    finally:
        release_child.set()
        loop.stop()
        try:
            await asyncio.wait_for(consumer, timeout=timeout)
        finally:
            unsubscribe()


@pytest.mark.asyncio
async def test_owner_canonical_fallback_and_absent_context(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], final_content="private"))
    task_id = await spawn(manager, ctx)
    receipt = await manager.send(task_id, "owner", "follow-up")
    snapshot = manager.statuses_for_session("owner")[task_id]
    snapshot.state = "cancelled"
    snapshot.owner = "other"
    snapshot.receipts.clear()
    current = manager.runtime_statuses()[task_id]
    assert current.state == "queued"
    assert current.owner == "owner"
    assert current.receipts == {receipt["message_id"]: "accepted"}
    await settled(manager)
    terminal = await manager.cancel(task_id, "owner")
    terminal.result = "tampered"
    terminal.receipts.clear()
    current = manager.statuses_for_session("owner")[task_id]
    assert current.result == "private"
    assert current.receipts == {receipt["message_id"]: "undelivered"}
    denied = await SubagentTool(manager).execute("cancel", task_id)
    assert is_tool_error_result(denied)
    for stranger in [RequestContext("websocket", "route", session_key="other"),
                     RequestContext("websocket", "route"), RequestContext("", "")]:
        for action in [None, "send", "cancel"]:
            assert await control(manager, stranger, task_id, action, "secret") == denied
    assert await control(manager, ctx, "missing") == denied
    same_owner = RequestContext("cli", "different", session_key="owner")
    assert json.loads(await control(manager, same_owner, task_id))["result"] == "private"
    fallback = RequestContext("cli", "fallback", runtime=ctx.runtime)
    fallback_id = await spawn(manager, fallback)
    await settled(manager)
    assert await control(manager, fallback, fallback_id, "cancel") == denied
    explicit_owner = RequestContext("cli", "fallback", session_key="cli:fallback")
    assert json.loads(await control(manager, explicit_owner, fallback_id))["state"] == "done"
    assert manager.statuses_for_session(None) == {}
    await manager.close()


@pytest.mark.asyncio
async def test_bounded_fifo_snapshot_and_terminal_receipts(tmp_path):
    manager, _, ctx = setup(tmp_path)
    entered, drain, drained, finish = (asyncio.Event() for _ in range(4))
    snapshots = []

    async def run(spec):
        entered.set()
        await drain.wait()
        snapshots.append(await spec.injection_callback())
        drained.set()
        await finish.wait()
        return AgentRunResult(messages=[], final_content="done")

    manager.runner.run = run
    task_id = await spawn(manager, ctx)
    await entered.wait()
    for invalid in ["", "  ", "x" * 8193, "界" * 2731]:
        assert is_tool_error_result(await control(manager, ctx, task_id, "send", invalid))
    receipts = []
    for i in range(manager.MAX_INBOX):
        receipt = json.loads(await control(manager, ctx, task_id, "send", str(i)))
        assert receipt["receipt"] == "accepted" and not receipt["delivered"]
        receipts.append(receipt["message_id"])
    assert is_tool_error_result(await control(manager, ctx, task_id, "send", "overflow"))
    drain.set()
    await drained.wait()
    assert snapshots == [[{"role": "user", "content": str(i)} for i in range(manager.MAX_INBOX)]]
    last = json.loads(await control(manager, ctx, task_id, "send", "late"))["message_id"]
    finish.set()
    await settled(manager)
    status = json.loads(await control(manager, ctx, task_id))
    assert status["state"] == "done"
    assert status["receipts"] == {**dict.fromkeys(receipts, "delivered"), last: "undelivered"}
    assert is_tool_error_result(await control(manager, ctx, task_id, "send", "no restart"))
    notice = await manager.bus.consume_inbound()
    assert notice.metadata["subagent_task_id"] == task_id
    assert notice.metadata["subagent_message_receipts"] == status["receipts"]
    assert last in notice.content
    assert manager.bus.inbound_size == 0
    await manager.close()


@pytest.mark.asyncio
async def test_lifetime_receipt_limit_rejects_without_silent_eviction(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.MAX_MESSAGES = 2
    entered = asyncio.Event()
    specs = []

    async def run(spec):
        specs.append(spec)
        entered.set()
        await asyncio.Event().wait()

    manager.runner.run = run
    task_id = await spawn(manager, ctx)
    await entered.wait()
    for text in ["one", "two"]:
        assert not is_tool_error_result(await control(manager, ctx, task_id, "send", text))
        await specs[0].injection_callback()
    assert is_tool_error_result(await control(manager, ctx, task_id, "send", "three"))
    await control(manager, ctx, task_id, "cancel")
    assert len(json.loads(await control(manager, ctx, task_id))["receipts"]) == 2
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("begun", [False, True])
async def test_queued_cancel_is_idempotent_and_never_admitted(tmp_path, begun):
    manager, _, ctx = setup(tmp_path, max_concurrent_subagents=1)
    entered = asyncio.Event()
    calls = []

    async def run(spec):
        calls.append(spec.initial_messages[-1]["content"])
        entered.set()
        await asyncio.Event().wait()

    manager.runner.run = run
    await spawn(manager, ctx, "occupy")
    await entered.wait()
    queued = await spawn(manager, ctx, "queued")
    accepted = json.loads(await control(manager, ctx, queued, "send", "pending"))
    if begun:
        await asyncio.sleep(0)
    a, b = await asyncio.gather(control(manager, ctx, queued, "cancel"),
                                control(manager, ctx, queued, "cancel"))
    assert json.loads(a) == json.loads(b)
    assert json.loads(a)["state"] == "cancelled"
    assert json.loads(a)["receipts"][accepted["message_id"]] == "undelivered"
    assert calls == ["occupy"]
    assert manager.bus.inbound_size == 1
    await manager.close()
    assert manager.bus.inbound_size == 1
    with request_context(ctx):
        assert is_tool_error_result(await SubagentTool(manager).execute(action="create", task="after close"))


@pytest.mark.asyncio
async def test_real_runner_injects_followup_without_durable_child_session(tmp_path):
    manager, provider, ctx = setup(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def chat(**kwargs):
        requests.append([dict(m) for m in kwargs["messages"]])
        if len(requests) == 1:
            entered.set()
            await release.wait()
        return LLMResponse(content="answer")

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    task_id = await spawn(manager, ctx)
    await entered.wait()
    ids = [json.loads(await control(manager, ctx, task_id, "send", text))["message_id"]
           for text in ["first update", "second update"]]
    release.set()
    await settled(manager)
    assert len(requests) == 2
    user_text = "\n".join(m["content"] for m in requests[-1] if m["role"] == "user")
    assert user_text.index("first update") < user_text.index("second update")
    assert json.loads(await control(manager, ctx, task_id))["receipts"] == dict.fromkeys(ids, "delivered")
    assert not (tmp_path / "sessions").exists()
    await manager.close()


@pytest.mark.asyncio
async def test_iteration_limit_returns_partial_findings_without_success_notice(tmp_path):
    manager, provider, ctx = setup(tmp_path, max_iterations=1)
    (tmp_path / "note.txt").write_text("evidence", encoding="utf-8")
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="Found the relevant file; verification is still pending.",
        tool_calls=[ToolCallRequest("read", "read_file", {"path": "note.txt"})],
    ))
    task_id = await spawn(manager, ctx)
    await settled(manager)
    status = json.loads(await control(manager, ctx, task_id))
    assert status["state"] == "incomplete"
    assert status["stop_reason"] == "max_iterations"
    assert status["partial"] is True
    assert status["result"] == "Found the relevant file; verification is still pending."
    notice = await manager.bus.consume_inbound()
    assert "completed successfully" not in notice.content
    assert notice.metadata["subagent_state"] == "incomplete"
    assert notice.metadata["subagent_partial"] is True
    assert notice.metadata["subagent_stop_reason"] == "max_iterations"
    with request_context(ctx):
        inline = await SubagentTool(manager).execute("create", task="verify", wait=True)
    assert is_tool_error_result(inline)
    assert "max_iterations" in inline
    assert "verification is still pending" in inline
    # A non-persistent run has no saved history, so no resume is advertised.
    assert "follow_up" not in inline
    assert "Task ID" not in inline
    await manager.close()


@pytest.mark.asyncio
async def test_iteration_limit_stops_are_resumable_with_saved_history_and_fresh_budget(tmp_path):
    manager, provider, ctx = setup(
        tmp_path, max_iterations=2, session_manager=SessionManager(tmp_path),
    )
    runtime = LLMRuntime.capture(
        provider, "test", context_window_tokens=128000, model_preset="default",
    )
    ctx = RequestContext(
        ctx.channel, ctx.chat_id, session_key=ctx.session_key, runtime=runtime,
    )
    (tmp_path / "note.txt").write_text("evidence", encoding="utf-8")
    log = tmp_path / "executed.log"
    calls = 0
    requests: list[list[dict]] = []
    tool = SubagentTool(manager, resolve_model_preset=lambda preset: runtime)

    async def chat(**kwargs):
        nonlocal calls
        calls += 1
        requests.append([dict(m) for m in kwargs["messages"]])
        if calls == 6:
            return LLMResponse(content="All steps finished.")
        if calls % 2 == 1:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                f"step{calls}", "exec", {"command": f"echo step{calls} >> {shlex.quote(str(log))}"},
            )])
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(
            f"note{calls}", "read_file", {"path": "note.txt"},
        )])

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    manager_closed = False
    restarted: SubagentManager | None = None
    try:
        task_id = await spawn(manager, ctx)
        await settled(manager)
        status = json.loads(await control(manager, ctx, task_id))
        assert status["state"] == "incomplete"
        assert status["stop_reason"] == "max_iterations"
        assert status["iteration"] == 1
        assert log.read_text(encoding="utf-8").splitlines() == ["step1"]

        # A follow-up starts a linked run that begins with its own full two-round budget.
        with request_context(ctx):
            first = await tool.execute(
                "follow_up", task_id=task_id,
                message="Continue with the remaining steps", wait=True,
            )
        assert is_tool_error_result(first)
        assert "max_iterations" in first
        second = next(
            t for t in manager.statuses_for_session(ctx.session_key).values()
            if t.continued_from_task_id == task_id
        )
        # The inline error names the stopped run so the caller can resume exactly this task.
        assert second.task_id in first
        assert "follow_up" in first
        carried = requests[2]
        tool_results = [m["content"] for m in carried if m.get("role") == "tool"]
        assert any("evidence" in result for result in tool_results)
        tool_call_names = [
            tc["function"]["name"]
            for m in carried if m.get("role") == "assistant" and m.get("tool_calls")
            for tc in m["tool_calls"]
        ]
        assert "exec" in tool_call_names and "read_file" in tool_call_names
        assert carried[-1] == {"role": "user", "content": "Continue with the remaining steps"}
        assert log.read_text(encoding="utf-8").splitlines() == ["step1", "step3"]
        second_status = json.loads(await control(manager, ctx, second.task_id))
        assert second_status["state"] == "incomplete"
        assert second_status["stop_reason"] == "max_iterations"
        assert second_status["iteration"] == 1

        # Repeated follow-ups keep receiving the full per-run budget: the host
        # restarts before this run, which spends one tool round and a final
        # answer inside its own two-round budget.
        await manager.close()
        manager_closed = True
        restarted = SubagentManager(
            workspace=tmp_path, bus=MessageBus(), max_tool_result_chars=16000,
            consolidator=MagicMock(spec=Consolidator),
            max_iterations=2, session_manager=SessionManager(tmp_path),
        )
        reopened_tool = SubagentTool(restarted, resolve_model_preset=lambda preset: runtime)
        with request_context(ctx):
            finished = await reopened_tool.execute(
                "follow_up", task_id=second.task_id, message="Finish the task", wait=True,
            )
        assert finished == "All steps finished."
        # The saved history and tool results survive the re-open and are replayed.
        reopened = requests[4]
        assert any("evidence" in m["content"] for m in reopened if m.get("role") == "tool")
        assert reopened[-1] == {"role": "user", "content": "Finish the task"}
        # No replay: every old tool call executed exactly once, in order.
        assert log.read_text(encoding="utf-8").splitlines() == ["step1", "step3", "step5"]
        third = next(
            t for t in restarted.statuses_for_session(ctx.session_key).values()
            if t.continued_from_task_id == second.task_id
        )
        assert third.state == "done"
        assert third.result == "All steps finished."

        # The original trace stays intact with its saved tool results.
        original_status, history = restarted.read_session(task_id, ctx.session_key)
        assert original_status.state == "incomplete"
        original_tools = [m["content"] for m in history.messages if m.get("role") == "tool"]
        assert any("evidence" in result for result in original_tools)
    finally:
        if not manager_closed:
            await manager.close()
        if restarted is not None:
            await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True])
async def test_cancel_preserves_completed_iteration_output_and_receipts(tmp_path, inline):
    manager, provider, ctx = setup(tmp_path)
    (tmp_path / "note.txt").write_text("evidence", encoding="utf-8")
    entered = asyncio.Event()
    calls = 0

    async def chat(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(content="Found a possible cause; not yet verified.", tool_calls=[
                ToolCallRequest("read", "read_file", {"path": "note.txt"}),
            ])
        entered.set()
        await asyncio.Event().wait()

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    if inline:
        execution = asyncio.create_task(manager.run_inline(
            task="task", session_key=ctx.session_key, runtime=ctx.runtime,
        ))
    else:
        task_id = await spawn(manager, ctx)
    await entered.wait()
    if inline:
        task_id = next(iter(manager.statuses_for_session(ctx.session_key)))
    receipt = json.loads(await control(manager, ctx, task_id, "send", "please verify"))
    status = json.loads(await control(manager, ctx, task_id, "cancel"))
    assert status["state"] == "cancelled"
    assert status["partial"] is True
    assert status["result"] == "Found a possible cause; not yet verified."
    assert status["receipts"][receipt["message_id"]] == "undelivered"
    if inline:
        result = await execution
        assert is_tool_error_result(result)
        assert result == f"Task cancelled.\nPartial result:\n{status['result']}"
        assert manager.bus.inbound_size == 0
    else:
        notice = await manager.bus.consume_inbound()
        assert status["result"] in notice.content
        assert notice.metadata["subagent_state"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_empty_final_response_is_a_failure(tmp_path):
    manager, provider, ctx = setup(tmp_path)
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content=""))
    task_id = await spawn(manager, ctx)
    await settled(manager)
    status = json.loads(await control(manager, ctx, task_id))
    assert status["state"] == "error"
    assert status["stop_reason"] == "empty_final_response"
    assert status["error"]
    notice = await manager.bus.consume_inbound()
    assert "completed successfully" not in notice.content
    assert notice.metadata["subagent_state"] == "error"
    await manager.close()


@pytest.mark.asyncio
async def test_follow_up_points_to_send_only_while_task_accepts_messages(tmp_path):
    manager, provider, ctx = setup(tmp_path, session_manager=SessionManager(tmp_path))
    release = asyncio.Event()

    async def chat(**_kwargs):
        await release.wait()
        return LLMResponse(content="")

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    try:
        task_id = await spawn(manager, ctx)
        with request_context(ctx):
            tool = SubagentTool(manager)
            active = await tool.execute("follow_up", task_id=task_id, message="Any updates?")
            assert is_tool_error_result(active)
            assert "use send" in active
            release.set()
            await asyncio.wait_for(settled(manager), timeout=2)
            failed = await tool.execute("follow_up", task_id=task_id, message="Any updates?")
            assert is_tool_error_result(failed)
            assert "current state: error" in failed
            assert "use send" not in failed
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_host_without_background_consumer_returns_the_child_result(tmp_path):
    manager, _, ctx = setup(tmp_path)
    ctx = RequestContext("cli", "oneshot", session_key="cli:oneshot", runtime=ctx.runtime,
                         can_receive_background_results=False)
    manager.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], final_content="verified"))
    with request_context(ctx):
        result = await SubagentTool(manager).execute("create", task="verify")
    assert result == "verified"
    assert manager.get_running_count() == 0
    assert manager.bus.inbound_size == 0
    await manager.close()


@pytest.mark.asyncio
async def test_provider_suppressed_cancellation_cannot_execute_returned_tools(tmp_path):
    manager, provider, ctx = setup(tmp_path)
    entered = asyncio.Event()

    async def chat(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="write", name="write_file", arguments={"path": "resurrected.txt", "content": "bad"},
            )])

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    task_id = await spawn(manager, ctx)
    await entered.wait()
    assert json.loads(await control(manager, ctx, task_id, "cancel"))["state"] == "cancelled"
    assert provider.chat_stream_with_retry.await_count == 1
    assert not (tmp_path / "resurrected.txt").exists()
    assert manager.bus.inbound_size == 1
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["done", "error", "cancel"])
async def test_real_task_process_cleanup_preserves_sibling_parent_and_work_products(tmp_path, ending):
    manager, _, ctx = setup(tmp_path)
    script = tmp_path / "worker.py"
    script.write_text("import sys\nprint('READY', flush=True)\nfor line in sys.stdin:\n print(line.strip(), flush=True)\n")
    argv = [sys.executable, "-u", str(script)]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    kwargs = {"command": command, "yield_time_ms": 100, "timeout": 60}
    if os.name == "nt":
        kwargs["shell"] = "cmd"
    events = {name: asyncio.Event() for name in ["one", "two"]}
    releases = {name: asyncio.Event() for name in events}
    exec_ids = {}
    processes = {}

    async def run(spec):
        name = spec.initial_messages[-1]["content"]
        written = await spec.tools.execute("write_file", {"path": f"{name}-output.txt", "content": name})
        assert not is_tool_error_result(written), written
        output = await spec.tools.execute("exec", kwargs)
        assert not is_tool_error_result(output), output
        exec_ids[name] = re.search(r"session_id: (\w+)", output).group(1)
        exec_manager = spec.tools.get("exec")._session_manager
        processes[name] = exec_manager._sessions[exec_ids[name]].process
        events[name].set()
        await releases[name].wait()
        if name == "one" and ending == "error":
            raise RuntimeError("task failed")
        return AgentRunResult(messages=[], final_content="finished")

    manager.runner.run = run
    parent_exec = ExecTool(working_dir=str(tmp_path), session_manager=manager._exec_session_manager)
    try:
        with request_context(ctx):
            output = await parent_exec.execute(**kwargs)
        parent_id = re.search(r"session_id: (\w+)", output).group(1)
        one = await spawn(manager, ctx, "one")
        two = await spawn(manager, ctx, "two")
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in events.values())), 15)
        one_exec_manager = manager._tasks[one].exec_manager
        if ending == "cancel":
            await control(manager, ctx, one, "cancel")
        else:
            releases["one"].set()
            await asyncio.gather(manager._running_tasks[one], return_exceptions=True)
        expected = {"cancel": "cancelled", "error": "error", "done": "done"}[ending]
        assert json.loads(await control(manager, ctx, one))["state"] == expected
        assert await asyncio.wait_for(processes["one"].wait(), timeout=5) is not None
        with pytest.raises(KeyError):
            await one_exec_manager.write(
                session_id=exec_ids["one"], chars=None, close_stdin=False, terminate=False,
                yield_time_ms=0, max_output_chars=1000, owner_session_key=f"subagent:{one}",
            )
        for session_id, owner, exec_manager in [
            (exec_ids["two"], f"subagent:{two}", manager._tasks[two].exec_manager),
            (parent_id, "owner", manager._exec_session_manager),
        ]:
            poll = await exec_manager.write(
                session_id=session_id, chars="SURVIVED\n", close_stdin=False, terminate=False,
                yield_time_ms=1000, max_output_chars=1000, owner_session_key=owner,
            )
            assert not poll.done
            assert "SURVIVED" in poll.output
        assert script.exists()
        assert await manager.cancel_by_session("owner") == 1
        assert manager.bus.inbound_size == 1
        parent_poll = await manager._exec_session_manager.write(
            session_id=parent_id, chars="STILL ALIVE\n", close_stdin=False, terminate=False,
            yield_time_ms=1000, max_output_chars=1000, owner_session_key="owner",
        )
        assert not parent_poll.done and "STILL ALIVE" in parent_poll.output
    finally:
        await manager.close()
    assert script.exists()
    assert (tmp_path / "one-output.txt").read_text() == "one"
    assert (tmp_path / "two-output.txt").read_text() == "two"


@pytest.mark.asyncio
async def test_terminal_retention_and_active_capacity_are_bounded(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.MAX_TERMINAL = 2
    manager.MAX_ACTIVE = 1
    entered, release = asyncio.Event(), asyncio.Event()

    async def run(spec):
        entered.set()
        await release.wait()
        return AgentRunResult(messages=[], final_content="done")

    manager.runner.run = run
    first = await spawn(manager, ctx)
    await entered.wait()
    with request_context(ctx):
        assert is_tool_error_result(await SubagentTool(manager).execute(action="create", task="overflow"))
    release.set()
    await settled(manager)
    for _ in range(3):
        await spawn(manager, ctx)
        await settled(manager)
    assert len(manager.runtime_statuses()) == 2
    assert len(manager.statuses_for_session("owner")) == 2
    assert is_tool_error_result(await control(manager, ctx, first))
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_completion_wins_cancel_race_and_rejects_late_messages(tmp_path, shutdown):
    manager, _, ctx = setup(tmp_path)
    manager.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], final_content="finished"))
    cleaning, release = asyncio.Event(), asyncio.Event()
    task_id = await spawn(manager, ctx)
    exec_manager = manager._tasks[task_id].exec_manager
    cleanup = exec_manager.close_all

    async def gated_cleanup():
        cleaning.set()
        await release.wait()
        return await cleanup()

    exec_manager.close_all = gated_cleanup
    message_id = json.loads(await control(manager, ctx, task_id, "send", "too late"))["message_id"]
    await cleaning.wait()
    assert json.loads(await control(manager, ctx, task_id))["state"] == "stopping"
    assert is_tool_error_result(await control(manager, ctx, task_id, "send", "rejected"))
    cancellation = asyncio.create_task(manager.close() if shutdown else control(manager, ctx, task_id, "cancel"))
    await asyncio.sleep(0)
    release.set()
    await cancellation
    await settled(manager)
    state = json.loads(await control(manager, ctx, task_id))
    assert state["state"] == "done"
    assert state["receipts"][message_id] == "undelivered"
    assert state["result"] == "finished"
    assert manager.bus.inbound_size == (0 if shutdown else 1)
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [False, True])
async def test_cancel_wins_over_late_execution_result(tmp_path, late_failure):
    manager, _, ctx = setup(tmp_path)
    entered, stopping, release = (asyncio.Event() for _ in range(3))

    async def run(spec):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopping.set()
            await release.wait()
            if late_failure:
                raise RuntimeError("late failure")
            return AgentRunResult(messages=[], final_content="not success")

    manager.runner.run = run
    task_id = await spawn(manager, ctx)
    await entered.wait()
    message_id = json.loads(await control(manager, ctx, task_id, "send", "pending"))["message_id"]
    cancelled = asyncio.create_task(control(manager, ctx, task_id, "cancel"))
    await stopping.wait()
    assert is_tool_error_result(await control(manager, ctx, task_id, "send", "late"))
    release.set()
    state = json.loads(await cancelled)
    assert state["state"] == "cancelled"
    assert state["receipts"][message_id] == "undelivered"
    assert state["result"] == "Task cancelled."
    assert json.loads(await control(manager, ctx, task_id, "cancel")) == state
    assert manager.bus.inbound_size == 1
    await manager.close()


@pytest.mark.asyncio
async def test_inline_caller_cancel_closes_child_without_notice(tmp_path):
    manager, _, ctx = setup(tmp_path)
    entered = asyncio.Event()

    async def run(spec):
        entered.set()
        await asyncio.Event().wait()

    manager.runner.run = run
    with request_context(ctx):
        inline = asyncio.create_task(SubagentTool(manager).execute(action="create", task="wait", wait=True))
    await entered.wait()
    inline.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inline
    assert [s.state for s in manager.statuses_for_session("owner").values()] == ["cancelled"]
    assert manager.get_running_count() == 0
    assert manager.bus.inbound_size == 0
    await manager.close()


@pytest.mark.asyncio
async def test_cleanup_failure_survives_status_eviction_and_shutdown_retries(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.MAX_TERMINAL = 1
    manager.runner.run = AsyncMock(return_value=AgentRunResult(
        messages=[], final_content="x" * (manager.max_tool_result_chars + 1),
    ))
    task_id = await spawn(manager, ctx)
    exec_manager = manager._tasks[task_id].exec_manager
    original_cleanup = exec_manager.close_all
    exec_manager.close_all = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    message_id = json.loads(await control(manager, ctx, task_id, "send", "pending"))["message_id"]
    await settled(manager)
    state = json.loads(await control(manager, ctx, task_id))
    assert state["state"] == "error"
    assert "cleanup failed" in state["error"]
    assert state["receipts"][message_id] == "undelivered"
    assert json.loads(await control(manager, ctx, task_id, "cancel")) == state
    exec_manager.close_all.assert_awaited_once()
    assert manager.bus.inbound_size == 1
    assert task_id in manager._tasks
    assert len(manager._tasks[task_id].outcome.result) == manager.max_tool_result_chars
    newer = await spawn(manager, ctx)
    await settled(manager)
    assert json.loads(await control(manager, ctx, newer))["state"] == "done"
    assert is_tool_error_result(await control(manager, ctx, task_id, "cancel"))
    manager.MAX_ACTIVE = 1
    with request_context(ctx):
        assert is_tool_error_result(await SubagentTool(manager).execute(action="create", task="over capacity"))
    exec_manager.close_all = original_cleanup
    await manager.close()
    assert not manager._tasks
    assert list(manager.runtime_statuses()) == [newer]


@pytest.mark.asyncio
async def test_tool_suppressed_cancellation_cannot_start_next_tool(tmp_path):
    manager, provider, ctx = setup(tmp_path)
    entered = asyncio.Event()
    calls = []

    class PauseTool(Tool):
        name = "pause"
        description = "Wait for a signal."
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            calls.append("pause")
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "suppressed"

    registry = ToolRegistry()
    registry.register(PauseTool())
    manager._build_tools = lambda **kwargs: registry
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=str(i), name="pause", arguments={}) for i in range(2)],
    ))
    task_id = await spawn(manager, ctx)
    await entered.wait()
    assert json.loads(await control(manager, ctx, task_id, "cancel"))["state"] == "cancelled"
    assert calls == ["pause"]
    assert provider.chat_stream_with_retry.await_count == 1
    await manager.close()


@pytest.mark.asyncio
async def test_long_running_task_receipts_survive_newer_task_retention(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.MAX_TERMINAL = 1
    entered, release = asyncio.Event(), asyncio.Event()

    async def run(spec):
        if spec.initial_messages[-1]["content"] == "old":
            entered.set()
            await release.wait()
        return AgentRunResult(messages=[], final_content="done")

    manager.runner.run = run
    old = await spawn(manager, ctx, "old")
    await entered.wait()
    message_id = json.loads(await control(manager, ctx, old, "send", "pending"))["message_id"]
    newer = await spawn(manager, ctx, "new")
    await manager._running_tasks[newer]
    release.set()
    await settled(manager)
    assert is_tool_error_result(await control(manager, ctx, newer))
    assert json.loads(await control(manager, ctx, old))["receipts"][message_id] == "undelivered"
    await manager.bus.consume_inbound()
    notice = await manager.bus.consume_inbound()
    assert notice.metadata["subagent_task_id"] == old
    assert notice.metadata["subagent_message_receipts"] == {message_id: "undelivered"}
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel", "session", "close", "inline"])
async def test_resistant_dependency_has_bounded_stop_and_no_late_result(tmp_path, operation):
    manager, provider, ctx = setup(tmp_path, max_concurrent_subagents=1)
    manager.CANCEL_WAIT_SECONDS = 0.02
    entered, cancelled, release = (asyncio.Event() for _ in range(3))

    async def chat(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            return LLMResponse(content="must not publish", tool_calls=[ToolCallRequest(
                id="late", name="write_file", arguments={"path": "late.txt", "content": "bad"},
            )])

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    inline = None
    if operation == "inline":
        with request_context(ctx):
            inline = asyncio.create_task(SubagentTool(manager).execute(action="create", task="wait", wait=True))
        await entered.wait()
        task_id = next(iter(manager.statuses_for_session("owner")))
    else:
        task_id = await spawn(manager, ctx)
        await entered.wait()
    record = manager._tasks[task_id]
    status = record.status
    try:
        if operation == "cancel":
            result = await asyncio.wait_for(control(manager, ctx, task_id, "cancel"), 1)
            assert json.loads(result)["state"] == "stopping"
        elif operation == "session":
            assert await asyncio.wait_for(manager.cancel_by_session("owner"), 1) == 1
        elif operation == "close":
            await asyncio.wait_for(manager.close(), 1)
        else:
            inline.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(inline, 1)
        assert cancelled.is_set()
        assert status.state == "stopping"
        assert manager.get_running_count() == 1
        assert manager._run_slots.locked()
        assert record.cleanup_task.done()
        assert record.exec_manager._closed
        assert is_tool_error_result(await control(manager, ctx, task_id, "send", "rejected"))
        assert json.loads(await control(manager, ctx, task_id, "cancel"))["state"] == "stopping"
        assert manager.bus.inbound_size == 0
    finally:
        release.set()
        await asyncio.wait_for(settled(manager), 2)
        await manager.close()
    assert status.state == "cancelled"
    assert not (tmp_path / "late.txt").exists()
    assert provider.chat_stream_with_retry.await_count == 1
    assert manager.get_running_count() == 0
    assert manager.bus.inbound_size == (1 if operation == "cancel" else 0)


@pytest.mark.asyncio
async def test_targeted_inline_cancel_does_not_cancel_waiting_parent(tmp_path):
    manager, _, ctx = setup(tmp_path)
    entered = asyncio.Event()

    async def run(spec):
        entered.set()
        await asyncio.Event().wait()

    manager.runner.run = run
    with request_context(ctx):
        parent = asyncio.create_task(SubagentTool(manager).execute(action="create", task="wait", wait=True))
    await entered.wait()
    task_id = next(iter(manager.statuses_for_session("owner")))
    await control(manager, ctx, task_id, "cancel")
    assert is_tool_error_result(await parent)
    assert not parent.cancelled()
    assert manager.bus.inbound_size == 0
    await manager.close()


@pytest.mark.asyncio
async def test_cleanup_wait_is_bounded_and_repeated_cancel_does_not_interrupt_it(tmp_path):
    manager, _, ctx = setup(tmp_path)
    manager.CANCEL_WAIT_SECONDS = 0.02
    manager.runner.run = AsyncMock(return_value=AgentRunResult(messages=[], final_content="done"))
    cleaning, release = asyncio.Event(), asyncio.Event()
    task_id = await spawn(manager, ctx)
    record = manager._tasks[task_id]
    status = record.status
    original = record.exec_manager.close_all

    async def cleanup():
        cleaning.set()
        await release.wait()
        return await original()

    record.exec_manager.close_all = cleanup
    await cleaning.wait()
    try:
        assert json.loads(await asyncio.wait_for(control(manager, ctx, task_id, "cancel"), 1))["state"] == "stopping"
        await asyncio.wait_for(manager.close(), 1)
        assert status.state == "stopping"
        assert manager.get_running_count() == 1
        assert not record.cleanup_task.done()
    finally:
        release.set()
        await asyncio.wait_for(settled(manager), 2)
    assert status.state == "done"
    assert manager.bus.inbound_size == 0


def test_subagent_schema_exposes_task_actions(tmp_path):
    manager, _, _ = setup(tmp_path)
    properties = SubagentTool(manager).parameters["properties"]
    assert properties["action"]["enum"] == ["create", "follow_up", "send", "cancel", "check"]
    assert "owner" not in properties
    tool = SubagentTool(manager)
    assert tool.validate_params({"action": "create", "task": "review"}) == []
    assert tool.validate_params({"action": "check"}) == []
    assert tool.validate_params({"action": "send", "task_id": "task", "message": "update"}) == []
    assert tool.validate_params({"action": "cancel", "task_id": "task"}) == []
