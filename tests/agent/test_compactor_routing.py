"""Routing for the optional compactor model preset.

``agents.defaults.compactor.modelOverride`` points context-pressure
compaction, provider-native local summarization, manual ``/compact``,
SDK session compaction, and idle archival at a named model preset.
Foreground turns and Dream keep their own runtimes.
"""

import asyncio
import re
from datetime import datetime, timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
)
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.events import ContextCompactionEvent, EventSink
from nanobot.llm_usage.types import LLMUsage
from nanobot.providers import factory
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderConversationState,
    ToolCallRequest,
)
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.providers.factory import ProviderSnapshot
from nanobot.sdk.clients import RuntimeClient
from nanobot.session.summary import (
    SUMMARY_CONTINUATION_TEXT,
    is_summary_checkpoint,
)

COMPACTOR_PRESET = ModelPresetConfig(
    model="compactor-model",
    context_window_tokens=200_000,
    max_tokens=4_096,
)
DREAM_PRESET = ModelPresetConfig(model="dream-model")


def _tool_definitions() -> list:
    return [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]


def _mock_provider(model: str, *, max_tokens: int = 4_096) -> MagicMock:
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = model
    provider.generation = GenerationSettings(max_tokens=max_tokens)
    provider.estimate_prompt_tokens.return_value = (100, "test-counter")
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="Checkpoint.", tool_calls=[]),
    )
    return provider


def _compactor_loader(provider: MagicMock, compactor_model: str = "compactor-model"):
    """Resolve named presets through the production preset-snapshot path."""

    def load_preset(name: str) -> ProviderSnapshot:
        if name == "compactor":
            return ProviderSnapshot(
                provider=provider,
                model=compactor_model,
                context_window_tokens=200_000,
                signature=("model_preset", "compactor", compactor_model),
                generation=GenerationSettings(max_tokens=4_096),
                model_preset="compactor",
            )
        if name == "dream":
            return ProviderSnapshot(
                provider=provider,
                model="dream-model",
                context_window_tokens=200_000,
                signature=("model_preset", "dream", "dream-model"),
                generation=GenerationSettings(max_tokens=4_096),
                model_preset="dream",
            )
        raise KeyError(f"unknown preset {name!r}")

    return load_preset


def _make_loop(
    tmp_path,
    *,
    context_window_tokens: int,
    fg_max_tokens: int = 0,
    session_ttl_minutes: int = 0,
    compactor: bool = True,
    dream: bool = False,
    shared_provider: bool = False,
    compactor_model: str = "compactor-model",
    next_turn_input_reserve_tokens: int = 0,
    min_verbatim_tail_tokens: int = 0,
):
    """Build a loop whose compactor preset resolves to a distinct provider."""
    foreground = _mock_provider("test-model", max_tokens=fg_max_tokens)
    presets = {}
    if compactor:
        presets["compactor"] = COMPACTOR_PRESET.model_copy(update={"model": compactor_model})
    if dream:
        presets["dream"] = DREAM_PRESET
    compactor_provider = (
        foreground if shared_provider else _mock_provider(compactor_model)
    ) if compactor else None
    loop = AgentLoop(
        bus=MessageBus(),
        provider=foreground,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=context_window_tokens,
        session_ttl_minutes=session_ttl_minutes,
        model_presets=presets or None,
        preset_snapshot_loader=(
            _compactor_loader(compactor_provider, compactor_model) if compactor else None
        ),
        compactor_model_preset="compactor" if compactor else None,
        dream_model_preset="dream" if dream else None,
        compactor_next_turn_input_reserve_tokens=next_turn_input_reserve_tokens,
        compactor_min_verbatim_tail_tokens=min_verbatim_tail_tokens,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    return foreground, loop


def _seed_history(loop: AgentLoop, key: str = "cli:test") -> None:
    session = loop.sessions.get_or_create(key)
    session.messages = [
        {"role": role, "content": f"old-{role}-{turn}"}
        for turn in range(6)
        for role in ("user", "assistant")
    ]
    loop.sessions.save(session)


def _pressure_estimates(provider: MagicMock) -> None:
    def estimate(messages, _tools, _model):
        contents = [str(message.get("content")) for message in messages]
        if contents and "SNIP" in contents[-1]:
            return 300, "test-counter"
        if any(content.startswith("old-") for content in contents):
            return 600, "test-counter"
        return 100, "test-counter"

    provider.estimate_prompt_tokens.side_effect = estimate


async def _run_pressure_turn(loop: AgentLoop, foreground: MagicMock) -> None:
    foreground.can_resume_conversation_state.return_value = False
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]
    _seed_history(loop)
    _pressure_estimates(foreground)
    await loop.process_direct("continue the task", session_key="cli:test")


def test_validator_accepts_compactor_model_preset() -> None:
    config = Config.model_validate({
        "modelPresets": {
            "compactor": {"model": "openai/gpt-4.1", "provider": "openai"},
        },
        "agents": {"defaults": {"compactor": {"modelOverride": "compactor"}}},
    })
    assert config.agents.defaults.compactor.model_override == "compactor"
    assert Config.model_validate({}).agents.defaults.compactor.model_override is None


def test_validator_rejects_unknown_compactor_model_preset() -> None:
    with pytest.raises(ValueError, match="Compactor model preset 'unknown' not found"):
        Config.model_validate({
            "agents": {"defaults": {"compactor": {"modelOverride": "unknown"}}},
        })


def test_validator_rejects_default_preset_name() -> None:
    with pytest.raises(ValueError, match="Compactor model preset 'default' not found"):
        Config.model_validate({
            "agents": {"defaults": {"compactor": {"modelOverride": "default"}}},
        })


def test_validator_reads_next_turn_input_reserve_tokens() -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"compactor": {"nextTurnInputReserveTokens": 512}}},
    })
    assert config.agents.defaults.compactor.next_turn_input_reserve_tokens == 512
    # Unset keeps the existing sizing (opt-in reserve).
    assert Config.model_validate({}).agents.defaults.compactor.next_turn_input_reserve_tokens == 0


def test_validator_rejects_negative_next_turn_input_reserve_tokens() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        Config.model_validate({
            "agents": {"defaults": {"compactor": {"nextTurnInputReserveTokens": -1}}},
        })


def test_validator_reads_min_verbatim_tail_tokens() -> None:
    config = Config.model_validate({
        "agents": {"defaults": {"compactor": {"minVerbatimTailTokens": 12_000}}},
    })
    assert config.agents.defaults.compactor.min_verbatim_tail_tokens == 12_000
    # Unset keeps the default 90k floor.
    assert Config.model_validate({}).agents.defaults.compactor.min_verbatim_tail_tokens == 90_000
    # 0 explicitly disables the floor (legacy full-prefix fallback).
    assert Config.model_validate({
        "agents": {"defaults": {"compactor": {"minVerbatimTailTokens": 0}}},
    }).agents.defaults.compactor.min_verbatim_tail_tokens == 0


@pytest.mark.parametrize("value", [-1, True, 1.5, "90000"])
def test_validator_rejects_invalid_min_verbatim_tail_tokens(value) -> None:
    with pytest.raises(ValueError, match="minVerbatimTailTokens"):
        Config.model_validate({
            "agents": {"defaults": {"compactor": {"minVerbatimTailTokens": value}}},
        })


def test_compactor_runtime_resolves_preset_without_changing_default(tmp_path) -> None:
    _, loop = _make_loop(tmp_path, context_window_tokens=200_000)

    runtime = loop.compactor_runtime()

    assert runtime is not None
    assert runtime.model == "compactor-model"
    assert runtime.model_preset == "compactor"
    assert loop.model == "test-model"
    assert loop.model_preset is None


def test_compactor_override_is_isolated_from_foreground_and_dream(tmp_path) -> None:
    _, loop = _make_loop(
        tmp_path, context_window_tokens=200_000, dream=True, shared_provider=True,
    )

    assert loop.compactor_runtime().model == "compactor-model"
    assert loop.dream_runtime().model == "dream-model"
    assert loop.llm_runtime().model == "test-model"
    selected, resumable, offers_tools = loop.select_compaction_runtime(loop.llm_runtime())
    assert selected.model == "compactor-model"
    # Same provider, different model: provider-private state is per-model, so
    # the compactor must summarize the transcript instead of resuming it, and
    # its request must not offer the task model's tool schemas.
    assert resumable is False
    assert offers_tools is False


async def test_pressure_compaction_uses_compactor_provider_with_its_limits(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=1_624, fg_max_tokens=100,
    )
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    compactor = loop.compactor_runtime().provider
    assert compactor is not foreground
    foreground.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=[]),
    )

    await _run_pressure_turn(loop, foreground)

    compactor.chat_stream_with_retry.assert_awaited_once()
    compactor_kwargs = compactor.chat_stream_with_retry.await_args.kwargs
    assert compactor_kwargs["model"] == "compactor-model"
    assert compactor_kwargs["max_tokens"] == 4_096
    # The configured compactor must not receive the task model's tool schemas,
    # and the request carries the explicit no-tools contract.
    assert compactor_kwargs["tools"] == []
    assert compactor_kwargs["tool_choice"] == "none"
    foreground.chat_stream_with_retry.assert_awaited_once()
    assert foreground.chat_stream_with_retry.await_args.kwargs["model"] == "test-model"
    assert foreground.chat_stream_with_retry.await_args.kwargs["max_tokens"] == 100
    # The turn itself keeps its own tools.
    assert foreground.chat_stream_with_retry.await_args.kwargs["tools"] == _tool_definitions()


async def test_pressure_compaction_default_uses_turn_runtime(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=1_624, fg_max_tokens=100, compactor=False,
    )
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    foreground.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="Current checkpoint.", tool_calls=[]),
        LLMResponse(content="done", tool_calls=[]),
    ])

    await _run_pressure_turn(loop, foreground)

    assert foreground.chat_stream_with_retry.await_count == 2
    for call in foreground.chat_stream_with_retry.await_args_list:
        assert call.kwargs["model"] == "test-model"
        assert call.kwargs["max_tokens"] == 100
    # Legacy no-override compaction still offers the session model's tools
    # without the no-tools contract.
    assert foreground.chat_stream_with_retry.await_args_list[0].kwargs["tools"] == _tool_definitions()
    assert foreground.chat_stream_with_retry.await_args_list[0].kwargs["tool_choice"] is None


async def _run_idle_check(loop: AgentLoop) -> None:
    loop._next_idle_compact_check_at = 0.0
    await loop._check_expired_sessions_if_due()
    tasks = list(loop._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


async def _seed_idle_session(loop: AgentLoop, key: str = "cli:idle") -> None:
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    session.updated_at = datetime.now() - timedelta(minutes=20)
    loop.sessions.save(session)


async def test_idle_archival_uses_compactor_provider(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=200_000, session_ttl_minutes=15,
    )
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    compactor = loop.compactor_runtime().provider

    await _seed_idle_session(loop)
    await _run_idle_check(loop)

    compactor.chat_stream_with_retry.assert_awaited_once()
    kwargs = compactor.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "compactor-model"
    assert kwargs["max_tokens"] == 4_096
    assert kwargs["tools"] == []
    assert kwargs["tool_choice"] == "none"
    foreground.chat_stream_with_retry.assert_not_awaited()
    loop.sessions.invalidate("cli:idle")
    refreshed = loop.sessions.get_or_create("cli:idle")
    assert refreshed.metadata["_last_summary"]["text"] == "Checkpoint."


async def test_idle_archival_default_uses_session_runtime(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path,
        context_window_tokens=200_000,
        compactor=False,
        session_ttl_minutes=15,
    )

    await _seed_idle_session(loop)
    await _run_idle_check(loop)

    foreground.chat_stream_with_retry.assert_awaited_once()
    assert foreground.chat_stream_with_retry.await_args_list[0].kwargs["model"] == "test-model"


def _native_state():
    state = SimpleNamespace(payload={"context_tokens": 100}, pending_messages=[])
    state.with_pending_messages = lambda messages: SimpleNamespace(
        payload=dict(state.payload), pending_messages=list(messages),
    )
    return state


async def _run_native_compaction(
    loop: AgentLoop, foreground: MagicMock,
) -> ContextCompactionState:
    foreground.can_resume_conversation_state.return_value = True
    accepted = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Inspect the source."},
        {"role": "assistant", "content": "ok"},
    ]
    turn_runtime = loop.llm_runtime()
    compaction_runtime, state_resumable, offers_tools = loop.select_compaction_runtime(turn_runtime)
    consolidate = partial(
        loop.consolidator.summarize_provider_compaction,
        runtime=compaction_runtime,
        session_key="cli:test",
        tools=loop.tools.get_definitions() if offers_tools else [],
        no_tool_contract=not offers_tools,
        persist=False,
        allow_provider_state=state_resumable,
    )
    compaction = ContextCompactionState.from_messages(accepted, consolidate, consolidate)
    config = ContextGovernanceConfig(
        provider=foreground, model="test-model", tools=loop.tools, workspace=None,
        session_key="cli:test", max_tool_result_chars=16_000,
        context_window_tokens=200_000, max_tokens=8_192,
    )
    state = ModelRequestState(
        config=config, compaction=compaction,
        conversation=ProviderConversationStateController(
            provider=foreground, model="test-model", messages=accepted,
            state=_native_state(), session_id="cli:test",
        ),
    )
    response = LLMResponse(
        content="done", tool_calls=[],
        provider_compaction_applied=True,
        provider_compaction_state=_native_state(),
        provider_compaction_scope="prior_context",
    )
    await ContextGovernor().summarize_provider_compaction(
        state, response, current_request_boundary=None,
    )
    return compaction


async def test_native_compaction_with_foreign_compactor_summarizes_transcript(tmp_path) -> None:
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    compactor = loop.compactor_runtime().provider
    assert compactor is not foreground
    selected, resumable, offers_tools = loop.select_compaction_runtime(loop.llm_runtime())
    assert selected.model == "compactor-model"
    assert resumable is False
    assert offers_tools is False

    compaction = await _run_native_compaction(loop, foreground)

    compactor.chat_stream_with_retry.assert_awaited_once()
    kwargs = compactor.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "compactor-model"
    # Local summarization of a native-compacted state must not offer the
    # task model's tool schemas to the configured compactor.
    assert kwargs["tools"] == []
    assert kwargs["tool_choice"] == "none"
    foreground.chat_stream_with_retry.assert_not_awaited()
    assert compaction.summary_checkpoint is not None
    # The transcript path produced a real LLM summary, not the raw fallback.
    assert compaction.summary_checkpoint.summary == "Checkpoint."


async def test_native_compaction_with_resumable_compactor_forbids_tools(tmp_path) -> None:
    """A configured compactor that can resume the provider state still
    summarizes tool-free.

    The native-state (resumable) request shape must carry the same no-tools
    contract as the full-replay shape, so a configured extra_body cannot
    re-enable tool calls on the replay. Provider state persists items, not
    tool schemas, so a tool-free request still replays the transcript.
    """
    foreground, loop = _make_loop(
        tmp_path,
        context_window_tokens=200_000,
        shared_provider=True,
        compactor_model="test-model",
    )
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    compactor = loop.compactor_runtime().provider
    assert compactor is foreground
    selected, resumable, offers_tools = loop.select_compaction_runtime(loop.llm_runtime())
    assert selected.model == "test-model"
    assert resumable is True
    assert offers_tools is False

    compaction = await _run_native_compaction(loop, foreground)

    kwargs = foreground.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "test-model"
    # The configured compactor summarizes the resumable state without the
    # task model's tool schemas and with the explicit no-tools contract.
    assert kwargs["tools"] == []
    assert kwargs["tool_choice"] == "none"
    assert compaction.summary_checkpoint is not None
    assert compaction.summary_checkpoint.summary == "Checkpoint."


async def test_native_compaction_default_preserves_provider_state_path(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=200_000, fg_max_tokens=4_096, compactor=False,
    )
    selected, resumable, offers_tools = loop.select_compaction_runtime(loop.llm_runtime())
    assert selected.model == "test-model"
    assert resumable is True
    assert offers_tools is True

    compaction = await _run_native_compaction(loop, foreground)

    foreground.chat_stream_with_retry.assert_awaited_once()
    assert foreground.chat_stream_with_retry.await_args.kwargs["model"] == "test-model"
    assert compaction.summary_checkpoint is not None
    assert compaction.summary_checkpoint.summary == "Checkpoint."


async def test_manual_compact_command_uses_compactor_preset(tmp_path) -> None:
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    await _seed_idle_session(loop, key="cli:manual")

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="manual", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    compactor = loop.compactor_runtime().provider
    compactor.chat_stream_with_retry.assert_awaited_once()
    kwargs = compactor.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "compactor-model"
    assert kwargs["max_tokens"] == 4_096
    # Regression: /compact must not offer the session model's tool schemas to
    # the configured compactor, which may start calling them instead of
    # summarizing and degrade to a raw checkpoint.
    assert kwargs["tools"] == []
    assert kwargs["tool_choice"] == "none"
    # The input budget estimate must not count the task model's tool schemas.
    assert compactor.estimate_prompt_tokens.call_args.args[1] == []
    foreground.chat_stream_with_retry.assert_not_awaited()
    loop.sessions.invalidate("cli:manual")
    refreshed = loop.sessions.get_or_create("cli:manual")
    assert refreshed.metadata["_last_summary"]["text"] == "Checkpoint."


async def test_manual_compact_command_default_uses_session_runtime(tmp_path) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=200_000, compactor=False,
    )
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    await _seed_idle_session(loop, key="cli:manual")

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="manual", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    foreground.chat_stream_with_retry.assert_awaited_once()
    kwargs = foreground.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "test-model"
    # Legacy no-override: the session model still receives its own tools
    # without the no-tools contract.
    assert kwargs["tools"] == _tool_definitions()
    assert kwargs["tool_choice"] is None


@pytest.mark.parametrize("trigger", ["manual", "scheduled_idle", "sdk"])
async def test_session_compaction_retains_a_replayable_tail(tmp_path, trigger) -> None:
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100, session_ttl_minutes=1,
    )
    compactor = loop.compactor_runtime().provider
    key = "cli:tail"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    session.add_message("assistant", "")  # Skipped in model history, but not in the raw index.
    session.add_message("user", "recent question")
    session.add_message("assistant", "", tool_calls=[{
        "id": "recent-call", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    session.add_message("tool", "recent result", tool_call_id="recent-call", name="read_file")
    session.add_message("assistant", "recent answer")
    expected_tail = session.get_history()[2:]
    session.updated_at = datetime.now() - timedelta(minutes=10)
    loop.sessions.save(session)

    scheduled = []
    try:
        if trigger == "manual":
            await loop._process_message(
                InboundMessage(channel="cli", sender_id="user", chat_id="tail", content="/compact"),
                runtime=loop.llm_runtime(),
            )
        elif trigger == "sdk":
            await RuntimeClient(loop).compact_session(key)
        else:
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert len(scheduled) == 1
            await scheduled.pop()
            # The preserved tail is still unarchived, but the idle scan must not compact it again.
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert not scheduled

        foreground.chat_stream_with_retry.assert_not_awaited()
        compactor.chat_stream_with_retry.assert_awaited_once()
        request = compactor.chat_stream_with_retry.await_args.kwargs
        assert request["tool_choice"] == "none"
        assert request["messages"][:-1] == [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        loop.sessions.invalidate(key)
        reloaded = loop.sessions.get_or_create(key)
        assert reloaded.last_archived == 3
        assert reloaded.metadata["_last_summary"]["text"] == "Checkpoint."
        assert reloaded.get_history() == expected_tail
    finally:
        for coro in scheduled:
            coro.close()
        await loop.aclose()


def _seed_floor_session(loop, key: str) -> list[dict]:
    """Same layout as the retained-tail fixture, idle so the scan can fire."""
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    session.add_message("assistant", "")  # Skipped in model history, but not in the raw index.
    session.add_message("user", "recent question")
    session.add_message("assistant", "", tool_calls=[{
        "id": "recent-call", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    session.add_message("tool", "recent result", tool_call_id="recent-call", name="read_file")
    session.add_message("assistant", "recent answer")
    full_history = session.get_history()
    session.updated_at = datetime.now() - timedelta(minutes=10)
    loop.sessions.save(session)
    return full_history


@pytest.mark.parametrize("trigger", ["manual", "scheduled_idle", "sdk"])
async def test_session_compaction_floor_skips_when_tail_below_floor(
    tmp_path, trigger, monkeypatch,
) -> None:
    # Deterministic per-message cost: the largest legal tail (five messages at
    # the "old answer" assistant boundary) is 500 tokens, so a 501-token floor
    # cannot be met by any legal user or assistant cut.
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 100,
    )
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100,
        session_ttl_minutes=1, min_verbatim_tail_tokens=501,
    )
    compactor = loop.compactor_runtime().provider
    key = "cli:floor-skip"
    full_history = _seed_floor_session(loop, key)

    scheduled = []
    try:
        if trigger == "manual":
            reply = await loop._process_message(
                InboundMessage(channel="cli", sender_id="user", chat_id="floor-skip", content="/compact"),
                runtime=loop.llm_runtime(),
            )
            assert reply is not None
            assert reply.content == "Context unchanged: nothing to compact while keeping the configured minimum of recent history."
            assert loop.bus.outbound_size == 0  # No compaction lifecycle events.
        elif trigger == "sdk":
            await RuntimeClient(loop).compact_session(key)
        else:
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert len(scheduled) == 1
            await scheduled.pop()
            # Reload the skip decision before the next idle scan.
            loop.sessions.invalidate(key)
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert not scheduled

        foreground.chat_stream_with_retry.assert_not_awaited()
        compactor.chat_stream_with_retry.assert_not_awaited()
        loop.sessions.invalidate(key)
        reloaded = loop.sessions.get_or_create(key)
        # A normal no-op: the history is preserved in full, nothing is archived
        # or summarized.
        assert reloaded.last_archived == 0
        assert reloaded.get_history() == full_history
        assert "_last_summary" not in reloaded.metadata

        if trigger == "scheduled_idle":
            # Resetting the policy must not inherit a skip from the old floor.
            loop.compactor_min_verbatim_tail_tokens = 0
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert len(scheduled) == 1
            await scheduled.pop()
            compactor.chat_stream_with_retry.assert_awaited_once()
    finally:
        for coro in scheduled:
            coro.close()
        await loop.aclose()


@pytest.mark.parametrize("floor", [499, 500])
async def test_session_compaction_floor_proceeds_when_tail_meets_floor(
    tmp_path, monkeypatch, floor,
) -> None:
    # The oldest legal cut now leaves a 500-token assistant-leading tail.
    # Equality is allowed when the overall history is above the floor.
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 100,
    )
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100,
        min_verbatim_tail_tokens=floor,
    )
    compactor = loop.compactor_runtime().provider
    key = "cli:floor-ok"
    full_history = _seed_floor_session(loop, key)
    expected_tail = full_history[1:]

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="floor-ok", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    foreground.chat_stream_with_retry.assert_not_awaited()
    compactor.chat_stream_with_retry.assert_awaited_once()
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.last_archived == 1
    assert reloaded.metadata["_last_summary"]["text"] == "Checkpoint."
    assert reloaded.get_history() == expected_tail
    await loop.aclose()


@pytest.mark.parametrize("trigger", ["manual", "sdk"])
async def test_session_compaction_below_floor_logs_no_op_with_estimate(tmp_path, monkeypatch, trigger) -> None:
    # The six replayable messages cost exactly the 600-token floor, so the
    # skip is a floor no-op carrying its estimate, not a "no compaction fit"
    # rejection, and the log names the trigger.
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 100,
    )
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100,
        min_verbatim_tail_tokens=600,
    )
    compactor = loop.compactor_runtime().provider
    key = "cli:floor-noop"
    _seed_floor_session(loop, key)

    logs: list[str] = []
    sink = logger.add(logs.append, level="INFO", format="{message}")
    try:
        if trigger == "manual":
            await loop._process_message(
                InboundMessage(channel="cli", sender_id="user", chat_id="floor-noop", content="/compact"),
                runtime=loop.llm_runtime(),
            )
        else:
            assert await RuntimeClient(loop).compact_idle_session(key) == ""
        assert any(
            f"Skipping session compaction for {key} ({trigger}): est. 600 replayable tokens, "
            "at or below the configured 600-token verbatim-tail floor; nothing to archive"
            in line for line in logs
        )
    finally:
        logger.remove(sink)
        await loop.aclose()

    foreground.chat_stream_with_retry.assert_not_awaited()
    compactor.chat_stream_with_retry.assert_not_awaited()


async def test_sdk_compaction_maps_trigger_to_no_wire_reason(tmp_path) -> None:
    # "sdk" is a logging-only trigger: lifecycle events keep the
    # client-facing CompactionReason set, so the wire events carry no reason.
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    await _seed_idle_session(loop, key="cli:sdk-events")
    session = loop.sessions.get_or_create("cli:sdk-events")
    runtime, tools, no_tool_contract, replay_config = await loop.compaction_runtime_for_session(session)
    events: list[ContextCompactionEvent] = []

    async def collect(event) -> None:
        events.append(event)

    try:
        summary = await loop.consolidator.compact_idle_session(
            "cli:sdk-events",
            runtime=runtime,
            tools=tools,
            no_tool_contract=no_tool_contract,
            replay_config=replay_config,
            events=EventSink(publish=collect),
            reason="sdk",
        )
    finally:
        await loop.aclose()

    assert summary == "Checkpoint."
    assert [event.phase for event in events] == ["started", "succeeded"]
    assert all(event.reason is None for event in events)


@pytest.mark.parametrize("trigger", ["manual", "scheduled_idle"])
async def test_session_compaction_success_logs_replay_prompt_estimates(tmp_path, monkeypatch, trigger) -> None:
    # Byte-sized probes make the before/after replay estimates deterministic:
    # both views are sized with the same replay model, wrapper, tools, and
    # projection, so the logged delta is the archived prefix minus the
    # summary and continuation the replay gains.
    def cost(message):
        return 4 + sum(len(str(message.get(field, "")).encode()) for field in (
            "content", "tool_calls", "tool_call_id", "name",
        ))

    for module in ("context_governance", "memory"):
        monkeypatch.setattr(f"nanobot.agent.{module}.estimate_message_tokens", cost)
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=120_000, fg_max_tokens=100,
        session_ttl_minutes=1,
    )
    foreground.estimate_prompt_tokens.side_effect = lambda messages, tools, model: (
        sum(map(cost, messages)), "synthetic-byte-counter",
    )
    key = "cli:estimates"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question " + "x" * 4_000)
    session.add_message("assistant", "old answer " + "y" * 4_000)
    session.add_message("user", "recent question")
    session.add_message("assistant", "recent answer")
    session.updated_at = datetime.now() - timedelta(minutes=10)
    loop.sessions.save(session)

    scheduled = []
    logs: list[str] = []
    sink = logger.add(logs.append, level="INFO", format="{message}")
    try:
        if trigger == "manual":
            await loop._process_message(
                InboundMessage(channel="cli", sender_id="user", chat_id="estimates", content="/compact"),
                runtime=loop.llm_runtime(),
            )
        else:
            await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
            assert len(scheduled) == 1
            await scheduled.pop()
        reason = "idle" if trigger == "scheduled_idle" else "manual"
        [line] = [line for line in logs if line.startswith(f"Session compact for {key} ({reason}):")]
        match = re.search(
            r"archived=\d+, replay_tail=\d+, retained=\d+, summary=True, stages=1, "
            r"est\. replay prompt tokens (\d+) -> (\d+), saved (\d+) tokens \((\d+)%\)$",
            line,
        )
        assert match
        before, after, saved, pct = map(int, match.groups())
        assert before - after == saved > 0
        assert abs(pct - saved * 100 / before) < 1
        loop.sessions.invalidate(key)
        reloaded = loop.sessions.get_or_create(key)
        assert reloaded.metadata["_last_summary"]["text"] == "Checkpoint."
    finally:
        for coro in scheduled:
            coro.close()
        logger.remove(sink)
        await loop.aclose()


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_compaction_success_log_survives_probe_failure(tmp_path, monkeypatch, phase) -> None:
    # A failed replay sizing probe (before publication or after the committed
    # checkpoint) drops the metric but must not flip the reported outcome.
    def cost(message):
        return 4 + sum(len(str(message.get(field, "")).encode()) for field in (
            "content", "tool_calls", "tool_call_id", "name",
        ))

    for module in ("context_governance", "memory"):
        monkeypatch.setattr(f"nanobot.agent.{module}.estimate_message_tokens", cost)
    foreground, loop = _make_loop(tmp_path, context_window_tokens=120_000, fg_max_tokens=100)
    foreground.estimate_prompt_tokens.side_effect = lambda messages, tools, model: (
        sum(map(cost, messages)), "synthetic-byte-counter",
    )

    def probe(self, session, provider, model):
        # The live pre-compaction session has no committed summary; the
        # post-compaction projection carries one, so it marks the phase.
        if (session.metadata.get("_last_summary") is not None) == (phase == "after"):
            raise RuntimeError("probe failed")
        return sum(map(cost, session.get_text_history())), "synthetic-byte-counter"

    monkeypatch.setattr("nanobot.agent.memory.Consolidator._session_prompt_probe", probe)
    key = "cli:probe-failure"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question " + "x" * 4_000)
    session.add_message("assistant", "old answer " + "y" * 4_000)
    session.add_message("user", "recent question")
    session.add_message("assistant", "recent answer")
    loop.sessions.save(session)

    logs: list[str] = []
    sink = logger.add(logs.append, level="INFO", format="{message}")
    try:
        await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="probe-failure", content="/compact"),
            runtime=loop.llm_runtime(),
        )
        [line] = [entry for entry in logs if entry.startswith(f"Session compact for {key} (manual):")]
        assert line.rstrip().endswith("summary=True, stages=1")
        assert "est. replay prompt tokens" not in line
    finally:
        logger.remove(sink)
        await loop.aclose()

    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.metadata["_last_summary"]["text"] == "Checkpoint."


@pytest.mark.parametrize(("floor", "reserve", "compactor_prompt_tokens"), [
    (600, 0, 100),  # Existing replayable history is exactly at the floor.
    (601, 0, 100),  # The skipped placeholder must not count as history.
    (400, 100_000, 100),  # No projected tail can fit alongside the next turn.
    (400, 0, 300_000),  # The actual compactor request is over budget.
])
async def test_sdk_compaction_cannot_keep_floor(tmp_path, monkeypatch, floor, reserve, compactor_prompt_tokens):
    monkeypatch.setattr("nanobot.agent.context_governance.estimate_message_tokens", lambda message: 100)
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100,
        min_verbatim_tail_tokens=floor, next_turn_input_reserve_tokens=reserve,
    )
    try:
        compactor = loop.compactor_runtime().provider
        compactor.estimate_prompt_tokens.side_effect = lambda messages, *_: (
            compactor_prompt_tokens if len(messages) > 1 else 100, "test-counter",
        )
        key = "cli:floor-impossible"
        full_history = _seed_floor_session(loop, key)

        assert await RuntimeClient(loop).compact_idle_session(key) == ""

        foreground.chat_stream_with_retry.assert_not_awaited()
        compactor.chat_stream_with_retry.assert_not_awaited()
        loop.sessions.invalidate(key)
        reloaded = loop.sessions.get_or_create(key)
        assert reloaded.last_archived == 0
        assert reloaded.get_history() == full_history
    finally:
        await loop.aclose()


async def test_session_compaction_retries_bounded_on_projected_budget_mismatch(tmp_path, monkeypatch):
    """A projected replay overrun never makes session sizing scan unbounded.

    The projected estimate is a constant above the foreground input budget,
    so every legal cut is rejected; the bounded retry probes only a small
    number of the oldest candidates instead of all of them, and no
    summarizer call is issued.
    """
    monkeypatch.setattr("nanobot.agent.context_governance.estimate_message_tokens", lambda _: 100)
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=12_000, fg_max_tokens=100, min_verbatim_tail_tokens=400,
    )
    try:
        key = "cli:projection-mismatch"
        session = loop.sessions.get_or_create(key)
        for _ in range(100):
            session.add_message("user", "short question")
            session.add_message("assistant", "short answer")
        loop.sessions.save(session)
        projected_sizes = []

        def estimate(messages, *_):
            if any(message.get("content") == "short question" for message in messages):
                projected_sizes.append(len(messages))
                return (12_000, "test-counter")
            return (100, "test-counter")

        foreground.estimate_prompt_tokens.side_effect = estimate
        logs: list[str] = []
        sink = logger.add(logs.append, level="INFO", format="{message}")
        try:
            assert await RuntimeClient(loop).compact_idle_session(key) == ""
        finally:
            logger.remove(sink)

        # Many legal tail candidates existed, but only the bounded oldest set
        # was projected.
        assert len(projected_sizes) == 3
        assert projected_sizes[0] > 10
        # The skip line consumes the replay_over_budget diagnostics instead of
        # a second planner log.
        assert any(
            re.search(
                r"Skipping session compaction for " + re.escape(key) + r" \(sdk\): "
                r"no projected replay fit within the foreground input budget "
                r"\(last: \d+ > \d+ tokens including summary and reserve; 3 request\(s\) probed\)",
                line,
            )
            for line in logs
        )
        foreground.chat_stream_with_retry.assert_not_awaited()
        loop.compactor_runtime().provider.chat_stream_with_retry.assert_not_awaited()
        assert (await loop.sessions.state.get(key)).last_archived == 0
    finally:
        await loop.aclose()


async def test_session_compaction_reserve_falls_back_to_full_prefix(tmp_path) -> None:
    """A next-turn reserve larger than the replay headroom keeps the full-prefix path.

    The same session retains a verbatim tail with the reserve unset (see
    ``test_session_compaction_retains_a_replayable_tail``); charging the
    reserve against the foreground replay headroom leaves no tail that also
    holds the fresh input, so compaction summarizes everything instead.
    """
    foreground, loop = _make_loop(
        tmp_path,
        context_window_tokens=12_000,
        fg_max_tokens=100,
        next_turn_input_reserve_tokens=100_000,
    )
    key = "cli:reserve"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    session.add_message("assistant", "")  # Skipped in model history, but not in the raw index.
    session.add_message("user", "recent question")
    session.add_message("assistant", "", tool_calls=[{
        "id": "recent-call", "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }])
    session.add_message("tool", "recent result", tool_call_id="recent-call", name="read_file")
    session.add_message("assistant", "recent answer")
    loop.sessions.save(session)

    runtime, _tools, _no_tools, replay_config = await loop.compaction_runtime_for_session(session)
    assert replay_config.next_turn_input_reserve_tokens == 100_000

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="reserve", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    compactor = loop.compactor_runtime().provider
    compactor.chat_stream_with_retry.assert_awaited_once()
    foreground.chat_stream_with_retry.assert_not_awaited()
    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    # Full prefix: no retained verbatim tail survives the compaction.
    assert reloaded.last_archived == len(session.messages)
    assert reloaded.metadata["_last_summary"]["text"] == "Checkpoint."
    assert reloaded.get_history() == []
    await loop.aclose()


async def test_sdk_compact_session_uses_compactor_preset(tmp_path) -> None:
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    session = loop.sessions.get_or_create("cli:sdk")
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    loop.sessions.save(session)

    await RuntimeClient(loop).compact_session("cli:sdk")

    compactor = loop.compactor_runtime().provider
    compactor.chat_stream_with_retry.assert_awaited_once()
    kwargs = compactor.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "compactor-model"
    assert kwargs["tools"] == []
    assert kwargs["tool_choice"] == "none"
    foreground.chat_stream_with_retry.assert_not_awaited()


async def test_sdk_compact_idle_session_uses_compactor_preset(tmp_path) -> None:
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    session = loop.sessions.get_or_create("cli:sdk")
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    loop.sessions.save(session)

    summary = await RuntimeClient(loop).compact_idle_session("cli:sdk")

    compactor = loop.compactor_runtime().provider
    compactor.chat_stream_with_retry.assert_awaited_once()
    assert compactor.chat_stream_with_retry.await_args.kwargs["model"] == "compactor-model"
    foreground.chat_stream_with_retry.assert_not_awaited()
    assert summary == "Checkpoint."


async def test_manual_compact_unsolicited_compactor_tool_call_falls_back_to_raw_once(tmp_path) -> None:
    """A tool-free compactor request gets the raw checkpoint, not a re-echo.

    When the compactor answers the tool-free summarization request with a tool
    call, the archiver must not replay that call transcript (which previously
    produced a second tool call and only then a raw checkpoint). It falls back
    to the raw checkpoint immediately, preserving the archived evidence.
    """
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    loop.tools.get_definitions = MagicMock(return_value=_tool_definitions())
    compactor = loop.compactor_runtime().provider
    compactor.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="call-1", name="read_file", arguments={"path": "x"})],
        finish_reason="tool_calls",
    ))
    await _seed_idle_session(loop, key="cli:manual")

    await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="manual", content="/compact"),
        runtime=loop.llm_runtime(),
    )

    compactor.chat_stream_with_retry.assert_awaited_once()
    assert compactor.chat_stream_with_retry.await_args.kwargs["tools"] == []
    loop.sessions.invalidate("cli:manual")
    refreshed = loop.sessions.get_or_create("cli:manual")
    summary = refreshed.metadata["_last_summary"]["text"]
    assert summary.startswith("[RAW]")
    assert "old question" in summary
    assert "old answer" in summary
    foreground.chat_stream_with_retry.assert_not_awaited()


async def test_new_background_archive_keeps_session_runtime_and_tools(tmp_path) -> None:
    """The /new background archive is not a compactor request: session model + tools."""
    foreground, loop = _make_loop(tmp_path, context_window_tokens=200_000)
    await _seed_idle_session(loop, key="cli:newchat")

    scheduled = []
    loop.schedule_background = scheduled.append  # type: ignore[method-assign]
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="newchat", content="/new")
    response = await loop._process_message(msg, runtime=loop.llm_runtime())

    assert response is not None
    assert len(scheduled) == 1
    await scheduled[0]

    foreground.chat_stream_with_retry.assert_awaited_once()
    kwargs = foreground.chat_stream_with_retry.await_args.kwargs
    assert kwargs["model"] == "test-model"
    # Legacy /new archival still runs on the session model with its tools
    # and without the no-tools contract.
    assert kwargs["tools"], "/new archival must offer the session model's tools"
    assert kwargs["tool_choice"] is None
    assert any(
        tool.get("function", {}).get("name") == "read_file"
        for tool in kwargs["tools"]
    )


async def test_from_config_routes_compaction_to_distinct_local_endpoint(tmp_path, monkeypatch) -> None:
    """Real Config -> preset/provider factory; only the transport is mocked."""
    config = Config.model_validate({
        "agents": {"defaults": {
            "workspace": str(tmp_path / "workspace"),
            "provider": "foreground",
            "model": "qwen/foreground",
            "contextWindowTokens": 1_624,
            "maxTokens": 100,
            "compactor": {"modelOverride": "compactor"},
        }},
        "providers": {
            "foreground": {"apiKey": "foreground-key", "apiBase": "http://localhost:8081/v1"},
            "local-lfm3": {"apiKey": "local-key", "apiBase": "http://localhost:8082/v1"},
        },
        "modelPresets": {
            "compactor": {
                "provider": "local-lfm3",
                "model": "lfm3-1.7b-int8",
                "maxTokens": 2_048,
                "contextWindowTokens": 16_384,
            },
        },
    })
    created: list[MagicMock] = []

    def construct(owned, *, preset, model=None):
        factory.validate_provider_setup(owned, preset=preset, model=model)
        provider = MagicMock(spec=LLMProvider)
        provider._image_input_error.return_value = None
        provider.generation = preset.to_generation_settings()
        provider.supports_images = preset.supports_images
        provider.provider_name = owned.get_provider_name(preset=preset)
        provider.get_default_model.return_value = model or preset.model
        provider.estimate_prompt_tokens = MagicMock(return_value=(100, "test-counter"))
        provider.chat_stream_with_retry = AsyncMock(
            return_value=LLMResponse(content="Checkpoint.", tool_calls=[]),
        )
        provider.can_resume_conversation_state.return_value = False
        provider.aclose = AsyncMock()
        created.append(provider)
        return provider

    monkeypatch.setattr(factory, "_make_provider_core", construct)
    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry())
    try:
        foreground = loop.llm_runtime()
        compactor = loop.compactor_runtime()
        assert foreground.model == "qwen/foreground"
        assert compactor.model == "lfm3-1.7b-int8"
        assert compactor.provider is not foreground.provider
        assert compactor.generation.max_tokens == 2_048
        assert compactor.context_window_tokens == 16_384
        # The real config resolution keeps the two endpoints distinct.
        assert config.get_api_base("qwen/foreground") == "http://localhost:8081/v1"
        assert config.get_api_base(
            "lfm3-1.7b-int8", preset=config.model_presets["compactor"],
        ) == "http://localhost:8082/v1"

        # A pressure turn keeps the foreground model; compaction goes to the
        # local endpoint with the compactor preset's budget.
        loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]
        _seed_history(loop)
        _pressure_estimates(foreground.provider)
        await loop.process_direct("continue the task", session_key="cli:test")

        compactor_provider = compactor.provider
        compactor_provider.chat_stream_with_retry.assert_awaited_once()
        compactor_kwargs = compactor_provider.chat_stream_with_retry.await_args.kwargs
        assert compactor_kwargs["model"] == "lfm3-1.7b-int8"
        assert compactor_kwargs["max_tokens"] == 2_048
        foreground.provider.chat_stream_with_retry.assert_awaited_once()
        assert foreground.provider.chat_stream_with_retry.await_args.kwargs["model"] == "qwen/foreground"
    finally:
        await loop.aclose()


async def test_from_config_wires_next_turn_input_reserve(tmp_path, monkeypatch) -> None:
    config = Config.model_validate({
        "agents": {"defaults": {
            "workspace": str(tmp_path / "workspace"),
            "provider": "foreground",
            "model": "qwen/foreground",
            "compactor": {"nextTurnInputReserveTokens": 512},
        }},
        "providers": {"foreground": {"apiKey": "key", "apiBase": "http://localhost:8081/v1"}},
    })

    def construct(owned, *, preset, model=None):
        factory.validate_provider_setup(owned, preset=preset, model=model)
        provider = MagicMock(spec=LLMProvider)
        provider._image_input_error.return_value = None
        provider.generation = preset.to_generation_settings()
        provider.supports_images = preset.supports_images
        provider.provider_name = owned.get_provider_name(preset=preset)
        provider.get_default_model.return_value = model or preset.model
        provider.estimate_prompt_tokens = MagicMock(return_value=(100, "test-counter"))
        provider.chat_stream_with_retry = AsyncMock(
            return_value=LLMResponse(content="Checkpoint.", tool_calls=[]),
        )
        provider.can_resume_conversation_state.return_value = False
        provider.aclose = AsyncMock()
        return provider

    monkeypatch.setattr(factory, "_make_provider_core", construct)
    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry())
    try:
        assert loop.compactor_next_turn_input_reserve_tokens == 512
        session = loop.sessions.get_or_create("cli:reserve")
        _runtime, _tools, _no_tools, replay_config = (
            await loop.compaction_runtime_for_session(session)
        )
        assert replay_config.next_turn_input_reserve_tokens == 512
    finally:
        await loop.aclose()


async def test_from_config_wires_min_verbatim_tail(tmp_path, monkeypatch) -> None:
    config = Config.model_validate({
        "agents": {"defaults": {
            "workspace": str(tmp_path / "workspace"),
            "provider": "foreground",
            "model": "qwen/foreground",
            "compactor": {"minVerbatimTailTokens": 12_000},
        }},
        "providers": {"foreground": {"apiKey": "key", "apiBase": "http://localhost:8081/v1"}},
    })

    def construct(owned, *, preset, model=None):
        factory.validate_provider_setup(owned, preset=preset, model=model)
        provider = MagicMock(spec=LLMProvider)
        provider._image_input_error.return_value = None
        provider.generation = preset.to_generation_settings()
        provider.supports_images = preset.supports_images
        provider.provider_name = owned.get_provider_name(preset=preset)
        provider.get_default_model.return_value = model or preset.model
        provider.estimate_prompt_tokens = MagicMock(return_value=(100, "test-counter"))
        provider.chat_stream_with_retry = AsyncMock(
            return_value=LLMResponse(content="Checkpoint.", tool_calls=[]),
        )
        provider.can_resume_conversation_state.return_value = False
        provider.aclose = AsyncMock()
        return provider

    monkeypatch.setattr(factory, "_make_provider_core", construct)
    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry())
    try:
        assert loop.compactor_min_verbatim_tail_tokens == 12_000
        session = loop.sessions.get_or_create("cli:floor")
        _runtime, _tools, _no_tools, replay_config = (
            await loop.compaction_runtime_for_session(session)
        )
        assert replay_config.min_verbatim_tail_tokens == 12_000
    finally:
        await loop.aclose()


@pytest.fixture
async def oversized_import(tmp_path, monkeypatch):
    foreground, loop = _make_loop(
        tmp_path, context_window_tokens=120_000, fg_max_tokens=100,
        next_turn_input_reserve_tokens=4_000, min_verbatim_tail_tokens=90_000,
    )
    compactor = loop.compactor_runtime().provider
    key = "cli:oversized-import"

    def cost(message):
        return 4 + sum(len(str(message.get(field, "")).encode()) for field in (
            "content", "tool_calls", "tool_call_id", "name",
        ))

    def estimate(messages, tools, model):
        return sum(map(cost, messages)), "synthetic-byte-counter"

    for module in ("context_governance", "memory"):
        monkeypatch.setattr(f"nanobot.agent.{module}.estimate_message_tokens", cost)
    foreground.estimate_prompt_tokens.side_effect = estimate
    compactor.estimate_prompt_tokens.side_effect = estimate
    imported = [{"role": "user", "content": "Review these synthetic batches."}]
    for i in range(17):
        imported.extend([
            {"role": "assistant", "content": f"Batch {i}: " + "x" * 23_000,
             "tool_calls": [{"id": f"call-{i}", "type": "function", "function": {
                 "name": "read_file", "arguments": "{}",
             }}]},
            {"role": "tool", "tool_call_id": f"call-{i}", "name": "read_file",
             "content": "synthetic result " + "y" * 1_000},
        ])
    imported.append({"role": "assistant", "content": "All batches read."})
    session = loop.sessions.get_or_create(key)
    session.provider_state = ProviderConversationState(
        kind="openai_responses", provider="openai:test", model="test-model",
        version=1, payload={"items": []},
    )
    loop.sessions.save(session)
    try:
        session = await loop.sessions.state.import_messages(key, imported, require_empty=True)
        assert session.provider_state is not None
        yield loop, compactor, foreground, key, session, cost
    finally:
        await loop.aclose()


@pytest.mark.parametrize(
    "sizing_mismatch", [None, "stage", "replay"],
    ids=["additive-fit", "earlier-exact-stage-fit", "later-exact-replay-fit"],
)
async def test_oversized_import_dry_run_and_manual_apply_use_the_same_stages(oversized_import, sizing_mismatch):
    loop, compactor, foreground, key, original, cost = oversized_import
    full_history = original.get_history()
    input_budget = 200_000 - 4_096
    probed_batch_counts = []
    replay_batch_counts = []
    replay_overhead = 0
    tail_floor = 90_000
    first_retained_batch = 13

    if sizing_mismatch == "replay":
        # Rejected replay: the gateway's prompt estimate carries fixed
        # overhead the additive sizing does not. Size the overhead so the
        # four-bundle additive cut is rejected and the next legal cut (three
        # bundles, above the floor) fits with equality.
        runtime, _tools, _no_tools, replay_config = await loop.compaction_runtime_for_session(
            original, replay_runtime=loop.llm_runtime(),
        )
        compactor_budget = loop.consolidator.compactor_input_budget(
            runtime, None, len(full_history),
        )
        assert compactor_budget is not None
        foreground_input_budget = 120_000 - 100 - 1024  # window - max output - safety buffer
        wrapper_cost = sum(map(cost, loop.consolidator._session_replay_wrapper(original, "")))
        continuation_cost = cost({"role": "user", "content": SUMMARY_CONTINUATION_TEXT})
        replay_tail_budget = (
            foreground_input_budget
            - wrapper_cost - continuation_cost
            - compactor_budget.max_summary_tokens
            - replay_config.next_turn_input_reserve_tokens
        )
        tail_costs = [cost(m) for m in full_history]
        batch_starts = {
            int(str(message["tool_calls"][0]["id"]).split("-")[1]): i
            for i, message in enumerate(full_history)
            if message.get("role") == "assistant" and message.get("tool_calls")
        }
        three_bundle_tail = sum(tail_costs[batch_starts[14]:])  # Batches 14..16 plus reply.
        four_bundle_tail = sum(tail_costs[batch_starts[13]:])  # Batches 13..16 plus reply.
        # The four-bundle cut is the oldest additive fit; the overhead must
        # reject exactly its replay so the next legal cut decides the sizing.
        assert three_bundle_tail < four_bundle_tail <= replay_tail_budget
        tail_floor = three_bundle_tail // 2
        loop.compactor_min_verbatim_tail_tokens = tail_floor
        replay_overhead = replay_tail_budget - three_bundle_tail
        first_retained_batch = 14

    def estimate_replay(messages, tools, model):
        tokens = sum(map(cost, messages))
        batches = sum(bool(message.get("tool_calls")) for message in messages)
        if batches in (3, 4) and messages[-1].get("content") == SUMMARY_CONTINUATION_TEXT:
            replay_batch_counts.append(batches)
            tokens += replay_overhead
        return tokens, "synthetic-replay-counter"

    if sizing_mismatch == "replay":
        foreground.estimate_prompt_tokens.side_effect = estimate_replay
        assert replay_overhead > 0

    def estimate(messages, tools, model):
        if sizing_mismatch == "stage" and messages and messages[0].get("content") == full_history[0]["content"]:
            batches = sum(bool(message.get("tool_calls")) for message in messages)
            probed_batch_counts.append(batches)
            # The additive fit is eight bundles; exact sizing fits seven, with
            # no headroom. Don't change the later carry-stage estimates.
            if batches in (7, 8):
                return input_budget + (batches - 7), "synthetic-exact-counter"
        return sum(map(cost, messages)), "synthetic-byte-counter"

    compactor.estimate_prompt_tokens.side_effect = estimate

    def summarize(**request):
        batch = [m for m in request["messages"] if m.get("tool_calls")][-1]
        return LLMResponse(
            content=f"Checkpoint through {batch['tool_calls'][0]['id']}.", finish_reason="stop",
            usage=LLMUsage.reported(input_tokens=sum(map(cost, request["messages"])), output_tokens=10)
            .with_timing(generation_ms=400, ttft_ms=50),
        )

    compactor.chat_stream_with_retry.side_effect = summarize

    async def command(content):
        return await loop._process_message(
            InboundMessage(channel="cli", sender_id="user", chat_id="oversized-import", content=content),
            runtime=loop.llm_runtime(),
        )

    reply = await command("/compact --dry-run")
    assert reply is not None
    assert "summary generated by the compactor (discarded)" in reply.content
    requests = [call.kwargs for call in compactor.chat_stream_with_retry.call_args_list]
    assert len(requests) == 2
    if sizing_mismatch == "stage":
        assert probed_batch_counts[:3] == [13, 8, 7]
        assert sum(bool(m.get("tool_calls")) for m in requests[0]["messages"]) == 7
    if sizing_mismatch == "replay":
        assert replay_batch_counts[:2] == [4, 3]
    assert "- Stage 1:" in reply.content and "- Stage 2:" in reply.content
    assert "25.0 output tokens/s (measured streaming)" in reply.content
    assert "10 output tokens" in reply.content
    loop.sessions.invalidate(key)
    unchanged = loop.sessions.get_or_create(key)
    assert unchanged.get_history() == full_history
    assert unchanged.last_archived == 0
    assert unchanged.provider_state == original.provider_state
    assert loop.consolidator.store.read_unprocessed_history(0) == []

    compactor.chat_stream_with_retry.reset_mock()
    logs: list[str] = []
    sink = logger.add(logs.append, level="INFO", format="{message}")
    try:
        await command("/compact")
    finally:
        logger.remove(sink)
    assert [call.kwargs for call in compactor.chat_stream_with_retry.call_args_list] == requests
    foreground.chat_stream_with_retry.assert_not_awaited()
    # The staged success log keeps the per-stage counts and adds the same
    # comparable before/after replay estimates as the single-request path.
    [line] = [entry for entry in logs if entry.startswith(f"Session compact for {key} (manual):")]
    metrics = re.search(
        r"stages=2, est\. replay prompt tokens (\d+) -> (\d+), saved (\d+) tokens \((\d+)%\)$", line,
    )
    assert metrics
    before, after, saved, pct = map(int, metrics.groups())
    assert before - after == saved > 0
    assert abs(pct - saved * 100 / before) < 1
    prefix = []
    carry = None
    for i, request in enumerate(requests):
        assert request["tools"] == [] and request["tool_choice"] == "none"
        estimated, _ = estimate(request["messages"], request["tools"], request["model"])
        assert estimated + request["max_tokens"] <= 200_000
        messages = request["messages"][:-1]
        if i:
            assert messages[0]["content"] == f"[Archived Context Summary]\n\n{carry}"
            messages = messages[1:]
        calls = {call["id"] for m in messages for call in m.get("tool_calls", [])}
        assert calls == {m["tool_call_id"] for m in messages if m["role"] == "tool"}
        prefix.extend(messages)
        carry = summarize(**request).content
    loop.sessions.invalidate(key)
    applied = loop.sessions.get_or_create(key)
    assert prefix == full_history[:applied.last_archived]
    assert applied.get_history() == full_history[applied.last_archived:]
    assert applied.get_history()[0]["role"] == "assistant"
    assert sum(map(cost, applied.get_history())) >= tail_floor
    assert applied.provider_state is None
    final_summary = f"Checkpoint through call-{first_retained_batch - 1}."
    assert applied.metadata["_last_summary"]["text"] == final_summary
    assert sum(is_summary_checkpoint(m) for m in applied.messages) == 1
    [entry] = loop.consolidator.store.read_unprocessed_history(0)
    assert entry["content"] == final_summary

    # A new foreground turn resumes from the persisted summary and retained
    # assistant-leading tail, not from an intermediate stage or old raw prefix.
    await command("What is the latest status?")
    foreground.chat_stream_with_retry.assert_awaited_once()
    replay = foreground.chat_stream_with_retry.call_args.kwargs["messages"]
    assert final_summary in replay[0]["content"]
    assert {m["tool_call_id"] for m in replay if m["role"] == "tool"} == {
        f"call-{i}" for i in range(first_retained_batch, 17)
    }


async def test_staged_sdk_checkpoint_succeeds_if_journal_cursor_write_fails(oversized_import):
    loop, compactor, _, key, original, _ = oversized_import
    store = loop.consolidator.store
    # A directory at the cursor path makes write_text fail after the journal
    # row was appended. The caller must not see failure or retry the append.
    store._cursor_file.mkdir()
    warnings = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        assert await RuntimeClient(loop).compact_idle_session(key) == "Checkpoint."
    finally:
        logger.remove(sink)

    assert compactor.chat_stream_with_retry.await_count == 2
    assert any(f"Staged checkpoint applied for {key} but updating the memory journal failed" in w for w in warnings)
    loop.sessions.invalidate(key)
    applied = loop.sessions.get_or_create(key)
    assert applied.metadata["_last_summary"]["text"] == "Checkpoint."
    assert applied.provider_state is None
    assert applied.last_archived > 0
    assert applied.get_history() == original.get_history()[applied.last_archived:]
    [entry] = store.read_unprocessed_history(0)
    assert entry["content"] == "Checkpoint."


@pytest.mark.parametrize("failure", [
    "provider", "oversized_carry", "replay_over_budget", "cancelled", "timeout", "stage_limit",
])
async def test_staged_sdk_failure_preserves_imported_session(oversized_import, monkeypatch, failure):
    loop, compactor, foreground, key, original, cost = oversized_import
    second_stage = asyncio.Event()

    async def summarize(**request):
        if compactor.chat_stream_with_retry.await_count == 1:
            return LLMResponse(content="Carry.")
        if failure == "replay_over_budget":
            return LLMResponse(content="Final checkpoint.")
        second_stage.set()
        if failure in {"cancelled", "timeout"}:
            await asyncio.Future()
        raise RuntimeError("synthetic provider failure")

    compactor.chat_stream_with_retry.side_effect = summarize

    def estimate_with_mismatch(marker):
        def estimate(messages, tools, model):
            if any(marker in m.get("content", "") for m in messages):
                return 200_000, "synthetic-counter-mismatch"
            return sum(map(cost, messages)), "synthetic-byte-counter"
        return estimate

    if failure == "oversized_carry":
        compactor.estimate_prompt_tokens.side_effect = estimate_with_mismatch("Carry.")
    elif failure == "replay_over_budget":
        foreground.estimate_prompt_tokens.side_effect = estimate_with_mismatch("Final checkpoint.")
    elif failure == "stage_limit":
        monkeypatch.setattr(loop.consolidator, "_STAGED_MAX_STAGES", 1)
    elif failure == "timeout":
        monkeypatch.setattr(loop.consolidator, "_STAGED_MAX_SECONDS", 0.15)

    attempt = RuntimeClient(loop).compact_idle_session(key)
    if failure == "cancelled":
        task = asyncio.create_task(attempt)
        try:
            await asyncio.wait_for(second_stage.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    elif failure == "stage_limit":
        assert await attempt == ""
        compactor.chat_stream_with_retry.assert_not_awaited()
    else:
        assert await attempt is None
        assert compactor.chat_stream_with_retry.await_count == (1 if failure == "oversized_carry" else 2)

    loop.sessions.invalidate(key)
    reloaded = loop.sessions.get_or_create(key)
    assert reloaded.messages == original.messages
    assert reloaded.provider_state == original.provider_state
    assert reloaded.last_archived == original.last_archived == 0
    assert "_last_summary" not in reloaded.metadata
    assert loop.consolidator.store.read_unprocessed_history(0) == []
