"""Regression tests for oldest-prefix cut selection on the foreground
context-pressure path.

When the full transcript does not fit the foreground input budget, the
governor may compact only the oldest raw prefix and keep a recent tail
verbatim. The cut must sit on a legal raw boundary (no split tool
call/result pair), the retained tail must fit the foreground headroom
(including the summary's reserved output), and the compacted prefix must
fit the history consolidator's actual input budget. When no cut satisfies
both budgets, the previous full-prefix compaction behavior is preserved.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
)
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import Consolidator
from nanobot.agent.runner import AgentRunner
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults
from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.sdk.clients import RuntimeClient
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.manager import Session
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT, CompactorInputBudget
from nanobot.utils.llm_runtime import LLMRuntime

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


def _assistant_tool_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "exec", "arguments": "{}"},
        }],
    }


def _governor_state(
    raw: list[dict],
    *,
    delta: list[dict],
    budget: CompactorInputBudget | None,
    window: int = 2000,
    max_tokens: int = 100,
) -> tuple[ModelRequestState, ContextCompactionState]:
    provider = MagicMock(spec=LLMProvider)
    tools = MagicMock()
    tools.get_definitions.return_value = []
    config = ContextGovernanceConfig(
        provider=provider,
        model="test-model",
        tools=tools,
        workspace=None,
        session_key="cut-test",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        context_window_tokens=window,
        max_tokens=max_tokens,
    )
    consolidate = AsyncMock(return_value="fresh checkpoint")
    compaction = ContextCompactionState.from_messages(
        raw,
        consolidate,
        None,
        compactor_input_budget=budget,
    )
    state = ModelRequestState(
        config=config,
        compaction=compaction,
        conversation=ProviderConversationStateController(
            provider=provider,
            model="test-model",
            messages=list(raw),
        ),
    )
    return state, compaction


def test_compactor_input_budget_reflects_compactor_runtime_capacity(monkeypatch):
    monkeypatch.setattr(
        "nanobot.agent.memory.estimate_prompt_tokens_chain",
        lambda *_args: (75, "test-counter"),
    )
    consolidator = Consolidator(
        MagicMock(),
        MagicMock(),
        lambda *args, **kwargs: [],
        lambda: [],
    )
    runtime = LLMRuntime(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        generation=GenerationSettings(max_tokens=200),
        context_window_tokens=1000,
    )

    budget = consolidator.compactor_input_budget(runtime, [], 1)

    assert budget == CompactorInputBudget(
        input_budget=800,
        fixed_overhead_tokens=75,
        max_summary_tokens=200,
    )

    # A window that cannot hold a single output token has no usable capacity.
    tight = LLMRuntime(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        generation=GenerationSettings(max_tokens=1000),
        context_window_tokens=1000,
    )
    assert consolidator.compactor_input_budget(tight, [], 1) is None

    # Overhead larger than the input budget leaves no transcript room.
    narrow = LLMRuntime(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        generation=GenerationSettings(max_tokens=200),
        context_window_tokens=250,
    )
    assert consolidator.compactor_input_budget(narrow, [], 1) is None

    # Non-numeric generation settings cannot be derived into a budget.
    loose = LLMRuntime(
        provider=MagicMock(),
        model="test-model",
        generation=MagicMock(),
        context_window_tokens=1000,
    )
    assert consolidator.compactor_input_budget(loose, [], 1) is None


def test_legal_cut_indices_keep_tool_pairs_whole():
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        _assistant_tool_call("c1"),
        {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "r1"},
        _assistant_tool_call("c2"),
        {"role": "tool", "tool_call_id": "c2", "name": "exec", "content": "r2"},
    ]

    legal = ContextGovernor.legal_cut_indices(raw, len(raw))

    # Cutting before a result whose call is in the prefix splits the pair.
    assert legal[5] is False
    assert legal[3] is False
    # Cutting right after a complete pair, or at a plain message, is safe.
    assert legal[4] is True
    assert legal[2] is True
    assert legal[1] is True

    orphan = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "tool", "tool_call_id": "unknown", "name": "exec", "content": "r"},
    ]
    assert ContextGovernor.legal_cut_indices(orphan, len(orphan)) == [
        False, False, False,
    ]


def _cost_model(monkeypatch, heavy: str = "old", heavy_cost: int = 500, light_cost: int = 10):
    """Deterministic token costs: heavy-content messages vs everything else."""

    def estimate(_provider, _model, messages, _tools):
        return (
            sum(
                heavy_cost if heavy in str(message.get("content") or "") else light_cost
                for message in messages
            ),
            "test-counter",
        )

    def estimate_message(message):
        return (
            heavy_cost if heavy in str(message.get("content") or "") else light_cost
        )

    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        estimate,
    )
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        estimate_message,
    )


def test_select_prefix_cut_protects_recent_verbatim_tail(monkeypatch):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    budget = CompactorInputBudget(
        input_budget=1000,
        fixed_overhead_tokens=100,
        max_summary_tokens=200,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)

    cut = ContextGovernor()._select_prefix_cut(
        state, state.compaction, delta, None,
    )

    # The oldest legal cut whose tail (q2, a2-old, q3, a3) still fits the
    # foreground headroom; adding a1-old would exceed it.
    assert cut == 3


def test_select_prefix_cut_skips_cut_that_splits_tool_pair(monkeypatch):
    _cost_model(monkeypatch, heavy_cost=10, light_cost=10)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        _assistant_tool_call("c1"),
        {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "r1"},
        {"role": "user", "content": "q2"},
        _assistant_tool_call("c2"),
        {"role": "tool", "tool_call_id": "c2", "name": "exec", "content": "r2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    # Tail budget (876 - 10 wrapper - 10 delta - 821 reserved output = 35)
    # reaches index 6 but not index 5, so the only cost-legal cut would sit
    # between the c2 call and its result.
    budget = CompactorInputBudget(
        input_budget=2000,
        fixed_overhead_tokens=100,
        max_summary_tokens=821,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)

    cut = ContextGovernor()._select_prefix_cut(
        state, state.compaction, delta, None,
    )

    # The cut lands on the older legal boundary (user q3), never on index 6.
    assert cut == 7
    tail = raw[cut:9]
    assert all(
        message.get("tool_call_id") is None
        or message.get("tool_call_id") in {
            call_id
            for kept in tail
            if kept.get("role") == "assistant"
            for call_id in (
                call.get("id") for call in kept.get("tool_calls") or []
            )
        }
        for message in tail
        if message.get("role") == "tool"
    )


async def test_select_prefix_cut_falls_back_when_compactor_prefix_exceeds_budget(
    monkeypatch,
):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    # The most protective cut compacts 510 tokens; a 500-token compactor
    # capacity cannot hold even that, so no useful cut exists.
    budget = CompactorInputBudget(
        input_budget=600,
        fixed_overhead_tokens=100,
        max_summary_tokens=200,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)

    cut = ContextGovernor()._select_prefix_cut(
        state, state.compaction, delta, None,
    )

    assert cut is None


async def test_select_prefix_cut_falls_back_without_foreground_headroom(monkeypatch):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    # Reserved summary output leaves no foreground headroom for any tail.
    budget = CompactorInputBudget(
        input_budget=1000,
        fixed_overhead_tokens=100,
        max_summary_tokens=1000,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)

    cut = ContextGovernor()._select_prefix_cut(
        state, state.compaction, delta, None,
    )

    assert cut is None


async def test_select_prefix_cut_falls_back_without_compactor_budget(monkeypatch):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    state, _ = _governor_state(raw, delta=delta, budget=None)

    cut = ContextGovernor()._select_prefix_cut(
        state, state.compaction, delta, None,
    )

    assert cut is None


_SESSION_HISTORY = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1-old"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2-old"},
    {"role": "user", "content": "q3"},
    {"role": "assistant", "content": "a3"},
]


def _session_cut(monkeypatch, *, reserve: int, budget: CompactorInputBudget) -> int | None:
    """Call ``select_prefix_cut`` the way session compaction does: no unsent delta.

    The session path supplies the replayable history itself; the wrapper is
    only the post-compaction replay prefix, never part of the compacted raw.
    """
    _cost_model(monkeypatch)
    config = ContextGovernanceConfig(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        tools=MagicMock(get_definitions=MagicMock(return_value=[])),
        workspace=None,
        session_key="cut-test",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        context_window_tokens=2000,
        max_tokens=100,
    )
    wrapper = [{"role": "system", "content": "wrapper"}]
    raw = list(_SESSION_HISTORY)
    return ContextGovernor().select_prefix_cut(
        config, raw, len(raw), budget, wrapper, None, [], None,
        session_compaction=True,
        next_turn_input_reserve_tokens=reserve,
    ).cut


def test_session_cut_charges_reserve_against_replay_headroom(monkeypatch):
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    # Without a reserve the oldest legal tail (q2..a3, 530 tokens) fits the
    # 656-token headroom.
    assert _session_cut(monkeypatch, reserve=0, budget=budget) == 3
    # The same headroom minus 126 tokens still holds the tail exactly; one
    # more reserved token leaves no legal user boundary, so compaction falls
    # back to the full prefix.
    assert _session_cut(monkeypatch, reserve=126, budget=budget) == 3
    assert _session_cut(monkeypatch, reserve=127, budget=budget) is None


def test_session_cut_reserve_does_not_reduce_compactor_capacity(monkeypatch):
    # 600-token compactor capacity (700 input minus 100 prompt) holds the 520
    # tokens the cut compacts, and 556-token replay headroom holds the 530
    # tail: the 200-token reserve must not be charged to the compactor.
    budget = CompactorInputBudget(
        input_budget=700, fixed_overhead_tokens=100, max_summary_tokens=100,
    )
    assert _session_cut(monkeypatch, reserve=0, budget=budget) == 3
    assert _session_cut(monkeypatch, reserve=200, budget=budget) == 3


def _session_cut_with_floor(
    monkeypatch,
    history: list[dict],
    *,
    floor: int,
    budget: CompactorInputBudget,
    reserve: int = 0,
    window: int = 10_000,
    heavy: str = "old",
) -> int | None:
    """Call ``select_prefix_cut`` like session compaction, with a verbatim floor."""
    _cost_model(monkeypatch, heavy=heavy)
    config = ContextGovernanceConfig(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        tools=MagicMock(get_definitions=MagicMock(return_value=[])),
        workspace=None,
        session_key="cut-test",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        context_window_tokens=window,
        max_tokens=100,
        min_verbatim_tail_tokens=floor,
    )
    wrapper = [{"role": "system", "content": "wrapper"}]
    raw = list(history)
    return ContextGovernor().select_prefix_cut(
        config, raw, len(raw), budget, wrapper, None, [], None,
        session_compaction=True,
        next_turn_input_reserve_tokens=reserve,
    ).cut


_TOOL_TURN_HISTORY = [
    {"role": "user", "content": "HEAVY q1"},
    _assistant_tool_call("c1"),
    {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "HEAVY r1"},
    _assistant_tool_call("c2"),
    {"role": "tool", "tool_call_id": "c2", "name": "exec", "content": "HEAVY r2"},
    {"role": "assistant", "content": "done"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]


def test_session_cut_uses_tool_pair_safe_assistant_boundary_for_tool_heavy_turn(monkeypatch):
    """A long tool-heavy first turn that a user-only cut cannot keep.

    The only eligible user boundary (q2) keeps 20 tokens, below the 900-token
    floor, so a user-only cut would skip the compaction (the first user turn
    cannot start a nonempty archive prefix). The assistant reply that opens
    the tool-heavy turn keeps 1050 tokens with every tool pair whole, so the
    cut lands there and the compaction proceeds.
    """
    budget = CompactorInputBudget(
        input_budget=5000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    cut = _session_cut_with_floor(monkeypatch, _TOOL_TURN_HISTORY, floor=900, budget=budget, heavy="HEAVY")

    raw = list(_TOOL_TURN_HISTORY)
    assert cut == 1
    assert raw[cut]["role"] == "assistant"
    assert raw[cut]["tool_calls"][0]["id"] == "c1"
    tail = raw[cut:]
    declared = {
        call["id"]
        for message in tail
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    assert {m["tool_call_id"] for m in tail if m.get("role") == "tool"} == declared


def test_session_mid_turn_cut_must_fit_both_budgets_and_reserve(monkeypatch):
    # The only tail meeting the 900-token floor (the tool-heavy turn opening
    # with c1) costs 1050 tokens; its prefix costs 500. The 1070-token
    # projected request (wrapper, tail, continuation) plus the 200 reserved
    # summary output fills the 1376-token headroom, so reserving one more
    # input token, or shrinking the compactor below the 500-token prefix,
    # independently makes the cut impossible.
    def cut(reserve: int, compactor_input: int) -> int | None:
        return _session_cut_with_floor(
            monkeypatch, _TOOL_TURN_HISTORY, floor=900, reserve=reserve,
            window=2500, heavy="HEAVY",
            budget=CompactorInputBudget(compactor_input, 100, 200),
        )

    assert cut(106, 610) == 1
    assert cut(107, 610) is None
    # A 500-token prefix exactly fills the 500 capacity; one less token does not.
    assert cut(106, 600) == 1
    assert cut(106, 599) is None


def test_session_cut_floor_disabled_keeps_legacy_user_boundary_sizing(monkeypatch):
    budget = CompactorInputBudget(
        input_budget=5000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    cut = _session_cut_with_floor(monkeypatch, _TOOL_TURN_HISTORY, floor=0, budget=budget, heavy="HEAVY")

    raw = list(_TOOL_TURN_HISTORY)
    # Legacy sizing: the oldest user boundary (q2) wins and the mid-turn
    # assistant boundary stays out of play without a floor.
    assert cut == 6
    assert raw[cut]["role"] == "user"


def test_session_cut_reports_below_floor_no_op_distinctly_from_no_legal_cut(monkeypatch):
    """A history at or below the floor is a no-op, not a sizing rejection.

    Both outcomes keep the full history, but the skip log must tell "nothing
    to archive by design" (with its estimate) apart from a tail that
    genuinely cannot meet the floor.
    """
    _cost_model(monkeypatch, heavy="HEAVY", heavy_cost=2000, light_cost=10)
    config = ContextGovernanceConfig(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        tools=MagicMock(get_definitions=MagicMock(return_value=[])),
        workspace=None,
        session_key="cut-test",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        context_window_tokens=10_000,
        max_tokens=100,
        min_verbatim_tail_tokens=6050,
    )
    budget = CompactorInputBudget(input_budget=5_000, fixed_overhead_tokens=100, max_summary_tokens=200)
    wrapper = [{"role": "system", "content": "wrapper"}]
    raw = list(_TOOL_TURN_HISTORY)

    # 3x2000 + 5x10 = 6050 replayable tokens; at exactly that floor the whole
    # history is protected, so there is nothing to archive. The decision comes
    # from the additive per-message sizing, without a projected probe.
    result = ContextGovernor().select_prefix_cut(
        config, raw, len(raw), budget, wrapper, None, [], None,
        session_compaction=True,
    )
    assert (result.cut, result.reason, result.history_tokens) == (None, "below_floor", 6050)

    # One token lower admits no legal tail (the largest is 4050 tokens), so
    # this is a genuine no-legal-cut, not a floor no-op.
    config.min_verbatim_tail_tokens = 6049
    result = ContextGovernor().select_prefix_cut(
        config, raw, len(raw), budget, wrapper, None, [], None,
        session_compaction=True,
    )
    assert (result.cut, result.reason, result.history_tokens) == (None, "no_legal_cut", None)

    # A thin (single-message) history can never be cut; at the floor it is the
    # same no-op, and above it stays a no-history rejection.
    result = ContextGovernor().select_prefix_cut(
        config, [dict(raw[0])], 1, budget, wrapper, None, [], None,
        session_compaction=True,
    )
    assert (result.cut, result.reason, result.history_tokens) == (None, "below_floor", 2000)
    config.min_verbatim_tail_tokens = 1500
    result = ContextGovernor().select_prefix_cut(
        config, [dict(raw[0])], 1, budget, wrapper, None, [], None,
        session_compaction=True,
    )
    assert (result.cut, result.reason, result.history_tokens) == (None, "no_history", None)


def _replay_probe_cost_model(monkeypatch, heavy: set[str]) -> list[int]:
    """Flat additive costs with inflated projected chain costs on ``heavy``.

    The additive tail sizing ignores the mismatch; only the projected replay
    probes see it. Returns the projected probe call sizes, in probe order.
    """
    probes: list[int] = []

    def estimate(_provider, _model, messages, _tools):
        contents = [str(message.get("content") or "") for message in messages]
        if "a3" in contents:
            probes.append(len(messages))
        return sum(3000 if content in heavy else 10 for content in contents), "test-counter"

    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain", estimate,
    )
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 10,
    )
    return probes


def _replay_probe_config(window: int = 4000) -> ContextGovernanceConfig:
    return ContextGovernanceConfig(
        provider=MagicMock(spec=LLMProvider),
        model="test-model",
        tools=MagicMock(get_definitions=MagicMock(return_value=[])),
        workspace=None,
        session_key="replay-probe-test",
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        context_window_tokens=window,
        max_tokens=100,
    )


_REPLAY_PROBE_HISTORY = [
    {"role": "user", "content": "q0"},
    {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
    {"role": "user", "content": "q3"},
    {"role": "assistant", "content": "a3"},
]


def test_session_replay_probe_retries_next_legal_cut_on_estimator_mismatch(monkeypatch):
    """An offline fit can be rejected by the gateway's projected replay estimate.

    The oldest additive cut carries an old message whose chain estimate
    overruns the foreground input budget; the next legal user boundary
    drops it and fits, so session compaction must retry it.
    """
    probes = _replay_probe_cost_model(monkeypatch, {"a1"})
    config = _replay_probe_config()
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )

    cut = ContextGovernor().select_prefix_cut(
        config, list(_REPLAY_PROBE_HISTORY), len(_REPLAY_PROBE_HISTORY), budget,
        [{"role": "system", "content": "wrapper"}], None, [], None,
        session_compaction=True,
    ).cut

    assert cut == 4
    assert probes == [8, 6]  # The rejected oldest cut, then the next legal one.


def test_session_replay_probe_retry_is_bounded_and_reports_rejection(monkeypatch):
    """When every probed cut is rejected, the retry stays bounded and reports.

    Four legal cuts exist, but only the three oldest are probed; the newer
    cut that would fit is never reached, so sizing work stays independent of
    transcript length.
    """
    probes = _replay_probe_cost_model(monkeypatch, {"a1", "a3"})
    config = _replay_probe_config()
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    history = list(_REPLAY_PROBE_HISTORY) + [
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]
    result = ContextGovernor().select_prefix_cut(
        config, history, len(history), budget,
        [{"role": "system", "content": "wrapper"}], None, [], None,
        session_compaction=True,
    )

    assert result.cut is None
    assert result.reason == "replay_over_budget"
    assert probes == [10, 8, 6]  # Bounded to three projected probes, not four.
    # The estimate/budget diagnostics ride on the result so the session skip
    # log can report them without a second planner log line.
    assert result.rejected_probes == 3
    assert result.last_required_tokens == 3250  # 3050-token probe + 200 summary.
    assert result.input_budget_tokens == 2876  # 4000 - 100 max output - 1024 buffer.


def test_foreground_replay_probe_keeps_single_probe(monkeypatch):
    """In-turn compaction keeps the historical single-probe behavior."""
    probes = _replay_probe_cost_model(monkeypatch, {"a1"})
    config = _replay_probe_config()
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )

    cut = ContextGovernor().select_prefix_cut(
        config, list(_REPLAY_PROBE_HISTORY), len(_REPLAY_PROBE_HISTORY), budget,
        [{"role": "system", "content": "wrapper"}], None,
        [{"role": "user", "content": "new input"}], None,
    ).cut

    # The rejected oldest cut is never retried in-turn; the next legal cut
    # would fit, but foreground sizing keeps one probe.
    assert cut is None
    assert probes == [8]


def test_foreground_cut_ignores_next_turn_reserve(monkeypatch):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)
    state.config.next_turn_input_reserve_tokens = 999

    # In-turn compaction already sizes against the real pending input, so
    # the session reserve must not shrink its protected tail.
    assert ContextGovernor()._select_prefix_cut(state, state.compaction, delta, None) == 3


def test_foreground_cut_ignores_min_verbatim_tail_floor(monkeypatch):
    _cost_model(monkeypatch)
    raw = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1-old"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2-old"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]
    delta = [{"role": "user", "content": "continue the current task"}]
    budget = CompactorInputBudget(
        input_budget=1000, fixed_overhead_tokens=100, max_summary_tokens=200,
    )
    state, _ = _governor_state(raw, delta=delta, budget=budget)
    # Safety wins: in-turn compaction must still use the best safe cut even
    # when the configured verbatim-tail floor cannot be met.
    state.config.min_verbatim_tail_tokens = 999

    assert ContextGovernor()._select_prefix_cut(state, state.compaction, delta, None) == 3


def _build_transcript(transcript: TranscriptInput) -> list[dict]:
    system = (
        transcript.session_summary["text"]
        if transcript.session_summary is not None
        else "system"
    )
    messages = [{"role": "system", "content": system}, *transcript.history]
    if transcript.current_message is not None:
        messages.append({
            "role": transcript.current_role,
            "content": transcript.current_message,
        })
    return messages


def _make_run_spec(
    provider,
    *,
    history: list[dict],
    budget: CompactorInputBudget | None,
    consolidate,
    session_summary: dict | None = None,
):
    return make_run_spec(
        provider,
        initial_messages=None,
        transcript_input=TranscriptInput(
            history=history,
            current_message="continue the current task",
            session_summary=session_summary,
        ),
        transcript_builder=_build_transcript,
        consolidate_history=consolidate,
        tools=MagicMock(get_definitions=MagicMock(return_value=[])),
        model="test-model",
        context_window_tokens=2000,
        max_tokens=100,
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        compactor_input_budget=budget,
    )


_HEAVY_HISTORY = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1-old"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2-old"},
    {"role": "user", "content": "q3"},
    {"role": "assistant", "content": "a3"},
]


async def test_runner_prefix_cut_keeps_projected_tail_verbatim(monkeypatch):
    _cost_model(monkeypatch)
    provider = MagicMock(spec=LLMProvider)
    provider.can_resume_conversation_state.return_value = False
    requests: list[list[dict]] = []

    async def request(*, messages, **_kwargs):
        requests.append(messages)
        return LLMResponse(content="done")

    provider.chat_stream_with_retry = request
    consolidate = AsyncMock(return_value="fresh checkpoint")

    result = await AgentRunner().run(_make_run_spec(
        provider,
        history=_HEAVY_HISTORY,
        budget=CompactorInputBudget(
            input_budget=1000,
            fixed_overhead_tokens=100,
            max_summary_tokens=200,
        ),
        consolidate=consolidate,
    ))

    # Only the oldest projected prefix is summarized.
    consolidate.assert_awaited_once_with(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1-old"},
        ],
        None,
    )
    # The protected tail survives verbatim, followed by the current delta.
    assert [message.get("content") for message in requests[0]] == [
        requests[0][0].get("content"),
        "q2",
        "a2-old",
        "q3",
        "a3",
        "continue the current task",
    ]
    assert "fresh checkpoint" in requests[0][0].get("content")
    assert result.summary_checkpoint is not None
    assert result.summary_checkpoint.summary == "fresh checkpoint"
    assert result.summary_checkpoint.transcript_boundary == 3


async def test_runner_prefix_cut_falls_back_to_full_prefix_when_compactor_budget_too_small(
    monkeypatch,
):
    _cost_model(monkeypatch)
    provider = MagicMock(spec=LLMProvider)
    provider.can_resume_conversation_state.return_value = False
    requests: list[list[dict]] = []

    async def request(*, messages, **_kwargs):
        requests.append(messages)
        return LLMResponse(content="done")

    provider.chat_stream_with_retry = request
    consolidate = AsyncMock(return_value="fresh checkpoint")

    result = await AgentRunner().run(_make_run_spec(
        provider,
        history=_HEAVY_HISTORY,
        budget=CompactorInputBudget(
            input_budget=600,
            fixed_overhead_tokens=100,
            max_summary_tokens=200,
        ),
        consolidate=consolidate,
    ))

    # No useful cut: the previous full-prefix compaction behavior is kept.
    consolidate.assert_awaited_once_with(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1-old"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2-old"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ],
        None,
    )
    assert [message.get("content") for message in requests[0]][1:] == [
        "continue the current task",
    ]
    assert result.summary_checkpoint is not None
    assert result.summary_checkpoint.transcript_boundary == 7


async def test_runner_prefix_cut_counts_prior_summary_against_compactor_capacity(
    monkeypatch,
):
    _cost_model(monkeypatch)
    provider = MagicMock(spec=LLMProvider)
    provider.can_resume_conversation_state.return_value = False
    requests: list[list[dict]] = []

    async def request(*, messages, **_kwargs):
        requests.append(messages)
        return LLMResponse(content="done")

    provider.chat_stream_with_retry = request
    consolidate = AsyncMock(return_value="fresh checkpoint")

    result = await AgentRunner().run(_make_run_spec(
        provider,
        history=_HEAVY_HISTORY,
        # Same capacity that admits the cut without a prior summary (900 -
        # 510 prefix), but the prior summary costs 500 of it, so no useful
        # cut remains and full-prefix compaction is kept.
        budget=CompactorInputBudget(
            input_budget=1000,
            fixed_overhead_tokens=100,
            max_summary_tokens=200,
        ),
        consolidate=consolidate,
        session_summary={
            "text": "old summary",
            "last_active": "2026-08-30T00:00:00",
        },
    ))

    consolidate.assert_awaited_once_with(
        [
            {"role": "system", "content": "old summary"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1-old"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2-old"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ],
        "old summary",
    )
    assert [message.get("content") for message in requests[0]][1:] == [
        "continue the current task",
    ]
    assert result.summary_checkpoint is not None
    assert result.summary_checkpoint.transcript_boundary == 7


def test_message_index_at_history_position_maps_projected_positions_to_persisted_messages():
    session = Session(
        key="chan:chat",
        messages=[
            {"role": "assistant", "content": "a0"},  # trimmed: before first user
            {"role": "user", "content": "q1"},
            _assistant_tool_call("c1"),
            {"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "r1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": ""},  # trimmed: empty assistant turn
            {"role": "user", "content": "q4"},
        ],
    )

    # Model-facing positions: 0=q1, 1=call, 2=result, 3=a2, 4=q3, 5=q4.
    assert session.message_index_at_history_position(0) == 1
    assert session.message_index_at_history_position(3) == 4
    assert session.message_index_at_history_position(5) == 7
    assert session.message_index_at_history_position(6) is None
    assert session.message_index_at_history_position(-1) is None


async def test_prefix_cut_persists_summary_at_mapped_boundary_and_replays_tail(
    tmp_path, monkeypatch,
) -> None:
    """A prefix-cut boundary maps into the persisted session, not by length.

    The summary checkpoint lands at the raw session index that produced the
    last message before the cut; the protected tail (including one tool
    call/result pair) is kept verbatim exactly once, and a fresh load from
    disk replays summary + tail + current turn.
    """
    old1 = "HEAVY old1 " + "The quick brown fox jumps over the lazy dog. " * 40
    old2 = "HEAVY old2 " + "Pack my box with five dozen liquor jugs. " * 40
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.can_resume_conversation_state.return_value = False
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="Current checkpoint.", tool_calls=[]),
        LLMResponse(content="done", tool_calls=[]),
    ])

    def estimate(messages, _tools, _model):
        total = 0
        for message in messages:
            content = str(message.get("content") or "")
            if "HEAVY" in content:
                total += 2000
            elif content.strip():
                total += 10
        return max(total, 10), "test-counter"

    provider.estimate_prompt_tokens = estimate
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=4300,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator._SAFETY_BUFFER = 0
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 2000 if "HEAVY" in str(message.get("content") or "") else 10,
    )

    session = loop.sessions.get_or_create("cli:cut")
    session.messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": old1},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": old2},
        {"role": "user", "content": "q3"},
        _assistant_tool_call("t1"),
        {"role": "tool", "tool_call_id": "t1", "name": "exec", "content": "result ok"},
        {"role": "assistant", "content": "a3"},
    ]
    loop.sessions.save(session)

    result = await loop.process_direct("continue the task", session_key="cli:cut")

    assert result.content == "done"
    # The compactor summarized only the oldest prefix; the second request
    # carries the verbatim protected tail, including the tool pair.
    request = provider.chat_stream_with_retry.await_args_list[1].kwargs["messages"]
    assert "Current checkpoint." in request[0]["content"]
    assert [message.get("content") for message in request] == [
        request[0]["content"],
        "q2",
        old2,
        "q3",
        "",
        "result ok",
        "a3",
        request[-1]["content"],
    ]
    assert "continue the task" in request[-1]["content"]
    assert old1 not in [str(message.get("content")) for message in request]

    # A fresh load from disk: the checkpoint maps into the persisted history
    # (before the current turn's save window), the tail is intact once.
    reloaded = loop.sessions.get_or_create("cli:cut")
    assert reloaded.metadata["_last_summary"]["text"] == "Current checkpoint."
    assert reloaded.last_archived == 2
    assert reloaded.messages[2]["content"] == SUMMARY_CONTINUATION_TEXT
    assert [message.get("content") for message in reloaded.messages[3:9]] == [
        "q2", old2, "q3", "", "result ok", "a3",
    ]
    assert [message.get("content") for message in reloaded.get_history(
        include_runtime_context=False,
    )] == [
        "q2", old2, "q3", "", "result ok", "a3", "continue the task", "done",
    ]


@pytest.mark.parametrize("trigger", ["scheduled_idle", "sdk"])
async def test_mid_turn_cut_persists_checkpoint_before_assistant_tail(
    tmp_path, monkeypatch, trigger,
) -> None:
    """Session compaction cuts at the tool-pair-safe assistant boundary.

    The tool-heavy first turn is long enough that no user boundary keeps the
    configured verbatim-tail floor; the assistant reply opening the turn does.
    The committed checkpoint lands before that assistant, and a fresh reload
    replays the assistant-leading tail verbatim with every tool pair intact —
    the next foreground turn sends it as-is after the summary.
    """
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=100)
    provider.can_resume_conversation_state.return_value = False
    provider.chat_stream_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="Mid-turn checkpoint.", tool_calls=[]),
        LLMResponse(content="done", tool_calls=[]),
    ])

    def estimate(messages, _tools, _model):
        total = 0
        for message in messages:
            content = str(message.get("content") or "")
            if "HEAVY" in content:
                total += 2000
            elif content.strip():
                total += 10
        return max(total, 10), "test-counter"

    provider.estimate_prompt_tokens = estimate
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=20_000,
        session_ttl_minutes=1,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator._SAFETY_BUFFER = 0
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]
    loop.compactor_min_verbatim_tail_tokens = 4000

    def message_cost(message):
        return 2000 if "HEAVY" in str(message.get("content") or "") else 10

    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens", message_cost,
    )
    monkeypatch.setattr("nanobot.session.manager.estimate_message_tokens", message_cost)

    session = loop.sessions.get_or_create("cli:mid")
    session.messages = [
        {"role": "user", "content": "HEAVY first instruction"},
        _assistant_tool_call("t1"),
        {"role": "tool", "tool_call_id": "t1", "name": "exec", "content": "HEAVY result one"},
        _assistant_tool_call("t2"),
        {"role": "tool", "tool_call_id": "t2", "name": "exec", "content": "HEAVY result two"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    session.updated_at = datetime.now() - timedelta(minutes=10)
    loop.sessions.save(session)
    if trigger == "sdk":
        assert await RuntimeClient(loop).compact_idle_session(session.key) == "Mid-turn checkpoint."
    else:
        scheduled = []
        await loop.auto_compact.check_expired(scheduled.append, loop.compaction_runtime_for_session)
        assert len(scheduled) == 1
        await scheduled.pop()

    # A fresh load: the checkpoint maps to the persisted boundary just before
    # the assistant reply, and the tool-heavy tail survives verbatim, including
    # its leading assistant tool call and every result.
    loop.sessions.invalidate("cli:mid")
    reloaded = loop.sessions.get_or_create("cli:mid")
    assert reloaded.metadata["_last_summary"]["text"] == "Mid-turn checkpoint."
    assert reloaded.last_archived == 1
    assert is_hidden_history_message(reloaded.messages[1])
    assert reloaded.messages[1]["content"] == SUMMARY_CONTINUATION_TEXT
    replayed = reloaded.get_history(include_runtime_context=False)
    assert [message.get("content") for message in replayed] == [
        "", "HEAVY result one", "", "HEAVY result two", "done", "q2", "a2",
    ]
    assert replayed[0]["role"] == "assistant"
    assert replayed[0]["tool_calls"][0]["id"] == "t1"
    declared = {
        call["id"]
        for message in replayed
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    assert {m["tool_call_id"] for m in replayed if m.get("role") == "tool"} == declared

    # The next foreground turn replays the assistant-leading tail right after
    # the summary, without dropping its leading message.
    result = await loop.process_direct("continue the task", session_key="cli:mid")
    assert result.content == "done"
    request = loop.provider.chat_stream_with_retry.call_args.kwargs["messages"]
    assert "Mid-turn checkpoint." in request[0]["content"]
    assert request[1]["role"] == "assistant"
    assert request[1]["tool_calls"][0]["id"] == "t1"
    assert request[2]["role"] == "tool"
    assert request[2]["tool_call_id"] == "t1"
    assert request[2]["content"] == "HEAVY result one"
    await loop.aclose()


async def test_anthropic_replays_and_recompacts_checkpointed_tool_tail(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.utils.helpers._get_token_encoding", lambda: None)
    provider = AnthropicProvider(api_key="sk-test")
    provider.generation = GenerationSettings(max_tokens=100)
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path,
        model=provider.get_default_model(), context_window_tokens=48_000,
        compactor_min_verbatim_tail_tokens=1_000,
    )
    tools = [tool for tool in loop.tools.get_definitions() if tool["function"]["name"] == "exec"]
    monkeypatch.setattr(loop.tools, "get_definitions", lambda: tools)
    key = "cli:anthropic-mid-turn"
    session = loop.sessions.get_or_create(key)
    session.messages = [
        {"role": "user", "content": "Investigate the processing logs."},
        *[message for i in range(3) for message in (
            _assistant_tool_call(f"batch-{i}"),
            {"role": "tool", "name": "exec", "tool_call_id": f"batch-{i}",
             "content": f"shard={i} status=processed\n" * 300},
        )],
        {"role": "assistant", "content": "All shards processed."},
    ]
    # Reopen a genuinely committed checkpoint from an earlier compaction.
    session.commit_summary_checkpoint("Previous investigation.", insert_at=1)
    loop.sessions.save(session)
    loop.sessions.invalidate(key)
    requests = []

    def open_stream(**kwargs):
        requests.append(kwargs)
        text = "Foreground answer." if len(requests) == 1 else "Second checkpoint."
        stream = MagicMock()
        stream.__anext__.side_effect = StopAsyncIteration
        stream.get_final_message = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        ))
        cm = MagicMock()
        cm.__aenter__.return_value = stream
        return cm

    monkeypatch.setattr(provider._client.messages, "stream", open_stream)
    monkeypatch.setattr(provider._client.messages, "create", AsyncMock(side_effect=AssertionError("Unexpected SDK call")))
    try:
        session = loop.sessions.get_or_create(key)
        original_tail = session.get_history()
        reply = await loop.process_direct("What did the logs show?", session_key=key)
        assert reply.content == "Foreground answer."
        assert loop.sessions.get_or_create(key).get_history()[:len(original_tail)] == original_tail

        # With no override, Anthropic also summarizes the oldest retained tool
        # round. Set the reserve to force one whole round out of the replay tail.
        loop.compactor_next_turn_input_reserve_tokens = 25_000
        preview = await loop.process_direct("/compact --dry-run", session_key=key)
        assert "summary generated by the compactor (discarded)" in preview.content
        assert await RuntimeClient(loop).compact_idle_session(key) == "Second checkpoint."
        assert len(requests) == 3
        assert requests[1] == requests[2]
        assert "Previous investigation." in requests[2]["system"][0]["text"]
        for request, expected_ids in zip(requests[:2], [{"batch-0", "batch-1", "batch-2"}, {"batch-0"}], strict=True):
            messages = request["messages"]  # System instructions are a separate Anthropic field.
            assert messages[0]["role"] == "user"
            pairs = set()
            for index, message in enumerate(messages):
                content = message["content"]
                if not isinstance(content, list):
                    continue
                calls = {block["id"] for block in content if block["type"] == "tool_use"}
                if calls:
                    assert message["role"] == "assistant"
                    assert messages[index + 1]["role"] == "user"
                    results = [block["tool_use_id"] for block in messages[index + 1]["content"]
                               if block["type"] == "tool_result"]
                    assert sorted(results) == sorted(calls)
                    pairs.update(calls)
            assert pairs == expected_ids
        loop.sessions.invalidate(key)
        assert loop.sessions.get_or_create(key).get_history()[:len(original_tail) - 2] == original_tail[2:]
    finally:
        await loop.aclose()
