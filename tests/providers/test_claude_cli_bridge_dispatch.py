"""The Claude CLI tool bridge executes through the active turn's shared
tool-dispatch boundary — the same ``ToolDispatch`` the agent runner uses for
its own model-initiated calls. These tests pin the equivalence contracts:
one turn of hosted calls shares one per-turn policy accounting (exactly as
across a normal turn's iterations), the boundary cannot be rewired by its
consumer, and each call settles with exactly one start and one terminal
observable event.
"""

import asyncio
from typing import Any

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import ToolCallRequest
from nanobot.providers.claude_cli_bridge import ClaudeCodeToolBridge
from nanobot.providers.claude_cli_tool_dispatch import ToolDispatch


class _FakeTool(Tool):
    def __init__(self, name: str, counter: list[Any] | None = None) -> None:
        self._name = name
        self.counter = counter if counter is not None else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> Any:
        self.counter.append(kwargs)
        return f"{self._name}({sorted(kwargs)})"


class _RecordingHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[AgentHookContext] = []

    async def on_tool_execution_event(self, context: AgentHookContext) -> None:
        self.events.append(context)


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _hook_context() -> AgentHookContext:
    return AgentHookContext(iteration=1, messages=[], session_key="test")


def _dispatch(registry: ToolRegistry, hook: AgentHook | None = None) -> ToolDispatch:
    return ToolDispatch(
        tools=registry,
        hook=hook or AgentHook(),
        context=_hook_context(),
    )


def test_dispatch_rejects_a_substituted_runner():
    # The execution closure is fixed at construction; a consumer cannot
    # supply its own runner.
    with pytest.raises((TypeError, ValueError)):
        ToolDispatch(
            tools=_registry(),
            hook=AgentHook(),
            context=_hook_context(),
            _run="substituted",  # type: ignore[call-arg]
        )


async def test_boundary_cannot_be_rewired_after_construction():
    """Even if a consumer replaces the registry attribute, execution still
    resolves through the registry captured at construction — the same
    guarantee an ordinary turn's calls get."""
    legit = _FakeTool("read_file", counter=[])
    registry = _registry(legit)
    dispatch = _dispatch(registry)
    # Impersonation attempt: swap the registry attribute after construction
    # with one that serves a different tool under the same name.
    impersonator = _FakeTool("read_file", counter=[])
    dispatch.tools = _registry(impersonator)
    call = ToolCallRequest(id="c1", name="read_file", arguments={"path": "fixture://x"})
    entry = await dispatch.dispatch(call)
    assert legit.counter == [{"path": "fixture://x"}]
    assert impersonator.counter == []
    assert entry.event["status"] == "ok"


async def test_one_turn_of_hosted_calls_shares_one_policy_budget():
    """Two identical bypass attempts in one provider turn escalate exactly as
    the same attempts spread across one ordinary turn's iterations would: the
    per-turn accounting is shared across every call dispatched on the turn."""
    counter: list[Any] = []

    class _ViolatingTool(Tool):
        @property
        def name(self) -> str:
            return "a"

        @property
        def description(self) -> str:
            return "always violates"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, **kwargs: Any) -> Any:
            counter.append(kwargs)
            raise RuntimeError("Error: path is outside the configured workspace: /etc/hosts")

    registry = _registry(_ViolatingTool())
    bridge = ClaudeCodeToolBridge(allowed_tool_names=frozenset({"a"}))
    turn = await bridge.start_turn(
        tool_names=["a"],
        tool_dispatch=_dispatch(registry),
        request_context=RequestContext(channel="test", chat_id="c", workspace="/workspace"),
    )
    try:
        first = await turn._execute_tool("a", {"path": "/etc/hosts"}, 0.0)
        await turn._execute_tool("a", {"path": "/etc/hosts"}, 0.0)
        third = await turn._execute_tool("a", {"path": "/etc/hosts"}, 0.0)
    finally:
        await turn.aclose()
    # The third identical attempt in the same turn hits the per-turn budget.
    assert "refusing repeated workspace-bypass attempts" in str(third)
    assert "refusing repeated workspace-bypass attempts" not in str(first)
    assert len(counter) == 3  # every attempt executed exactly once, no retries


async def test_each_call_settles_with_exactly_one_start_and_one_terminal_event():
    hook = _RecordingHook()
    registry = _registry(_FakeTool("a"))
    bridge = ClaudeCodeToolBridge(allowed_tool_names=frozenset({"a"}))
    turn = await bridge.start_turn(
        tool_names=["a"],
        tool_dispatch=_dispatch(registry, hook),
        request_context=RequestContext(channel="test", chat_id="c", workspace="/workspace"),
    )
    try:
        result = await turn._execute_tool("a", {"path": "fixture://a"}, 0.0)
        assert "Error" not in str(result)
    finally:
        await turn.aclose()
    assert len(hook.events) == 2
    start, terminal = hook.events
    assert start.tool_results == [] and start.tool_events == []
    assert start.tool_calls[0].name == "a"
    assert terminal.tool_events[0]["status"] == "ok"
    assert "a(" in str(terminal.tool_results[0])


async def test_cancel_settles_with_one_terminal_event_and_no_silent_retry():
    entered = asyncio.Event()

    class _SlowTool(Tool):
        @property
        def name(self) -> str:
            return "a"

        @property
        def description(self) -> str:
            return "slow read"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, **kwargs: Any) -> Any:
            entered.set()
            await asyncio.sleep(30)

    registry = _registry(_SlowTool())
    hook = _RecordingHook()
    bridge = ClaudeCodeToolBridge(allowed_tool_names=frozenset({"a"}), tool_timeout_s=60.0)
    turn = await bridge.start_turn(
        tool_names=["a"],
        tool_dispatch=_dispatch(registry, hook),
        request_context=RequestContext(channel="test", chat_id="c", workspace="/workspace"),
    )
    task = asyncio.create_task(turn._execute_tool("a", {"path": "fixture://slow"}, 0.0))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await turn.aclose()
    # The boundary observed exactly one start and one terminal (cancellation)
    # event; nothing was retried or replayed.
    assert len(hook.events) == 2
    assert hook.events[-1].tool_events[0] == {
        "name": "a",
        "status": "error",
        "detail": "cancelled",
    }
