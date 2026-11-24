"""Tests for the per-turn, localhost-authenticated Claude CLI tool bridge.

The bridge is exercised with the real shared per-turn tool dispatch boundary
(``ToolDispatch`` — the same object the agent runner uses for its own
model-initiated calls) and a real MCP client over the loopback SSE endpoint —
no live Claude Code, no credentials, no network beyond 127.0.0.1. Generic
dispatch, auth, and isolation tests widen the surface explicitly with synthetic
tool fixtures (``allowed_tool_names``); the production allowlist and the
``read_file`` scope proof are tested separately against the real,
workspace-scoped read tool with harmless temp fixtures.
"""

import asyncio
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext, current_request_context
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.claude_cli_bridge import (
    BridgeError,
    ClaudeCodeToolBridge,
)
from nanobot.providers.claude_cli_tool_dispatch import ToolDispatch
from nanobot.security.workspace_access import default_workspace_scope


class _FakeTool(Tool):
    def __init__(
        self,
        name: str,
        payload: str = "ok",
        *,
        delay: float = 0.0,
        raise_error: bool = False,
        error_text: str = "boom",
        counter: list[Any] | None = None,
    ) -> None:
        self._name = name
        self._payload = payload
        self._delay = delay
        self._raise_error = raise_error
        self._error_text = error_text
        self.counter = counter if counter is not None else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        self.counter.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise_error:
            raise RuntimeError(self._error_text)
        return f"{self._payload}({json.dumps(kwargs, sort_keys=True)})"


class _RecordingHook(AgentHook):
    """Records every per-call event the dispatch boundary emits."""

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


def _dispatch_for(registry: ToolRegistry, hook: AgentHook | None = None) -> ToolDispatch:
    return ToolDispatch(
        tools=registry,
        hook=hook or AgentHook(),
        context=AgentHookContext(iteration=1, messages=[], session_key="test"),
    )


async def _start_bridge(
    registry: ToolRegistry,
    *,
    tool_names: list[str],
    hook: AgentHook | None = None,
    request_context: RequestContext | None = None,
    model_messages: list[dict[str, Any]] | None = None,
    **bridge_kwargs: Any,
) -> tuple[ClaudeCodeToolBridge, Any]:
    bridge = ClaudeCodeToolBridge(**bridge_kwargs)
    turn = await bridge.start_turn(
        tool_names=tool_names,
        tool_dispatch=_dispatch_for(registry, hook),
        request_context=request_context,
        model_messages=model_messages,
    )
    return bridge, turn


# -------------------------------------------------------------------------- #
# Snapshot policy
# -------------------------------------------------------------------------- #
async def test_turn_requires_at_least_one_registered_tool():
    with pytest.raises(BridgeError):
        await _start_bridge(_registry(), tool_names=[])
    with pytest.raises(BridgeError):
        await _start_bridge(_registry(_FakeTool("a")), tool_names=["unknown"])


async def test_snapshot_exposes_only_opted_in_registered_tools():
    a, b = _FakeTool("a"), _FakeTool("b")
    registry = _registry(a, b)
    _, turn = await _start_bridge(
        registry, tool_names=["a"], allowed_tool_names=frozenset({"a", "b"})
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["a"]
    finally:
        await turn.aclose()


async def test_snapshot_is_frozen_at_turn_start():
    a = _FakeTool("a")
    registry = _registry(a)
    _, turn = await _start_bridge(registry, tool_names=["a"], allowed_tool_names=frozenset({"a"}))
    try:
        # A tool registered after the turn started must not join the surface.
        registry.register(_FakeTool("late"))
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
        assert [t.name for t in tools.tools] == ["a"]
    finally:
        await turn.aclose()


# -------------------------------------------------------------------------- #
# Authentication
# -------------------------------------------------------------------------- #
async def test_missing_or_wrong_token_is_rejected():
    _, turn = await _start_bridge(
        _registry(_FakeTool("a")), tool_names=["a"], allowed_tool_names=frozenset({"a"})
    )
    try:
        async with httpx.AsyncClient() as client:
            no_auth = await client.get(turn.url)
            wrong = await client.get(turn.url, headers={"Authorization": "Bearer wrong"})
        assert no_auth.status_code == 401
        assert wrong.status_code == 401
    finally:
        await turn.aclose()


async def test_stale_token_cannot_act_after_turn_ends():
    _, turn = await _start_bridge(
        _registry(_FakeTool("a")), tool_names=["a"], allowed_tool_names=frozenset({"a"})
    )
    url, token = turn.url, turn.token
    await turn.aclose()
    assert turn.closed
    # The server is shut down (connection refused) or the token revoked (401);
    # in neither case can the stale token act.
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        assert response.status_code == 401
    except httpx.ConnectError:
        pass  # server fully shut down
    # The endpoint must not serve the (closed) turn: a fresh client session must
    # fail to initialize a usable tool call.
    with pytest.raises(Exception):
        async with sse_client(url, headers={"Authorization": f"Bearer {token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("a", {"path": "x"})


# -------------------------------------------------------------------------- #
# Dispatch: the shared boundary, denial, bounds, no retries
# -------------------------------------------------------------------------- #
async def test_tool_call_dispatches_once_through_the_shared_boundary():
    counter: list[Any] = []
    tool = _FakeTool("a", payload="fixture", counter=counter)
    _, turn = await _start_bridge(
        _registry(tool), tool_names=["a"], allowed_tool_names=frozenset({"a"})
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("a", {"path": "fixture://note"})
        assert result.isError is False
        text = result.content[0].text
        assert "fixture({\"path\": \"fixture://note\"})" in text
        assert counter == [{"path": "fixture://note"}]  # exactly one effect, no retry
    finally:
        await turn.aclose()


async def test_unknown_tool_name_cannot_invoke_and_names_nothing_available():
    a, secret = _FakeTool("a"), _FakeTool("secret")
    registry = _registry(a, secret)
    _, turn = await _start_bridge(
        registry, tool_names=["a"], allowed_tool_names=frozenset({"a", "secret"})
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("secret", {"path": "x"})
        assert result.isError is True
        message = result.content[0].text
        assert message == "Tool not available in this turn."
        # The fixed refusal names no available tool, and nothing was executed.
        for available in ("a", "secret"):
            assert available + "(" not in message  # no tool-name/args enumeration
        assert secret.counter == []
    finally:
        await turn.aclose()


async def test_tool_error_is_surfaced_and_never_retried():
    counter: list[Any] = []
    tool = _FakeTool("a", raise_error=True, counter=counter)
    _, turn = await _start_bridge(
        _registry(tool), tool_names=["a"], allowed_tool_names=frozenset({"a"})
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("a", {"path": "x"})
        assert result.isError is True
        assert len(counter) == 1  # no retry on failure
    finally:
        await turn.aclose()


async def test_tool_call_is_bounded_in_time():
    counter: list[Any] = []
    tool = _FakeTool("a", delay=5.0, counter=counter)
    started = time.monotonic()
    _, turn = await _start_bridge(
        _registry(tool),
        tool_names=["a"],
        tool_timeout_s=0.5,
        allowed_tool_names=frozenset({"a"}),
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("a", {"path": "x"})
        assert result.isError is True
        assert "timed out" in result.content[0].text
        assert time.monotonic() - started < 3.0
    finally:
        await turn.aclose()


async def test_result_text_is_bounded_and_event_detail_is_bounded():
    tool = _FakeTool("a", payload="x" * 10_000)
    hook = _RecordingHook()
    _, turn = await _start_bridge(
        _registry(tool),
        tool_names=["a"],
        hook=hook,
        max_result_bytes=1024,
        allowed_tool_names=frozenset({"a"}),
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("a", {"path": "x"})
        wire = result.content[0].text
        # Capped at max_result_bytes plus a bounded truncation note.
        assert len(wire.encode("utf-8")) <= 1024 + 64
        assert "truncated" in wire
        # The hook observes the call's terminal event with the boundary's own
        # bounded detail (never the full payload).
        terminal = hook.events[-1]
        detail = terminal.tool_events[0]["detail"]
        assert len(detail) <= 120 + len("...")
    finally:
        await turn.aclose()


async def test_repeated_workspace_violations_share_the_turn_budget():
    """Two identical bypass attempts in one provider turn escalate exactly as
    the same attempts spread across one ordinary turn's iterations would."""
    counter: list[Any] = []
    tool = _FakeTool(
        "a",
        raise_error=True,
        error_text="Error: path is outside the configured workspace: /etc/passwd",
        counter=counter,
    )
    _, turn = await _start_bridge(
        _registry(tool), tool_names=["a"], allowed_tool_names=frozenset({"a"})
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                first = await session.call_tool("a", {"path": "/etc/passwd"})
                second = await session.call_tool("a", {"path": "/etc/passwd"})
                third = await session.call_tool("a", {"path": "/etc/passwd"})
        assert first.isError is True
        assert second.isError is True
        assert third.isError is True
        # The repeated violation escalates on the third identical attempt.
        assert "refusing repeated workspace-bypass attempts" in third.content[0].text
        assert "refusing repeated workspace-bypass attempts" not in first.content[0].text
        assert len(counter) == 3  # every attempt executed exactly once
    finally:
        await turn.aclose()


# -------------------------------------------------------------------------- #
# Turn isolation
# -------------------------------------------------------------------------- #
async def test_turns_are_isolated_endpoints_with_distinct_tokens():
    registry = _registry(_FakeTool("a"))
    bridge = ClaudeCodeToolBridge(allowed_tool_names=frozenset({"a"}))
    turn1 = await bridge.start_turn(tool_names=["a"], tool_dispatch=_dispatch_for(registry))
    turn2 = await bridge.start_turn(tool_names=["a"], tool_dispatch=_dispatch_for(registry))
    try:
        assert turn1.token != turn2.token
        assert turn1.url != turn2.url
        # turn1's token cannot act on turn2's endpoint (distinct bearer tokens).
        async with httpx.AsyncClient() as client:
            response = await client.get(turn2.url, headers={"Authorization": f"Bearer {turn1.token}"})
        assert response.status_code == 401
    finally:
        await turn1.aclose()
        await turn2.aclose()


# -------------------------------------------------------------------------- #
# Cancellation revokes access, cleans up, and settles exactly once
# -------------------------------------------------------------------------- #
async def test_closing_the_turn_cancels_in_flight_calls_and_revokes():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class _SlowTool(Tool):
        @property
        def name(self) -> str:
            return "slow"

        @property
        def description(self) -> str:
            return "slow"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"path": {"type": "string"}}}

        async def execute(self, **kwargs: Any) -> Any:
            entered.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    hook = _RecordingHook()
    registry = _registry(_SlowTool())
    _, turn = await _start_bridge(
        registry,
        tool_names=["slow"],
        hook=hook,
        tool_timeout_s=60.0,
        allowed_tool_names=frozenset({"slow"}),
    )
    reached_tear_down = False
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                call = asyncio.create_task(session.call_tool("slow", {"path": "x"}))
                await asyncio.wait_for(entered.wait(), timeout=5.0)
                await turn.aclose()
                # The in-flight call must be torn down with the turn, not left to
                # run: it either receives the bridge's cancellation (an isError
                # result) or the client transport is severed (an exception).
                try:
                    result = await asyncio.wait_for(call, timeout=5.0)
                except BaseException:
                    pass
                else:
                    assert result.isError is True
                reached_tear_down = True
    except BaseException:
        # Only tolerate the client transport being severed as the turn is torn
        # down; any earlier failure is a real one and is re-raised.
        if not reached_tear_down:
            raise
    assert turn.closed
    assert entered.is_set() and cancelled.is_set()  # the tool was cancelled, not completed
    # The boundary settled the call exactly once: a start and one terminal
    # cancellation event, and nothing after.
    assert len(hook.events) == 2
    assert hook.events[-1].tool_events[0] == {
        "name": "slow",
        "status": "error",
        "detail": "cancelled",
    }


# -------------------------------------------------------------------------- #
# Turn context: tools see the active turn's own context
# -------------------------------------------------------------------------- #
async def test_each_turn_binds_its_own_request_context():
    class _ContextTool(Tool):
        @property
        def name(self) -> str:
            return "ctx"

        @property
        def description(self) -> str:
            return "records the bound context"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> Any:
            ctx = current_request_context()
            return f"session={ctx.session_key if ctx else None}"

    tool = _ContextTool()
    registry = _registry(tool)
    bridge = ClaudeCodeToolBridge(allowed_tool_names=frozenset({"ctx"}))
    turn1 = await bridge.start_turn(
        tool_names=["ctx"],
        tool_dispatch=_dispatch_for(registry),
        request_context=RequestContext(channel="c1", chat_id="a", session_key="turn-1"),
    )
    try:
        async with sse_client(turn1.url, headers={"Authorization": f"Bearer {turn1.token}"}) as (
            read, write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result1 = await session.call_tool("ctx", {})
    finally:
        await turn1.aclose()
    assert result1.content[0].text == "session=turn-1"

    # A later turn of the same conversation has its own context; the first
    # turn's context is never carried over.
    turn2 = await bridge.start_turn(
        tool_names=["ctx"],
        tool_dispatch=_dispatch_for(registry),
        request_context=RequestContext(channel="c1", chat_id="a", session_key="turn-2"),
    )
    try:
        async with sse_client(turn2.url, headers={"Authorization": f"Bearer {turn2.token}"}) as (
            read, write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result2 = await session.call_tool("ctx", {})
    finally:
        await turn2.aclose()
    assert result2.content[0].text == "session=turn-2"


# -------------------------------------------------------------------------- #
# Turn events: the boundary's hook chain sees each call exactly once
# -------------------------------------------------------------------------- #
async def test_each_call_emits_one_start_and_one_terminal_event_to_the_hook():
    hook = _RecordingHook()
    _, turn = await _start_bridge(
        _registry(_FakeTool("a", payload="v")),
        tool_names=["a"],
        hook=hook,
        allowed_tool_names=frozenset({"a"}),
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("a", {"path": "x"})
        assert result.isError is False
    finally:
        await turn.aclose()
    # Exactly one start and exactly one terminal observation, in order —
    # never synthesized, never replayed.
    assert len(hook.events) == 2
    start, terminal = hook.events
    assert start.tool_calls[0].name == "a"
    assert start.tool_results == [] and start.tool_events == []
    assert terminal.tool_events[0]["name"] == "a"
    assert terminal.tool_events[0]["status"] == "ok"
    assert "v(" in str(terminal.tool_results[0])


# -------------------------------------------------------------------------- #
# Production allowlist: zero tools or the workspace-only read_file
# -------------------------------------------------------------------------- #
async def test_default_bridge_refuses_names_outside_the_production_allowlist():
    # Even a registered tool is refused: the default surface is exact.
    with pytest.raises(BridgeError, match="not in the authorized tool surface"):
        await _start_bridge(_registry(_FakeTool("exec")), tool_names=["exec"])
    with pytest.raises(BridgeError, match="not in the authorized tool surface"):
        await _start_bridge(
            _registry(_FakeTool("read_file")), tool_names=["read_file", "exec"]
        )


async def test_custom_same_named_read_file_is_refused_fail_closed():
    # A same-named custom or replacement tool is not the trusted
    # workspace-scoped read tool: no endpoint is opened.
    registry = _registry(_FakeTool("read_file"))
    with pytest.raises(BridgeError):
        await _start_bridge(
            registry,
            tool_names=["read_file"],
            request_context=_read_file_context(Path(".")),
        )


def _read_file_context(
    workspace: Path,
    *,
    scoped: bool = True,
    context_workspace: Path | None = None,
) -> RequestContext:
    ctx = RequestContext(
        channel="test",
        chat_id="chat-1",
        workspace=context_workspace if context_workspace is not None else workspace,
    )
    if scoped:
        ctx = dataclasses.replace(
            ctx,
            workspace_scope=default_workspace_scope(workspace, restrict_to_workspace=True),
        )
    return ctx


async def test_read_file_requires_a_trusted_turn_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    tool = ReadFileTool(workspace=ws)
    registry = _registry(tool)

    # No turn context at all: the workspace cannot be proven, so refuse.
    with pytest.raises(BridgeError):
        await _start_bridge(registry, tool_names=["read_file"])
    # A context with no workspace and no scope proves nothing.
    with pytest.raises(BridgeError):
        await _start_bridge(
            registry,
            tool_names=["read_file"],
            request_context=RequestContext(channel="test", chat_id="c", workspace=None),
        )
    # Unscoped context naming a different workspace than the tool's: the
    # turn's read_file is not bound to this turn's workspace, so refuse.
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(BridgeError):
        await _start_bridge(
            registry,
            tool_names=["read_file"],
            request_context=_read_file_context(other, scoped=False),
        )


async def test_scoped_read_file_serves_workspace_reads_and_denies_escapes(tmp_path):
    """The live test: a real ``read_file`` through the actual scoped open.

    The allowed read and every denial go through the same ``WorkspaceFiles``
    open that serves the read — check and open are one operation, so no
    check-then-reopen window exists.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "note.txt").write_text("line one\nline two\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("top secret outside the workspace\n", encoding="utf-8")
    os.symlink(secret, ws / "link.txt")  # escaping symlink

    tool = ReadFileTool(workspace=ws)
    registry = _registry(tool)
    _, turn = await _start_bridge(
        registry,
        tool_names=["read_file"],
        request_context=_read_file_context(ws),
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                allowed = await session.call_tool("read_file", {"path": "note.txt"})
                absolute_inside = await session.call_tool(
                    "read_file", {"path": str(ws / "note.txt")}
                )
                absolute_outside = await session.call_tool(
                    "read_file", {"path": str(secret)}
                )
                traversal = await session.call_tool("read_file", {"path": "../../outside/secret.txt"})
                escaping_link = await session.call_tool("read_file", {"path": "link.txt"})
        # The allowed reads (relative and absolute, both inside the workspace)
        # succeed through the scoped open.
        assert allowed.isError is False
        assert "line one" in allowed.content[0].text
        assert absolute_inside.isError is False
        assert "line one" in absolute_inside.content[0].text
        # Absolute outside paths, relative traversal, and escaping symlinks are
        # denied by the same scoped open; the secret never crosses the bridge.
        for denied in (absolute_outside, traversal, escaping_link):
            assert denied.isError is True
            assert "top secret" not in denied.content[0].text
    finally:
        await turn.aclose()


async def test_unscoped_read_file_bound_to_the_turn_workspace_is_served(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "note.txt").write_text("payload\n", encoding="utf-8")
    tool = ReadFileTool(workspace=ws)
    registry = _registry(tool)
    # No scope, but the turn's trusted workspace positively equals the tool's
    # construction workspace: the binding is proven, so the turn may start.
    _, turn = await _start_bridge(
        registry,
        tool_names=["read_file"],
        request_context=_read_file_context(ws, scoped=False),
    )
    try:
        async with sse_client(turn.url, headers={"Authorization": f"Bearer {turn.token}"}) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("read_file", {"path": "note.txt"})
        assert result.isError is False
        assert "payload" in result.content[0].text
    finally:
        await turn.aclose()
