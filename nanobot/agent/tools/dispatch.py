"""The per-turn tool execution boundary shared by every tool caller.

One :class:`ToolDispatch` is created per agent-iteration and bundles the
turn's own execution boundary: the tool registry, the turn's hook chain
(audit/progress/consent lifecycle), the iteration context, and the turn's
policy accounting (repeated external-lookup blocks and workspace-violation
escalation).

Every tool call — whether initiated by the model in an ordinary runner
iteration or initiated by a hosted client inside a provider turn (the
Claude CLI tool bridge) — executes through :meth:`ToolDispatch.dispatch`,
which applies the same registration/schema validation, hooks, read
receipts, violation escalation, error shaping, cancellation handling, and
exactly-once observable events. The runner attaches the active turn's
dispatch to the provider call context so such consumers can reach the
*active* turn's boundary; calls made outside an agent turn carry no
dispatch and cannot authorize tool execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import tool_log_content_allowed
from nanobot.agent.tools.file_state import file_read_context
from nanobot.agent.tools.registry import ToolRegistry, is_tool_error_result, with_tool_error_hint
from nanobot.providers.base import ToolCallRequest
from nanobot.utils.runtime import (
    repeated_external_lookup_error,
    repeated_workspace_violation_error,
)

# SSRF is a hard security block at the tool boundary, but the agent turn
# should recover conversationally instead of aborting the runtime.
_SSRF_MARKERS: tuple[str, ...] = (
    "internal/private url detected",
    "private/internal address",
    "private address",
)
_SSRF_BOUNDARY_NOTE = (
    "This is a non-bypassable security boundary. Stop trying to access "
    "private/internal URLs. Do not retry with curl, wget, encoded IPs, "
    "alternate DNS, redirects, proxies, or another tool. Ask the user for "
    "local files, logs, screenshots, or an explicit safe public URL instead. "
    "If the user explicitly trusts this private URL, ask them to whitelist "
    "the exact IP/CIDR via tools.ssrfWhitelist."
)
_WORKSPACE_VIOLATION_MARKERS: tuple[str, ...] = (
    "outside the configured workspace",
    "outside allowed directory",
    "working_dir is outside",
    "working_dir could not be resolved",
    "path outside working dir",
    "path traversal detected",
)


@dataclass(frozen=True)
class ToolCallResult:
    tool_call: ToolCallRequest
    result: Any
    event: dict[str, str]


@dataclass(eq=False)
class ToolDispatch:
    """One turn's boundary for executing tool calls.

    ``external_lookup_counts`` and ``workspace_violation_counts`` are the
    turn's policy accounting: every call dispatched through this object
    shares the same per-turn budgets, so a hosted client that makes several
    calls in one provider turn is limited exactly as a model that makes the
    same calls across one turn's iterations. ``model_messages`` indexes the
    model-visible tool results for read receipts; ``None`` means no receipts
    are visible yet (the hosted consumer supplies its own view when it has
    one).
    """

    tools: ToolRegistry
    hook: AgentHook
    context: AgentHookContext
    concurrent: bool = True
    model_messages: list[dict[str, Any]] | None = None
    compacted_tool_results: set[str] | None = None
    external_lookup_counts: dict[str, int] | None = None
    workspace_violation_counts: dict[str, int] | None = None

    def __init__(
        self,
        tools: ToolRegistry,
        hook: AgentHook,
        context: AgentHookContext,
        *,
        concurrent: bool = True,
        model_messages: list[dict[str, Any]] | None = None,
        compacted_tool_results: set[str] | None = None,
        external_lookup_counts: dict[str, int] | None = None,
        workspace_violation_counts: dict[str, int] | None = None,
    ) -> None:
        self.tools = tools
        self.hook = hook
        self.context = context
        self.concurrent = concurrent
        self.model_messages = model_messages
        self.compacted_tool_results = compacted_tool_results
        # The turn's policy accounting is shared by reference: every dispatch
        # created for the same run (one per iteration, plus hosted consumer
        # views) limits the turn as one unit.
        self.external_lookup_counts = (
            external_lookup_counts if external_lookup_counts is not None else {}
        )
        self.workspace_violation_counts = (
            workspace_violation_counts if workspace_violation_counts is not None else {}
        )
        self._read_index: dict[str, str] | None = None
        # The execution closure is built once here and captures this turn's
        # registry (both its validation/preparation and its execution): no
        # consumer of the boundary (including a hosted provider turn) can
        # substitute the runner or rewire tool resolution after construction.
        registry = self.tools
        prepare_call = getattr(self.tools, "prepare_call", None)
        self._prepare_call = prepare_call if callable(prepare_call) else None

        async def _run(tool_call: ToolCallRequest, tool: Any, params: Any) -> Any:
            if tool is not None:
                return await tool.execute(**params)
            return await registry.execute(tool_call.name, params)

        self._run = _run

    def _read_results(self) -> dict[str, str]:
        """Index the model-visible read receipts once, on first use."""
        if self._read_index is None:
            self._read_index = {
                message["tool_call_id"]: message["content"]
                for message in self.model_messages or []
                if message.get("role") == "tool"
                and isinstance(message.get("tool_call_id"), str)
                and isinstance(message.get("content"), str)
                and message["tool_call_id"] not in (self.compacted_tool_results or ())
            }
        return self._read_index

    async def dispatch(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Run one tool call through this turn's full execution boundary.

        Emits the call's start event, executes with the turn's validation,
        hooks, receipts, violation policy, and error shaping, emits exactly
        one terminal event (result, error, or cancellation), and returns the
        settled result. Cancellation propagates after the terminal event.
        """
        call_context, result, event = await self._settle(tool_call)
        # The settled result is recorded before the terminal event, so a
        # failing or cancelled terminal notification never discards it; the
        # event is still emitted before the result is returned, so a consumer
        # never observes the effect twice or out of order.
        entry = ToolCallResult(tool_call, result, event)
        await self._notify_terminal(call_context, entry)
        return entry

    async def _settle(self, tool_call: ToolCallRequest) -> tuple[AgentHookContext, Any, dict[str, str]]:
        """Start event plus execution; return the call context and settled result.

        On cancellation, emits the terminal cancellation event (without waiting
        for a blocked consumer) and re-raises; a cancelled call settles to no
        result, so its checkpoint recovery marks it interrupted.
        """
        call_context = replace(self.context, tool_calls=[tool_call], tool_results=[], tool_events=[])
        try:
            await self.hook.on_tool_execution_event(call_context)
            result, event = await self._execute_call(tool_call)
        except asyncio.CancelledError:
            cancelled = replace(call_context, tool_results=["Tool execution cancelled"], tool_events=[{
                "name": tool_call.name, "status": "error", "detail": "cancelled",
            }])
            # Give immediate consumers a terminal event, but do not wait for a
            # blocked consumer during cancellation. Own and drain the attempt.
            notification = asyncio.create_task(self.hook.on_tool_execution_event(cancelled))
            try:
                await asyncio.sleep(0)
            finally:
                notification.cancel()
                await asyncio.gather(notification, return_exceptions=True)
            raise
        return call_context, result, event

    async def _notify_terminal(self, call_context: AgentHookContext, entry: ToolCallResult) -> None:
        await self.hook.on_tool_execution_event(replace(
            call_context, tool_results=[entry.result], tool_events=[entry.event],
        ))

    async def dispatch_batches(
        self,
        tool_calls: list[ToolCallRequest],
        *,
        on_settled: Callable[[list[ToolCallResult]], Awaitable[None]] | None = None,
    ) -> list[ToolCallResult]:
        """Own a batch of calls through settlement, in the turn's ordering.

        Concurrent batches are formed from the tools' ``concurrency_safe``
        marks, exactly as an ordinary turn orders its model-initiated calls.
        ``on_settled`` runs on every exit with the results settled so far,
        including results whose notification failed.
        """
        tool_results: list[ToolCallResult | None] = [None] * len(tool_calls)

        async def execute(index: int, tool_call: ToolCallRequest) -> None:
            call_context, result, event = await self._settle(tool_call)
            # Record the settled result before the terminal event: a
            # notification failure (or cancellation landing on the event) must
            # not keep it from on_settled.
            tool_results[index] = ToolCallResult(tool_call, result, event)
            await self._notify_terminal(call_context, tool_results[index])

        try:
            offset = 0
            for batch in _partition_tool_batches(self.tools, tool_calls, concurrent=self.concurrent):
                if self.concurrent and len(batch) > 1:
                    tasks = [
                        asyncio.create_task(execute(offset + index, tool_call))
                        for index, tool_call in enumerate(batch)
                    ]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        # gather propagates ordinary errors before siblings finish.
                        # The run must not release ownership while they can act.
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                else:
                    for index, tool_call in enumerate(batch):
                        await execute(offset + index, tool_call)
                offset += len(batch)
        finally:
            settled = [result for result in tool_results if result is not None]
            if settled and on_settled is not None:
                await on_settled(settled)
        return [result for result in tool_results if result is not None]

    async def _execute_call(self, tool_call: ToolCallRequest) -> tuple[Any, dict[str, str]]:
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            self.external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            return with_tool_error_hint(lookup_error), event

        prepare_call = cast(
            Callable[[str, Any], object] | None,
            self._prepare_call,
        )
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            prepared = prepare_call(tool_call.name, tool_call.arguments)
            if isinstance(prepared, tuple):
                prepared_tuple = cast(tuple[object, ...], prepared)
                if len(prepared_tuple) == 3:
                    tool, params, prep_error = cast(tuple[Any, Any, str | None], prepared_tuple)
        if prep_error:
            payload = with_tool_error_hint(prep_error)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            handled = _classify_violation(
                raw_text=prep_error,
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=self.workspace_violation_counts,
            )
            if handled is not None:
                return handled
            return payload, event

        await self.hook.before_execute_tool(self.context, tool_call, tool, params)
        try:
            with file_read_context(tool_call.id, self._read_results):
                result = await self._run(tool_call, tool, params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.hook.on_execute_tool_error(self.context, tool_call, tool, params, exc)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            payload = with_tool_error_hint(f"Error: {type(exc).__name__}: {exc}")
            handled = _classify_violation(
                raw_text=str(exc),
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=self.workspace_violation_counts,
            )
            if handled is not None:
                return handled
            return payload, event

        if is_tool_error_result(result):
            await self.hook.on_execute_tool_error(self.context, tool_call, tool, params, result)
            payload = with_tool_error_hint(result)
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            if isinstance(result, ToolResult) and result.error_code:
                event["error_code"] = result.error_code
            handled = _classify_violation(
                raw_text=result,
                soft_payload=payload,
                event=event,
                tool_call=tool_call,
                workspace_violation_counts=self.workspace_violation_counts,
            )
            if handled is not None:
                return handled
            return payload, event

        await self.hook.after_execute_tool(self.context, tool_call, tool, params, result)

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}


def is_ssrf_violation(text: str) -> bool:
    """Return whether a tool error describes a blocked private-network request."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _SSRF_MARKERS)


def _is_workspace_violation(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if is_ssrf_violation(lowered):
        return True
    return any(marker in lowered for marker in _WORKSPACE_VIOLATION_MARKERS)


def _classify_violation(
    *,
    raw_text: str,
    soft_payload: str,
    event: dict[str, str],
    tool_call: ToolCallRequest,
    workspace_violation_counts: dict[str, int],
) -> tuple[Any, dict[str, str]] | None:
    if is_ssrf_violation(raw_text):
        logger.warning(
            "Tool {} blocked by SSRF guard; returning non-retryable tool error: {}",
            tool_call.name,
            raw_text.replace("\n", " ").strip()[:200]
            if tool_log_content_allowed() else "[content hidden]",
        )
        event["detail"] = _event_detail("ssrf_violation: ", raw_text)
        return _ssrf_soft_payload(raw_text), event

    if _is_workspace_violation(raw_text):
        escalation = repeated_workspace_violation_error(
            tool_call.name,
            tool_call.arguments,
            workspace_violation_counts,
        )
        event["detail"] = _event_detail("workspace_violation: ", raw_text)
        if escalation is not None:
            logger.warning(
                "Tool {} hit workspace boundary repeatedly; escalating hint",
                tool_call.name,
            )
            event["detail"] = _event_detail(
                "workspace_violation_escalated: ",
                raw_text,
            )
            return escalation, event
        return soft_payload, event

    return None


def _ssrf_soft_payload(raw_text: str) -> str:
    text = raw_text.strip() or "Error: request blocked by SSRF guard"
    return f"{text}\n\n{_SSRF_BOUNDARY_NOTE}"


def _event_detail(prefix: str, text: str, limit: int = 160) -> str:
    return (prefix + text.replace("\n", " ").strip())[:limit]


def _partition_tool_batches(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
) -> list[list[ToolCallRequest]]:
    if not concurrent:
        return [[tool_call] for tool_call in tool_calls]

    batches: list[list[ToolCallRequest]] = []
    current: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        get_tool = cast(Callable[[str], Any] | None, getattr(tools, "get", None))
        tool = get_tool(tool_call.name) if callable(get_tool) else None
        can_batch = bool(tool and tool.concurrency_safe)
        if can_batch:
            current.append(tool_call)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([tool_call])
    if current:
        batches.append(current)
    return batches
