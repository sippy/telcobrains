"""Ordinary-turn tool call execution, delegated to the shared dispatch boundary.

The per-call behavior (validation, hooks, receipts, violation policy, error
shaping, ordering, settlement) lives in :mod:`nanobot.agent.tools.dispatch`;
this module keeps the runner-facing batch entry point.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.dispatch import (
    ToolCallResult,
    ToolDispatch,
    _classify_violation,
    is_ssrf_violation,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import ToolCallRequest


async def execute_tool_calls(
    tools: ToolRegistry,
    tool_calls: list[ToolCallRequest],
    *,
    concurrent: bool,
    external_lookup_counts: dict[str, int],
    workspace_violation_counts: dict[str, int],
    hook: AgentHook,
    context: AgentHookContext,
    model_messages: list[dict[str, Any]] | None = None,
    compacted_tool_results: set[str] | None = None,
    on_settled: Callable[[list[ToolCallResult]], Awaitable[None]] | None = None,
    dispatch: ToolDispatch | None = None,
) -> None:
    """Own tool tasks through settlement, including results whose notification failed."""
    if dispatch is None:
        dispatch = ToolDispatch(
            tools=tools,
            hook=hook,
            context=context,
            concurrent=concurrent,
            model_messages=model_messages,
            compacted_tool_results=compacted_tool_results,
            external_lookup_counts=external_lookup_counts,
            workspace_violation_counts=workspace_violation_counts,
        )
    await dispatch.dispatch_batches(tool_calls, on_settled=on_settled)
