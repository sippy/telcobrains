# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nanobot.agent.subagent import SubagentControlError
from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context, current_request_session_key
from nanobot.agent.tools.schema import (
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.session.manager import SessionPolicy

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.context import ToolContext
    from nanobot.utils.llm_runtime import LLMRuntime


@tool_parameters(tool_parameters_schema(
    action=StringSchema(
        "create starts a task; follow_up starts a linked run from the saved history of a completed or cancelled task "
        "(cancelled history may be partial), or of a task stopped at its iteration limit: the new run keeps the saved "
        "conversation and tool results, starts with a fresh full iteration budget, and never reruns old tool calls or "
        "restores old processes; send queues a message; cancel requests an idempotent stop; "
        "check returns task status, receipts, and results.",
        enum=("create", "follow_up", "send", "cancel", "check"),
    ),
    task=StringSchema("Required for create: the task for the subagent to complete.", min_length=1),
    label=StringSchema("Display label for create or follow_up."),
    model_preset=StringSchema(
        "For create or follow_up: child-only preset. Omit to inherit the parent on create, or reuse the saved preset on follow_up.",
        min_length=1,
    ),
    temperature=NumberSchema(
        description="Sampling temperature for create or follow_up; uses the selected runtime's temperature if omitted.",
        minimum=0.0,
        maximum=2.0,
    ),
    wait=BooleanSchema(
        description=(
            "For create or follow_up: true waits for the result; false runs in background with a completion notification. "
            "Temporary sessions and hosts unable to receive background results always wait."
        ),
        default=False,
    ),
    task_id=StringSchema(
        "Full task ID returned by create. Required for follow_up/send/cancel; optional for check. "
        "Omit for check to list the current session's active and recently finished tasks.",
        min_length=1,
    ),
    message=StringSchema(
        "Required for follow_up or send: text, at most 8192 UTF-8 bytes. For send, accepted means queued, not delivered. "
        "Delivery receipts mean injected into the task transcript, not that the task acted on the message.",
        max_length=8192,
    ),
    required=["action"],
))
class SubagentTool(Tool):
    """Manage the current session's private subagent tasks."""

    def __init__(
        self,
        manager: SubagentManager,
        resolve_model_preset: Callable[[str], LLMRuntime] | None = None,
    ):
        self._manager = manager
        self._resolve_model_preset = resolve_model_preset

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        if ctx.subagent_manager is None:
            raise RuntimeError("SubagentTool requires an initialized subagent manager")
        return cls(ctx.subagent_manager, ctx.resolve_model_preset)

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def description(self) -> str:
        return (
            "Manage the current session's private subagent tasks. "
            "Background results arrive automatically; avoid repeated polling. "
            "Check with a known task_id to read older results. "
            "Partial results or iteration limits do not mean completion. "
            "A task stopped at its iteration limit can be resumed with follow_up, "
            "which continues from its saved history with a fresh iteration budget."
        )

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(
        self, action: str, task_id: str | None = None, message: str | None = None,
        task: str | None = None, label: str | None = None,
        temperature: float | None = None, wait: bool = False,
        model_preset: str | None = None, **kwargs: Any,
    ) -> str:
        if action == "create":
            if not task or not task.strip():
                return ToolResult.error("Error: create requires a non-empty task")
            return await self._create_task(task, label, temperature, wait, model_preset)
        owner = current_request_session_key()
        try:
            if action == "follow_up":
                if not task_id or not message or not message.strip() or len(message.encode("utf-8")) > 8192:
                    return ToolResult.error("Error: follow_up requires task_id and a non-empty message of at most 8192 UTF-8 bytes.")
                request_ctx = current_request_context()
                if request_ctx is None or not request_ctx.persist_session:
                    return ToolResult.error("Error: follow_up requires a persistent conversation.")
                source = self._manager.follow_up_source(task_id, owner)
                preset = model_preset if model_preset is not None else source.model_preset
                if preset is None:
                    return ToolResult.error("Error: the original task has no saved model preset; specify model_preset for the follow-up.")
                return await self._create_task(message, label, temperature, wait, preset, continued_from_task_id=task_id)
            if action == "check":
                if not owner:
                    raise SubagentControlError("task unavailable")
                if task_id is not None:
                    return json.dumps(self._manager.check(task_id, owner).as_dict(), ensure_ascii=False)
                return json.dumps({
                    "tasks": [status.as_dict()
                              for status in self._manager.statuses_for_session(owner, include_history=False).values()],
                }, ensure_ascii=False)
            if action in {"send", "cancel"}:
                if not task_id:
                    return ToolResult.error(f"Error: {action} requires task_id")
                if action == "send":
                    return json.dumps(await self._manager.send(task_id, owner, message))
                status = await self._manager.cancel(task_id, owner)
                return json.dumps(status.as_dict(), ensure_ascii=False)
            return ToolResult.error("Error: unknown subagent action")
        except SubagentControlError as exc:
            return ToolResult.error(f"Error: {exc}")

    async def _create_task(
        self, task: str, label: str | None, temperature: float | None, wait: bool,
        model_preset: str | None,
        *, continued_from_task_id: str | None = None,
    ) -> str:
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            return ToolResult.error("Error: create requires an active model runtime")
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or (
            f"{origin_channel}:{origin_chat_id}" if origin_channel and origin_chat_id else None
        )
        if not session_key:
            return ToolResult.error("Error: create requires an active session identity")
        runtime = request_ctx.runtime
        if model_preset is not None:
            if self._resolve_model_preset is None:
                return ToolResult.error("Error: subagent model preset selection is unavailable.")
            try:
                runtime = self._resolve_model_preset(model_preset)
            except KeyError as exc:
                detail = exc.args[0] if exc.args else f"Model preset {model_preset!r} not found."
                return ToolResult.error(f"Error: {detail}")
            except ValueError as exc:
                return ToolResult.error(f"Error: could not resolve subagent model preset {model_preset!r}: {exc}")
        inline = wait or not request_ctx.can_receive_background_results or not request_ctx.persist_session
        method = self._manager.run_inline if inline else self._manager.spawn
        continuation = {"continued_from_task_id": continued_from_task_id} if continued_from_task_id else {}
        return await method(
            task=task,
            runtime=runtime,
            label=label,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            origin_turn_id=request_ctx.turn_id,
            temperature=temperature,
            workspace_scope=current_workspace_scope(),
            session_policy=SessionPolicy(
                persist=request_ctx.persist_session,
                log_content=request_ctx.log_content,
            ),
            **continuation,
        )
