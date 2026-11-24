"""The provider-side view of the shared per-turn tool dispatch boundary.

The Claude CLI tool bridge executes the CLI's tool calls through the active
turn's :class:`~nanobot.agent.tools.dispatch.ToolDispatch` — the same object
the agent runner uses for its own model-initiated calls — so registration,
schema validation, hooks (audit/progress/consent), read receipts, violation
escalation, error shaping, cancellation, and exactly-once observable events
are identical for both. This module is a thin re-export kept in
``providers/`` so provider code depends only on the providers package surface
and never imports agent internals beyond it.
"""

from nanobot.agent.tools.dispatch import ToolDispatch

__all__ = ["ToolDispatch"]
