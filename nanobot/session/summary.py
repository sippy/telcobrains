from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict, cast

from nanobot.session.history_visibility import is_hidden_history_message

SUMMARY_CONTINUATION_TEXT = (
    "Continue the active task from the working-memory checkpoint above."
)

def is_summary_checkpoint(message: Mapping[str, Any]) -> bool:
    """Identify the durable boundary of a replacement summary."""
    return (
        is_hidden_history_message(message)
        and message.get("content") == SUMMARY_CONTINUATION_TEXT
    )


class SessionSummary(TypedDict):
    text: str
    last_active: str


@dataclass(frozen=True, slots=True)
class SessionSummaryCheckpoint:
    """A replacement summary and the raw transcript boundary it covers."""

    summary: str
    transcript_boundary: int


@dataclass(frozen=True, slots=True)
class CompactorInputBudget:
    """The history consolidator's actual input capacity for a transcript prefix.

    ``input_budget`` is the compactor's context window minus its reserved
    output (mirroring ``Consolidator.summarize_transcript``). ``fixed_overhead_tokens``
    covers the summarization prompt and the tool definitions offered to the
    compactor. ``max_summary_tokens`` caps the summary the compactor can emit;
    the foreground reserves it when checking headroom for a retained tail.
    """

    input_budget: int
    fixed_overhead_tokens: int
    max_summary_tokens: int


def session_summary_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    fallback_last_active: datetime,
) -> SessionSummary | None:
    raw: object = metadata.get("_last_summary") if metadata is not None else None
    if not isinstance(raw, Mapping):
        return None
    summary_data = cast(Mapping[str, object], raw)
    text = summary_data.get("text")
    if not isinstance(text, str) or not text:
        return None
    raw_last_active = summary_data.get("last_active")
    if isinstance(raw_last_active, str):
        try:
            datetime.fromisoformat(raw_last_active)
            last_active = raw_last_active
        except ValueError:
            last_active = fallback_last_active.isoformat()
    else:
        last_active = fallback_last_active.isoformat()
    return {"text": text, "last_active": last_active}
