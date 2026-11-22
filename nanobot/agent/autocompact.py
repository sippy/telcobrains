from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from nanobot.agent.memory import min_tail_skip_marker
from nanobot.events import NO_EVENTS, EventSink
from nanobot.session.keys import is_dream_session
from nanobot.session.manager import Session, SessionManager
from nanobot.session.summary import (
    SessionSummary,
    session_summary_from_metadata,
)

if TYPE_CHECKING:
    from nanobot.agent.context_governance import ContextGovernanceConfig
    from nanobot.agent.memory import Consolidator
    from nanobot.utils.llm_runtime import LLMRuntime

SessionEventFactory = Callable[[str], EventSink]


class AutoCompact:
    def __init__(
        self,
        sessions: SessionManager,
        consolidator: Consolidator,
        session_ttl_minutes: int = 0,
        bind_events: SessionEventFactory | None = None,
        *,
        session_lock: Callable[[str], asyncio.Lock],
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, SessionSummary] = {}
        self._bind_events = bind_events
        self._session_lock = session_lock

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def _is_expired(self, ts: datetime | str | None,
                    now: datetime | None = None) -> bool:
        if not self.enabled or not ts:
            return False
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            current = now or datetime.now()
            if getattr(ts, "tzinfo", None) is not None or current.tzinfo is not None:
                idle_seconds = current.timestamp() - ts.timestamp()
            else:
                idle_seconds = (current - ts).total_seconds()
        except (OSError, OverflowError, TypeError, ValueError):
            # list_sessions() forwards raw persisted metadata; an unusable value
            # must not escape the idle scan and stop the agent loop.
            return False
        return idle_seconds >= self._ttl * 60

    @staticmethod
    def _already_compacted(session: Session) -> bool:
        # A protected suffix is still unarchived, but it is not new activity.
        summary = session_summary_from_metadata(
            session.metadata, fallback_last_active=session.created_at,
        )
        return summary is not None and summary["last_active"] == session.updated_at.isoformat()

    @staticmethod
    def _floor_already_blocked(
        session: Session, runtime: LLMRuntime, replay: ContextGovernanceConfig | None,
    ) -> bool:
        marker = session.metadata.get("_min_tail_skip")
        return (
            isinstance(marker, list)
            and replay is not None
            and replay.min_verbatim_tail_tokens > 0
            and marker == min_tail_skip_marker(session, runtime, replay)
        )

    async def check_expired(
        self,
        schedule_background: Callable[[Coroutine[Any, Any, None]], None],
        resolve_runtime: Callable[
            [Session], Awaitable[tuple[LLMRuntime, list[dict[str, Any]], bool, ContextGovernanceConfig]]
        ],
        active_session_keys: Collection[str] = (),
    ) -> None:
        """Schedule idle archival without blocking the event loop."""
        if not self.enabled:
            return
        now = datetime.now()
        active_keys = set(active_session_keys)
        for info in await self.sessions.state.list_sessions():
            key = info.get("key", "")
            if (
                not key
                or is_dream_session(key)
                or key.startswith("subagent:")
                or key in self._archiving
            ):
                continue
            if key in active_keys or not self._is_expired(info.get("updated_at"), now):
                continue
            if not await self.sessions.state.has_unarchived_messages(key):
                continue
            session = await self.sessions.state.get_header(key)
            if self._already_compacted(session):
                continue
            try:
                runtime, tools, no_tool_contract, replay_config = await resolve_runtime(session)
            except (KeyError, ValueError):
                continue
            if self._floor_already_blocked(session, runtime, replay_config):
                continue
            self._archiving.add(key)
            schedule_background(self._archive(
                key,
                runtime=runtime,
                tools=tools,
                no_tool_contract=no_tool_contract,
                replay_config=replay_config,
            ))

    async def _archive(
        self,
        key: str,
        *,
        runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        replay_config: ContextGovernanceConfig | None = None,
    ) -> None:
        if is_dream_session(key):
            self._archiving.discard(key)
            return
        try:
            async with self._session_lock(key):
                # A turn may have refreshed activity while this archive waited.
                session = await self.sessions.state.get_header(key)
                if (
                    not self._is_expired(session.updated_at)
                    or self._already_compacted(session)
                    or self._floor_already_blocked(session, runtime, replay_config)
                ):
                    return
                # Idle compaction and turns share one transcript boundary.
                summary = await self.consolidator.compact_idle_session(
                    key,
                    runtime=runtime,
                    tools=tools,
                    no_tool_contract=no_tool_contract,
                    replay_config=replay_config,
                    events=self._bind_events(key) if self._bind_events else NO_EVENTS,
                    reason="idle",
                )
                if summary:
                    session = await self.sessions.state.get(key)
                    stored = session_summary_from_metadata(
                        session.metadata,
                        fallback_last_active=session.updated_at,
                    )
                    if stored is not None:
                        self._summaries[key] = stored
        except Exception:
            logger.exception("Auto-compact: failed for {}", key)
        finally:
            self._archiving.discard(key)

    async def prepare_session(
        self,
        session: Session,
        key: str,
    ) -> tuple[Session, SessionSummary | None]:
        """Resolve a summary while preserving the turn's owner-issued draft."""
        if is_dream_session(key):
            self._archiving.discard(key)
            self._summaries.pop(key, None)
            return session, None
        entry = self._summaries.pop(key, None)
        if entry:
            return session, entry
        # Persisted metadata may outlive schema changes; a malformed summary must
        # not abort turn preparation.
        return session, session_summary_from_metadata(
            session.metadata,
            fallback_last_active=session.updated_at,
        )
