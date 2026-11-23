# Tool schemas are installed by the ``@tool_parameters`` class decorator at
# runtime; static analyzers cannot observe that it clears ``parameters`` from
# ``__abstractmethods__`` before these classes are instantiated.
# pyright: reportAbstractUsage=false, reportPrivateUsage=false

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import weakref
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, cast
from uuid import uuid4

from loguru import logger

from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    PrefixCutResult,
)
from nanobot.config.paths import DREAM_CURSOR_PATH, MEMORY_CURSOR_PATH, MEMORY_JOURNAL_PATH
from nanobot.events import NO_EVENTS, CompactionReason, ContextCompactionEvent, EventSink
from nanobot.llm_usage.context import llm_usage_source
from nanobot.llm_usage.types import LLMUsage
from nanobot.providers.base import LLMProvider, LLMResponse, ProviderConversationState
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.runtime_context import public_history_messages
from nanobot.session.manager import Session, SessionManager
from nanobot.session.message_content import history_message_text
from nanobot.session.summary import (
    SUMMARY_CONTINUATION_TEXT,
    CompactorInputBudget,
    is_summary_checkpoint,
    session_summary_from_metadata,
)
from nanobot.utils.cancellation import shield_and_drain
from nanobot.utils.gitstore import MEMORY_CONTENT_PATHS, MEMORY_WIKI_PATH, GitStore
from nanobot.utils.helpers import (
    atomic_write_lines,
    build_assistant_message,
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    strip_think,
    truncate_text,
    truncate_text_to_tokens,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.workspace_prompts import (
    WORKSPACE_PROMPT_MAX_CHARS,
    has_workspace_prompt_override,
    load_workspace_prompt_override,
    workspace_prompt_file,
)

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.utils.llm_runtime import LLMRuntime

# "sdk" is a logging-only trigger: SDK compactions run the same lifecycle,
# but wire events keep the client-facing CompactionReason set, so events map
# it to no reason.
CompactionTrigger = CompactionReason | Literal["sdk"]


class MemoryStore:
    """File I/O for the memory journal, resident memory, and on-demand wiki."""

    _DEFAULT_MAX_HISTORY = 1000
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.wiki_dir = workspace / MEMORY_WIKI_PATH
        self.history_file = workspace / MEMORY_JOURNAL_PATH
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = workspace / MEMORY_CURSOR_PATH
        self._dream_cursor_file = workspace / DREAM_CURSOR_PATH
        self._corruption_logged = False
        self._malformed_entry_logged = False
        self._oversize_logged = False
        self._dream_prompt_oversize_logged = False
        self._append_lock = threading.Lock()
        self.dream_lock = asyncio.Lock()
        self._git = GitStore.for_memory(workspace)
        self._maybe_migrate_legacy_history()

    @property
    def git(self) -> GitStore:
        return self._git


    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._cursor_file.write_text(str(last_cursor), encoding="utf-8")
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                self._dream_cursor_file.write_text(str(last_cursor), encoding="utf-8")

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate


    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")


    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")


    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        """Return long-term memory formatted for context injection."""
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""


    def _normalize_history_entry(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
    ) -> str:
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        raw = entry.rstrip()
        content = strip_think(raw)
        if len(content) > limit:
            if not self._oversize_logged:
                self._oversize_logged = True
                logger.warning(
                    "history entry exceeds {} chars ({}); truncating. "
                    "Usually means a caller forgot its own cap; "
                    "further occurrences suppressed.",
                    limit,
                    len(content),
                )
            content = truncate_text(content, limit)
        return content

    def append_history(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor.

        Entries are passed through `strip_think` to drop template-level leaks
        (e.g. unclosed `<think` prefixes, `<channel|>` markers) before being
        persisted. If the cleaned content is empty but the raw entry wasn't,
        the record is persisted with an empty string rather than falling back
        to the raw leak — otherwise `strip_think`'s guarantees would be
        undone when Dream consumes the journal entry.

        A defensive cap (*max_chars*, default ``_HISTORY_ENTRY_HARD_CAP``) is
        applied as a final safety net: individual callers should cap their own
        content more tightly; this default only exists to catch unintentional
        large writes (e.g. an LLM echoing its input back as a "summary").
        """
        content = self._normalize_history_entry(entry, max_chars=max_chars)
        if entry.rstrip() and not content:
            logger.debug(
                "history entry stripped to empty (likely template leak); "
                "persisting empty content to avoid re-polluting Dream input",
            )
        return self._append_history_record(content, session_key=session_key)

    def _append_history_record(self, content: str, *, session_key: str | None = None) -> int:
        """Persist already sanitized, bounded content without rewriting chunk edges."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Cursor allocation and the append must be atomic: concurrent writers
        # could otherwise read the same current cursor and emit duplicates.
        with self._append_lock:
            cursor = self._next_cursor()
            record = {"cursor": cursor, "timestamp": ts, "content": content}
            if session_key:
                record["session_key"] = session_key
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        """Non-negative int cursors only; reject bool (``isinstance(True, int)`` is True)."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _iter_valid_entries(self) -> Iterator[tuple[range, dict[str, Any], int]]:
        poisoned: Any = None
        malformed_cursor: int | None = None
        for lines, entry in self._iter_entries():
            raw = entry.get("cursor")
            if raw is None:
                continue
            cursor = self._valid_cursor(raw)
            if cursor is None:
                poisoned = raw
                continue
            if not self._valid_history_payload(entry):
                malformed_cursor = cursor
                continue
            yield lines, entry, cursor
        if poisoned is not None and not self._corruption_logged:
            self._corruption_logged = True
            logger.warning(
                "history.jsonl contains an invalid cursor ({!r}); dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                poisoned,
            )
        if malformed_cursor is not None and not self._malformed_entry_logged:
            self._malformed_entry_logged = True
            logger.warning(
                "history.jsonl contains a malformed entry at cursor {}; dropping it. "
                "Usually caused by an external writer; further occurrences suppressed.",
                malformed_cursor,
            )

    @staticmethod
    def _valid_history_payload(entry: dict[str, Any]) -> bool:
        if not isinstance(entry.get("timestamp"), str):
            return False
        if not isinstance(entry.get("content"), str):
            return False
        session_key = entry.get("session_key")
        return session_key is None or isinstance(session_key, str)

    def _read_cursor_counter(self) -> int | None:
        if not self._cursor_file.exists():
            return None
        with suppress(ValueError, OSError):
            cursor = int(self._cursor_file.read_text(encoding="utf-8").strip())
            if cursor >= 0:
                return cursor
        return None

    def _next_cursor(self) -> int:
        cursor_counter = self._read_cursor_counter()
        last = self._read_last_entry() or {}
        last_cursor = self._valid_cursor(last.get("cursor"))
        if cursor_counter is not None:
            if last_cursor is not None:
                return max(cursor_counter, last_cursor) + 1
            max_history_cursor = max((c for _, _, c in self._iter_valid_entries()), default=0)
            return max(cursor_counter, max_history_cursor) + 1

        # Fast path: trust the tail when intact.  Otherwise scan the whole
        # file and take ``max`` — that stays correct even if the monotonic
        # invariant was broken by external writes.
        if last_cursor is not None:
            return last_cursor + 1
        return max((c for _, _, c in self._iter_valid_entries()), default=0) + 1

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with a valid cursor > *since_cursor*."""
        return [e for _, e, c in self._iter_valid_entries() if c > since_cursor]

    def compact_history(self) -> None:
        """Drop oldest processed entries without discarding pending Dream input."""
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        last_dream_cursor = self.get_last_dream_cursor()
        first_unprocessed = next(
            (
                index
                for index, entry in enumerate(entries)
                if (
                    (cursor := self._valid_cursor(entry.get("cursor"))) is not None
                    and cursor > last_dream_cursor
                )
            ),
            len(entries),
        )
        keep_from = min(len(entries) - self.max_history_entries, first_unprocessed)
        kept = entries[keep_from:]
        if len(kept) > self.max_history_entries:
            logger.warning(
                "History compaction retained {} unprocessed entries beyond the configured "
                "limit of {}",
                len(kept),
                self.max_history_entries,
            )
        self._write_entries(kept)


    def _iter_entries(self) -> Iterator[tuple[range, dict[str, Any]]]:
        with suppress(FileNotFoundError):
            with open(self.history_file, "r", encoding="utf-8") as f:
                line_number = 1
                for line in f:
                    # Match read_file's splitlines, including Unicode separators
                    # that JSON leaves unescaped inside string values.
                    lines = range(line_number, line_number + len(line.splitlines()))
                    line_number = lines.stop
                    line = line.strip()
                    if line:
                        try:
                            parsed: object = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(parsed, dict):
                            yield lines, cast(dict[str, Any], parsed)

    def _read_entries(self) -> list[dict[str, Any]]:
        return [entry for _, entry in self._iter_entries()]

    def _read_last_entry(self) -> dict[str, Any] | None:
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [line for line in data.split("\n") if line.strip()]
                if not lines:
                    return None
                parsed: object = json.loads(lines[-1])
                return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        atomic_write_lines(
            self.history_file,
            (json.dumps(entry, ensure_ascii=False) for entry in entries),
        )


    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            with suppress(ValueError, OSError):
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    def get_latest_cursor(self) -> int:
        return max(self._next_cursor() - 1, 0)

    @property
    def dream_prompt_file(self) -> Path:
        return workspace_prompt_file(self.workspace, "dream")

    def has_dream_prompt_override(self) -> bool:
        return has_workspace_prompt_override(self.dream_prompt_file)

    @staticmethod
    def default_dream_prompt() -> str:
        from nanobot.agent.skill_metadata import BUILTIN_SKILLS_DIR

        return render_template(
            "agent/dream.md",
            strip=True,
            skill_creator_path=str(BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"),
        )

    def _dream_template(self) -> str:
        text, original_chars = load_workspace_prompt_override(self.dream_prompt_file)
        if text is not None:
            if (
                original_chars > WORKSPACE_PROMPT_MAX_CHARS
                and not self._dream_prompt_oversize_logged
            ):
                self._dream_prompt_oversize_logged = True
                logger.warning(
                    "workspace Dream prompt exceeds {} chars ({}); truncating. "
                    "Further occurrences suppressed.",
                    WORKSPACE_PROMPT_MAX_CHARS, original_chars,
                )
            return text
        return self.default_dream_prompt()

    def build_dream_prompt(self) -> tuple[str, int] | None:
        """Describe the pending journal range for one file-reading Dream task.

        Appends preserve these read_file line numbers. Dream is serialized, and
        history compaction runs only after the task finishes reading its range.
        """
        since_cursor = self.get_last_dream_cursor()
        with self._append_lock:
            pending = [
                (lines, cursor)
                for lines, _, cursor in self._iter_valid_entries()
                if cursor > since_cursor
            ]
        if not pending:
            return None

        last_cursor = max(cursor for _, cursor in pending)
        history_prompt = render_template(
            "agent/dream_history.md", strip=True,
            path=str(self.history_file.resolve()),
            start_line=pending[0][0].start, end_line=pending[-1][0].stop - 1,
            count=len(pending), since_cursor=since_cursor, last_cursor=last_cursor,
        )
        return f"{self._dream_template()}\n\n{history_prompt}", last_cursor

    def dream_content_diff(self) -> str:
        """Structured summary of uncommitted changes to the durable memory files.

        Returns "" when git is unavailable or no content file changed. This is
        the ground-truth input for diff-grounded Dream commit messages.
        """
        if not self._git.is_initialized():
            return ""
        return self._git.summarize_working_tree(
            list(MEMORY_CONTENT_PATHS), directories=[MEMORY_WIKI_PATH],
        )

    def build_dream_tools(self) -> ToolRegistry:
        """Build the restricted tool registry used by Dream runs."""
        from nanobot.agent.skill_metadata import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.apply_patch import ApplyPatchTool
        from nanobot.agent.tools.file_state import FileStates
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
        from nanobot.agent.tools.registry import ToolRegistry

        tools = ToolRegistry()
        file_states = FileStates()
        workspace = self.workspace
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        editable_files = [self.memory_file, self.soul_file, self.user_file]

        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_read_allowed_dirs=extra_read,
            file_states=file_states,
        ))
        tools.register(EditFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_dirs=[self.wiki_dir],
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        tools.register(ApplyPatchTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_dirs=[self.wiki_dir],
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        tools.register(WriteFileTool(
            workspace=workspace,
            allowed_dir=skills_dir,
            extra_write_allowed_dirs=[self.wiki_dir],
            extra_write_allowed_files=editable_files,
            file_states=file_states,
        ))
        return tools

    @staticmethod
    def dream_run_completed(
        resp: object | None,
    ) -> bool:
        """Return True when the Dream agent reached a normal terminal response."""
        metadata = getattr(resp, "metadata", None)
        if not isinstance(metadata, dict):
            return False
        return cast(dict[str, Any], metadata).get("_stop_reason") == "completed"

    @staticmethod
    def dream_incompletion_reason(
        resp: object | None,
    ) -> str:
        """Human-readable explanation of why a Dream run cannot advance."""
        metadata = getattr(resp, "metadata", None)
        if isinstance(metadata, dict):
            stop_reason = cast(dict[str, Any], metadata).get("_stop_reason", "unknown")
        else:
            stop_reason = "missing response metadata"
        return f"stop_reason: {stop_reason}"


    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            content = history_message_text(message)
            if not content:
                continue
            tools_used = message.get("tools_used")
            tools = (
                f" [tools: {', '.join(cast(list[str], tools_used))}]"
                if tools_used
                else ""
            )
            raw_timestamp = message.get("timestamp")
            timestamp = str(raw_timestamp) if raw_timestamp is not None else "?"
            role = str(message.get("role") or "unknown")
            lines.append(f"[{timestamp[:16]}] {role.upper()}{tools}: {content}")
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> str:
        """Persist raw messages in bounded chunks and return a checkpoint."""
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        chunk_size = min(max(1, limit), _HISTORY_ENTRY_HARD_CAP - 1_000)
        # Clean the complete text first: a thinking block may span chunk boundaries.
        formatted = strip_think(self._format_messages(public_history_messages(messages)))
        chunks = [
            formatted[start:start + chunk_size]
            for start in range(0, len(formatted), chunk_size)
        ] or [""]
        for part, chunk in enumerate(chunks, start=1):
            suffix = f" (part {part}/{len(chunks)})" if len(chunks) > 1 else ""
            # The whole text is sanitized and each record fits the hard cap.
            # Normalizing again would trim meaningful whitespace at chunk edges.
            self._append_history_record(
                f"[RAW] {len(messages)} messages{suffix}\n{chunk}",
                session_key=session_key,
            )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages in {} entries",
            len(messages),
            len(chunks),
        )
        checkpoint = self._normalize_history_entry(
            f"[RAW] {len(messages)} messages\n{formatted}",
            max_chars=limit,
        )
        return checkpoint

    def _build_raw_checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
    ) -> str:
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        checkpoint = (
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(public_history_messages(messages))}"
        )
        return self._normalize_history_entry(checkpoint, max_chars=limit)


    @staticmethod
    def dream_session_key() -> str:
        """Give every batch its own session, including batches in the same second."""
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"

    @staticmethod
    def build_dream_commit_message(prefix: str, diff_body: str) -> str:
        """Build a Dream commit message grounded in the real working-tree diff.

        *diff_body* is a structured, machine-derived summary of the actual file
        changes (see :meth:`dream_content_diff` /
        :meth:`GitStore.summarize_working_tree`). The LLM narrative is
        deliberately excluded so the audit record (``/dream-log``) reflects the
        filesystem's truth, not the model's self-report.

        An empty *diff_body* yields the bare *prefix*, which ``auto_commit``
        turns into a no-op when there is nothing to stage.
        """
        diff_body = (diff_body or "").strip()
        if not diff_body:
            return prefix
        return f"{prefix}\n\n{diff_body}"


# Raw fallbacks use a tighter cap. Completed model summaries may scale with the
# configured generation budget, while append_history() still enforces the
# emergency hard cap against pathological provider output.
_RAW_ARCHIVE_MAX_CHARS = 16_000
_HISTORY_ENTRY_HARD_CAP = 64_000
_ARCHIVE_TOOL_RESULT = (
    "Session archival does not execute tools. Use only the supplied conversation and "
    "return the requested compact checkpoint now; do not call another tool."
)


def _archive_prompt(archive_count: int) -> dict[str, str]:
    return {"role": "user", "content": render_template(
        "agent/consolidator_archive.md", strip=True, archive_count=archive_count,
    )}


class MemoryArchiver:
    """Generate transcript checkpoints and optionally journal their source.

    The archiver deliberately has no SessionManager dependency: it may read a
    captured transcript batch and, for durable sessions, append to history.jsonl,
    but it cannot mutate provider continuation state or advance a session watermark.
    """

    def __init__(
        self,
        store: MemoryStore,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        resolve_prompt_context: Callable[[Session], tuple[str | None, Path | None]] | None = None,
    ) -> None:
        self.store = store
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._resolve_prompt_context = resolve_prompt_context

    def _raw_checkpoint(
        self,
        messages: list[dict[str, Any]],
        *,
        session_key: str,
        previous_summary: str | None,
        max_tokens: int,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
    ) -> str:
        if telemetry is not None:
            telemetry.raw_fallback = True
        raw = (
            self.store.raw_archive(messages, session_key=session_key)
            if persist
            else self.store._build_raw_checkpoint(messages)
        )
        return self._combine_raw_checkpoint(
            raw,
            previous_summary=previous_summary,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _combine_raw_checkpoint(
        raw: str,
        *,
        previous_summary: str | None,
        max_tokens: int,
    ) -> str:
        """Return a bounded checkpoint that preserves prior and newly archived context."""
        token_limit = max(1, max_tokens)
        if not previous_summary:
            return truncate_text_to_tokens(raw, token_limit)

        combined = (
            "[Previous archived context]\n"
            f"{previous_summary}\n\n"
            "[Newly archived raw context]\n"
            f"{raw}"
        )
        bounded = truncate_text_to_tokens(combined, token_limit)
        if bounded == combined:
            return combined

        # Keep evidence from both sides when their full concatenation cannot fit.
        section_limit = max(1, (token_limit - 32) // 2)
        return truncate_text_to_tokens(
            "[Previous archived context]\n"
            f"{truncate_text_to_tokens(previous_summary, section_limit)}\n\n"
            "[Newly archived raw context]\n"
            f"{truncate_text_to_tokens(raw, section_limit)}",
            token_limit,
        )

    async def archive(
        self,
        source_messages: list[dict[str, Any]],
        *,
        runtime: LLMRuntime,
        session_key: str,
        history: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        previous_summary: str | None = None,
        input_token_budget: int | None = None,
        fallback_max_tokens: int | None = None,
        provider_state: ProviderConversationState | None = None,
        no_tool_contract: bool = False,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
        allow_raw_fallback: bool = True,
    ) -> str | None:
        """Generate a replacement checkpoint and optionally persist it.

        ``no_tool_contract`` marks a configured-compactor request: its
        summarization must be tool-free in every request shape (full
        transcript replay or native-state replay), and adapters are told so
        with an explicit ``tool_choice='none'`` that survives configured
        ``extra_body`` merges. Legacy no-override compaction keeps its
        existing request shape, including an empty session tool registry.

        ``allow_raw_fallback=False`` is the strict contract staged compaction
        runs each stage under: a request that does not fit, a failed provider
        call, or unusable output yields ``None`` instead of a bounded raw
        checkpoint, so one failed stage can never masquerade as a summary.
        """
        if not source_messages:
            return None

        fallback_action = "using raw checkpoint" if allow_raw_fallback else "aborting staged compaction"

        def raw_fallback() -> str | None:
            if not allow_raw_fallback:
                return None
            return self._raw_checkpoint(
                source_messages,
                session_key=session_key,
                previous_summary=previous_summary,
                max_tokens=(
                    fallback_max_tokens
                    if fallback_max_tokens is not None
                    else runtime.generation.max_tokens
                ),
                persist=persist,
                telemetry=telemetry,
            )

        prompt_message = _archive_prompt(len(source_messages))
        provider_context = None
        state_controller: ProviderConversationStateController | None = None
        state_messages: list[dict[str, Any]] = []
        call_tools = request_tools
        if provider_state is not None:
            instruction_messages: list[dict[str, Any]] = []
            for message in history:
                if message.get("role") not in {"system", "developer"}:
                    break
                instruction_messages.append(dict(message))
            request_messages = [*instruction_messages, prompt_message]
            state_controller = ProviderConversationStateController(
                provider=runtime.provider,
                model=runtime.model,
                messages=state_messages,
                state=provider_state,
                session_id=session_key,
            )
            state_messages.append(dict(prompt_message))
            provider_context = state_controller.prepare_request(
                state_messages,
                context_window_tokens=runtime.context_window_tokens,
            )
            if provider_context is None or provider_context.conversation_state is None:
                return raw_fallback()
            call_tools = []
        else:
            request_messages = [
                *[dict(message) for message in history],
                prompt_message,
            ]
        response: LLMResponse | None = None
        for attempt in range(2):
            if input_token_budget is not None and provider_context is None and (
                attempt == 0 or not allow_raw_fallback
            ):
                estimated, source = estimate_prompt_tokens_chain(
                    runtime.provider, runtime.model, request_messages, call_tools,
                )
                if input_token_budget <= 0 or estimated > input_token_budget:
                    logger.debug(
                        "Memory archive input does not fit for {}: {}/{} via {}; {}",
                        session_key, estimated, input_token_budget, source, fallback_action,
                    )
                    return raw_fallback()
            try:
                if telemetry is not None:
                    telemetry.provider_calls += 1
                with llm_usage_source("dream"):
                    response = await runtime.provider.chat_stream_with_retry(
                        model=runtime.model,
                        messages=request_messages,
                        tools=call_tools,
                        temperature=runtime.generation.temperature,
                        max_tokens=runtime.generation.max_tokens,
                        reasoning_effort=runtime.generation.reasoning_effort,
                        # A configured compactor summarizes without tools in
                        # every request shape, including native-state replay
                        # (provider state persists items, not tool schemas, so
                        # a tool-free request still replays). The explicit
                        # contract tells adapters to keep the request tool-free
                        # even when a configured extra_body would merge tools
                        # or tool_choice into it.
                        tool_choice="none" if no_tool_contract else None,
                        provider_context=provider_context,
                    )
            except Exception:
                phase = "provider call" if attempt == 0 else "tool-call recovery"
                logger.warning("Memory archive {} failed; {}", phase, fallback_action)
                return raw_fallback()
            if telemetry is not None and response.usage is not None:
                telemetry.usage = (
                    response.usage
                    if telemetry.usage is None
                    else telemetry.usage + response.usage
                )
            if response.should_execute_tools is not True or attempt == 1 or not request_tools:
                # Tool-free requests (a configured compactor, or a session
                # without tools) never recover: echoing an unsolicited
                # tool-call transcript back only risks another call, and the
                # raw checkpoint below preserves the full evidence.
                break

            logger.info(
                "Memory archive provider returned {} tool call(s); requesting checkpoint",
                len(response.tool_calls),
            )
            assistant_message = build_assistant_message(
                response.content,
                tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            tool_messages = [
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": _ARCHIVE_TOOL_RESULT,
                }
                for call in response.tool_calls
            ]
            request_messages = [
                *request_messages,
                assistant_message,
                *tool_messages,
            ]
            if state_controller is not None:
                state_controller.observe_response(
                    response,
                    state_messages,
                )
                state_messages.extend([
                    state_controller.project_response_message(
                        dict(assistant_message),
                        response,
                    ),
                    *[dict(message) for message in tool_messages],
                ])
                provider_context = state_controller.prepare_request(
                    state_messages,
                    context_window_tokens=runtime.context_window_tokens,
                )
                if provider_context is None or provider_context.conversation_state is None:
                    return raw_fallback()
        assert response is not None
        if response.finish_reason in {"error", "length"}:
            logger.warning(
                "Memory archive provider did not complete ({}); {}",
                response.finish_reason, fallback_action,
            )
            return raw_fallback()
        if response.has_tool_calls is True:
            logger.warning("Memory archive provider returned tool calls; {}", fallback_action)
            return raw_fallback()
        summary = response.content
        if not summary or not summary.strip():
            logger.warning("Memory archive provider returned no summary; {}", fallback_action)
            return raw_fallback()
        summary = self.store._normalize_history_entry(summary)
        if not summary:
            logger.warning(
                "Memory archive provider summary was not safe to replay; {}", fallback_action
            )
            return raw_fallback()
        if persist and summary != "(nothing)":
            self.store.append_history(summary, session_key=session_key)
        return summary

    async def archive_session(
        self,
        session: Session,
        *,
        archive_end: int,
        runtime: LLMRuntime,
        input_token_budget: int,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
    ) -> str | None:
        """Archive a captured session prefix without mutating the session.

        ``tools`` is the tool set offered with the summarization request.
        ``None`` keeps the legacy registry defaults; a configured compactor
        passes ``[]`` (with ``no_tool_contract``) so the request offers no
        tool calls at all. ``persist`` controls the history journal exactly
        as in :meth:`archive`, so dry runs can discard their result.
        """
        messages = [
            message for message in session.messages[session.last_archived:archive_end]
            if not message.get("_command") and not is_summary_checkpoint(message)
        ]
        if not messages:
            return None
        session_summary = session_summary_from_metadata(
            session.metadata,
            fallback_last_active=session.updated_at,
        )
        previous_summary = session_summary["text"] if session_summary else None

        if input_token_budget <= 0:
            logger.debug(
                "Memory archive has no safe input budget for {}; raw-dumping",
                session.key,
            )
            return self._raw_checkpoint(
                messages,
                session_key=session.key,
                previous_summary=previous_summary,
                max_tokens=runtime.generation.max_tokens,
                persist=persist,
                telemetry=telemetry,
            )
        prefix = Session(
            key=session.key,
            messages=list(session.messages[:archive_end]),
            last_consolidated=session.last_archived,
        )
        history = prefix.get_text_history(max_tokens=input_token_budget)
        archive_history = Session(
            key=session.key,
            messages=messages,
        ).get_text_history()
        if not archive_history or history[-len(archive_history):] != archive_history:
            logger.debug(
                "Memory archive cannot replay the full chunk for {}; raw-dumping",
                session.key,
            )
            return self._raw_checkpoint(
                messages,
                session_key=session.key,
                previous_summary=previous_summary,
                max_tokens=runtime.generation.max_tokens,
                persist=persist,
                telemetry=telemetry,
            )
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        workspace: Path | None = None
        if self._resolve_prompt_context is not None:
            channel, workspace = self._resolve_prompt_context(session)
        history_messages = self._build_messages(
            history=history,
            current_message=None,
            channel=channel,
            session_summary=session_summary,
            workspace=workspace,
        )
        request_tools = self._get_tool_definitions() if tools is None else tools
        return await self.archive(
            messages,
            runtime=runtime,
            session_key=session.key,
            history=history_messages,
            request_tools=request_tools,
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
            no_tool_contract=no_tool_contract,
            persist=persist,
            telemetry=telemetry,
        )


@dataclass
class CompactionStageReport:
    """One summarizer stage of a staged compaction attempt."""

    stage: int
    elapsed_s: float = 0.0
    provider_calls: int = 0
    usage: LLMUsage | None = None


@dataclass
class CompactionCallTelemetry:
    """Measure the summarizer calls behind one compaction attempt.

    Filled in by :meth:`MemoryArchiver.archive` so a dry-run preview can
    report provider usage and distinguish an LLM summary from a locally
    built raw checkpoint. Staged compaction aggregates one report per
    stage into ``stages`` and the totals into the other fields.
    """

    provider_calls: int = 0
    usage: LLMUsage | None = None
    raw_fallback: bool = False
    stages: list[CompactionStageReport] = field(default_factory=list)


@dataclass
class CompactionPlan:
    """Archive boundary chosen for one session compaction attempt.

    ``prefix`` is ``None`` on the legacy full-prefix path, where the whole
    raw unarchived suffix is archived. ``stages`` is ``None`` for the single
    summarizer call; a non-None value partitions the prepared prefix into
    consecutive stages that each fit one compactor request, summarized
    chronologically with a carry-forward summary. ``skip_reason`` marks a
    sizing no-op rather than a legacy full-prefix archive.
    """

    session: Session
    previous_summary: str | None
    archive_start: int
    archive_end: int
    prefix: list[dict[str, Any]] | None = None
    unarchived_count: int = 0
    skip_reason: str | None = None
    replay_config: ContextGovernanceConfig | None = None
    stages: list[list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class _SessionPrefixCut:
    """One session replay boundary sized for the compactor.

    ``planner`` is the shared planner's result: ``planner.cut`` is ``None``
    when no boundary was selected, and ``planner.reason`` then carries the
    stable no-cut code (for instance ``below_floor`` when the replayable
    history already fits under the configured verbatim-tail floor).
    """

    archive_end: int | None
    prefix: list[dict[str, Any]] | None
    stages: list[list[dict[str, Any]]] | None
    planner: PrefixCutResult


@dataclass
class CompactionPreview:
    """Outcome of one dry-run compaction whose checkpoint was discarded."""

    outcome: Literal["no_history", "skipped", "summarized", "raw_fallback", "error"] = "no_history"
    reason: str | None = None
    model: str = ""
    summary: str | None = None
    elapsed_s: float = 0.0
    usage: LLMUsage | None = None
    provider_calls: int = 0
    unarchived_count: int = 0
    archived_count: int = 0
    retained_count: int = 0
    estimated_prompt_tokens: int = 0
    estimated_summary_tokens: int = 0
    estimated_prompt_tokens_after: int = 0
    stages: list[CompactionStageReport] = field(default_factory=list)


def min_tail_skip_marker(
    session: Session, runtime: LLMRuntime, replay: ContextGovernanceConfig,
) -> list[str | int | None]:
    """Key a skipped cut to its replay boundary and session sizing policy."""
    return [
        session.updated_at.isoformat(), session.last_archived,
        replay.min_verbatim_tail_tokens, replay.next_turn_input_reserve_tokens,
        replay.model, replay.context_window_tokens, replay.max_tokens,
        runtime.model, runtime.context_window_tokens, runtime.generation.max_tokens,
    ]


class Consolidator:
    """Coordinate session Memory checkpoints through ``MemoryArchiver``."""

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift
    # Finite staged-compaction policy: at most this many summarizer stages per
    # attempt, within this wall-clock budget for all summarizer calls.
    _STAGED_MAX_STAGES = 8
    _STAGED_MAX_SECONDS: float = 1800
    _STAGED_BOUNDARY_PROBES = 4  # Exact sizing probes per stage, newest legal cut first.

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        resolve_prompt_context: Callable[[Session], tuple[str | None, Path | None]] | None = None,
    ):
        self.store = store
        self.sessions = sessions
        self._build_messages = build_messages
        self._resolve_prompt_context = resolve_prompt_context
        self._get_tool_definitions = get_tool_definitions
        self._context_governor = ContextGovernor()
        self.archiver = MemoryArchiver(
            store=store,
            build_messages=build_messages,
            get_tool_definitions=get_tool_definitions,
            resolve_prompt_context=resolve_prompt_context,
        )
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def compactor_input_budget(
        self,
        runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None,
        archive_count: int,
    ) -> CompactorInputBudget | None:
        """Describe the consolidator's actual input capacity for a transcript prefix.

        Mirrors ``summarize_transcript``: the compactor's window minus its
        reserved output, minus the summarization prompt and offered tools.
        ``None`` means no positive capacity can be derived, so callers keep
        the full-prefix compaction behavior.
        """
        max_output_tokens = runtime.generation.max_tokens
        context_window = runtime.context_window_tokens
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or isinstance(context_window, bool)
            or not isinstance(context_window, int)
        ):
            return None
        input_budget = context_window - max(0, max_output_tokens)
        if input_budget <= 0:
            return None
        request_tools = self._get_tool_definitions() if tools is None else tools
        overhead, _ = estimate_prompt_tokens_chain(
            runtime.provider,
            runtime.model,
            [_archive_prompt(max(0, archive_count))],
            request_tools,
        )
        if input_budget - overhead <= 0:
            return None
        return CompactorInputBudget(
            input_budget=input_budget,
            fixed_overhead_tokens=overhead,
            max_summary_tokens=max_output_tokens,
        )

    async def summarize_transcript(
        self,
        accepted_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
        provider_state: ProviderConversationState | None = None,
        no_tool_contract: bool = False,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
    ) -> str | None:
        """Summarize the exact transcript prefix already accepted by the model."""
        source_messages = [
            dict(message)
            for message in accepted_messages
            if message.get("role") != "system"
        ]
        if not source_messages:
            return None

        max_output_tokens = max(0, runtime.generation.max_tokens)
        input_token_budget = runtime.context_window_tokens - max_output_tokens
        checkpoint_tokens = min(
            max_output_tokens,
            max(1, (input_token_budget - self._SAFETY_BUFFER) // 2),
        )

        summary = await self.archiver.archive(
            source_messages,
            runtime=runtime,
            session_key=session_key,
            history=accepted_messages,
            request_tools=tools,
            previous_summary=previous_summary,
            input_token_budget=input_token_budget,
            fallback_max_tokens=max(1, checkpoint_tokens),
            provider_state=provider_state,
            no_tool_contract=no_tool_contract,
            persist=persist,
            telemetry=telemetry,
        )
        if summary is None:
            return None
        return truncate_text_to_tokens(summary, max(1, max_output_tokens))

    async def summarize_provider_compaction(
        self,
        state: ProviderConversationState,
        fallback_messages: list[dict[str, Any]],
        previous_summary: str | None,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]],
        persist: bool = True,
        allow_provider_state: bool = True,
        no_tool_contract: bool = False,
    ) -> str | None:
        """Prompt a native compacted state without replaying its raw history.

        ``allow_provider_state`` is ``False`` when *runtime* is a separate
        provider from the one that produced *state*; the archiver then
        summarizes the materialized transcript instead of resuming a state it
        cannot replay.
        """
        return await self.summarize_transcript(
            fallback_messages,
            previous_summary,
            runtime=runtime,
            session_key=session_key,
            tools=tools,
            provider_state=state if allow_provider_state else None,
            no_tool_contract=no_tool_contract,
            persist=persist,
        )

    def estimate_session_prompt_tokens(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
    ) -> tuple[int, str]:
        """Estimate prompt size from the full replayable session history."""
        return self._session_prompt_probe(session, runtime.provider, runtime.model)

    def _session_prompt_probe(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
    ) -> tuple[int, str]:
        """Estimate one full replay prompt locally, without provider inference.

        Same wrapper, tools, and history projection for any two views of the
        session, so before/after figures stay comparable.
        """
        history = session.get_text_history()
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        summary = session_summary_from_metadata(
            session.metadata,
            fallback_last_active=session.updated_at,
        )
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            session_summary=summary,
        )
        return estimate_prompt_tokens_chain(
            provider,
            model,
            probe_messages,
            self._get_tool_definitions(),
        )

    def _input_token_budget(self, runtime: LLMRuntime) -> int:
        return (
            runtime.context_window_tokens
            - runtime.generation.max_tokens
            - self._SAFETY_BUFFER
        )

    async def archive_session(
        self,
        session: Session,
        *,
        archive_end: int,
        runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
    ) -> str | None:
        """Archive one captured session range through the shared Memory path."""
        return await self.archiver.archive_session(
            session,
            archive_end=archive_end,
            runtime=runtime,
            input_token_budget=self._input_token_budget(runtime),
            tools=tools,
            no_tool_contract=no_tool_contract,
            persist=persist,
            telemetry=telemetry,
        )

    def _session_replay_wrapper(self, session: Session, summary: str) -> list[dict[str, Any]]:
        channel = session.key.split(":", 1)[0] if ":" in session.key else None
        workspace = None
        if self._resolve_prompt_context is not None:
            channel, workspace = self._resolve_prompt_context(session)
        return self._build_messages(
            history=[], current_message=None, channel=channel, workspace=workspace,
            session_summary={"text": summary, "last_active": session.updated_at.isoformat()},
        )

    def _session_prefix_cut(
        self,
        session: Session,
        config: ContextGovernanceConfig,
        budget: CompactorInputBudget,
        previous_summary: str | None,
        runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None,
    ) -> _SessionPrefixCut:
        """Choose one replay boundary and size a single request before staging it."""
        history = session.get_history()
        planner = self._context_governor.select_prefix_cut(
            config, history, len(history), budget, self._session_replay_wrapper(session, ""),
            None, [], config.tools.get_definitions(), session_compaction=True,
            next_turn_input_reserve_tokens=config.next_turn_input_reserve_tokens,
            check_prefix_capacity=False,
        )
        if planner.cut is None:
            return _SessionPrefixCut(archive_end=None, prefix=None, stages=None, planner=planner)

        archive_end = session.message_index_at_history_position(planner.cut)
        assert archive_end is not None  # The cut indexes the same untrimmed replay view.
        prefix = self._context_governor.prepare_messages_for_model(config, history[:planner.cut])
        request_tools = self._get_tool_definitions() if tools is None else tools
        single_request = prefix
        if previous_summary:
            single_request = [
                {"role": "system", "content": f"Previous conversation summary:\n{previous_summary}"},
                *prefix,
            ]
        archive_count = sum(message.get("role") != "system" for message in prefix)
        if archive_count:
            estimated, _ = estimate_prompt_tokens_chain(
                runtime.provider, runtime.model,
                [*single_request, _archive_prompt(archive_count)], request_tools,
            )
            if estimated <= budget.input_budget:
                return _SessionPrefixCut(
                    archive_end=archive_end, prefix=single_request, stages=None, planner=planner,
                )
        stages = self._partition_stages(prefix, previous_summary, budget, runtime, request_tools)
        return _SessionPrefixCut(
            archive_end=archive_end, prefix=None, stages=stages, planner=planner,
        )

    @staticmethod
    def _with_stage_carry(messages: list[dict[str, Any]], summary: str | None) -> list[dict[str, Any]]:
        if not summary:
            return messages
        # Match the checkpoint merger's existing prompt contract.
        return [{"role": "system", "content": f"[Archived Context Summary]\n\n{summary}"}, *messages]

    def _partition_stages(
        self,
        prefix: list[dict[str, Any]],
        previous_summary: str | None,
        budget: CompactorInputBudget,
        runtime: LLMRuntime,
        request_tools: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]] | None:
        """Bound the plan using both actual request overhead and a carry reserve."""
        carry_wrapper, _ = estimate_prompt_tokens_chain(
            runtime.provider, runtime.model, self._with_stage_carry([], " "), None,
        )
        first_carry_cost = 0
        if previous_summary:
            first_carry_cost, _ = estimate_prompt_tokens_chain(
                runtime.provider, runtime.model, self._with_stage_carry([], previous_summary), None,
            )
        carry_reserve = carry_wrapper + budget.max_summary_tokens
        base_capacity = budget.input_budget - budget.fixed_overhead_tokens
        legal = self._context_governor.legal_cut_indices(prefix, len(prefix))
        costs = [estimate_message_tokens(message) for message in prefix]
        stages: list[list[dict[str, Any]]] = []
        start = 0
        while start < len(prefix):
            stage_capacity = base_capacity - (carry_reserve if start else first_carry_cost)
            candidates: deque[int] = deque(maxlen=self._STAGED_BOUNDARY_PROBES)
            used = 0
            for candidate in range(start + 1, len(prefix) + 1):
                used += costs[candidate - 1]
                if used > stage_capacity:
                    break
                if candidate == len(prefix) or legal[candidate]:
                    candidates.append(candidate)
            # The additive estimate can undercount the projected request. Try
            # a bounded number of earlier legal cuts rather than probing every
            # prefix or giving up at the first mismatch.
            for end in reversed(candidates):
                stage = prefix[start:end]
                count = sum(message.get("role") != "system" for message in stage)
                if not count:
                    return None  # Shorter prefixes would also have no archive source.
                request = self._with_stage_carry(stage, previous_summary) if start == 0 else stage
                estimated, _ = estimate_prompt_tokens_chain(
                    runtime.provider, runtime.model, [*request, _archive_prompt(count)], request_tools,
                )
                if estimated + (carry_reserve if start else 0) <= budget.input_budget:
                    stages.append(stage)
                    start = end
                    break
            else:
                return None
            if len(stages) == self._STAGED_MAX_STAGES and start < len(prefix):
                return None
        return stages if len(stages) > 1 else None

    async def plan_idle_compaction(
        self,
        session: Session,
        *,
        runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None = None,
        replay_config: ContextGovernanceConfig | None = None,
    ) -> CompactionPlan | None:
        """Choose the exact archive boundary for one session compaction attempt.

        Shared by :meth:`compact_idle_session` and its dry run so both pick
        identical boundaries, compactor budgets, and summarizer requests.
        Returns ``None`` when there is no replayable activity to archive.
        """
        archive_start = session.last_archived
        messages_to_archive = list(session.messages[archive_start:])
        if not any(
            not message.get("_command") and not is_summary_checkpoint(message)
            for message in messages_to_archive
        ):
            return None
        stored_summary = session_summary_from_metadata(
            session.metadata, fallback_last_active=session.updated_at,
        )
        previous_summary = stored_summary["text"] if stored_summary else None
        budget = (
            self.compactor_input_budget(runtime, tools, len(messages_to_archive))
            if replay_config is not None else None
        )
        cut = (
            await self._context_governor.prepare_in_background(
                self._session_prefix_cut, session, replay_config, budget,
                previous_summary, runtime, tools,
            )
            if replay_config is not None and budget is not None else None
        )
        skip_reason = None
        if cut is None:
            archive_end = archive_start + len(messages_to_archive)
            prefix = None
            stages = None
            if (
                replay_config is not None
                and replay_config.min_verbatim_tail_tokens > 0
                and budget is None
            ):
                skip_reason = (
                    "the compactor's input capacity cannot be derived for the verbatim-tail policy"
                )
        else:
            # A planner rejection keeps the legacy full-range archive boundary
            # (or a floor no-op); a chosen cut fixes the exact archive end.
            archive_end = (
                cut.archive_end
                if cut.archive_end is not None
                else archive_start + len(messages_to_archive)
            )
            prefix, stages = cut.prefix, cut.stages
            floor = replay_config.min_verbatim_tail_tokens
            if cut.planner.cut is not None and prefix is None and stages is None:
                skip_reason = "the protected prefix exceeds the bounded compactor capacity"
            elif cut.planner.cut is None and floor > 0:
                if cut.planner.reason == "below_floor":
                    # A no-op by design, not a budget rejection: the whole
                    # replayable history already fits under the floor.
                    skip_reason = (
                        f"est. {cut.planner.history_tokens} replayable tokens, at or below the "
                        f"configured {floor}-token verbatim-tail floor; nothing to archive"
                    )
                elif cut.planner.reason == "replay_over_budget":
                    # The skip line carries the estimate/budget diagnostics so
                    # the rejection is not logged twice.
                    skip_reason = (
                        "no projected replay fit within the foreground input budget "
                        f"(last: {cut.planner.last_required_tokens} > "
                        f"{cut.planner.input_budget_tokens} tokens including summary and reserve; "
                        f"{cut.planner.rejected_probes} request(s) probed)"
                    )
                else:
                    skip_reason = (
                        "no compaction fit while keeping the configured minimum of "
                        f"{floor} verbatim tail tokens"
                    )
            elif (
                cut.planner.cut is None
                and cut.planner.reason == "replay_over_budget"
                and skip_reason is None
            ):
                # Floor disabled: the rejection falls back to full-prefix
                # archival, so the diagnostics stay visible once here.
                logger.info(
                    "Session compaction replay rejected for {}: {} projected request(s) exceeded "
                    "the foreground input budget (last: {} > {} tokens including summary and reserve)",
                    session.key, cut.planner.rejected_probes,
                    cut.planner.last_required_tokens, cut.planner.input_budget_tokens,
                )
        unarchived_count = sum(
            1 for message in messages_to_archive
            if not message.get("_command") and not is_summary_checkpoint(message)
        )
        return CompactionPlan(
            session=session,
            previous_summary=previous_summary,
            archive_start=archive_start,
            archive_end=archive_end,
            prefix=prefix,
            unarchived_count=unarchived_count,
            skip_reason=skip_reason,
            replay_config=replay_config,
            stages=stages,
        )

    async def _execute_compaction(
        self,
        plan: CompactionPlan,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        persist: bool = True,
        telemetry: CompactionCallTelemetry | None = None,
    ) -> str | None:
        """Run the summarizer for a planned compaction at the plan's exact boundary."""
        if plan.stages is not None:
            try:
                async with asyncio.timeout(self._STAGED_MAX_SECONDS):
                    return await self._summarize_stages(
                        plan, runtime=runtime, session_key=session_key, tools=tools,
                        no_tool_contract=no_tool_contract, telemetry=telemetry,
                    )
            except TimeoutError:
                logger.warning("Staged compaction timed out for {}; retaining history", session_key)
                return None
        if plan.prefix is not None:
            return await self.summarize_transcript(
                plan.prefix, plan.previous_summary, runtime=runtime, session_key=session_key,
                tools=self._get_tool_definitions() if tools is None else tools,
                no_tool_contract=no_tool_contract,
                persist=persist, telemetry=telemetry,
            )
        return await self.archive_session(
            plan.session, archive_end=plan.archive_end, runtime=runtime, tools=tools,
            no_tool_contract=no_tool_contract,
            persist=persist, telemetry=telemetry,
        )

    async def _summarize_stages(
        self,
        plan: CompactionPlan,
        *,
        runtime: LLMRuntime,
        session_key: str,
        tools: list[dict[str, Any]] | None,
        no_tool_contract: bool,
        telemetry: CompactionCallTelemetry | None,
    ) -> str | None:
        """Run every stage without publishing anything until the caller commits."""
        stages = plan.stages
        assert stages is not None
        request_tools = self._get_tool_definitions() if tools is None else tools
        input_budget = runtime.context_window_tokens - max(0, runtime.generation.max_tokens)
        carry = plan.previous_summary
        for index, stage in enumerate(stages, start=1):
            stage_messages = self._with_stage_carry(stage, carry)
            stage_telemetry = CompactionCallTelemetry()
            stage_started = time.monotonic()
            try:
                summary = await self.archiver.archive(
                    [dict(message) for message in stage_messages if message.get("role") != "system"],
                    runtime=runtime,
                    session_key=session_key,
                    history=[dict(message) for message in stage_messages],
                    request_tools=request_tools,
                    previous_summary=None,
                    input_token_budget=input_budget,
                    no_tool_contract=no_tool_contract,
                    persist=False,
                    telemetry=stage_telemetry,
                    allow_raw_fallback=False,
                )
            finally:
                stage_elapsed = time.monotonic() - stage_started
                if telemetry is not None:
                    telemetry.stages.append(CompactionStageReport(
                        stage=index,
                        elapsed_s=stage_elapsed,
                        provider_calls=stage_telemetry.provider_calls,
                        usage=stage_telemetry.usage,
                    ))
                    telemetry.provider_calls += stage_telemetry.provider_calls
                    if stage_telemetry.usage is not None:
                        telemetry.usage = (
                            stage_telemetry.usage
                            if telemetry.usage is None
                            else telemetry.usage + stage_telemetry.usage
                        )
            if summary is None:
                logger.warning(
                    "Staged compaction for {} failed at stage {}/{}; retaining the original history",
                    session_key, index, len(stages),
                )
                return None
            carry = summary
        assert carry is not None
        if not await self._context_governor.prepare_in_background(self._staged_replay_fits, plan, carry):
            logger.warning("Staged checkpoint would exceed the replay budget for {}; retaining history", session_key)
            return None
        return carry

    def _staged_replay_fits(self, plan: CompactionPlan, summary: str) -> bool:
        config = plan.replay_config
        assert config is not None  # Staging requires a bounded replay plan.
        projected = self._project_checkpoint(plan, summary)
        probe = self._context_governor.prepare_messages_for_model(config, [
            *self._session_replay_wrapper(plan.session, summary),
            *projected.get_history(),
            {"role": "user", "content": SUMMARY_CONTINUATION_TEXT},
        ])
        estimated, _ = estimate_prompt_tokens_chain(
            config.provider, config.model, probe, config.tools.get_definitions(),
        )
        return (
            estimated + max(0, config.next_turn_input_reserve_tokens)
            <= self._context_governor.input_budget(config)
        )

    @staticmethod
    def _project_checkpoint(plan: CompactionPlan, summary: str) -> Session:
        session = plan.session
        projected = dataclasses_replace(session, messages=list(session.messages), metadata=dict(session.metadata))
        projected.commit_summary_checkpoint(summary, insert_at=plan.archive_end, last_active=session.updated_at)
        return projected

    def _projected_prompt_probe(
        self,
        plan: CompactionPlan,
        summary: str,
        provider: LLMProvider,
        model: str,
    ) -> tuple[int, str]:
        """Estimate the post-compaction replay prompt on a throwaway projection."""
        return self._session_prompt_probe(
            self._project_checkpoint(plan, summary), provider, model,
        )

    async def _commit_staged_summary(self, plan: CompactionPlan, summary: str) -> None:
        await self.sessions.state.commit_summary(plan.session, summary, archive_end=plan.archive_end)
        if summary != "(nothing)":
            try:
                self.store.append_history(summary, session_key=plan.session.key)
            except OSError:
                # The journal and its cursor are separate from the session
                # transaction. An append can fail after writing the entry;
                # don't retry it or report that the checkpoint was rolled back.
                logger.opt(exception=True).warning(
                    "Staged checkpoint applied for {} but updating the memory journal failed",
                    plan.session.key,
                )

    async def compact_idle_session(
        self,
        session_key: str,
        *,
        runtime: LLMRuntime,
        max_suffix: int = 0,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        replay_config: ContextGovernanceConfig | None = None,
        events: EventSink = NO_EVENTS,
        notify: bool = False,
        reason: CompactionTrigger | None = None,
    ) -> str | None:
        """Replace an old prefix with a summary, retaining a tail when both budgets permit.

        ``max_suffix`` is accepted for SDK compatibility; it does not control
        the retained tail.
        ``tools`` overrides the tool set offered to the summarizer; a
        configured compactor receives ``[]`` (no tool calls offered) with
        ``no_tool_contract`` set.
        ``reason`` is the trigger for logs; ``"sdk"`` is logging-only and maps
        to no reason on wire events.
        """
        lock = self.get_lock(session_key)
        event_reason = None if reason == "sdk" else reason
        async with lock:
            await self.sessions.state.discard(session_key)
            session = await self.sessions.state.get(session_key)

            # Read-only sizing before announcing: resolve the compactor budget and
            # the protected-tail cut so a floor no-op needs no lifecycle events.
            # If sizing fails or is cancelled, emit the lifecycle pair before
            # propagating the error.
            compaction_id = uuid4().hex
            started = ContextCompactionEvent(
                compaction_id=compaction_id, phase="started", notify=notify, reason=event_reason,
            )
            try:
                plan = await self.plan_idle_compaction(
                    session, runtime=runtime, tools=tools, replay_config=replay_config,
                )
            except (Exception, asyncio.CancelledError) as exc:
                await events.emit(started)
                await events.emit(
                    ContextCompactionEvent(
                        compaction_id=compaction_id,
                        phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        notify=notify,
                        reason=event_reason,
                    ),
                )
                raise

            if plan is None:
                return ""
            if plan.skip_reason:
                # A normal no-op, not a failure or a full-prefix archive.
                # Remember it for idle scans only while the history and sizing policy match.
                marker = min_tail_skip_marker(session, runtime, replay_config)
                if session.metadata.get("_min_tail_skip") != marker:
                    try:
                        await self.sessions.state.update_metadata(
                            session_key, {"_min_tail_skip": marker}, expected_generation=session.generation,
                        )
                    except Exception:
                        logger.debug("Could not record min-tail skip marker for {}", session_key, exc_info=True)
                logger.info(
                    "Skipping session compaction for {}{}: {}",
                    session_key,
                    f" ({reason})" if reason is not None else "",
                    plan.skip_reason,
                )
                return ""

            # Best-effort replay sizing before publication: tokenization is
            # offloaded, and sizing trouble only drops the metric, never the
            # compaction (cancellation here is a real pre-publication
            # cancellation and propagates as usual).
            before_tokens: int | None = None
            if replay_config is not None:
                try:
                    before_tokens, _ = await self._context_governor.prepare_in_background(
                        self._session_prompt_probe, session, replay_config.provider, replay_config.model,
                    )
                except Exception:
                    before_tokens = None

            await events.emit(started)
            try:
                summary = await self._execute_compaction(
                    plan, runtime=runtime, session_key=session_key, tools=tools,
                    no_tool_contract=no_tool_contract,
                )
                if summary and plan.stages is not None:
                    # Once publication is accepted, finish the checkpoint and
                    # best-effort journal update before propagating cancellation.
                    await shield_and_drain(self._commit_staged_summary(plan, summary))
                elif summary:
                    await self.sessions.state.commit_summary(session, summary, archive_end=plan.archive_end)
            except (Exception, asyncio.CancelledError) as exc:
                await events.emit(
                    ContextCompactionEvent(
                        compaction_id=compaction_id,
                        phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        notify=notify,
                        reason=event_reason,
                    ),
                )
                raise
            if not summary:
                await events.emit(
                    ContextCompactionEvent(
                        compaction_id=compaction_id, phase="failed", notify=notify, reason=event_reason,
                    ),
                )
                return None

            await events.emit(
                ContextCompactionEvent(
                    compaction_id=compaction_id,
                    phase="succeeded",
                    notify=notify,
                    reason=event_reason,
                ),
            )

            # Projected replay sizing after the success event: the checkpoint
            # is committed and no lifecycle event follows it, so sizing trouble
            # only drops the metric and cancellation propagates without
            # re-reporting the outcome.
            replay_metrics = ""
            if replay_config is not None and before_tokens is not None and before_tokens > 0:
                after_tokens: int | None = None
                try:
                    after_tokens, _ = await self._context_governor.prepare_in_background(
                        self._projected_prompt_probe, plan, summary,
                        replay_config.provider, replay_config.model,
                    )
                except Exception:
                    after_tokens = None
                if after_tokens is not None and after_tokens > 0:
                    change = before_tokens - after_tokens
                    if change > 0:
                        replay_metrics = (
                            f", est. replay prompt tokens {before_tokens} -> {after_tokens}, "
                            f"saved {change} tokens ({change / before_tokens * 100:.0f}%)"
                        )
                    elif change < 0:
                        replay_metrics = (
                            f", est. replay prompt tokens {before_tokens} -> {after_tokens}, "
                            f"added {-change} tokens ({-change / before_tokens * 100:.0f}%)"
                        )
                    else:
                        replay_metrics = (
                            f", est. replay prompt tokens {before_tokens} -> {after_tokens}, no change"
                        )

            logger.info(
                "Session compact for {}{}: archived={}, replay_tail={}, retained={}, summary={}, stages={}{}",
                session_key,
                f" ({reason})" if reason is not None else "",
                plan.archive_end - plan.archive_start,
                len(session.messages) - plan.archive_end,
                len(session.messages),
                bool(summary),
                len(plan.stages) if plan.stages is not None else 1,
                replay_metrics,
            )

            return summary

    async def compact_idle_session_dry_run(
        self,
        session_key: str,
        *,
        runtime: LLMRuntime,
        replay_runtime: LLMRuntime,
        tools: list[dict[str, Any]] | None = None,
        no_tool_contract: bool = False,
        replay_config: ContextGovernanceConfig | None = None,
    ) -> CompactionPreview:
        """Run one real compaction attempt and discard its result.

        Shares :meth:`plan_idle_compaction` and :meth:`_execute_compaction`
        with :meth:`compact_idle_session`, so it runs the same preflight and
        the same summarizer requests; the configured compactor is called
        for real, so provider inference and its cost can occur. The only
        differences are the publication effects: no summary commit,
        history-journal entry, skip marker, or lifecycle event, and no
        session metadata. Token figures are estimates.
        """
        preview = CompactionPreview(model=runtime.model)
        lock = self.get_lock(session_key)
        async with lock:
            session = await self.sessions.state.get(session_key)
            try:
                plan = await self.plan_idle_compaction(
                    session, runtime=runtime, tools=tools, replay_config=replay_config,
                )
            except Exception as exc:
                logger.exception("Compaction dry-run sizing failed for {}", session_key)
                preview.outcome = "error"
                preview.reason = type(exc).__name__
                return preview
            if plan is None:
                preview.outcome = "no_history"
                return preview
            preview.unarchived_count = plan.unarchived_count
            preview.estimated_prompt_tokens, _ = self.estimate_session_prompt_tokens(
                session, runtime=replay_runtime,
            )
            if plan.skip_reason:
                preview.outcome = "skipped"
                preview.reason = plan.skip_reason
                return preview

            preview.archived_count = plan.archive_end - plan.archive_start
            preview.retained_count = len(session.messages) - plan.archive_end

            telemetry = CompactionCallTelemetry()
            started = time.monotonic()
            try:
                summary = await self._execute_compaction(
                    plan, runtime=runtime, session_key=session_key, tools=tools,
                    no_tool_contract=no_tool_contract,
                    persist=False, telemetry=telemetry,
                )
            except Exception as exc:
                logger.exception("Compaction dry run failed for {}", session_key)
                preview.outcome = "error"
                preview.reason = type(exc).__name__
                return preview
            finally:
                preview.elapsed_s = time.monotonic() - started
                preview.usage = telemetry.usage
                preview.provider_calls = telemetry.provider_calls
                preview.stages = telemetry.stages
            if not summary:
                preview.outcome = "error"
                preview.reason = (
                    "staged compaction could not finish safely; the original history was retained"
                    if plan.stages is not None else "the compactor produced no checkpoint"
                )
                return preview

            preview.summary = summary
            preview.outcome = "raw_fallback" if telemetry.raw_fallback else "summarized"
            preview.estimated_summary_tokens, _ = estimate_prompt_tokens_chain(
                replay_runtime.provider, replay_runtime.model,
                [{"role": "user", "content": summary}], None,
            )
            # Project the post-compaction replay on a throwaway copy, exactly as
            # commit_summary_checkpoint would; nothing is written for it.
            projected = self._project_checkpoint(plan, summary)
            preview.estimated_prompt_tokens_after, _ = self.estimate_session_prompt_tokens(
                projected, runtime=replay_runtime,
            )
            return preview
