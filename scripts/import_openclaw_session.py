"""Copy one OpenClaw SQLite conversation into an existing nanobot WebUI workspace."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from nanobot.config.loader import get_config_path, load_config, resolve_config_env_vars
from nanobot.session.display_history import HistoryFact, HistoryKind
from nanobot.session.location import SessionLocation
from nanobot.session.manager import Session
from nanobot.session.sqlite_store import SqliteSessionStore
from nanobot.webui.session_identity import webui_session_key

_SOURCE_ID = "openclaw_source_session_id"


def read_transcript(source: Path, session_id: str) -> tuple[str, list[dict[str, Any]]]:
    # A read transaction keeps the selected window and active branch consistent.
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        window = db.execute(
            """SELECT w.display_name, n.label FROM session_windows w
               LEFT JOIN session_nodes n ON n.session_key=w.session_key AND n.current_session_id=w.session_id
               WHERE w.session_id=?""", (session_id,),
        ).fetchone()
        if window is None:
            raise ValueError("session UUID not found in this OpenClaw agent database")
        events = db.execute(
            """SELECT e.seq, e.event_json, e.event_zstd, e.message_role
               FROM session_transcript_active_events a JOIN transcript_events e
                 ON e.session_id=a.session_id AND e.seq=a.event_seq
               WHERE a.session_id=? AND e.message_role IN ('user', 'assistant')
               ORDER BY a.active_position""", (session_id,),
        )
        messages = []
        for seq, raw, packed, role in events:
            if packed is not None:
                try:
                    from compression import zstd
                except ImportError as exc:
                    raise ValueError("compressed OpenClaw events require Python 3.14 or newer") from exc
                raw = zstd.decompress(packed)
            event = json.loads(raw)
            message = event["message"]
            if message.get("display") is False or (message.get("provenance") or {}).get("kind") in {
                "internal_system", "inter_session",
            }:
                continue
            text = message.get("content")
            if isinstance(text, list):
                text = "\n".join(
                    block["text"] for block in text if isinstance(block, dict)
                    and block.get("type") == "text" and isinstance(block.get("text"), str)
                )
            if not isinstance(text, str) or not text.strip():
                continue
            messages.append({
                "role": role, "content": text, "timestamp": event["timestamp"],
                "source": "openclaw", "source_session_id": session_id, "source_event_seq": seq,
            })
    if not messages:
        raise ValueError("no visible text in the active transcript (archived transcripts are not supported)")
    name, label = window
    return f"{label or name or session_id} (OpenClaw)", messages


def display_facts(messages: list[dict[str, Any]]) -> list[HistoryFact]:
    """Group each user input and its following answers into a completed replay turn."""
    facts: list[HistoryFact] = []
    for message in messages:
        seq = message["source_event_seq"]
        timestamp = int(datetime.fromisoformat(message["timestamp"]).timestamp() * 1000)
        is_user = message["role"] == "user"
        if not facts or is_user:
            if facts:
                last = facts[-1]
                facts.append(HistoryFact(HistoryKind.COMPLETION, turn_id=last.turn_id,
                                         phase="complete", created_at_ms=last.created_at_ms))
            turn_id = f"openclaw-{seq}"
        content = {"text": message["content"]}
        if is_user:
            content["input_id"] = f"openclaw-{seq}"
        facts.append(HistoryFact(HistoryKind.INPUT if is_user else HistoryKind.ANSWER, content,
                                 turn_id=turn_id, phase="user" if is_user else "answer",
                                 created_at_ms=timestamp))
    if facts:
        last = facts[-1]
        facts.append(HistoryFact(HistoryKind.COMPLETION, turn_id=last.turn_id,
                                 phase="complete", created_at_ms=last.created_at_ms))
    return facts


def import_session(
    store: SqliteSessionStore, session_id: str, title: str, messages: list[dict[str, Any]],
) -> str:
    key = webui_session_key(f"openclaw-{session_id}")
    facts = display_facts(messages)

    def commit() -> str:
        session = store.load(key)
        if session is None:
            session = Session(key, messages=messages, metadata={
                "title": title, "title_user_edited": True, _SOURCE_ID: session_id,
            })
            store.save(session)
            outcome = "Imported"
        else:
            if session.metadata.get(_SOURCE_ID) != session_id:
                raise ValueError(f"destination {key} already exists and was not imported from this source")
            if store.project_history(key, lambda reader: reader.turn_count):
                return "Already imported"
            # Repair only an unchanged canonical-only import, never an active/edited chat.
            if session.messages != messages:
                raise ValueError(f"destination {key} has no display history but its messages differ from the source")
            outcome = "Repaired WebUI transcript for"
        store.append_history(key, facts, expected_generation=session.generation)
        return outcome

    return store.run_write(commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", type=UUID, help="OpenClaw session UUID")
    parser.add_argument("--source-db", type=Path, default=Path.home() / ".openclaw/agents/main/agent/openclaw-agent.sqlite",
                        help="OpenClaw agent SQLite database (defaults to the main agent)")
    parser.add_argument("--config", type=Path, default=get_config_path(), help="nanobot config file")
    parser.add_argument("--workspace", type=Path, help="override the configured nanobot workspace")
    parser.add_argument("--dry-run", action="store_true", help="read and count source messages without opening the destination")
    args = parser.parse_args()
    session_id = str(args.session_id)
    try:
        title, messages = read_transcript(args.source_db, session_id)
        counts = Counter(message["role"] for message in messages)
        if args.dry_run:
            outcome = "Would import"
        else:
            config_path = args.config.expanduser().resolve()
            if not config_path.is_file():
                raise ValueError(f"nanobot config does not exist: {config_path}")
            config = resolve_config_env_vars(load_config(config_path), config_path=config_path)
            workspace = args.workspace or Path(config.agents.defaults.workspace)
            location = SessionLocation(workspace, sessions_root=config_path.parent / "sessions")
            store = SqliteSessionStore(location.sessions_dir)
            if not store.path.is_file():
                raise ValueError(f"no nanobot session database at {store.path}; start the target workspace first")
            outcome = import_session(store, session_id, title, messages)
    except (KeyError, ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    print(f"{outcome} {title!r} (source: {counts['user']} user, {counts['assistant']} assistant messages)")
    print(f"Session: {webui_session_key(f'openclaw-{session_id}')}")


if __name__ == "__main__":
    main()
