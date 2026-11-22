import json
import sqlite3
from contextlib import closing

import pytest

from nanobot.session.display_history import MAX_HISTORY_FACT_BYTES
from nanobot.session.manager import Session
from nanobot.session.sqlite_store import SqliteSessionStore
from nanobot.webui.transcript import build_webui_thread_response
from scripts.import_openclaw_session import import_session, read_transcript

SID = "f87241d4-40cf-4b05-9c60-d90e299ffd70"
KEY = f"websocket:openclaw-{SID}"
TIMESTAMP = "2026-11-11T22:23:18.819Z"


@pytest.fixture
def transcript(tmp_path):
    source = tmp_path / "openclaw.sqlite"
    with closing(sqlite3.connect(source)) as db, db:
        db.executescript("""
            CREATE TABLE session_windows(session_id, session_key, display_name);
            CREATE TABLE session_nodes(session_key, current_session_id, label);
            CREATE TABLE session_transcript_active_events(session_id, active_position, event_seq);
            CREATE TABLE transcript_events(session_id, seq, event_json, event_zstd, message_role);
        """)
        db.execute("INSERT INTO session_windows VALUES (?, 'agent', 'Generated title')", (SID,))
        db.execute("INSERT INTO session_nodes VALUES ('agent', ?, 'Test chat')", (SID,))
        events = [
            ("user", "first question", {}),
            ("assistant", "old branch", {}),
            ("assistant", [{"type": "toolCall"}, {"type": "text", "text": "first answer"}], {}),
            ("user", "another session's event", {"provenance": {"kind": "inter_session"}}),
            ("user", "second question", {}),
            ("assistant", "second answer", {}),
        ]
        for seq, (role, content, extra) in enumerate(events):
            event = json.dumps({"timestamp": TIMESTAMP, "message": {"role": role, "content": content, **extra}})
            db.execute("INSERT INTO transcript_events VALUES (?, ?, ?, NULL, ?)", (SID, seq, event, role))
            if seq != 1:
                db.execute("INSERT INTO session_transcript_active_events VALUES (?, ?, ?)", (SID, seq, seq))
    return source


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    destination = SqliteSessionStore(sessions_dir)
    destination.initialize(workspace)
    return destination


@pytest.mark.parametrize("repair", [False, True])
def test_import_is_visible_through_webui_pagination_and_repeatable(transcript, store, repair):
    title, messages = read_transcript(transcript, SID)
    assert title == "Test chat (OpenClaw)"
    if repair:
        store.save(Session(KEY, messages=messages, metadata={"openclaw_source_session_id": SID, "title": title}))
        assert store.project_history(KEY, lambda reader: reader.turn_count) == 0
    assert import_session(store, SID, title, messages) == ("Repaired WebUI transcript for" if repair else "Imported")
    assert import_session(store, SID, title, messages) == "Already imported"
    saved = store.load(KEY)
    assert saved is not None and saved.messages == messages

    def page(before=None):
        return store.project_history(
            KEY, lambda history: build_webui_thread_response(KEY, history=history, limit=1, before=before),
        )

    latest = page()
    assert [e["text"] for e in latest["events"] if "text" in e] == ["second question", "second answer"]
    oldest = page(latest["page"]["before_cursor"])
    assert [e["text"] for e in oldest["events"] if "text" in e] == ["first question", "first answer"]
    assert not oldest["page"]["has_more_before"]


def test_invalid_display_fact_rolls_back_canonical_import(transcript, store):
    title, messages = read_transcript(transcript, SID)
    messages[-1]["content"] = "x" * MAX_HISTORY_FACT_BYTES
    with pytest.raises(ValueError, match="storage limit"):
        import_session(store, SID, title, messages)
    assert store.load(KEY) is None
