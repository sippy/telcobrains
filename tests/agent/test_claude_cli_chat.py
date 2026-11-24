"""End-to-end Nanobot chat path for the explicit ``claude_cli`` preset.

Drives two real agent turns through the actual loop/runner boundary with a fake
``claude`` executable (no live Claude Code, no authenticated calls):

- Turn one opens a native CLI session; the random marker in the user input must
  be delivered over stdin and the recorded state must carry the native session id.
- Turn two resumes the exact recorded native session (``--resume``) and sends
  only the new user input on the wire — never a transcript replay.

Cross-conversation, cross-workspace, and cross-account resumes are denied
fail-closed, and cancellation must not leave orphan CLI processes.
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import uuid
from pathlib import Path

import pytest

from agent.session_helpers import run_session
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import ProviderConversationState
from nanobot.providers.claude_cli_bridge import ClaudeCodeToolBridge
from nanobot.providers.claude_cli_provider import (
    IDENTITY_CHANGED_MESSAGE,
    IDENTITY_UNVERIFIED_MESSAGE,
    SESSION_BINDING_MESSAGE,
    SESSION_INTERRUPTED_MESSAGE,
    SESSION_STATE_KIND,
    ClaudeCLIProvider,
)

_MODEL = "claude_cli/fake-model"
NATIVE_ID = "native-abc"

_CHAT_FAKE_BODY = """
if [ "$1" = "auth" ]; then
  echo auth >> "$CLI_LOG_DIR/auth.log"
  if [ -f "$CLI_LOG_DIR/noauth" ]; then
    echo "not logged in" >&2
    exit 1
  fi
  if [ -f "$CLI_LOG_DIR/badauth" ]; then
    echo "this is not json"
    exit 0
  fi
  email=$(cat "$CLI_LOG_DIR/email" 2>/dev/null || echo "fake@example.com")
  printf '{"authMethod":"claude.ai","email":"%s","orgId":"org-1","configDirectory":"/home/fake/.claude"}\\n' "$email"
  exit 0
fi
d="$CLI_LOG_DIR/calls"
mkdir -p "$d"
n=$(ls -1 "$d" 2>/dev/null | wc -l | tr -d " ")
c="$d/$n"
mkdir "$c"
printf "%s\\n" "$@" > "$c/args"
cat > "$c/stdin"
env | sort > "$c/env"
echo "$$" > "$c/pid"
if [ -f "$CLI_LOG_DIR/slow" ]; then
  sleep 30
fi
sid="native-abc"
if [ -f "$CLI_LOG_DIR/sid" ]; then
  sid=$(cat "$CLI_LOG_DIR/sid")
fi
resumed=$(awk '/^--resume$/{getline; print; exit}' "$c/args")
if [ -n "$resumed" ]; then
  sid="$resumed"
  if [ -n "${FAKE_RESUME_REPORT:-}" ]; then
    sid="$FAKE_RESUME_REPORT"
  fi
  if [ -f "$CLI_LOG_DIR/losess" ]; then
    echo "No conversation found with session ID: $resumed" >&2
    exit 1
  fi
fi
printf '{"type":"system","subtype":"init","session_id":"%s"}\\n' "$sid"
printf '{"type":"result","subtype":"success","is_error":false,"result":"ok","session_id":"%s","usage":{"input_tokens":1,"output_tokens":1}}\\n' "$sid" "$sid"
"""


def _write_fake_cli(path: Path, body: str = _CHAT_FAKE_BODY) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


def _make_loop(tmp_path: Path, fake: Path, log_dir: Path, workspace: Path) -> AgentLoop:
    os.environ["CLI_LOG_DIR"] = str(log_dir)
    provider = ClaudeCLIProvider(default_model=_MODEL, command=str(fake), cwd=str(workspace))
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=workspace, model=_MODEL)
    return loop


def _msg(chat_id: str, content: str) -> InboundMessage:
    return InboundMessage(channel="test", sender_id="user-1", chat_id=chat_id, content=content)


def _last_assistant(session) -> str:
    for message in reversed(session.messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else str(content)
    return ""


async def _outbound_contents(loop: AgentLoop, count: int = 1) -> list[str]:
    """Collect the next *count* delivered chat replies from the loop's bus."""
    contents = []
    for _ in range(count):
        while True:
            try:
                message = await asyncio.wait_for(loop.bus.outbound.get(), timeout=5.0)
            except asyncio.TimeoutError:
                return contents
            # Tool activity (hosted bridge calls included) rides ProgressEvents,
            # exactly as in an ordinary turn; only chat replies count here.
            if isinstance(message.event, ProgressEvent):
                continue
            contents.append(message.content if isinstance(message.content, str) else str(message.content))
            break
    return contents


def _call_count(log_dir: Path) -> int:
    calls = log_dir / "calls"
    return len(list(calls.iterdir())) if calls.exists() else 0


def _call_args(log_dir: Path, index: int) -> list[str]:
    return (log_dir / "calls" / str(index) / "args").read_text(encoding="utf-8").splitlines()


def _call_stdin(log_dir: Path, index: int) -> str:
    return (log_dir / "calls" / str(index) / "stdin").read_text(encoding="utf-8")


_TOOL_FAKE_CLI = r'''
import json, os, sys

log_dir = os.environ["CLI_LOG_DIR"]
call_dir = os.path.join(log_dir, "calls")
os.makedirs(call_dir, exist_ok=True)

if sys.argv[1:2] == ["auth"]:
    with open(os.path.join(log_dir, "auth.log"), "a", encoding="utf-8") as f:
        f.write("auth\n")
    email_path = os.path.join(log_dir, "email")
    email = open(email_path, encoding="utf-8").read().strip() if os.path.exists(email_path) else "fake@example.com"
    print(json.dumps({"authMethod": "claude.ai", "email": email, "orgId": "org-1", "configDirectory": "/home/fake/.claude"}))
    sys.exit(0)

n = len(os.listdir(call_dir))
c = os.path.join(call_dir, str(n))
os.makedirs(c)
argv = sys.argv[1:]
with open(os.path.join(c, "args"), "w", encoding="utf-8") as f:
    f.write("\n".join(argv) + "\n")
with open(os.path.join(c, "stdin"), "w", encoding="utf-8") as f:
    f.write(sys.stdin.read())

def find_flag(name):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None

sid = "native-abc"
resumed = find_flag("--resume")
if resumed:
    sid = resumed

result_text = "no-bridge"
mcp_config = find_flag("--mcp-config")
if mcp_config:
    if os.path.exists(mcp_config):
        with open(mcp_config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = json.loads(mcp_config)
    with open(os.path.join(c, "mcp_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    servers = config.get("mcpServers", {})
    if servers:
        import asyncio
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        server = next(iter(servers.values()))
        async def drive():
            async with sse_client(server["url"], headers=server.get("headers", {})) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    names = sorted(t.name for t in (await session.list_tools()).tools)
                    with open(os.path.join(c, "tools.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(names) + "\n")
                    if not names:
                        return "NO_TOOLS"
                    # Exercise the real workspace-scoped read through the turn's
                    # shared dispatch boundary: one allowed workspace read and one
                    # escaping path that must be denied by the same operation.
                    outcomes = []
                    allowed_text = None
                    for label, args in (
                        ("allowed", {"path": "note.txt"}),
                        ("denied", {"path": "../outside/outside.txt"}),
                    ):
                        result = await session.call_tool(names[0], args)
                        text = result.content[0].text
                        with open(os.path.join(c, "tool_%s.txt" % label), "w", encoding="utf-8") as f:
                            f.write(text)
                        with open(os.path.join(c, "tool_%s_iserror.txt" % label), "w") as f:
                            f.write(str(result.isError))
                        if label == "allowed":
                            allowed_text = text
                        outcomes.append("%s isError=%s" % (label, result.isError))
                    return " ; ".join(outcomes) + " || " + (allowed_text or "")
        result_text = "bridge-saw: " + asyncio.run(drive())

print(json.dumps({"type": "system", "subtype": "init", "session_id": sid}))
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": result_text,
    "session_id": sid,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}))
'''


def _write_fake_cli_py(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


@pytest.mark.asyncio
async def test_tool_turn_and_resumed_turn_run_through_the_shared_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real AgentLoop + fake Claude CLI + real MCP bridge: the RESUMED turn
    re-authorizes its tools through its own active turn's shared dispatch
    boundary.

    The opted-in ``read_file`` is served only through the active turn's shared
    tool dispatch boundary (the same object the loop's own model-initiated
    calls run through). A resumed turn must re-authorize the surface through
    its own turn — never the previous turn's endpoint — with the same
    workspace-scoped allow/deny semantics, and the built-in CLI tool surface
    stays empty.
    """
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    monkeypatch.setenv("CLI_LOG_DIR", str(log_dir))
    fake = tmp_path / "claude"
    _write_fake_cli_py(fake, _TOOL_FAKE_CLI)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "note.txt").write_text("loop fixture payload\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.txt").write_text("top secret outside\n", encoding="utf-8")

    provider = ClaudeCLIProvider(
        default_model=_MODEL,
        command=str(fake),
        cwd=str(workspace),
        tools=["read_file"],
        tool_bridge=ClaudeCodeToolBridge(),
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model=_MODEL,
        restrict_to_workspace=True,
    )
    try:
        # Turn one: a fresh native session; its hosted tool calls already run
        # through the loop's shared per-turn dispatch boundary (the allowed
        # read / denied escape surface is covered by the config-level e2e
        # suite). Here we only need the reply to carry the read outcome.
        await run_session(loop, _msg("chat-tools", "read the fixture"))
        reply = " ".join(await _outbound_contents(loop, count=1))
        assert "loop fixture payload" in reply

        call0 = log_dir / "calls" / "0"
        args0 = (call0 / "args").read_text(encoding="utf-8").splitlines()
        assert args0[args0.index("--tools") + 1] == ""  # built-in surface stays empty
        assert args0[args0.index("--allowedTools") + 1] == "mcp__nanobot__*"

        # Turn two: the resumed turn is likewise authorized through its own
        # active turn's shared boundary — never the first turn's endpoint.
        await run_session(loop, _msg("chat-tools", "read it again"))
        reply2 = " ".join(await _outbound_contents(loop, count=1))
        assert "loop fixture payload" in reply2
        assert _call_count(log_dir) == 2
        args1 = _call_args(log_dir, 1)
        assert args1[args1.index("--resume") + 1] == NATIVE_ID
        assert (log_dir / "calls" / "1" / "tools.txt").read_text(encoding="utf-8").splitlines() == ["read_file"]
        assert (log_dir / "calls" / "1" / "tool_allowed_iserror.txt").read_text().strip() == "False"
        assert "loop fixture payload" in (log_dir / "calls" / "1" / "tool_allowed.txt").read_text(encoding="utf-8")
        assert (log_dir / "calls" / "1" / "tool_denied_iserror.txt").read_text().strip() == "True"
        assert "top secret outside" not in (log_dir / "calls" / "1" / "tool_denied.txt").read_text(encoding="utf-8")
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_two_turn_chat_opens_and_resumes_native_session(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)

    marker = f"marker-{uuid.uuid4().hex}"
    try:
        # Turn one: fresh native session.
        await run_session(loop, _msg("chat-1", f"remember the {marker}"))
        session = loop.sessions.get_or_create("test:chat-1")
        state = session.provider_state
        assert state is not None
        assert state.kind == SESSION_STATE_KIND
        assert state.payload["session_id"] == NATIVE_ID
        assert state.payload["nanobot_session"] == "test:chat-1"
        assert state.payload["workspace"] == str(workspace)
        # The verified login identity is bound as a one-way fingerprint.
        assert state.payload["identity"]
        assert "fake@example.com" not in str(state.to_private_record())

        args0 = _call_args(log_dir, 0)
        assert "--resume" not in args0
        assert "--no-session-persistence" not in args0
        stdin0 = _call_stdin(log_dir, 0)
        assert marker in stdin0
        assert (log_dir / "auth.log").read_text().count("auth") == 1

        # Turn two: resume the exact recorded session with only the new input.
        await run_session(loop, _msg("chat-1", "what was the marker?"))
        session = loop.sessions.get_or_create("test:chat-1")
        assert session.provider_state is not None
        assert session.provider_state.payload["session_id"] == NATIVE_ID

        assert _call_count(log_dir) == 2
        args1 = _call_args(log_dir, 1)
        assert args1[args1.index("--resume") + 1] == NATIVE_ID
        assert "--system-prompt-file" not in args1
        # Exactly the new user input on the wire — no transcript replay.
        assert _call_stdin(log_dir, 1) == "what was the marker?"
        # The login was re-verified for the resumed turn.
        assert (log_dir / "auth.log").read_text().count("auth") == 2
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_denied_across_conversations(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)
    try:
        await run_session(loop, _msg("chat-1", "first message"))
        state = loop.sessions.get_or_create("test:chat-1").provider_state
        assert state is not None

        # A different conversation attempts to adopt the recorded session.
        session2 = await loop.sessions.state.get("test:chat-2")
        session2.provider_state = ProviderConversationState.from_private_record(
            state.to_private_record()
        )
        await loop.sessions.state.prepare_input(session2)
        await run_session(loop, _msg("chat-2", "adopt my session"))

        # The denial is fail-closed: the CLI never ran a resume, the recorded
        # state was discarded, and the fixed diagnostic was delivered.
        assert _call_count(log_dir) == 1
        session2 = await loop.sessions.state.get("test:chat-2")
        assert session2.provider_state is None
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert SESSION_BINDING_MESSAGE in delivered
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_denied_across_workspaces(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    ws_a = tmp_path / "ws-a"
    ws_b = tmp_path / "ws-b"
    ws_a.mkdir()
    ws_b.mkdir()

    loop_a = _make_loop(tmp_path, fake, log_dir, ws_a)
    loop_b = _make_loop(tmp_path, fake, log_dir, ws_b)
    try:
        await run_session(loop_a, _msg("chat-1", "first message"))
        state = loop_a.sessions.get_or_create("test:chat-1").provider_state
        assert state is not None
        assert state.payload["workspace"] == str(ws_a)

        session_b = await loop_b.sessions.state.get("test:chat-1")
        session_b.provider_state = ProviderConversationState.from_private_record(
            state.to_private_record()
        )
        await loop_b.sessions.state.prepare_input(session_b)
        await run_session(loop_b, _msg("chat-1", "adopt my session"))

        assert _call_count(log_dir) == 1
        session_b = await loop_b.sessions.state.get("test:chat-1")
        assert session_b.provider_state is None
        delivered = " ".join(await _outbound_contents(loop_b, count=2))
        assert SESSION_BINDING_MESSAGE in delivered
    finally:
        await loop_a.aclose()
        await loop_b.aclose()


@pytest.mark.asyncio
async def test_resume_denied_when_login_changed(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)
    try:
        await run_session(loop, _msg("chat-1", "first message"))
        assert loop.sessions.get_or_create("test:chat-1").provider_state is not None

        # A different account now owns the CLI login.
        (log_dir / "email").write_text("other-account@example.com", encoding="utf-8")
        await run_session(loop, _msg("chat-1", "second message"))

        assert _call_count(log_dir) == 1
        session = loop.sessions.get_or_create("test:chat-1")
        assert session.provider_state is None
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert IDENTITY_CHANGED_MESSAGE in delivered
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_denied_when_identity_unverifiable(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)
    try:
        await run_session(loop, _msg("chat-1", "first message"))
        assert loop.sessions.get_or_create("test:chat-1").provider_state is not None

        # The CLI can no longer attest the login (unsupported/unparseable output).
        (log_dir / "badauth").touch()
        await run_session(loop, _msg("chat-1", "second message"))

        assert _call_count(log_dir) == 1
        session = loop.sessions.get_or_create("test:chat-1")
        assert session.provider_state is None
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert IDENTITY_UNVERIFIED_MESSAGE in delivered
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_cancellation_cleans_up_cli_process(tmp_path: Path) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (log_dir / "slow").touch()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)
    try:
        await loop._enqueue_session_message(_msg("chat-slow", "slow please"))
        calls = log_dir / "calls"
        call_dir = None
        for _ in range(200):
            if calls.exists() and list(calls.iterdir()):
                call_dir = list(calls.iterdir())[0]
                break
            await asyncio.sleep(0.05)
        assert call_dir is not None, "the CLI was never started"
        pid = int((call_dir / "pid").read_text().strip())

        for task in list(loop._active_tasks.get("test:chat-slow", [])):
            task.cancel()
        await asyncio.gather(*loop._active_tasks.get("test:chat-slow", []), return_exceptions=True)
        await asyncio.sleep(0.3)
        # No orphan CLI process may survive cancellation.
        assert not Path(f"/proc/{pid}").exists()
    finally:
        loop.stop()
        await loop.aclose()


@pytest.mark.asyncio
async def test_cancelled_resume_leaves_actionable_guidance_and_no_replay(tmp_path: Path) -> None:
    """Cancel a resumed turn, then the next inbound is refused fail-closed.

    The cancelled turn's staged message is left in the provider state, so the next
    turn sees two pending messages. The provider must refuse with actionable
    'start a new chat' guidance — no transcript replay and no extra completion
    spawn — rather than a technical shape error.
    """
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = _make_loop(tmp_path, fake, log_dir, workspace)

    try:
        # Turn one: fresh, fast. Records the native session (call #0).
        await run_session(loop, _msg("chat-resume", "first message"))
        assert loop.sessions.get_or_create("test:chat-resume").provider_state is not None
        assert _call_count(log_dir) == 1

        # Turn two: a resume that hangs in the CLI, then gets cancelled. The CLI
        # starts (call #1) but never completes; its staged message stays pending.
        (log_dir / "slow").touch()
        await loop._enqueue_session_message(_msg("chat-resume", "slow resume"))
        calls = log_dir / "calls"
        for _ in range(200):
            if calls.exists() and len(list(calls.iterdir())) >= 2:
                break
            await asyncio.sleep(0.05)
        assert _call_count(log_dir) >= 2, "the resume CLI was never started"

        for task in list(loop._active_tasks.get("test:chat-resume", [])):
            task.cancel()
        await asyncio.gather(*loop._active_tasks.get("test:chat-resume", []), return_exceptions=True)
        await asyncio.sleep(0.3)
        (log_dir / "slow").unlink()

        # Turn three: the next inbound. It must be refused with actionable guidance
        # and must not spawn another completion (call count stays at 2).
        await run_session(loop, _msg("chat-resume", "third message"))
        assert _call_count(log_dir) == 2
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert SESSION_INTERRUPTED_MESSAGE in delivered
    finally:
        loop.stop()
        await loop.aclose()
