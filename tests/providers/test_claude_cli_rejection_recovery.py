"""Recovery for Claude CLI turns refused before generation (usage/rate limit).

When the CLI's terminal result is a structured pre-generation refusal, no
assistant output or tool work happened, but the CLI may have recorded the user
input in its native session history. These tests drive a fake CLI that models
both observed persistence semantics — the input is recorded natively, or it is
not — and prove, end to end through the provider-state controller, that:

- the provider state survives the refusal so the conversation is not dead,
- a new message resumes the native session with only the new input (no replay
  of the refused input, no duplicated native records),
- without proof in the native session file the resume is refused fail-closed
  with actionable guidance, and the CLI is never invoked for it.

No live Claude Code and no authenticated calls: the fake records argv, stdin,
and its native session file, and emits the real 2.1.317-shaped terminal
results.
"""

import json
import os
import stat
from dataclasses import replace

import pytest

from nanobot.providers.base import (
    LLMProvider,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.claude_cli_provider import (
    REJECTED_INPUT_UNVERIFIED_MESSAGE,
    SESSION_INTERRUPTED_MESSAGE,
    ClaudeCLIProvider,
)
from nanobot.providers.conversation_state import (
    MODEL_ERROR_PLACEHOLDER,
    ProviderConversationStateController,
)
from nanobot.utils.helpers import build_assistant_message

_MODEL = "claude_cli/claude-sonnet-4-5-20250929"
_NATIVE_SESSION_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"

# A failed print run: init record, then the real 2.1.317-shaped usage-limit
# terminal result (is_error, api_error, no assistant output). FAKE_REJECT
# selects this path. FAKE_NO_RECORD additionally models the CLI refusing
# before it persisted the user turn to its native session file.
_RECOVERY_FAKE_BODY = """
if [ "$1" = "auth" ]; then
  printf '{"authMethod":"claude.ai","email":"fake@example.com","orgId":"org-1","configDirectory":"/home/fake/.claude"}\\n'
  exit 0
fi
mkdir -p "$CLAUDE_CLI_FAKE_DIR/calls"
n=$(ls -1 "$CLAUDE_CLI_FAKE_DIR/calls" 2>/dev/null | wc -l | tr -d " ")
c="$CLAUDE_CLI_FAKE_DIR/calls/$n"
mkdir "$c"
printf "%s\\n" "$@" > "$c/args"
cat > "$c/stdin"
resumed=$(awk '/^--resume$/{getline; print; exit}' "$c/args")
sid="__NATIVE_SESSION_ID__"
[ -n "$resumed" ] && sid="$resumed"
proj=$(printf %s "$PWD" | sed 's/[^A-Za-z0-9]/-/g')
native_dir="$HOME/.claude/projects/$proj"
mkdir -p "$native_dir"
record_user() {
  python3 - "$native_dir/$sid.jsonl" "$sid" "$c/stdin" <<'PY'
import json, sys
path, sid, stdin_file = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(stdin_file, encoding="utf-8").read()
rec = {
    "type": "user",
    "sessionId": sid,
    "isSidechain": False,
    "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    "uuid": "u-" + sid[:8],
    "timestamp": "2026-02-18T12:00:00.000Z",
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\\n")
PY
}
if [ -n "${FAKE_REJECT:-}" ]; then
  if [ -z "${FAKE_NO_RECORD:-}" ]; then
    record_user
  fi
  printf '{"type":"system","subtype":"init","session_id":"%s"}\\n' "$sid"
  printf '{"type":"result","subtype":"success","is_error":true,"api_error_status":429,'
  printf '"api_error_origin":"first_party","api_error":"usage_limit_reached",'
  printf '"terminal_reason":"api_error","result":"Usage limit reached",'
  printf '"session_id":"%s","usage":{"input_tokens":0,"output_tokens":0}}\\n' "$sid"
  exit 1
fi
record_user
python3 - "$native_dir/$sid.jsonl" "$sid" <<'PY'
import json, sys
path, sid = sys.argv[1], sys.argv[2]
rec = {
    "type": "assistant",
    "sessionId": sid,
    "isSidechain": False,
    "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    "uuid": "a-" + sid[:8],
    "timestamp": "2026-02-18T12:00:01.000Z",
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\\n")
PY
printf '{"type":"system","subtype":"init","session_id":"%s"}\\n' "$sid"
printf '{"type":"result","subtype":"success","is_error":false,"result":"done",'
printf '"session_id":"%s","usage":{"input_tokens":2,"output_tokens":1}}\\n' "$sid"
""".replace("__NATIVE_SESSION_ID__", _NATIVE_SESSION_ID)


def _write_fake_cli(path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


def _provider(command: str) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(default_model=_MODEL, command=command)


def _ctx(conversation_state: "ProviderConversationState | None" = None) -> ProviderCallContext:
    return ProviderCallContext(
        conversation_state=conversation_state,
        session_id="webui:chat-1",
        workspace="/tmp/ws",
    )


def _native_file(tmp_path):
    # Mirror the provider's sanitization exactly (every non-alphanumeric to "-").
    import re

    proj = re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(os.getcwd()))
    return tmp_path / "home" / ".claude" / "projects" / proj / f"{_NATIVE_SESSION_ID}.jsonl"


def _native_records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _placeholder() -> dict:
    return build_assistant_message(MODEL_ERROR_PLACEHOLDER)


async def _turn(provider, messages, state, session_id="webui:chat-1"):
    """Run one turn the way the runner does: prepare, call, observe, finish.

    *messages* is the full transcript including the new user message of the
    turn, exactly as the runner hands it to the controller and the provider.
    Returns the finished state, the response, and the transcript after the
    turn (with the runner's local error placeholder appended on failure).
    """
    controller = ProviderConversationStateController(
        provider=provider,
        model=_MODEL,
        messages=messages,
        state=state,
        session_id=session_id,
    )
    context = replace(
        controller.prepare_request(
            messages,
            context_window_tokens=None,
            supplemental_messages=[messages[-1]],
        ),
        workspace="/tmp/ws",
    )
    response = await provider.chat_stream(
        messages=messages,
        provider_context=context,
    )
    controller.observe_response(response, messages)
    after = list(messages)
    if response.finish_reason == "error":
        after.append(_placeholder())
    return controller.finish(after), response, after


async def _fresh_rejected_turn(provider, messages, tmp_path, monkeypatch, **env):
    """Turn one of a new chat refused before generation (manual fresh context)."""
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(tmp_path / "out"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    response = await provider.chat_stream(
        messages=messages,
        provider_context=_ctx(),
    )
    return response


@pytest.mark.asyncio
async def test_usage_limit_refusal_preserves_state_and_recovers_on_new_message(
    tmp_path, monkeypatch
) -> None:
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _RECOVERY_FAKE_BODY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    provider = _provider(str(fake))

    system, turn_one, turn_two = (
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
    )

    # Turn one (fresh chat) is refused before generation.
    response = await _fresh_rejected_turn(
        provider, [system, turn_one], tmp_path, monkeypatch, FAKE_REJECT="1",
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "usage_limit"
    assert LLMProvider.public_error_kind(response) == "usage_limit"
    # The refusal is provably pre-generation, so the provider state must
    # survive it: the conversation is not dead.
    assert response.preserve_provider_state_on_error is True
    assert response.provider_state is not None
    assert response.provider_state.payload["session_id"] == _NATIVE_SESSION_ID
    assert response.provider_state.pending_messages == []
    # No automatic retry: exactly one CLI print call so far.
    assert len(list((tmp_path / "out" / "calls").iterdir())) == 1

    # The CLI recorded the refused input in its native session history.
    records = _native_records(_native_file(tmp_path))
    assert [r["type"] for r in records] == ["user"]
    assert records[0]["message"]["content"] == [{"type": "text", "text": "turn one"}]

    # The runner closed the turn with its local error placeholder.
    transcript = [system, turn_one, _placeholder()]
    # Drive the state through the controller exactly as the runner does:
    controller = ProviderConversationStateController(
        provider=provider,
        model=_MODEL,
        messages=[system, turn_one],
        state=None,
        session_id="webui:chat-1",
    )
    controller.observe_response(response, [system, turn_one])
    state = controller.finish(transcript)
    assert state is not None
    assert state.pending_messages == [_placeholder()]

    # Turn two: a new message arrives after the limit clears. The native
    # session proves the refused input is its tail, so only the new input is
    # sent — the refused input is never replayed.
    monkeypatch.delenv("FAKE_REJECT")
    state, second, _ = await _turn(provider, transcript + [turn_two], state)
    assert second.finish_reason == "stop"
    assert state is not None

    args2 = (tmp_path / "out" / "calls" / "1" / "args").read_text(encoding="utf-8").splitlines()
    assert args2[args2.index("--resume") + 1] == _NATIVE_SESSION_ID
    stdin2 = (tmp_path / "out" / "calls" / "1" / "stdin").read_text(encoding="utf-8")
    assert stdin2 == "turn two"

    # The native history has the refused input exactly once (no duplication),
    # followed by the new input and the reply.
    records = _native_records(_native_file(tmp_path))
    texts = [r["message"]["content"][0]["text"] for r in records]
    assert texts == ["turn one", "turn two", "done"]
    assert [r["type"] for r in records] == ["user", "user", "assistant"]


@pytest.mark.asyncio
async def test_rejection_without_native_record_fails_closed(
    tmp_path, monkeypatch
) -> None:
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _RECOVERY_FAKE_BODY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    provider = _provider(str(fake))

    system, turn_one, turn_two = (
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
    )

    response = await _fresh_rejected_turn(
        provider, [system, turn_one], tmp_path, monkeypatch,
        FAKE_REJECT="1", FAKE_NO_RECORD="1",
    )
    assert response.error_kind == "usage_limit"
    assert response.preserve_provider_state_on_error is True
    assert response.provider_state is not None

    transcript = [system, turn_one, _placeholder()]
    controller = ProviderConversationStateController(
        provider=provider,
        model=_MODEL,
        messages=[system, turn_one],
        state=None,
        session_id="webui:chat-1",
    )
    controller.observe_response(response, [system, turn_one])
    state = controller.finish(transcript)

    # The native file has no proof of the refused input: resending it could
    # duplicate native history, and dropping it could lose it. Fail closed
    # with actionable guidance — and never invoke the CLI for the resume.
    state, second, _ = await _turn(provider, transcript + [turn_two], state)
    assert second.finish_reason == "error"
    assert second.error_kind == "bad_request"
    assert second.content == REJECTED_INPUT_UNVERIFIED_MESSAGE
    assert "Start a new chat" in second.content
    assert len(list((tmp_path / "out" / "calls").iterdir())) == 1


@pytest.mark.asyncio
async def test_rejection_during_resume_recovers_with_new_message(
    tmp_path, monkeypatch
) -> None:
    """The reported scenario: an established chat hits the limit mid-conversation."""
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _RECOVERY_FAKE_BODY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    provider = _provider(str(fake))

    system, turn_one, turn_two, turn_three = (
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
        {"role": "user", "content": "turn three"},
    )

    # Turn one completes and records the native session.
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(tmp_path / "out"))
    first = await provider.chat_stream(
        messages=[system, turn_one],
        provider_context=_ctx(),
    )
    assert first.finish_reason == "stop"
    state = first.provider_state
    assert state is not None

    # Turn two is refused before generation while resuming.
    transcript = [
        system, turn_one,
        build_assistant_message("done"),
        turn_two,
    ]
    monkeypatch.setenv("FAKE_REJECT", "1")
    state, rejected, transcript_after = await _turn(provider, transcript, state)
    assert rejected.finish_reason == "error"
    assert rejected.error_kind == "usage_limit"
    assert rejected.preserve_provider_state_on_error is True
    # The resume path keeps the existing state (no fresh one is recorded).
    assert rejected.provider_state is None
    assert state is not None
    assert state.pending_messages == [_placeholder()]

    # Turn three recovers: the native tail proves "turn two" was recorded.
    monkeypatch.delenv("FAKE_REJECT")
    transcript2 = transcript_after + [turn_three]
    state, third, _ = await _turn(provider, transcript2, state)
    assert third.finish_reason == "stop"

    args3 = (tmp_path / "out" / "calls" / "2" / "args").read_text(encoding="utf-8").splitlines()
    assert args3[args3.index("--resume") + 1] == _NATIVE_SESSION_ID
    stdin3 = (tmp_path / "out" / "calls" / "2" / "stdin").read_text(encoding="utf-8")
    assert stdin3 == "turn three"

    records = _native_records(_native_file(tmp_path))
    texts = [r["message"]["content"][0]["text"] for r in records]
    # The refused "turn two" appears exactly once — it was never replayed —
    # and each completed turn is followed by its reply.
    assert texts == ["turn one", "done", "turn two", "turn three", "done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("leftover", [
    {"role": "user", "content": "interrupted input"},
    {"role": "assistant", "content": "partial reply"},
])
async def test_unverifiable_leftover_still_refuses(
    tmp_path, monkeypatch, leftover: dict
) -> None:
    """Leftovers that are not the exact failed-turn shape stay fail-closed."""
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _RECOVERY_FAKE_BODY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(tmp_path / "out"))
    provider = _provider(str(fake))

    system, turn_one, turn_two = (
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
    )
    first = await provider.chat_stream(
        messages=[system, turn_one],
        provider_context=_ctx(),
    )
    assert first.finish_reason == "stop"

    # An interrupted (e.g. cancelled) turn left a non-placeholder message in
    # the pending delta: the native session cannot be resumed safely.
    state = first.provider_state.with_pending_messages([leftover, turn_two])
    second = await provider.chat_stream(
        messages=[system, turn_one, leftover, turn_two],
        provider_context=_ctx(conversation_state=state),
    )
    assert second.finish_reason == "error"
    assert second.error_kind == "bad_request"
    assert second.content == SESSION_INTERRUPTED_MESSAGE
    # The CLI was never invoked for the refused resume.
    assert len(list((tmp_path / "out" / "calls").iterdir())) == 1


@pytest.mark.asyncio
async def test_placeholder_without_matching_transcript_refuses(
    tmp_path, monkeypatch
) -> None:
    """A placeholder pending with a transcript that does not match it refuses."""
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _RECOVERY_FAKE_BODY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(tmp_path / "out"))
    provider = _provider(str(fake))

    system, turn_one, turn_two = (
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
    )
    first = await provider.chat_stream(
        messages=[system, turn_one],
        provider_context=_ctx(),
    )
    state = first.provider_state.with_pending_messages([_placeholder(), turn_two])

    # The transcript tail is [assistant, placeholder, user]: there is no
    # refused user input before the placeholder, so nothing can be verified.
    second = await provider.chat_stream(
        messages=[system, build_assistant_message("done"), _placeholder(), turn_two],
        provider_context=_ctx(conversation_state=state),
    )
    assert second.finish_reason == "error"
    assert second.error_kind == "bad_request"
    assert second.content == SESSION_INTERRUPTED_MESSAGE
    assert len(list((tmp_path / "out" / "calls").iterdir())) == 1
