"""Tests for the Claude Code CLI provider: fresh, tool-free text completions.

The CLI is exercised with a fake ``claude`` executable (no live Claude Code and no
authenticated calls). The fake records the argv, stdin, and environment it received
and emits a Claude Code ``stream-json`` stream, so the tests verify the provider's
bounded subprocess behavior, structured parsing, fail-closed request validation,
and routing isolation without touching credentials.
"""

import json
import os
import stat
import sys
import tempfile

import pytest
from pydantic import ValidationError

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import (
    ClaudeCLIProviderConfig,
    Config,
    ModelPresetConfig,
    ProvidersConfig,
)
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.claude_cli_bridge import ClaudeCodeToolBridge
from nanobot.providers.claude_cli_provider import (
    USAGE_LIMIT_MESSAGE,
    ClaudeCLIProvider,
    _classify_cli_failure,
    _parse_usage_limit_reset,
    _ParsedStream,
    build_claude_cli_argv,
    clean_claude_cli_env,
)
from nanobot.providers.claude_cli_tool_dispatch import ToolDispatch
from nanobot.providers.factory import make_provider
from nanobot.providers.routing import resolve_provider_route
from nanobot.security.workspace_access import default_workspace_scope

_MODEL = "claude_cli/claude-sonnet-4-5-20250929"


class _RecordingHook(AgentHook):
    """Records every per-call event the shared dispatch boundary emits."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[AgentHookContext] = []

    async def on_tool_execution_event(self, context: AgentHookContext) -> None:
        self.events.append(context)


def _write_fake_cli(path, body: str) -> None:
    """Write an executable fake ``claude`` that runs *body* (bash) on startup."""
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


def _happy_fake_body() -> str:
    return (
        'mkdir -p "$CLAUDE_CLI_FAKE_DIR"\n'
        'printf "%s\\n" "$@" > "$CLAUDE_CLI_FAKE_DIR/args.txt"\n'
        'cat > "$CLAUDE_CLI_FAKE_DIR/stdin.txt"\n'
        'env | sort > "$CLAUDE_CLI_FAKE_DIR/env.txt"\n'
        'printf \'{"type":"system","subtype":"init"}\\n\'\n'
        'printf \'{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"dup"}]}}\\n\'\n'
        'printf \'{"type":"result","subtype":"success","is_error":false,"result":"Hello from fake CLI",'
        '"usage":{"input_tokens":7,"output_tokens":3,"cache_read_input_tokens":0,'
        '"cache_creation_input_tokens":0}}\\n\'\n'
    )


def _system_user_fake_body() -> str:
    """Fake that records the --system-prompt-file target (copied) plus argv and stdin."""
    return (
        'mkdir -p "$CLAUDE_CLI_FAKE_DIR"\n'
        'SYSFILE=""\n'
        'prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--system-prompt-file" ]; then SYSFILE="$a"; fi\n'
        '  prev="$a"\n'
        'done\n'
        'if [ -n "$SYSFILE" ]; then cp "$SYSFILE" "$CLAUDE_CLI_FAKE_DIR/system.txt"; fi\n'
        'printf "%s\\n" "$@" > "$CLAUDE_CLI_FAKE_DIR/args.txt"\n'
        'cat > "$CLAUDE_CLI_FAKE_DIR/stdin.txt"\n'
        'printf \'{"type":"result","subtype":"success","is_error":false,"result":"summarized",'
        '"usage":{"input_tokens":5,"output_tokens":2}}\\n\'\n'
    )


def _provider(command: str, **kwargs) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(default_model=_MODEL, command=command, **kwargs)


# -------------------------------------------------------------------------- #
# argv and environment
# -------------------------------------------------------------------------- #
def test_argv_is_strict_empty_and_isolated() -> None:
    argv = build_claude_cli_argv("claude", _MODEL)
    # The provider prefix is stripped so the CLI receives the raw model id.
    assert argv[0] == "claude"
    assert argv[argv.index("--model") + 1] == "claude-sonnet-4-5-20250929"
    # Strict empty tool surface: no built-in tools and no MCP tools.
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--disallowedTools") + 1] == "mcp__*"
    # Only the empty supplied MCP config is honored; no inherited servers.
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    # No CLAUDE.md, skills, plugins, hooks, custom agents, or user/local settings.
    assert "--safe-mode" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    # Permission checks are never bypassed and no session is persisted to disk.
    assert argv[argv.index("--permission-mode") + 1] == "default"
    assert "--no-session-persistence" in argv
    # Print mode with a structured stream on stdout (stream-json requires --verbose).
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv


def test_argv_requires_a_model_id() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_claude_cli_argv("claude", "")


def test_clean_env_strips_credentials_and_is_authoritative() -> None:
    # The scrub is authoritative: credentials/routing vars are removed and there is no
    # post-scrub injection point (no extraEnv), so a gateway credential can never reach
    # the CLI.
    base = {
        "PATH": "/usr/bin",
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "ANTHROPIC_BASE_URL": "https://evil.example",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
        "HOME": "/home/u",
    }
    env = clean_claude_cli_env(base)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert name not in env
    assert env["HOME"] == "/home/u"
    assert env["PATH"] == "/usr/bin"


# -------------------------------------------------------------------------- #
# Stream parsing (single source of truth, no duplicated text)
# -------------------------------------------------------------------------- #
def test_parsed_stream_prefers_terminal_result_and_does_not_duplicate() -> None:
    parsed = _ParsedStream()
    parsed.ingest(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "partial"}]}}
    )
    parsed.ingest(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "final answer",
            "usage": {"input_tokens": 2, "output_tokens": 4},
        }
    )
    assert parsed.saw_result is True
    assert parsed.is_error is False
    assert parsed.result_text == "final answer"
    assert parsed.assistant_text == "partial"
    # The reply is never the concatenation of the two.
    assert "partial" not in parsed.result_text


def test_classify_failure_detects_auth_without_leaking_payloads() -> None:
    kind, message = _classify_cli_failure(["Error: not logged in", "use claude auth login"], 1)
    assert kind == "auth"
    assert "not authenticated" in message.lower()
    # The raw stderr is not embedded in the user-facing message.
    assert "use claude auth login" not in message


# -------------------------------------------------------------------------- #
# Fail-closed request validation
# -------------------------------------------------------------------------- #
async def test_rejects_multi_turn_history() -> None:
    provider = _provider("/nonexistent/claude")
    response = await provider.chat(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "bad_request"


async def test_rejects_tools_and_tool_choice() -> None:
    provider = _provider("/nonexistent/claude")
    response = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "bad_request"


async def test_rejects_image_content() -> None:
    provider = _provider("/nonexistent/claude")
    response = await provider.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ]
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "bad_request"


def test_does_not_advertise_image_support() -> None:
    assert ClaudeCLIProvider(default_model=_MODEL).supports_images is False
    assert ClaudeCLIProvider(default_model=_MODEL).supports_tools is False


# -------------------------------------------------------------------------- #
# End-to-end with a fake CLI
# -------------------------------------------------------------------------- #
async def test_happy_path_runs_cli_and_parses_result(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _happy_fake_body())

    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hello world"}])

    assert response.finish_reason == "stop"
    assert response.content == "Hello from fake CLI"
    assert response.usage is not None
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3

    # The prompt was delivered over stdin.
    assert (out_dir / "stdin.txt").read_text(encoding="utf-8").strip() == "hello world"
    # The strict argv reached the process (prefix stripped, empty tool surface).
    # The fake records ``"$@"`` (the args after the executable), so the first arg
    # is the print flag, not the command name.
    args = (out_dir / "args.txt").read_text(encoding="utf-8").splitlines()
    assert "-p" in args
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "claude-sonnet-4-5-20250929"
    # No gateway Anthropic credentials leaked into the CLI environment.
    env = (out_dir / "env.txt").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in env


async def test_missing_cli_reports_missing_cli() -> None:
    provider = _provider("/nonexistent/path/claude")
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "missing_cli"


async def test_auth_failure_is_classified_without_leaking(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        'echo "Error: not logged in, please run: claude auth login" >&2\n'
        "exit 1\n",
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "auth"
    # The user-facing message names the problem but not the raw CLI output.
    assert "not authenticated" in response.content.lower()


async def test_auth_failure_in_result_record_is_classified(tmp_path, monkeypatch) -> None:
    # The real CLI reports an unauthenticated run as an error *result* on stdout
    # (not stderr), e.g. result: "Not logged in · Please run /login".
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        'printf \'{"type":"result","subtype":"success","is_error":true,'
        '"num_turns":1,"result":"Not logged in \\u00b7 Please run /login",'
        '"usage":{"input_tokens":0,"output_tokens":0}}\\n\'\n',
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "auth"
    # A useful, generic auth hint (no raw payload, no leaked tokens).
    assert "authenticated" in response.content.lower()


async def test_non_json_stderr_does_not_leak_into_content(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        'echo "DEBUG: bearer=supersecrettoken xyz" >&2\n'
        'printf \'{"type":"result","subtype":"success","is_error":false,'
        '"result":"ok","usage":{"input_tokens":1,"output_tokens":1}}\\n\'\n',
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert "supersecrettoken" not in response.content


async def test_timeout_stops_and_reports(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, "sleep 30\n")
    provider = _provider(str(fake), timeout_s=0.5)
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "timeout"


async def test_fresh_system_plus_user_transport_shape(tmp_path, monkeypatch) -> None:
    # Provider-level transport coverage, NOT coverage of an actual nanobot call
    # path: it drives chat_stream directly with the accepted request shape
    # (one trusted system message + one user message, tool-free) and verifies
    # how each part is delivered (system prompt via --system-prompt-file, user
    # prompt via stdin). No nanobot utility call path has been verified or
    # configured to route to this provider: the real call sites (the main
    # agent turn, compaction archival, Dream) all send tools,
    # tool_choice, and/or prior conversation, which this route refuses, so
    # wiring a utility completion through it is deferred follow-on work.
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _system_user_fake_body())

    provider = _provider(str(fake))
    system_prompt = "You are a terse summarizer. MARKER_SYSTEM_INSTRUCTION"
    user_prompt = "Summarize: the archive has three entries."
    response = await provider.chat_stream(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[],
    )
    assert response.finish_reason == "stop"
    assert response.content == "summarized"

    # The system prompt was delivered via a file, never as command-line text.
    args = (out_dir / "args.txt").read_text(encoding="utf-8").splitlines()
    assert "--system-prompt-file" in args
    sysfile = args[args.index("--system-prompt-file") + 1]
    assert os.path.exists(sysfile) is False  # the temp file is removed after the run
    assert (out_dir / "system.txt").read_text(encoding="utf-8") == system_prompt
    assert "MARKER_SYSTEM_INSTRUCTION" not in " ".join(args)
    # The user prompt went over stdin.
    assert (out_dir / "stdin.txt").read_text(encoding="utf-8").strip() == user_prompt


async def test_error_response_never_returns_private_output(tmp_path, monkeypatch) -> None:
    # A failed run that embeds sentinel secrets in BOTH the result record and stderr
    # must surface only a fixed diagnostic, never the secrets from either surface.
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        'echo "DEBUG trace SECRET_STDERR_LEAK_abc123 near /home/user/secret path" >&2\n'
        'printf \'{"type":"result","subtype":"success","is_error":true,'
        '"result":"boom SECRET_RESULT_LEAK_xyz789 near /home/admin/keys",'
        '"usage":{"input_tokens":1,"output_tokens":1}}\\n\'\n'
        "exit 1\n",
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    # Fixed diagnostic text only: no secret from either surface leaks.
    for secret in ("SECRET_STDERR_LEAK_abc123", "SECRET_RESULT_LEAK_xyz789", "abc123", "xyz789"):
        assert secret not in response.content


def _usage_limit_record(result_text: str) -> str:
    return (
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":429,"api_error_origin":"first_party",'
        '"api_error":"usage_limit_reached","terminal_reason":"api_error",'
        f'"result":"{result_text}",'
        '"usage":{"input_tokens":0,"output_tokens":0}}'
    )


def _error_record_fake_body(record: str) -> str:
    """Fake CLI body emitting one error result record verbatim, then exit 1."""
    return "printf '%s\\n' '" + record + "'\nexit 1\n"


async def test_usage_limit_result_maps_to_usage_limit_with_clean_reset(
    tmp_path, monkeypatch,
) -> None:
    # The real 2.1.317 terminal result for a plan usage limit.
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        _error_record_fake_body(
            _usage_limit_record(
                "You\\u0027ve hit your 5-hour limit \u00b7 resets 1pm (America/Vancouver)",
            )
        ),
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.finish_reason == "error"
    assert response.error_kind == "usage_limit"
    assert LLMProvider.public_error_kind(response) == "usage_limit"
    # No automatic retry, fallback, or replay: the error is not transient.
    assert LLMProvider.is_transient_response(response) is False
    # Clean, validated display fields only.
    assert response.error_reset_time == "1pm"
    assert response.error_reset_timezone == "America/Vancouver"
    assert "1pm" in response.content
    assert "America/Vancouver" in response.content
    # The raw diagnostic is never forwarded.
    for fragment in ("5-hour", "You\'ve hit", "usage_limit_reached", "first_party"):
        assert fragment not in response.content


@pytest.mark.parametrize(
    "result_text",
    [
        "You\\u0027ve hit your limit",
        "You\\u0027ve hit your limit - resets 25pm (America/Vancouver)",
        "You\\u0027ve hit your limit - resets 1pm (Mars/Olympus)",
        "You\\u0027ve hit your limit - resets 1pm (America)",
        "You\\u0027ve hit your limit - resets 1 pm (America/Vancouver)",
        "You\\u0027ve hit your limit - resets 1:5pm (America/Vancouver)",
        "You\\u0027ve hit your limit - resets 12:60pm (America/Vancouver)",
        "You\\u0027ve hit your limit - resets 1pm (a/b/c/d)",
    ],
)
async def test_usage_limit_without_valid_reset_falls_back_to_safe_message(
    result_text: str, tmp_path, monkeypatch,
) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _error_record_fake_body(_usage_limit_record(result_text)))
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.finish_reason == "error"
    assert response.error_kind == "usage_limit"
    assert response.error_reset_time is None
    assert response.error_reset_timezone is None
    assert response.content == USAGE_LIMIT_MESSAGE
    # No partial or guessed reset details leak into the message.
    for fragment in ("1pm", "25pm", "America/Vancouver", "Mars"):
        assert fragment not in response.content


async def test_transient_429_maps_to_rate_limit_without_retry(
    tmp_path, monkeypatch,
) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        _error_record_fake_body(
            '{"type":"result","subtype":"success","is_error":true,'
            '"api_error_status":429,"api_error":"rate_limit_error",'
            '"result":"Too many requests",'
            '"usage":{"input_tokens":0,"output_tokens":0}}'
        ),
    )
    provider = _provider(str(fake))
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    # A plain 429 is a transient rate limit, distinct from the plan limit.
    assert response.finish_reason == "error"
    assert response.error_kind == "rate_limit"
    assert response.error_status_code == 429
    assert LLMProvider.public_error_kind(response) == "rate_limit"
    assert LLMProvider.is_transient_response(response) is False
    assert response.error_reset_time is None
    assert response.error_reset_timezone is None
    assert "Too many requests" not in response.content


async def test_usage_limit_error_is_never_retried_in_persistent_mode(
    tmp_path, monkeypatch,
) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(
        fake,
        _error_record_fake_body(
            _usage_limit_record(
                "You\\u0027ve hit your 5-hour limit \u00b7 resets 1pm (America/Vancouver)",
            )
        ),
    )
    provider = _provider(str(fake))
    first = await provider.chat(messages=[{"role": "user", "content": "hi"}])

    calls = 0

    async def call(**kw):
        nonlocal calls
        calls += 1
        return first

    result = await provider._run_with_retry(
        call,
        {},
        retry_mode="persistent",
        on_retry_wait=None,
        on_retry_exhausted=None,
        on_retry_status=None,
    )

    assert calls == 1
    assert result is first


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("You\'ve hit your 5-hour limit · resets 1pm (America/Vancouver)", ("1pm", "America/Vancouver")),
        ("resets 12am (UTC)", ("12am", "UTC")),
        ("resets 3:30pm (America/Argentina/Buenos_Aires)", ("3:30pm", "America/Argentina/Buenos_Aires")),
        ("resets 25pm (America/Vancouver)", None),
        ("resets 0pm (America/Vancouver)", None),
        ("resets 1pm (Mars/Olympus)", None),
        ("resets 1pm (America)", None),
        ("resets 1pm (a/b/c/d)", None),
        ("resets 1pm (America/Van couver)", None),
        ("no reset information", None),
        ("", None),
    ],
)
def test_parse_usage_limit_reset(text: str, expected) -> None:
    assert _parse_usage_limit_reset(text) == expected


# -------------------------------------------------------------------------- #
# Routing: explicit opt-in only; normal routing preserved
# -------------------------------------------------------------------------- #
def _cli_config(command: str = "claude") -> Config:
    return Config(
        providers=ProvidersConfig(claude_cli=ClaudeCLIProviderConfig(command=command)),
        model_presets={
            "cli": ModelPresetConfig(model=_MODEL, provider="claude_cli"),
        },
    )


def test_claude_cli_preset_routes_to_claude_cli_backend() -> None:
    config = _cli_config()
    route = resolve_provider_route(config, preset=config.model_presets["cli"])
    assert route.backend == "claude_cli"
    assert route.provider_name == "claude_cli"
    assert isinstance(route.provider_config, ClaudeCLIProviderConfig)


def test_make_provider_builds_claude_cli_without_api_key() -> None:
    config = _cli_config()
    provider = make_provider(config, preset=config.model_presets["cli"])
    assert isinstance(provider, ClaudeCLIProvider)
    assert provider.supports_images is False


def test_make_provider_default_stays_tool_free() -> None:
    config = _cli_config()
    provider = make_provider(config, preset=config.model_presets["cli"])
    assert provider.tools == ()
    assert provider.tool_bridge is None


def test_make_provider_wires_opted_in_tools_bridge() -> None:
    config = Config(
        providers=ProvidersConfig(
            claude_cli=ClaudeCLIProviderConfig(command="claude", tools=["read_file"]),
        ),
        model_presets={
            "cli": ModelPresetConfig(model=_MODEL, provider="claude_cli"),
        },
    )
    provider = make_provider(config, preset=config.model_presets["cli"])
    assert provider.tools == ("read_file",)
    assert isinstance(provider.tool_bridge, ClaudeCodeToolBridge)


def test_config_refuses_tool_names_outside_the_production_allowlist() -> None:
    # The allowlist is exact: zero tools or the workspace-only read_file.
    assert ClaudeCLIProviderConfig().tools == []
    assert ClaudeCLIProviderConfig(tools=["read_file"]).tools == ["read_file"]
    for name in ("exec", "write_file", "mcp__anything"):
        with pytest.raises(ValidationError, match="not allowed"):
            ClaudeCLIProviderConfig(tools=[name])
    for name in ("", "  "):
        with pytest.raises(ValidationError, match="non-empty"):
            ClaudeCLIProviderConfig(tools=[name])


def test_provider_wiring_refuses_tool_names_outside_the_production_allowlist() -> None:
    # Even a directly constructed provider (no config layer) is fail-closed.
    with pytest.raises(ValueError, match="not allowed"):
        ClaudeCLIProvider(default_model=_MODEL, tools=["exec"])
    assert ClaudeCLIProvider(default_model=_MODEL, tools=[]).tools == ()
    assert ClaudeCLIProvider(default_model=_MODEL, tools=["read_file"]).tools == ("read_file",)


def test_claude_cli_route_does_not_require_an_api_key() -> None:
    # No api_key is set anywhere on the claude_cli provider config, yet the route
    # resolves: the CLI owns its own local login.
    config = _cli_config()
    assert config.providers.claude_cli.api_key is None
    resolve_provider_route(config, preset=config.model_presets["cli"])  # must not raise


def test_anthropic_preset_still_routes_to_anthropic() -> None:
    from nanobot.config.schema import ProviderConfig

    config = Config(
        providers=ProvidersConfig(anthropic=ProviderConfig(api_key="sk-ant-test")),
        model_presets={
            "anth": ModelPresetConfig(model="anthropic/claude-sonnet-4-5", provider="anthropic"),
        },
    )
    route = resolve_provider_route(config, preset=config.model_presets["anth"])
    assert route.backend == "anthropic"
    assert route.provider_name == "anthropic"


def test_registry_has_direct_claude_cli_spec() -> None:
    from nanobot.providers.registry import find_by_name

    spec = find_by_name("claude_cli")
    assert spec is not None
    assert spec.backend == "claude_cli"
    assert spec.is_direct is True
    assert spec.request_apis == ()


# -------------------------------------------------------------------------- #
# Session-resume argv shape
# -------------------------------------------------------------------------- #
def test_session_argv_persists_session_and_resumes() -> None:
    fresh = build_claude_cli_argv("claude", _MODEL, persist_session=True)
    assert "--no-session-persistence" not in fresh
    assert "--resume" not in fresh

    resume = build_claude_cli_argv(
        "claude", _MODEL, persist_session=True, resume_session_id="native-123"
    )
    assert "--no-session-persistence" not in resume
    assert resume[resume.index("--resume") + 1] == "native-123"
    # A resumed session keeps its own system prompt: the two are never combined.
    assert "--system-prompt-file" not in resume


def test_resume_requires_persistence() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_claude_cli_argv("claude", _MODEL, resume_session_id="native-123")


def test_stateless_argv_is_unchanged() -> None:
    argv = build_claude_cli_argv("claude", _MODEL)
    assert "--no-session-persistence" in argv
    assert "--resume" not in argv


# -------------------------------------------------------------------------- #
# Native session resume with a fake CLI (auth status + print mode)
# -------------------------------------------------------------------------- #
_SESSION_FAKE_BODY = (
    'if [ "$1" = "auth" ]; then\n'
    '  if [ -n "${FAKE_AUTH_FAIL:-}" ]; then\n'
    '    echo "not logged in" >&2\n'
    "    exit 1\n"
    "  fi\n"
    '  if [ -n "${FAKE_AUTH_BROKEN:-}" ]; then\n'
    '    echo "not json at all"\n'
    "    exit 0\n"
    "  fi\n"
    '  email="${FAKE_EMAIL:-fake@example.com}"\n'
    '  org="${FAKE_ORG:-org-1}"\n'
    '  if [ -n "${FAKE_DROP_ORG:-}" ]; then\n'
    '    printf \'{"authMethod":"claude.ai","email":"%s","configDirectory":"/home/fake/.claude"}\\n\' "$email"\n'
    "  else\n"
    '    printf \'{"authMethod":"claude.ai","email":"%s","orgId":"%s","configDirectory":"/home/fake/.claude"}\\n\' "$email" "$org"\n'
    "  fi\n"
    "  exit 0\n"
    "fi\n"
    'mkdir -p "$CLAUDE_CLI_FAKE_DIR/calls"\n'
    'n=$(ls -1 "$CLAUDE_CLI_FAKE_DIR/calls" 2>/dev/null | wc -l | tr -d " ")\n'
    'c="$CLAUDE_CLI_FAKE_DIR/calls/$n"\n'
    'mkdir "$c"\n'
    'printf "%s\\n" "$@" > "$c/args"\n'
    'cat > "$c/stdin"\n'
    'sid="native-123"\n'
    'resumed=$(awk \'/^--resume$/{getline; print; exit}\' "$c/args")\n'
    'if [ -n "$resumed" ]; then\n'
    '  sid="$resumed"\n'
    '  if [ -n "${FAKE_LOSS:-}" ]; then\n'
    '    echo "No conversation found with session ID: $resumed" >&2\n'
    "    exit 1\n"
    "  fi\n"
    '  if [ -n "${FAKE_REPORT_SID:-}" ]; then sid="$FAKE_REPORT_SID"; fi\n'
    "fi\n"
    'if [ -n "${FAKE_NO_SID:-}" ]; then\n'
    '  printf \'{"type":"system","subtype":"init"}\\n\'\n'
    '  printf \'{"type":"result","subtype":"success","is_error":false,"result":"done"}\\n\'\n'
    "  exit 0\n"
    "fi\n"
    'printf \'{"type":"system","subtype":"init","session_id":"%s"}\\n\' "$sid"\n'
    'printf \'{"type":"result","subtype":"success","is_error":false,"result":"done",'
    '"session_id":"%s","usage":{"input_tokens":2,"output_tokens":1}}\\n\' "$sid" "$sid"\n'
)


def _ctx(
    session_id: str | None = "webui:chat-1",
    workspace: str = "/tmp/ws",
    conversation_state: "ProviderConversationState | None" = None,
) -> "ProviderCallContext":
    from nanobot.providers.base import ProviderCallContext

    return ProviderCallContext(
        conversation_state=conversation_state,
        session_id=session_id,
        workspace=workspace,
    )


async def _fresh_session(provider, monkeypatch, out_dir) -> "LLMResponse":
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    return await provider.chat_stream(
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "turn one"},
        ],
        provider_context=_ctx(),
    )


async def test_fresh_chat_with_context_records_verified_session(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import SESSION_STATE_KIND

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    provider = _provider(str(fake))

    response = await _fresh_session(provider, monkeypatch, out_dir)
    assert response.finish_reason == "stop"

    state = response.provider_state
    assert state is not None
    assert state.kind == SESSION_STATE_KIND
    assert state.model == _MODEL
    assert state.payload["session_id"] == "native-123"
    assert state.payload["nanobot_session"] == "webui:chat-1"
    assert state.payload["workspace"] == "/tmp/ws"
    # A one-way fingerprint is recorded, never the raw account identifier.
    assert state.payload["identity"]
    assert "fake@example.com" not in state.payload["identity"]
    assert "fake@example.com" not in str(state.to_private_record())

    # The fresh session run must persist the native session (no opt-out flag).
    args = (out_dir / "calls" / "0" / "args").read_text(encoding="utf-8").splitlines()
    assert "--no-session-persistence" not in args
    assert "--resume" not in args
    assert (out_dir / "calls" / "0" / "stdin").read_text(encoding="utf-8").strip() == "turn one"


async def test_stateless_chat_records_no_session_and_skips_auth(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    body = _SESSION_FAKE_BODY.replace(
        'if [ "$1" = "auth" ]; then\n',
        'if [ "$1" = "auth" ]; then\n  echo auth >> "$CLAUDE_CLI_FAKE_DIR/auth_calls"\n',
    )
    _write_fake_cli(fake, body)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    # No provider context: the stateless one-shot transport is unchanged — no
    # login attestation, no session recorded, session persistence disabled.
    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])
    assert response.finish_reason == "stop"
    assert response.provider_state is None
    args = (out_dir / "calls" / "0" / "args").read_text(encoding="utf-8").splitlines()
    assert "--no-session-persistence" in args
    assert not (out_dir / "auth_calls").exists()


async def test_resume_sends_only_new_input_with_recorded_id(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state
    assert state is not None

    second = await provider.chat_stream(
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "turn two"},
        ],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert second.finish_reason == "stop"

    # Turn two resumed the exact recorded native session.
    args2 = (out_dir / "calls" / "1" / "args").read_text(encoding="utf-8").splitlines()
    assert args2[args2.index("--resume") + 1] == "native-123"
    assert "--system-prompt-file" not in args2
    # Only the new user input went on the wire — never the transcript.
    stdin2 = (out_dir / "calls" / "1" / "stdin").read_text(encoding="utf-8")
    assert stdin2 == "turn two"

    # The binding survives across turns.
    assert second.provider_state is not None
    assert second.provider_state.payload["session_id"] == "native-123"
    assert second.provider_state.payload["identity"] == state.payload["identity"]


@pytest.mark.parametrize("overrides", [
    {"session_id": "webui:chat-2"},
    {"workspace": "/tmp/other-ws"},
])
async def test_resume_denied_out_of_scope_bindings(
    tmp_path, monkeypatch, overrides: dict
) -> None:
    from nanobot.providers.base import ProviderCallContext
    from nanobot.providers.claude_cli_provider import SESSION_BINDING_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    base = {"session_id": "webui:chat-1", "workspace": "/tmp/ws"}
    base.update(overrides)
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=ProviderCallContext(
            conversation_state=state.with_pending_messages([{"role": "user", "content": "turn two"}]),
            session_id=base["session_id"],
            workspace=base["workspace"],
        ),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "session_binding"
    assert response.content == SESSION_BINDING_MESSAGE
    # The CLI was never invoked for the denied resume (no print call).
    assert not (out_dir / "calls" / "1").exists()


async def test_resume_denied_when_login_changed(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import IDENTITY_CHANGED_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    monkeypatch.setenv("FAKE_EMAIL", "someone-else@example.com")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "identity_changed"
    assert response.content == IDENTITY_CHANGED_MESSAGE
    assert not (out_dir / "calls" / "1").exists()


async def test_resume_denied_when_org_changed(tmp_path, monkeypatch) -> None:
    """Same account, a different organization now owns the login: refuse."""
    from nanobot.providers.claude_cli_provider import IDENTITY_CHANGED_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    monkeypatch.setenv("FAKE_ORG", "org-a")
    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state
    assert state is not None

    # Same email, but a different org now owns the login.
    monkeypatch.setenv("FAKE_ORG", "org-b")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "identity_changed"
    assert response.content == IDENTITY_CHANGED_MESSAGE
    # Refused before the prompt is sent: the CLI print mode never runs.
    assert not (out_dir / "calls" / "1").exists()


async def test_resume_denied_when_org_disappears(tmp_path, monkeypatch) -> None:
    """A recorded org is no longer attested: refuse fail-closed."""
    from nanobot.providers.claude_cli_provider import IDENTITY_CHANGED_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    # First turn records a session whose login attests org-1.
    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state
    assert state is not None

    # The login still names the same user but no longer reports any org.
    monkeypatch.setenv("FAKE_DROP_ORG", "1")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "identity_changed"
    assert response.content == IDENTITY_CHANGED_MESSAGE
    assert not (out_dir / "calls" / "1").exists()


async def test_resume_allowed_when_org_unchanged(tmp_path, monkeypatch) -> None:
    """Same account and same organization: the resume proceeds normally."""
    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    monkeypatch.setenv("FAKE_ORG", "org-a")
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state
    assert state is not None

    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "stop"
    assert response.provider_state is not None
    # The binding survives with an unchanged organization.
    assert response.provider_state.payload["identity"] == state.payload["identity"]
    args2 = (out_dir / "calls" / "1" / "args").read_text(encoding="utf-8").splitlines()
    assert args2[args2.index("--resume") + 1] == "native-123"


async def test_resume_denied_for_legacy_state_without_org(
    tmp_path, monkeypatch
) -> None:
    """A session recorded before org binding (a v1 fingerprint) is refused.

    The fingerprint is versioned, so an older, weaker state that bound only the
    account can never match a current attestation that also binds the org.
    """
    import hashlib

    from nanobot.providers.base import ProviderConversationState
    from nanobot.providers.claude_cli_provider import (
        IDENTITY_CHANGED_MESSAGE,
        SESSION_STATE_KIND,
        SESSION_STATE_VERSION,
    )

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    # A v1-style fingerprint bound the account only (no org, v1 version tag).
    legacy_identity = hashlib.sha256(
        b"claude-cli-v1|claude.ai|fake@example.com"
    ).hexdigest()
    legacy = ProviderConversationState(
        kind=SESSION_STATE_KIND,
        provider=f"claude_cli:{fake}",
        model=_MODEL,
        version=SESSION_STATE_VERSION,
        payload={
            "session_id": "native-123",
            "nanobot_session": "webui:chat-1",
            "workspace": "/tmp/ws",
            "cli": str(fake),
            "identity": legacy_identity,
        },
    ).with_pending_messages([{"role": "user", "content": "turn two"}])

    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=legacy),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "identity_changed"
    assert response.content == IDENTITY_CHANGED_MESSAGE
    # Refused before the prompt is sent: the CLI print mode never runs.
    assert not (out_dir / "calls").exists()


async def test_resume_denied_when_identity_unverifiable(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import IDENTITY_UNVERIFIED_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    monkeypatch.setenv("FAKE_AUTH_BROKEN", "1")
    provider = _provider(str(fake))

    # A session route fails closed without a verifiable login identity: no
    # session is opened, no state is recorded, and the CLI print mode never runs.
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn one"}],
        provider_context=_ctx(),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "identity_unverified"
    assert response.content == IDENTITY_UNVERIFIED_MESSAGE
    assert not (out_dir / "calls").exists()


async def test_resume_denied_when_not_logged_in(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import AUTH_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    monkeypatch.setenv("FAKE_AUTH_FAIL", "1")
    provider = _provider(str(fake))

    state = None
    # Establish a recorded session first, under a good login.
    monkeypatch.delenv("FAKE_AUTH_FAIL", raising=False)
    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state
    assert state is not None

    monkeypatch.setenv("FAKE_AUTH_FAIL", "1")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "auth"
    assert response.content == AUTH_MESSAGE
    assert not (out_dir / "calls" / "1").exists()


async def test_resume_reports_session_lost_when_cli_refuses(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import SESSION_LOST_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    monkeypatch.setenv("FAKE_LOSS", "1")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "session_lost"
    assert response.content == SESSION_LOST_MESSAGE


async def test_resume_rejects_forked_session_id(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import SESSION_MISMATCH_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    monkeypatch.setenv("FAKE_REPORT_SID", "forked-999")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "session_mismatch"
    assert response.content == SESSION_MISMATCH_MESSAGE
    assert response.provider_state is None


async def test_resume_denied_when_session_unconfirmed(tmp_path, monkeypatch) -> None:
    from nanobot.providers.claude_cli_provider import SESSION_MISMATCH_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    # The run "succeeded" but never confirmed which native session it was in:
    # an unconfirmed resume is denied fail-closed and never re-recorded.
    monkeypatch.setenv("FAKE_NO_SID", "1")
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "turn two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "session_mismatch"
    assert response.content == SESSION_MISMATCH_MESSAGE
    assert response.provider_state is None


async def test_foreign_state_shape_refused_before_cli(tmp_path, monkeypatch) -> None:
    from nanobot.providers.base import ProviderCallContext, ProviderConversationState
    from nanobot.providers.claude_cli_provider import SESSION_BINDING_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    # A foreign provider's state handed to this provider is refused with a
    # fixed diagnostic before the CLI is ever resolved or invoked.
    foreign = ProviderConversationState(
        kind="openai_responses",
        provider="openai_responses:example",
        model=_MODEL,
        version=1,
        payload={},
    ).with_pending_messages([{"role": "user", "content": "turn two"}])
    response = await provider.chat(
        messages=[{"role": "user", "content": "turn two"}],
        provider_context=ProviderCallContext(
            conversation_state=foreign,
            session_id="webui:chat-1",
            workspace="/tmp/ws",
        ),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "session_binding"
    assert response.content == SESSION_BINDING_MESSAGE
    assert not (out_dir / "calls").exists()


async def test_resume_leftover_from_interrupted_turn_gets_actionable_guidance(
    tmp_path, monkeypatch
) -> None:
    """A leftover staged message plus the new one (the cancel repro) is refused
    with actionable 'start a new chat' guidance, not a technical shape error, and
    without spawning another completion."""
    from nanobot.providers.claude_cli_provider import SESSION_INTERRUPTED_MESSAGE

    out_dir = tmp_path / "out"
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    provider = _provider(str(fake))

    first = await _fresh_session(provider, monkeypatch, out_dir)
    state = first.provider_state

    # The previous (cancelled) turn's message is still staged, and the new turn's
    # message was staged on top of it: two pending messages.
    response = await provider.chat(
        messages=[{"role": "user", "content": "x"}],
        provider_context=_ctx(conversation_state=state.with_pending_messages([
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ])),
    )
    assert response.finish_reason == "error"
    assert response.error_kind == "bad_request"
    assert response.content == SESSION_INTERRUPTED_MESSAGE
    # No replay and no automatic retry: no completion is spawned.
    assert not (out_dir / "calls" / "1").exists()


def test_can_resume_checks_kind_provider_and_model(tmp_path) -> None:
    from nanobot.providers.base import ProviderConversationState
    from nanobot.providers.claude_cli_provider import (
        _SESSION_STATE_KEYS,
        SESSION_STATE_KIND,
        SESSION_STATE_VERSION,
    )

    fake = tmp_path / "claude"
    _write_fake_cli(fake, _SESSION_FAKE_BODY)
    provider = _provider(str(fake))

    def _state(**overrides) -> ProviderConversationState:
        payload = {key: "v" for key in _SESSION_STATE_KEYS}
        data = dict(
            kind=SESSION_STATE_KIND,
            provider=f"claude_cli:{fake}",
            model=_MODEL,
            version=SESSION_STATE_VERSION,
            payload=payload,
        )
        data.update(overrides)
        return ProviderConversationState(**data)

    assert provider.can_resume_conversation_state(_state(), model=_MODEL) is True
    assert provider.can_resume_conversation_state(_state(kind="openai_responses"), model=_MODEL) is False
    assert provider.can_resume_conversation_state(_state(version=2), model=_MODEL) is False
    assert provider.can_resume_conversation_state(_state(provider="claude_cli:/other/cli"), model=_MODEL) is False
    assert provider.can_resume_conversation_state(_state(model="claude_cli/other-model"), model=_MODEL) is False
    assert provider.can_resume_conversation_state(
        _state(payload={"session_id": "x"}), model=_MODEL
    ) is False
    # The provider also never resumes a state bound to a CLI it cannot resolve.
    missing = _provider("/nonexistent/claude")
    assert missing.can_resume_conversation_state(
        _state(provider="claude_cli:/nonexistent/claude"), model=_MODEL
    ) is False


# -------------------------------------------------------------------------- #
# Opt-in tool bridge (fake CLI as MCP client + real bridge + real registry)
# -------------------------------------------------------------------------- #
def _write_fake_cli_py(path, body: str) -> None:
    """Write an executable fake ``claude`` that runs *body* (python3) on startup."""
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


_BRIDGE_FAKE_CLI = r'''
import json, os, sys

fake_dir = os.environ["CLAUDE_CLI_FAKE_DIR"]
os.makedirs(fake_dir, exist_ok=True)
argv = sys.argv[1:]
with open(os.path.join(fake_dir, "args.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(argv) + "\n")
with open(os.path.join(fake_dir, "stdin.txt"), "w", encoding="utf-8") as f:
    f.write(sys.stdin.read())

def find_flag(name):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None

mcp_config = find_flag("--mcp-config")
if mcp_config is None:
    sys.exit(2)
if os.path.exists(mcp_config):
    config_path = mcp_config
    with open(mcp_config, encoding="utf-8") as f:
        config = json.load(f)
else:
    config_path = None
    config = json.loads(mcp_config)
with open(os.path.join(fake_dir, "mcp_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f)
if config_path is not None:
    with open(os.path.join(fake_dir, "mcp_config_mode.txt"), "w", encoding="utf-8") as f:
        f.write(format(os.stat(config_path).st_mode & 0o777, "03o"))

result_text = "no-bridge"
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
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                with open(os.path.join(fake_dir, "tools.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(names) + "\n")
                if not names:
                    return "NO_TOOLS"
                # The fake CLI exercises the real workspace-scoped read tool
                # through the actual scoped open: one allowed workspace read and
                # one escaping path that must be denied by the same operation.
                outcomes = []
                allowed_text = None
                for label, args in (
                    ("allowed", {"path": "note.txt"}),
                    ("denied", {"path": "../outside/outside.txt"}),
                ):
                    result = await session.call_tool(names[0], args)
                    text = result.content[0].text
                    with open(os.path.join(fake_dir, f"tool_{label}.txt"), "w", encoding="utf-8") as f:
                        f.write(text)
                    with open(os.path.join(fake_dir, f"tool_{label}_iserror.txt"), "w") as f:
                        f.write(str(result.isError))
                    if label == "allowed":
                        allowed_text = text
                    outcomes.append(f"{label} isError={result.isError}")
                # The CLI folds the allowed tool result into its own answer.
                return " ; ".join(outcomes) + " || " + allowed_text
    result_text = "bridge-saw: " + asyncio.run(drive())

print(json.dumps({"type": "system", "subtype": "init"}))
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}}))
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": result_text,
    "session_id": "sess-1",
    "usage": {"input_tokens": 3, "output_tokens": 2},
}))
'''


async def test_bridge_turn_invokes_one_tool_and_surfaces_it_once(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    # The fake CLI (a python3 subprocess) must import the same ``mcp`` package the
    # test process uses; mirror the test's search path into its environment.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "note.txt").write_text("the fixture payload\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.txt").write_text("top secret outside\n", encoding="utf-8")
    fake = tmp_path / "claude"
    _write_fake_cli_py(fake, _BRIDGE_FAKE_CLI)

    # The production surface: the turn's registry carries the built-in,
    # workspace-scoped read tool for this turn's workspace.
    tool = ReadFileTool(workspace=ws)
    registry = ToolRegistry()
    registry.register(tool)
    bridge = ClaudeCodeToolBridge()
    provider = _provider(str(fake), tools=["read_file"], tool_bridge=bridge)

    hook = _RecordingHook()
    before = {
        p for p in os.listdir(tempfile.gettempdir()) if p.startswith("nanobot-claude-cli-mcp-")
    }
    # The active turn authorizes the tools: it supplies the shared tool
    # dispatch boundary (the same object its own model-initiated calls run
    # through) and the trusted turn context (workspace scope) that binds
    # read_file to the turn's workspace.
    dispatch = ToolDispatch(
        tools=registry,
        hook=hook,
        context=AgentHookContext(iteration=1, messages=[], session_key="test"),
    )
    with request_context(RequestContext(
        channel="test", chat_id="chat-1", workspace=ws,
        workspace_scope=default_workspace_scope(ws, restrict_to_workspace=True),
    )):
        response = await provider.chat_stream(
            messages=[{"role": "user", "content": "read the fixture"}],
            provider_context=ProviderCallContext(tool_dispatch=dispatch),
        )

    assert response.finish_reason == "stop"
    # The CLI saw the allowed tool result and folded it into its own answer.
    assert "the fixture payload" in response.content
    # The allowed read succeeded through the actual scoped open...
    allowed = (out_dir / "tool_allowed.txt").read_text(encoding="utf-8")
    assert (out_dir / "tool_allowed_iserror.txt").read_text().strip() == "False"
    assert "the fixture payload" in allowed
    # ...and the escaping path was denied by the same scoped open, with the
    # secret never crossing the bridge.
    denied = (out_dir / "tool_denied.txt").read_text(encoding="utf-8")
    assert (out_dir / "tool_denied_iserror.txt").read_text().strip() == "True"
    assert "outside allowed directory" in denied
    assert "top secret outside" not in denied
    # The real CLI activity surfaced through the turn's hook chain: exactly
    # one start and one terminal observation per call, in the call order.
    assert len(hook.events) == 4
    start1, end1, start2, end2 = hook.events
    assert start1.tool_calls[0].name == "read_file"
    assert start1.tool_calls[0].arguments == {"path": "note.txt"}
    assert start1.tool_results == [] and start1.tool_events == []
    assert end1.tool_events[0]["status"] == "ok"
    assert "the fixture payload" in str(end1.tool_results[0])
    assert start2.tool_calls[0].arguments == {"path": "../outside/outside.txt"}
    assert end2.tool_events[0]["status"] == "error"

    # The CLI received only the turn-local bridge: one SSE server named after the
    # bridge, carrying the turn token in a header (never on the command line).
    args = (out_dir / "args.txt").read_text(encoding="utf-8").splitlines()
    assert args[args.index("--tools") + 1] == ""  # built-in surface stays empty
    assert args[args.index("--allowedTools") + 1] == "mcp__nanobot__*"
    assert "--restricted" in args
    assert "--safe-mode" not in args
    config = json.loads((out_dir / "mcp_config.json").read_text(encoding="utf-8"))
    servers = config["mcpServers"]
    assert list(servers) == ["nanobot"]
    assert servers["nanobot"]["type"] == "sse"
    assert servers["nanobot"]["url"].startswith("http://127.0.0.1:")
    token = servers["nanobot"]["headers"]["Authorization"].removeprefix("Bearer ")
    assert token and token not in " ".join(args)
    # The config was the only form on the wire: the fake CLI saw the exposed tool.
    assert (out_dir / "tools.txt").read_text(encoding="utf-8").splitlines() == ["read_file"]
    # The turn secret was a 0600 file, and it is gone after the turn.
    assert (out_dir / "mcp_config_mode.txt").read_text(encoding="utf-8") == "600"
    leftovers = [
        p for p in os.listdir(tempfile.gettempdir())
        if p.startswith("nanobot-claude-cli-mcp-") and p not in before
    ]
    assert leftovers == []


async def test_tools_opted_in_without_bridge_refuses_and_runs_nothing(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _happy_fake_body())

    provider = _provider(str(fake), tools=["read_file"])  # no bridge
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "bridge_unavailable"
    # Fail closed before spawn: the CLI never ran and no endpoint was opened.
    assert not (out_dir / "args.txt").exists()


async def test_tools_opted_in_outside_an_active_turn_refuse_and_run_nothing(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli(fake, _happy_fake_body())

    bridge = ClaudeCodeToolBridge()
    provider = _provider(str(fake), tools=["read_file"], tool_bridge=bridge)
    # No provider_context: the call is not part of an active agent turn, so the
    # turn's registry — the only authorization source — is unavailable.
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "error"
    assert response.error_kind == "bridge_unavailable"
    assert "active Nanobot turn" in (response.content or "")
    # Fail closed before spawn: the CLI never ran and no endpoint was opened.
    assert not (out_dir / "args.txt").exists()


async def test_default_route_stays_tool_free_and_uses_no_bridge_file(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli_py(fake, _BRIDGE_FAKE_CLI)

    provider = _provider(str(fake))  # default: no tools
    response = await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert response.finish_reason == "stop"
    assert response.content == "no-bridge"
    config = json.loads((out_dir / "mcp_config.json").read_text(encoding="utf-8"))
    assert config == {"mcpServers": {}}
    assert not (out_dir / "mcp_config_mode.txt").exists()  # no file, inline JSON only


async def test_cancelled_bridge_turn_revokes_endpoint_and_cleans_up(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CLAUDE_CLI_FAKE_DIR", str(out_dir))
    fake = tmp_path / "claude"
    _write_fake_cli_py(
        fake,
        "import os, sys\n"
        "os.makedirs(os.environ['CLAUDE_CLI_FAKE_DIR'], exist_ok=True)\n"
        "open(os.path.join(os.environ['CLAUDE_CLI_FAKE_DIR'], 'args.txt'), 'w').close()\n"
        "sys.stdin.read()\n"
        "import time\n"
        "time.sleep(30)\n",
    )

    created: list[str] = []
    original = ClaudeCLIProvider._write_secret_file

    def _recording(prefix: str, content: str) -> str:
        path = original(prefix, content)
        created.append(path)
        return path

    monkeypatch.setattr(ClaudeCLIProvider, "_write_secret_file", staticmethod(_recording))

    class _RecordingBridge(ClaudeCodeToolBridge):
        def __init__(self) -> None:
            super().__init__()
            self.turns: list[object] = []

        async def start_turn(
            self,
            *,
            tool_names,
            tool_dispatch,
            request_context=None,
            model_messages=None,
        ):
            turn = await super().start_turn(
                tool_names=tool_names,
                tool_dispatch=tool_dispatch,
                request_context=request_context,
                model_messages=model_messages,
            )
            self.turns.append(turn)
            return turn

    ws = tmp_path / "ws"
    ws.mkdir()
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace=ws))
    bridge = _RecordingBridge()
    provider = _provider(str(fake), tools=["read_file"], tool_bridge=bridge, timeout_s=60.0)

    import asyncio

    # The task copies the current context, so the bound turn context (the
    # trusted workspace scope) authorizes the read_file surface for the turn.
    with request_context(RequestContext(
        channel="test", chat_id="chat-1", workspace=ws,
        workspace_scope=default_workspace_scope(ws, restrict_to_workspace=True),
    )):
        task = asyncio.create_task(
            provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                provider_context=ProviderCallContext(
                    tool_dispatch=ToolDispatch(
                        tools=registry,
                        hook=AgentHook(),
                        context=AgentHookContext(
                            iteration=1, messages=[], session_key="test"
                        ),
                    )
                ),
            )
        )
    await asyncio.sleep(0.5)  # let the bridge come up and the CLI start
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(bridge.turns) == 1
    assert bridge.turns[0].closed is True  # the endpoint was revoked and shut down
    assert created and all(not os.path.exists(path) for path in created)
    # The 0600 turn-secret file is gone; the CLI (still sleeping) was torn down by
    # the existing process-tree cleanup, so no live process holds the config.
