"""Claude Code CLI backend.

Runs the locally installed ``claude`` CLI as a bounded, argv-only subprocess. The
CLI owns its existing local login; nanobot never reads, copies, stores, or forwards
credentials and never falls back to the Anthropic HTTP API for this route.

Two request shapes are supported. Built-in tools are always disabled (no
permission bypass); the only tool surface is, when explicitly opted in in the
provider configuration, a per-turn, localhost-authenticated MCP bridge that
serves a fixed snapshot of the opted-in tools registered for the active
Nanobot turn and dispatches every call through the turn's own tool registry.
Tool opt-in is an exact production allowlist (zero tools or the
workspace-only ``read_file``); any other name is refused, as is a same-named
custom tool that is not the built-in workspace-scoped read tool:

- **Fresh completion** — one user prompt, optionally preceded by one trusted
  system prompt. With no provider context this is the stateless one-shot
  transport (no session is persisted or resumed). When the turn carries a
  conversation context, the fresh run opens a native CLI session and the
  returned response binds the recorded session state to the owning
  conversation, workspace, CLI, and a verified login identity.
- **Native session resume** — the exact recorded CLI session id is passed via
  ``--resume`` and only the single new user message is sent, never a transcript
  replay. The login's account *and* organization are re-verified and must match
  the recorded ones, and the run must confirm the exact recorded session id (a
  forked or unconfirmed id, a changed account or org, is refused fail-closed).

Any request outside those shapes is refused (fail closed). See
``docs/claude-cli.md`` for setup and capability limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.context import current_request_context
from nanobot.execution.processes import ProcessTree, spawn_process_tree
from nanobot.llm_usage.types import LLMUsage
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.claude_cli_bridge import (
    BRIDGE_SERVER_NAME,
    PRODUCTION_TOOL_ALLOWLIST,
    BridgeTurn,
    ToolBridge,
)
from nanobot.providers.claude_cli_tool_dispatch import ToolDispatch
from nanobot.runtime_context import RUNTIME_CONTEXT_MESSAGE_META, detach_runtime_context
from nanobot.utils.cancellation import shield_and_drain

PROVIDER_NAME = "claude_cli"
MODEL_PREFIX = "claude_cli"

SESSION_STATE_KIND = "claude_cli_session"
SESSION_STATE_VERSION = 1
_SESSION_STATE_KEYS = ("session_id", "nanobot_session", "workspace", "cli", "identity")

# User-identifying fields that ``claude auth status`` may report. A login is
# verified only when at least one of these carries a non-empty value that names a
# *user*: an auth method name, a config path, an opaque marker, an organization
# id, or an environment variable alone is not proof of which account owns the
# login. An org id in particular identifies the workspace, not the user.
_USER_IDENTITY_FIELDS = (
    "email",
    "accountId",
    "account_id",
    "userId",
    "user_id",
)

# Organization-identifying fields, bound separately from the user account so a
# same-user/different-org login (or a login that loses its org) is refused. An
# org id alone does not identify a user.
_ORG_IDENTITY_FIELDS = (
    "orgId",
    "organizationId",
    "organization_id",
    "orgUuid",
)

AUTH_MESSAGE = "Claude CLI is not authenticated on this host. Sign in with `claude auth login`."
IDENTITY_UNVERIFIED_MESSAGE = (
    "Claude CLI login could not be verified through the installed CLI; the completion was not started."
)
IDENTITY_CHANGED_MESSAGE = (
    "The Claude CLI login (account or organization) changed since this conversation started; "
    "the recorded session will not be resumed."
)
SESSION_INTERRUPTED_MESSAGE = (
    "The previous turn was interrupted, so this Claude CLI conversation can no longer be "
    "resumed. Start a new chat to continue."
)
SESSION_BINDING_MESSAGE = (
    "The recorded Claude CLI session does not match this conversation's scope; it will not be resumed."
)
SESSION_LOST_MESSAGE = (
    "The recorded Claude CLI session is no longer available; start a new conversation to continue."
)
SESSION_MISMATCH_MESSAGE = (
    "The CLI did not confirm the exact recorded session on resume; the result was not accepted."
)
BRIDGE_UNAVAILABLE_MESSAGE = (
    "Claude CLI tools are opted in, but no tool bridge is available in this runtime; "
    "the turn was not started. Tools on this route are disabled by default."
)
BRIDGE_TURN_UNAVAILABLE_MESSAGE = (
    "Claude CLI tools are opted in, but this call is not part of an active Nanobot turn, "
    "so no tools are authorized for it; the turn was not started. Tools on this route "
    "are only available inside agent turns and are disabled by default."
)

IDENTITY_TIMEOUT_S = 20.0
IDENTITY_MAX_BYTES = 64 * 1024

# Claude Code consults these environment variables before its local login state.
# Dropping them keeps a gateway process's Anthropic HTTP credentials and provider
# routing from steering the CLI toward a different provider, endpoint, or token
# source. The CLI's own local login (keychain / ``~/.claude``) is untouched.
CLAUDE_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_OLD",
    "ANTHROPIC_API_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
)

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 1 * 1024 * 1024
MAX_STREAM_LINES = 10_000
DIAGNOSTIC_MAX_LINES = 200

_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_DENY_MCP_TOOLS = "mcp__*"
# Standard system directories, used only as a PATH fallback when the gateway service
# environment provides no usable PATH at all.
_DEFAULT_CLI_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


class _StreamBoundError(Exception):
    """Raised when the CLI output exceeds a configured bound."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)


class _DeadlineExceededError(Exception):
    """Raised when the completion exceeds its wall-clock bound."""


def _strip_model_prefix(model: str) -> str:
    """Return the model id the CLI accepts, dropping the provider prefix."""
    candidate = (model or "").strip()
    if not candidate:
        return ""
    if "/" in candidate:
        prefix, _, rest = candidate.partition("/")
        if prefix.lower() == MODEL_PREFIX:
            candidate = rest
    return candidate.strip()


def build_claude_cli_argv(
    command: str,
    model: str,
    system_prompt_file: str | None = None,
    *,
    persist_session: bool = False,
    resume_session_id: str | None = None,
    mcp_config_file: str | None = None,
) -> list[str]:
    """Build the strict, isolated, print-mode argv for one completion.

    The surface is intentionally empty and isolated:

    - ``--tools ""``        — disable every built-in tool (strict empty tool surface).
    - ``--disallowedTools mcp__*`` — deny any MCP tool by name as a second guard.
    - ``--strict-mcp-config`` + ``--mcp-config '{"mcpServers":{}}'`` — only the
      (empty) supplied MCP config is honored; no inherited MCP servers.
    - ``--safe-mode``       — disable inherited customizations and all MCP in tool-free mode.
    - ``--restricted``      — ignore user/project/local settings but allow the explicit
      MCP configuration in bridge mode; managed policy still applies.
    - ``--setting-sources ""`` — load no user/project/local settings files.
    - ``--permission-mode default`` — never bypass permission checks.
    - ``--no-session-persistence`` — do not write a resumable session to disk
      (stateless one-shot transport only; a session run must persist).

    If *system_prompt_file* is given, the trusted system prompt is supplied from
    that file via ``--system-prompt-file`` (the prompt text is never placed on the
    command line); the prompt itself is delivered separately over stdin. Never
    combined with a resume: a resumed session keeps its own system prompt.

    When *resume_session_id* is given the run resumes that exact native session
    and *persist_session* must also be set; a fresh session run sets only
    *persist_session* so the CLI records the session for later resume.

    When *mcp_config_file* is given (the opt-in tool bridge), the strict empty
    MCP surface is replaced by the turn-local bridge config read from that file:
    the only MCP server is the per-turn loopback bridge, and the built-in tool
    surface stays empty (``--tools ""``). The ``mcp__*`` deny guard is dropped for
    the bridge (it would block the bridge's own tools) and replaced by an
    explicit ``--allowedTools`` entry scoped to the bridge server, so no other
    tool — built-in or foreign MCP — can be pre-authorized. The file path is the
    only form accepted: the config carries the turn's bearer token and must not
    appear on the command line.
    """
    cli_model = _strip_model_prefix(model)
    if not cli_model:
        raise ValueError(f"Claude CLI model id must be non-empty (got {model!r})")
    if resume_session_id is not None and not persist_session:
        raise ValueError("Resuming a session requires session persistence.")
    argv = [
        command,
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--model",
        cli_model,
        "--tools",
        "",
    ]
    if mcp_config_file is None:
        argv += [
            "--disallowedTools",
            _DENY_MCP_TOOLS,
            "--strict-mcp-config",
            "--mcp-config",
            _EMPTY_MCP_CONFIG,
        ]
    else:
        # Only the turn-local bridge is the MCP server, and only its tools are
        # pre-authorized; every built-in tool remains disabled above.
        argv += [
            "--strict-mcp-config",
            "--mcp-config",
            mcp_config_file,
            "--allowedTools",
            f"mcp__{BRIDGE_SERVER_NAME}__*",
        ]
    argv += [
        "--safe-mode" if mcp_config_file is None else "--restricted",
        "--setting-sources",
        "",
        "--permission-mode",
        "default",
    ]
    if resume_session_id is not None:
        argv += ["--resume", resume_session_id]
    elif system_prompt_file:
        argv += ["--system-prompt-file", system_prompt_file]
    if not persist_session:
        argv += ["--no-session-persistence"]
    return argv


def clean_claude_cli_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment for the CLI that preserves its local login.

    Provider-routing, credential, and telemetry overrides are removed so the CLI
    resolves its own existing local login. Disable CLAUDE.md discovery explicitly,
    including in bridge mode where safe mode cannot be used.
    """
    source = dict(base_env if base_env is not None else os.environ)
    for name in CLAUDE_CREDENTIAL_ENV_VARS:
        source.pop(name, None)
    source["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    return source


def resolve_claude_cli_command(command: str) -> str | None:
    """Resolve *command* to an absolute ``claude`` path, or None when not found.

    The gateway service PATH may omit the user's local install directory
    (e.g. ``~/.local/bin``), so a bare command is searched there too before
    falling back to the ambient PATH. A configured absolute path is used as-is.
    This only inspects the filesystem; it never spawns the CLI.
    """
    command = (command or "").strip()
    if not command:
        command = "claude"
    if os.path.isabs(command):
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return command
        return None
    home = Path.home()
    user_local_dirs = (
        str(home / ".local" / "bin"),
        str(home / ".claude" / "local" / "bin"),
    )
    ambient_path = os.environ.get("PATH", "")
    candidates = list(user_local_dirs)
    if ambient_path:
        candidates.append(ambient_path)
    return shutil.which(command, path=os.pathsep.join(candidates))


def _build_cli_path(ambient: str | None, executable_dir: str) -> str:
    """Build a usable PATH for the CLI process.

    The gateway service environment may omit a PATH or the user's local install
    directory. The CLI (and its interpreter) still need a usable PATH, so the
    resolved executable's directory comes first, then the ambient PATH, and finally
    the standard system directories when nothing usable is set.
    """
    parts: list[str] = []
    if executable_dir:
        parts.append(executable_dir)
    if ambient:
        parts.append(ambient)
    if not ambient:
        parts.append(_DEFAULT_CLI_PATH)
    return os.pathsep.join(dict.fromkeys(parts))


class _ParsedStream:
    """Terminal state accumulated from a Claude Code stream-json run.

    The terminal ``result`` record is the single source of truth for the completion
    text; the assistant-message text is accumulated only as a fallback so the two
    are never concatenated (which would duplicate the reply). ``session_id``
    captures the native session id (init record first, terminal result as the
    authoritative confirmation).
    """

    __slots__ = (
        "assistant_text",
        "result_record",
        "saw_result",
        "is_error",
        "result_text",
        "init_session_id",
        "result_session_id",
    )

    def __init__(self) -> None:
        self.assistant_text = ""
        self.result_record: dict[str, Any] | None = None
        self.saw_result = False
        self.is_error = False
        self.result_text: str | None = None
        self.init_session_id: str | None = None
        self.result_session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        return self.result_session_id or self.init_session_id

    def ingest(self, record: dict[str, Any]) -> None:
        record_type = record.get("type")
        if record_type == "assistant":
            message = record.get("message")
            if isinstance(message, dict):
                self._append_assistant_text(message.get("content"))
        elif record_type == "result":
            self.saw_result = True
            self.result_record = record
            is_error = record.get("is_error")
            self.is_error = is_error if isinstance(is_error, bool) else False
            result = record.get("result")
            self.result_text = result if isinstance(result, str) else None
            if isinstance(record.get("session_id"), str) and record["session_id"]:
                self.result_session_id = record["session_id"]
        elif record_type == "system" and record.get("subtype") == "init":
            if isinstance(record.get("session_id"), str) and record["session_id"]:
                self.init_session_id = record["session_id"]

    def _append_assistant_text(self, content: Any) -> None:
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    self.assistant_text += text


def _extract_usage(record: dict[str, Any]) -> LLMUsage | None:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return None

    def _int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int) and value >= 0:
            return value
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
        return parsed if parsed >= 0 else 0

    input_tokens = _int(usage.get("input_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    cache_read = _int(usage.get("cache_read_input_tokens"))
    cache_write = _int(usage.get("cache_creation_input_tokens"))
    # The logical input total includes cache reads and writes.
    logical_input = input_tokens + cache_read + cache_write
    if logical_input == 0 and output_tokens == 0:
        return None
    return LLMUsage.reported(
        input_tokens=logical_input,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read or None,
        cache_write_tokens=cache_write or None,
    )


def _looks_like_session_loss(text: str) -> bool:
    """Detect the CLI reporting that a resumed session no longer exists."""
    blob = (text or "").lower()
    return (
        "no conversation found" in blob
        or "no session found" in blob
        or "conversation not found" in blob
        or "session not found" in blob
    )


def _classify_cli_failure(diagnostics: list[str], returncode: int | None) -> tuple[str, str]:
    """Map a failed CLI run to a useful, bounded status without private payloads."""
    blob = " ".join(diagnostics).lower()
    if _looks_like_session_loss(blob):
        return "session_lost", SESSION_LOST_MESSAGE
    if "not logged in" in blob or "no auth" in blob or "unauthenticated" in blob:
        return "auth", "Claude CLI is not authenticated on this host. Sign in with `claude auth login`."
    if (
        "invalid api key" in blob
        or "invalid x-api-key" in blob
        or "401" in blob
        or "403" in blob
        or "authentication" in blob
        or "credential" in blob
    ):
        return "auth", "Claude CLI reported an authentication failure; check the local login."
    if "overloaded" in blob or "529" in blob or "503" in blob or "rate limit" in blob:
        return "server_error", "Claude CLI reported a transient upstream failure."
    if returncode in (137, -9):
        return "cli_error", "Claude CLI process was terminated (out of memory or signal 9)."
    if returncode == 0:
        return "cli_error", "Claude CLI exited successfully but returned no completion."
    return "cli_error", f"Claude CLI exited with code {returncode}."


def _looks_like_auth_failure(text: str) -> bool:
    """Detect an authentication failure from a bounded CLI status message."""
    blob = (text or "").lower()
    return (
        "not logged in" in blob
        or "please run /login" in blob
        or "unauthenticated" in blob
        or "authentication" in blob
        or "no auth" in blob
        or "invalid api key" in blob
    )


class ClaudeCLIProvider(LLMProvider):
    """Provider that performs fresh text completions via the Claude Code CLI.

    Completions are built-in-tool-free by default. When *tools* (explicit
    opt-in names, restricted to :data:`PRODUCTION_TOOL_ALLOWLIST` — zero tools
    or the workspace-only ``read_file``) and a *tool_bridge* are given, each
    turn is authorized to serve the opted-in tools through a per-turn loopback
    MCP endpoint, but only when the call belongs to an active Nanobot turn that
    supplies the shared tool dispatch boundary (the same execution/policy
    boundary every ordinary agent tool call runs through); every bridge call
    then dispatches through that boundary under the turn's real request
    context.
    """

    supports_images = False

    def __init__(
        self,
        *,
        default_model: str,
        command: str = "claude",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        cwd: str | None = None,
        tools: Sequence[str] = (),
        tool_bridge: ToolBridge | None = None,
        provider_name: str = PROVIDER_NAME,
    ) -> None:
        super().__init__(api_key=None, api_base=None, provider_name=provider_name)
        self.default_model = default_model
        self.supports_images = False
        self.supports_tools = False
        self.supports_tool_calls = False
        self.command = command or "claude"
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_output_bytes = max(1024, int(max_output_bytes))
        self.cwd = cwd or os.getcwd()
        # Explicit opt-in: the default (empty) stays strictly tool-free. The
        # production allowlist is exact (zero tools or the workspace-only
        # read_file): any other name is a wiring error, refused fail-closed
        # before a provider exists.
        for name in tools:
            if name not in PRODUCTION_TOOL_ALLOWLIST:
                raise ValueError(
                    f"claude_cli tools: '{name}' is not allowed; the only opt-in "
                    "tool is 'read_file' (workspace-only)"
                )
        self.tools: tuple[str, ...] = tuple(tools)
        self.tool_bridge = tool_bridge

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------ #
    # Request validation (fail closed)
    # ------------------------------------------------------------------ #
    def _fresh_request(self, messages: list[dict[str, Any]]) -> tuple[str | None, str]:
        """Return ``(system_prompt, user_prompt)`` for a fresh completion, or refuse.

        A fresh tool-free completion is exactly one user message with plain text,
        optionally preceded by a single trusted system message (the shape the
        main agent turn sends when the explicit Claude CLI preset is selected).
        Anything else — prior turns without a resumable native session,
        assistant or tool roles, images, or structured content — is refused
        rather than silently dropped or replayed across the CLI's shared login.
        """
        if not isinstance(messages, list):
            raise _RequestRefusedError(
                "The Claude CLI route only performs fresh, single-prompt completions; "
                "requests carrying prior conversation are refused."
            )
        if len(messages) == 2:
            system_msg, user_msg = messages[0], messages[1]
        elif len(messages) == 1:
            system_msg, user_msg = None, messages[0]
        else:
            raise _RequestRefusedError(
                "The Claude CLI route only performs fresh, single-prompt completions; "
                "requests carrying prior conversation are refused."
            )
        if not isinstance(user_msg, dict) or user_msg.get("role") != "user":
            raise _RequestRefusedError(
                "The Claude CLI route expects exactly one user message; other roles are refused."
            )
        prompt = _prompt_text(user_msg)
        if prompt is None or not prompt.strip():
            raise _RequestRefusedError(
                "The Claude CLI route needs a non-empty text prompt; images and structured "
                "content are not supported."
            )
        if system_msg is None:
            return None, prompt
        if not isinstance(system_msg, dict) or system_msg.get("role") not in {"system", "developer"}:
            raise _RequestRefusedError(
                "The Claude CLI route accepts at most one trusted system message followed by "
                "one user message; other shapes are refused."
            )
        system_text = system_msg.get("content")
        if not isinstance(system_text, str) or not system_text.strip():
            raise _RequestRefusedError(
                "The Claude CLI route needs a non-empty plain-text system message; structured "
                "system content is not supported."
            )
        return system_text, prompt

    @staticmethod
    def _refuse_tools(tools: list[dict[str, Any]] | None, tool_choice: Any) -> None:
        if tools:
            raise _RequestRefusedError(
                "The Claude CLI route does not accept tools; the completion surface is "
                "strictly empty."
            )
        if tool_choice is not None:
            raise _RequestRefusedError(
                "The Claude CLI route does not accept a tool_choice; the completion surface "
                "is strictly empty."
            )

    # ------------------------------------------------------------------ #
    # Conversation-state resume (opt-in, fail closed)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _state_provider_name(cli: str) -> str:
        return f"claude_cli:{cli}"

    def can_resume_conversation_state(
        self,
        state: ProviderConversationState,
        model: str | None = None,
    ) -> bool:
        """Whether *state* is a Claude CLI session this exact provider/model can own.

        The runtime binding (resolved CLI executable) and model are checked here;
        the conversation, workspace, and login-identity bindings are re-verified
        against the live call context in :meth:`_run_turn`, and fail closed on any
        mismatch. A saved session therefore can never be resumed by another
        conversation, another workspace, a moved CLI, or a different login.
        """
        if state.kind != SESSION_STATE_KIND or state.version != SESSION_STATE_VERSION:
            return False
        try:
            cli = self._resolve_cli_executable()
        except _SpawnError:
            return False
        if state.provider != self._state_provider_name(cli):
            return False
        if _strip_model_prefix(state.model) != _strip_model_prefix(model or self.default_model):
            return False
        return all(
            isinstance(state.payload.get(key), str) and state.payload[key]
            for key in _SESSION_STATE_KEYS
        )

    def _build_session_state(
        self,
        *,
        cli: str,
        model: str,
        session_id: str,
        context: ProviderCallContext,
        identity: str,
    ) -> ProviderConversationState:
        """Record the verified bindings for one native session.

        Only a one-way identity fingerprint is stored — never the account
        identifier, any credential, or any CLI output.
        """
        return ProviderConversationState(
            kind=SESSION_STATE_KIND,
            provider=self._state_provider_name(cli),
            model=model,
            version=SESSION_STATE_VERSION,
            payload={
                "session_id": session_id,
                "nanobot_session": context.session_id or "",
                "workspace": context.workspace or "",
                "cli": cli,
                "identity": identity,
            },
        )

    def _resume_prompt(self, pending_messages: list[dict[str, Any]]) -> str:
        """Return the exact new user input for a resume, or refuse.

        A resume sends only the single new user message over stdin. The prior
        conversation lives in the CLI's own session; a transcript replay — or a
        merged/queued batch of messages — is refused rather than folded in.
        """
        if not isinstance(pending_messages, list) or not pending_messages:
            raise _RequestRefusedError(
                "The Claude CLI resume route needs exactly one new user message; "
                "other shapes are refused."
            )
        if len(pending_messages) > 1:
            # Several pending messages can only come from a leftover of an
            # interrupted (e.g. cancelled) turn: the current message was staged
            # on top of the previous turn's. The native session can't be resumed
            # safely (that would replay or drop the leftover), so give the user an
            # actionable next step instead of a technical shape error.
            raise _RequestRefusedError(SESSION_INTERRUPTED_MESSAGE)
        message = pending_messages[0]
        if not isinstance(message, dict) or message.get("role") != "user":
            raise _RequestRefusedError(
                "The Claude CLI resume route expects one new user message; other roles are refused."
            )
        prompt = _prompt_text(message)
        if prompt is None or not prompt.strip():
            raise _RequestRefusedError(
                "The Claude CLI resume route needs a non-empty text message; images and "
                "structured content are not supported."
            )
        return prompt

    async def _verify_identity(self, executable: str, env: dict[str, str]) -> str:
        """Verify the CLI's login via its own ``auth status`` attestation.

        Returns a one-way fingerprint of the account *and* organization identity.
        Fails closed: an unusable command, non-zero exit, unparseable output, or
        output naming no user (an org id alone does not identify a user) is
        treated as *unverified*, never as "some account". The raw attestation
        (and the identifier and org inside it) is never logged or stored.
        """
        deadline = time.monotonic() + min(self.timeout_s, IDENTITY_TIMEOUT_S)
        try:
            tree = await spawn_process_tree(
                [executable, "auth", "status"],
                cwd=self.cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                capture_output=True,
            )
        except (OSError, _SpawnError):
            raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE)
        proc = tree.process
        output = b""
        returncode: int | None = None
        try:
            stdout = proc.stdout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or stdout is None:
                    raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE)
                try:
                    chunk = await asyncio.wait_for(stdout.read(4096), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE) from exc
                if not chunk:
                    break
                output += chunk
                if len(output) > IDENTITY_MAX_BYTES:
                    raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE)
            try:
                returncode = await self._wait_for_exit(proc, deadline)
            except _DeadlineExceededError as exc:
                raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE) from exc
        finally:
            await self._finish_tree(tree)
        if returncode != 0:
            raise _TurnError("auth", AUTH_MESSAGE)
        record = self._parse_identity_record(output.decode("utf-8", errors="replace"))
        if record is None:
            raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE)
        identifier = None
        for key in _USER_IDENTITY_FIELDS:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                identifier = value.strip()
                break
        if identifier is None:
            # An org id (or an auth method, path, or opaque marker) alone does not
            # identify a user: fail closed.
            raise _TurnError("identity_unverified", IDENTITY_UNVERIFIED_MESSAGE)
        org_id = ""
        for key in _ORG_IDENTITY_FIELDS:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                org_id = value.strip()
                break
        auth_method = record.get("authMethod")
        # The fingerprint binds the user account AND the organization. It is
        # versioned (v2): a recorded v1 fingerprint (account only) can never match,
        # so older, weaker state is refused fail-closed rather than downgraded.
        material = (
            "claude-cli-v2"
            f"|{auth_method if isinstance(auth_method, str) else ''}"
            f"|{identifier}"
            f"|{org_id}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_identity_record(text: str) -> dict[str, Any] | None:
        """Parse the ``auth status`` JSON object; None when unparseable."""
        stripped = text.strip()
        if not stripped:
            return None
        candidate = stripped
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                candidate = line
                break
        try:
            record = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        return await self._run_turn(
            messages=messages,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )

    async def chat_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.chat(provider_context=provider_context, **kwargs)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Any = None,
        on_thinking_delta: Any = None,
        on_tool_call_delta: Any = None,
        provider_context: ProviderCallContext | None = None,
    ) -> LLMResponse:
        # No incremental native deltas are exposed in this tranche; the terminal
        # text is delivered once as a single delta, so it is never duplicated.
        # *on_tool_call_delta* is accepted for runner compatibility but
        # intentionally unused: tool calls the CLI makes through the bridge are
        # real activity of this turn's shared dispatch boundary and are
        # surfaced exactly once through the turn's hook chain — the same path
        # as ordinary model-initiated tool calls — so forwarding them here as
        # hosted provider events would surface them twice.
        response = await self._run_turn(
            messages=messages,
            tools=tools,
            model=model,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if on_content_delta and response.content and response.finish_reason != "error":
            await on_content_delta(response.content)
        return response

    async def chat_stream_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.chat_stream(provider_context=provider_context, **kwargs)

    # ------------------------------------------------------------------ #
    # Turn execution
    # ------------------------------------------------------------------ #
    async def _run_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        tool_choice: str | dict[str, Any] | None,
        provider_context: ProviderCallContext | None,
    ) -> LLMResponse:
        try:
            self._refuse_tools(tools, tool_choice)
        except _RequestRefusedError as exc:
            return self._error("bad_request", str(exc))
        effective_model = model or self.default_model

        # Opt-in tool bridge: started only after every fail-closed validation
        # above has passed (a refused turn never opens the port) and always torn
        # down in the finally below — including on cancellation, which revokes
        # the turn token and shuts the server down.
        bridge_turn: BridgeTurn | None = None
        mcp_config_file: str | None = None
        if self.tools:
            if self.tool_bridge is None:
                return self._error("bridge_unavailable", BRIDGE_UNAVAILABLE_MESSAGE)
            # Tools are authorized by the active turn itself: its shared tool
            # dispatch boundary (supplied by the agent runner) is both the
            # source of the tool snapshot and the execution/policy boundary for
            # every call, and the turn's request context is bound for every
            # call. A call made outside an agent turn (stateless one-shot,
            # auxiliary call) carries no turn dispatch and is refused before
            # any endpoint is opened.
            turn_context = (
                provider_context if isinstance(provider_context, ProviderCallContext) else None
            )
            turn_dispatch = turn_context.tool_dispatch if turn_context is not None else None
            if not isinstance(turn_dispatch, ToolDispatch):
                return self._error("bridge_unavailable", BRIDGE_TURN_UNAVAILABLE_MESSAGE)

            try:
                bridge_turn = await self.tool_bridge.start_turn(
                    tool_names=self.tools,
                    tool_dispatch=turn_dispatch,
                    request_context=current_request_context(),
                    model_messages=messages,
                )
            except Exception as exc:
                logger.warning("Claude CLI bridge failed to start: {}", type(exc).__name__)
                return self._error("bridge_unavailable", BRIDGE_UNAVAILABLE_MESSAGE)
            # The config carries the turn's bearer token: it reaches the CLI only
            # through this 0600 file, never the command line.
            mcp_config_file = self._write_secret_file(
                "nanobot-claude-cli-mcp-", bridge_turn.mcp_config_json
            )
        try:
            return await self._run_turn_body(
                messages=messages,
                effective_model=effective_model,
                provider_context=provider_context,
                mcp_config_file=mcp_config_file,
            )
        finally:
            if mcp_config_file is not None:
                with suppress(OSError):
                    os.unlink(mcp_config_file)
            if bridge_turn is not None:
                # The turn may be cancelled while the bridge is being closed; the
                # endpoint must still be fully revoked and shut down before the
                # cancellation propagates.
                await shield_and_drain(bridge_turn.aclose())

    async def _run_turn_body(
        self,
        *,
        messages: list[dict[str, Any]],
        effective_model: str,
        provider_context: ProviderCallContext | None,
        mcp_config_file: str | None,
    ) -> LLMResponse:
        context = provider_context if isinstance(provider_context, ProviderCallContext) else None
        resume_requested = context is not None and context.conversation_state is not None
        resume_id: str | None = None
        system_prompt: str | None = None
        identity: str | None = None
        persist = context is not None

        # Validate the input shape before any subprocess is involved: a malformed
        # resume input is refused without touching the CLI at all.
        if resume_requested:
            state = context.conversation_state
            # Unsupported state shapes fail closed before any field access: a
            # foreign or malformed state is denied, never dereferenced.
            if (
                state.kind != SESSION_STATE_KIND
                or state.version != SESSION_STATE_VERSION
                or not all(
                    isinstance(state.payload.get(key), str) and state.payload[key]
                    for key in _SESSION_STATE_KEYS
                )
            ):
                return self._error("session_binding", SESSION_BINDING_MESSAGE)
            # Fail-closed scope binding: the recorded native session belongs to
            # exactly one conversation and workspace. A context missing either
            # cannot prove ownership and is denied.
            if (
                not context.session_id
                or context.session_id != state.payload["nanobot_session"]
                or not context.workspace
                or context.workspace != state.payload["workspace"]
            ):
                return self._error("session_binding", SESSION_BINDING_MESSAGE)
            try:
                prompt = self._resume_prompt(state.pending_messages)
            except _RequestRefusedError as exc:
                return self._error("bad_request", str(exc))
            resume_id = state.payload["session_id"]
            persist = True
        else:
            try:
                system_prompt, prompt = self._fresh_request(messages)
            except _RequestRefusedError as exc:
                return self._error("bad_request", str(exc))

        try:
            cli = self._resolve_cli_executable()
        except _SpawnError as exc:
            return self._error("missing_cli", exc.user_message)
        env = clean_claude_cli_env(os.environ)
        # Keep the resolved CLI (and any helpers it needs) findable even when the
        # gateway service PATH omits the user's local install directory.
        env["PATH"] = _build_cli_path(env.get("PATH"), os.path.dirname(cli))

        if resume_requested:
            # The runtime (exact CLI executable + model) must also match, and the
            # login must be re-verified live and still be the recorded one.
            if not self.can_resume_conversation_state(state, effective_model):
                return self._error("session_binding", SESSION_BINDING_MESSAGE)
            try:
                identity = await self._verify_identity(cli, env)
            except _TurnError as exc:
                return self._error(exc.kind, exc.message)
            if identity != state.payload["identity"]:
                return self._error("identity_changed", IDENTITY_CHANGED_MESSAGE)
        elif persist and context.session_id and context.workspace:
            try:
                identity = await self._verify_identity(cli, env)
            except _TurnError as exc:
                return self._error(exc.kind, exc.message)
        else:
            # Stateless one-shot transport: no session is recorded, so no
            # login attestation is required and no state is ever returned.
            persist = False
            identity = None

        try:
            argv = build_claude_cli_argv(
                self.command,
                effective_model,
                persist_session=persist,
                resume_session_id=resume_id,
                mcp_config_file=mcp_config_file,
            )
        except ValueError as exc:
            return self._bad_request(str(exc))

        started = time.monotonic()
        deadline = started + self.timeout_s
        try:
            parsed, returncode, diagnostics = await self._run_completion(argv, prompt, system_prompt, deadline, executable=cli)
        except _DeadlineExceededError:
            logger.warning("Claude CLI completion exceeded {}s for {}", self.timeout_s, self.command)
            return self._error("timeout", "Claude CLI completion timed out before returning a result.")
        except _StreamBoundError as exc:
            logger.warning("Claude CLI output bound exceeded ({}); stopping the process", exc.kind)
            return self._error(
                "output_limit",
                "Claude CLI output exceeded the configured bound and the run was stopped.",
            )
        except _SpawnError as exc:
            return self._error("missing_cli", exc.user_message)
        except asyncio.CancelledError:
            raise

        response = self._build_response(parsed, returncode, diagnostics, started)
        if (
            response.finish_reason == "stop"
            and resume_id is not None
            and parsed.session_id != resume_id
        ):
            # A resume must confirm the exact recorded session. A forked id — or
            # no confirmation at all — is not this conversation's session, so an
            # unconfirmed resume is denied instead of being re-recorded.
            return self._error("session_mismatch", SESSION_MISMATCH_MESSAGE)
        if response.finish_reason == "stop" and persist and identity is not None and context is not None:
            native_id = parsed.session_id or resume_id
            if native_id:
                response.provider_state = self._build_session_state(
                    cli=cli,
                    model=effective_model,
                    session_id=native_id,
                    context=context,
                    identity=identity,
                )
            else:
                logger.warning("Claude CLI returned no session id for {}", self.command)
        return response

    # ------------------------------------------------------------------ #
    # Subprocess execution
    # ------------------------------------------------------------------ #
    def _resolve_cli_executable(self) -> str:
        """Resolve the CLI to an absolute path, searching user-local install dirs.

        The gateway service PATH may omit the user's local install directory
        (e.g. ``~/.local/bin``), so a bare ``claude`` is searched there too before
        falling back to the ambient PATH. A configured absolute path is used as-is.
        """
        resolved = resolve_claude_cli_command(self.command)
        if resolved is not None:
            return resolved
        command = self.command or "claude"
        if os.path.isabs(command):
            raise _SpawnError(
                f"Claude CLI executable not found at the configured path: {command}. "
                "Set providers.claude_cli.command to the installed `claude`."
            )
        raise _SpawnError(
            f"Claude CLI executable not found: {command}. Install Claude Code or set "
            "providers.claude_cli.command to its full path."
        )

    async def _run_completion(
        self,
        argv: list[str],
        prompt: str,
        system_prompt: str | None,
        deadline: float,
        executable: str | None = None,
    ) -> tuple[_ParsedStream, int | None, list[str]]:
        executable = executable or self._resolve_cli_executable()
        env = clean_claude_cli_env(os.environ)
        # Ensure the resolved CLI (and any helpers it needs) stay findable even when the
        # gateway service PATH omits the user's local install directory.
        executable_dir = os.path.dirname(executable)
        env["PATH"] = _build_cli_path(env.get("PATH"), executable_dir)
        argv = [executable, *argv[1:]]
        # The trusted system prompt is passed via a 0600 temp file so its text never
        # appears on the command line; it is removed after the run regardless of outcome.
        system_file = self._write_system_prompt_file(system_prompt) if system_prompt is not None else None
        if system_file is not None:
            argv = [*argv, "--system-prompt-file", str(system_file)]
        try:
            tree = await spawn_process_tree(
                argv,
                cwd=self.cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                capture_output=True,
            )
            proc = tree.process
            diagnostics: list[str] = []
            try:
                await self._write_prompt(proc, prompt, deadline)
                parsed = await self._read_stream(proc, diagnostics, deadline)
                returncode = await self._wait_for_exit(proc, deadline)
                return parsed, returncode, diagnostics
            finally:
                await self._finish_tree(tree)
        finally:
            if system_file is not None:
                with suppress(OSError):
                    os.unlink(system_file)

    def _write_system_prompt_file(self, text: str) -> str:
        """Write the trusted system prompt to a 0600 temp file; return its path."""
        fd, path = tempfile.mkstemp(prefix="nanobot-claude-cli-sysprompt-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(path, 0o600)
        return path

    @staticmethod
    def _write_secret_file(prefix: str, content: str) -> str:
        """Write turn-secret content (the bridge MCP config with its token) to a
        0600 temp file; return its path."""
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(path, 0o600)
        return path

    async def _write_prompt(self, proc: asyncio.subprocess.Process, prompt: str, deadline: float) -> None:
        stdin = proc.stdin
        if stdin is None:
            raise _SpawnError("Claude CLI stdin was not available for prompt delivery.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceededError()
        try:
            stdin.write(prompt.encode("utf-8"))
            await asyncio.wait_for(stdin.drain(), timeout=remaining)
            # ``close()`` is synchronous (returns None) in some asyncio versions and
            # returns a future in others; handle both so the CLI reliably sees EOF.
            close_result = stdin.close()
            if close_result is not None:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(close_result, timeout=5.0)
        except asyncio.TimeoutError as exc:
            raise _DeadlineExceededError() from exc

    async def _read_stream(
        self,
        proc: asyncio.subprocess.Process,
        diagnostics: list[str],
        deadline: float,
    ) -> _ParsedStream:
        stdout = proc.stdout
        if stdout is None:
            raise _SpawnError("Claude CLI stdout was not available for result parsing.")
        stderr_task = asyncio.ensure_future(self._read_stderr(proc, diagnostics, deadline))
        parsed = _ParsedStream()
        total_bytes = 0
        line_count = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _DeadlineExceededError()
                try:
                    line_bytes = await asyncio.wait_for(stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise _DeadlineExceededError() from exc
                if not line_bytes:
                    break
                line_count += 1
                if line_count > MAX_STREAM_LINES:
                    raise _StreamBoundError("line_count")
                total_bytes += len(line_bytes)
                if total_bytes > self.max_output_bytes:
                    raise _StreamBoundError("bytes")
                if len(line_bytes) > MAX_LINE_BYTES:
                    raise _StreamBoundError("line_size")
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                record = _parse_record(line)
                if record is None:
                    # Non-JSON output on the result stream is diagnostics only.
                    if len(diagnostics) < DIAGNOSTIC_MAX_LINES:
                        diagnostics.append(line)
                    continue
                parsed.ingest(record)
        finally:
            await self._drain_stderr(stderr_task, proc, deadline)
        return parsed

    async def _read_stderr(
        self,
        proc: asyncio.subprocess.Process,
        diagnostics: list[str],
        deadline: float,
    ) -> None:
        stderr = proc.stderr
        if stderr is None:
            return
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line_bytes = await asyncio.wait_for(stderr.readline(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                if proc.returncode is not None:
                    continue
                break
            if not line_bytes:
                break
            if len(diagnostics) < DIAGNOSTIC_MAX_LINES:
                diagnostics.append(line_bytes.decode("utf-8", errors="replace").rstrip("\n"))

    async def _drain_stderr(self, stderr_task: asyncio.Task, proc: asyncio.subprocess.Process, deadline: float) -> None:
        if stderr_task.done():
            return
        remaining = deadline - time.monotonic()
        with suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(stderr_task, timeout=max(0.5, remaining))
        if not stderr_task.done():
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task

    async def _wait_for_exit(self, proc: asyncio.subprocess.Process, deadline: float) -> int | None:
        if proc.returncode is not None:
            return proc.returncode
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceededError()
        try:
            return await asyncio.wait_for(proc.wait(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise _DeadlineExceededError() from exc

    async def _finish_tree(self, tree: ProcessTree) -> None:
        """Ensure the process and its descendants are cleaned up on every path."""
        proc = tree.process
        try:
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            with suppress(Exception):
                await tree.terminate()
        finally:
            tree.finish()

    # ------------------------------------------------------------------ #
    # Response construction
    # ------------------------------------------------------------------ #
    def _build_response(
        self,
        parsed: _ParsedStream,
        returncode: int | None,
        diagnostics: list[str],
        started: float,
    ) -> LLMResponse:
        generation_ms = max(1, int((time.monotonic() - started) * 1000))
        if parsed.saw_result and not parsed.is_error:
            # Prefer the terminal result text; fall back to assistant text only when
            # the result carried none. Never concatenate both (that duplicates text).
            content = parsed.result_text
            if content is None or not content.strip():
                content = parsed.assistant_text.strip()
            usage = _extract_usage(parsed.result_record or {})
            if content.strip():
                return LLMResponse(
                    content=content,
                    finish_reason="stop",
                    usage=usage,
                    generation_ms=generation_ms,
                )
            # A reply-less successful result means the CLI ended the turn on purpose
            # with no text; report it as a completed empty turn rather than an error.
            return LLMResponse(
                content="",
                finish_reason="stop",
                usage=usage,
                generation_ms=generation_ms,
            )

        # Failure: classify internally and return fixed diagnostic text only. The CLI's
        # own error messages and stderr are never forwarded verbatim — they can carry
        # prompts, paths, or other private output — so only a bounded, fixed status is
        # surfaced to the caller.
        if parsed.saw_result and parsed.is_error and parsed.result_record is not None:
            status = _result_status_text(parsed.result_record)
            if _looks_like_auth_failure(status):
                return self._error(
                    "auth",
                    "Claude CLI is not authenticated on this host. Sign in with `claude auth login`.",
                )
            if _looks_like_session_loss(status):
                return self._error("session_lost", SESSION_LOST_MESSAGE)
            return self._error(
                "cli_error",
                "Claude CLI finished with an error before returning a completion.",
            )

        kind, message = _classify_cli_failure(diagnostics, returncode)
        return self._error(kind, message)

    def _error(self, kind: str, message: str) -> LLMResponse:
        return LLMResponse(
            content=message,
            finish_reason="error",
            error_kind=kind,
            error_should_retry=False,
        )

    def _bad_request(self, message: str) -> LLMResponse:
        return LLMResponse(
            content=message,
            finish_reason="error",
            error_kind="bad_request",
            error_should_retry=False,
        )


class _RequestRefusedError(Exception):
    """A request shape the route refuses to run (fail closed)."""


class _TurnError(Exception):
    """A refused turn with a fixed user-facing kind and message."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(message)


class _SpawnError(Exception):
    """The CLI could not be started."""

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        super().__init__(user_message)


def _prompt_text(message: dict[str, Any]) -> str | None:
    """Return the user-visible text of a message.

    The gateway decorates the current user message with a runtime-context
    suffix (workspace paths, tool policy). That scaffolding is meaningless to a
    tool-free external CLI and must not be persisted into its native session,
    so the validated suffix is detached before the text is accepted.
    """
    content = message.get("content")
    meta = message.get("_meta")
    if isinstance(meta, dict):
        marker = meta.get(RUNTIME_CONTEXT_MESSAGE_META)
        if isinstance(marker, dict):
            detached = detach_runtime_context(content, marker)
            if detached is not None:
                content = detached[0]
    return _content_as_prompt(content)


def _content_as_prompt(content: Any) -> str | None:
    """Accept a plain string or a single text block; reject images/structured content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = [block for block in content if isinstance(block, dict)]
        if len(content) != 1 or len(blocks) != 1:
            return None
        block = blocks[0]
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            return None
        return block["text"]
    return None


def _parse_record(line: str) -> dict[str, Any] | None:
    """Parse one stream-json line; return ``None`` for non-JSON or non-object lines."""
    stripped = line.strip()
    if not stripped or not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        record = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _result_status_text(record: dict[str, Any]) -> str:
    """Return the CLI's error status text for internal classification only.

    This value is never returned to the caller; it is used solely to detect an
    authentication failure from the terminal result record.
    """
    for key in ("error", "message", "result"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    subtype = record.get("subtype")
    return subtype if isinstance(subtype, str) else ""
