# Claude Code CLI provider

The **Claude CLI** provider runs the locally installed `claude` CLI as a bounded subprocess
for text completions. It is an **explicit, opt-in route**: it is selected only when a preset
names `provider: "claude_cli"` (or a `claude_cli/<model>` model reference). It never replaces the
normal Anthropic HTTP route, and it never falls back to the Anthropic HTTP API.

The CLI owns its **existing local login** (keychain or `~/.claude`). nanobot never reads, copies,
stores, or forwards credentials for this route, and no API key is used or required.

## What it supports today

Each call performs one **fresh, tool-free text completion** or one **native session resume**:

- One user message with plain text, sent to the CLI over **stdin**, optionally preceded by a
  single trusted system message (fresh runs only).
- The system prompt is supplied from a `0600` temp file via `--system-prompt-file` (its text
  never appears on the command line); the file is removed after the run regardless of outcome.
- No built-in CLI tools, no inherited MCP servers, no images. The only tool surface
  is the explicit opt-in [tool bridge](#tools-opt-in), off by default and limited to an
  exact allowlist (zero tools or the workspace-only `read_file`).
- The CLI is launched with a strict, isolated surface so it cannot inherit or act on the host's
  workspace, settings, hooks, skills, CLAUDE.md files, or custom agents (see [Isolation](#isolation)).
- The reply and token usage are read from the CLI's structured `stream-json` output.

**Multi-turn chat resumes the CLI's own session.** When the turn belongs to a conversation
(the explicit preset is selected on a normal chat), the first completion opens a native CLI
session (the CLI is allowed to persist it) and nanobot records that exact session id bound to
the owning conversation, workspace, CLI executable, model, and a one-way fingerprint of the
CLI's verified login. Every later turn resumes that exact session with `--resume` and sends
only the single new user message over stdin — never a transcript replay. Before each run the
CLI's login is re-attested through its own `auth status` output; if the attestation is
unavailable, unparseable, names no user (an org id alone does not identify a user), or
identifies a different account **or** organization than the recorded ones, the turn is refused
fail-closed before any prompt is sent. A resume must confirm the exact recorded session id in
its result; a forked or unconfirmed id is denied and never re-recorded. A recorded session can
never be resumed by another conversation, another workspace, a moved CLI, or a different login
(account or organization): any mismatch is denied with a fixed message instead of falling back.

A stateless one-shot transport is still available for calls with no conversation context
(auxiliary, context-free completions): those run with `--no-session-persistence`, never record
or resume a session, and never require a login attestation.

Requests outside those shapes are **refused** (fail closed) rather than silently dropped or
replayed: tools, a `tool_choice`, images, a prior conversation without a resumable native
session, or a resume carrying more or fewer than exactly one new user message. If you need any
of those, use a regular provider instead.

## Tools (opt-in)

Tools are **off by default**: the route is a strictly tool-free text completion surface unless
you explicitly opt in. The opt-in is an **exact production allowlist**: zero tools, or the
workspace-only `read_file`. Any other name is a configuration error (`providers.claude_cli.tools`
rejects it at load time, and the provider wiring refuses it before a provider exists). To give
the CLI workspace read access:

```json
{
  "providers": {
    "claude_cli": {
      "command": "claude",
      "tools": ["read_file"]
    }
  }
}
```

When `read_file` is opted in, each agent turn serves the CLI one loopback MCP endpoint
(`127.0.0.1`, ephemeral port) that is alive for exactly that turn:

- The tool surface is a **fixed snapshot** taken at turn start: a tool is exposed only when it
  is opted in by exact name **and** registered for the active turn. Unknown or non-allowed
  names are refused with a fixed message that names no available tool.
- `read_file` is exposed only when the turn's registered `read_file` is positively proven to
  carry the required scope: it must be the built-in, workspace-scoped read tool (the shared
  scoped-filesystem abstraction in `nanobot/execution/files.py`), and the turn's immutable
  request context must prove the turn's workspace. A same-named custom or replacement tool,
  a disabled tool, or a turn that cannot prove its workspace is refused fail-closed
  (`bridge_unavailable`) before any endpoint is opened.
- Every request to the endpoint must carry the turn's one-time bearer token (missing, wrong,
  or stale tokens get 401). When the turn ends — success, failure, or cancellation — the token
  is revoked, in-flight calls are cancelled, and the server is shut down.
- Every call dispatches through the turn's own tool registry (parameter validation, execution,
  shared error handling) with the turn's own request context and workspace scope, so the
  tool's own workspace policy applies exactly as in a normal turn: reads resolve against the
  turn's workspace from the trusted turn context (never from tool arguments), and absolute
  paths outside the workspace, relative traversal, and escaping symlinks are denied by the
  same scoped open that serves the read. No separate path pre-check, no raw registry calls,
  no separate shell, no permission bypass.
- Each call has a wall-clock bound (`toolTimeoutS`, default 60s) and a result-size bound
  (`maxToolResultBytes`, default 65536); the bridge never retries a call.
- Real tool activity is surfaced once per phase through the normal tool-event channel; nothing
  is synthesized or replayed.
- Tools are only authorized **inside an active Nanobot turn**: a stateless one-shot or
  auxiliary call carries no turn registry and is refused (`bridge_unavailable`) before any
  endpoint is opened.

The built-in CLI tool surface stays empty (`--tools ""`): the endpoint is the only tool source
the CLI can reach, and only `mcp__nanobot__*` is pre-authorized, so no foreign MCP tool or
built-in tool can be reached. The token-bearing endpoint config is written to a `0600` temp
file that the CLI receives as `--mcp-config <file>` (never on the command line) and is removed
after the turn.

## Setup

1. Install Claude Code and sign it in on this host. Verify it works on its own. Note that
   `--output-format stream-json` requires `--verbose`:

   ```bash
   claude auth status
   echo "hello" | claude -p --output-format stream-json --verbose --model claude-sonnet-4-5
   ```

2. Add the provider block and a preset. In **Settings → Models → New preset**, or in
   `config.json`:

   ```json
   {
     "providers": {
       "claude_cli": {
         "command": "claude",
         "timeoutS": 120,
         "maxOutputBytes": 8388608
       }
     },
     "modelPresets": {
       "claude-cli": {
         "model": "claude_cli/claude-sonnet-4-5-20250929",
         "provider": "claude_cli"
       }
     }
   }
   ```

3. Select that preset (pin `provider: "claude_cli"` explicitly).

### Configuration keys

| Key | Default | Meaning |
|---|---|---|
| `providers.claude_cli.command` | `"claude"` | The `claude` command, or a full path if the gateway's `PATH` does not include the install directory (for example `~/.local/bin`). |
| `providers.claude_cli.timeoutS` | `120` | Wall-clock bound for one completion. The process and its descendants are cleaned up when it is exceeded. |
| `providers.claude_cli.maxOutputBytes` | `8388608` | Bound on the CLI stream. Larger output stops the run and reports an `output_limit` error. |
| `providers.claude_cli.tools` | `[]` | Exact tool allowlist (see [Tools (opt-in)](#tools-opt-in)): empty keeps the route tool-free; the only allowed entry is `read_file` (workspace-only). Any other name is a configuration error. |
| `providers.claude_cli.toolTimeoutS` | `60` | Wall-clock bound for one tool call through the bridge. |
| `providers.claude_cli.maxToolResultBytes` | `65536` | Bound on a single tool result forwarded to the CLI. |

`apiKey`/`apiBase` are accepted for schema compatibility but are ignored: this route uses no
HTTP credentials. There is deliberately no way to inject extra environment into the CLI
process — the environment scrub is authoritative, so a gateway credential can never be
forwarded to the CLI.

### Finding the CLI when the service PATH omits it

If the gateway service does not have `~/.local/bin` (or wherever `claude` is installed) on its
`PATH`, set `providers.claude_cli.command` to the **full path** of the executable, e.g.
`"command": "/home/you/.local/bin/claude"`. A bare `claude` is also searched in
`~/.local/bin` and `~/.claude/local/bin` before the ambient `PATH`.

## How it works

For each completion nanobot starts one `claude` process with print mode and a structured stream:

- The prompt is written to **stdin** (no prompt in the command line, so it never appears in
  process listings or shell history).
- The CLI streams `stream-json` records to **stdout**. The terminal `result` record is the single
  source of truth for the reply text and token usage, so the reply is never duplicated.
- **stderr** is captured only as bounded diagnostics for internal classification. It is never
  returned to the chat reply and no raw CLI output or error text is ever forwarded: on failure
  the caller receives only a fixed diagnostic message (see [Errors](#errors)).

The run is bounded by the wall-clock timeout and the output-size bound, and the process tree
(including any descendants) is cleaned up on every exit path, including cancellation.

## Isolation

The subprocess is launched with a strict, isolated surface:

| Flag | Effect |
|---|---|
| `--tools ""` | Disable every built-in tool (strict empty tool surface). |
| `--disallowedTools "mcp__*"` | Deny any MCP tool by name as a second guard. |
| `--strict-mcp-config` + `--mcp-config '{"mcpServers":{}}'` | Only the (empty) supplied MCP config is honored; no inherited MCP servers. |
| `--safe-mode` | Disable CLAUDE.md, skills, plugins, hooks, custom commands/agents, and workflows. |
| `--setting-sources ""` | Load no user/project/local settings files. |
| `--permission-mode default` | Never bypass permission checks. |
| `--no-session-persistence` | Do not write a resumable session to disk. |

When tools are opted in, the strict empty MCP surface is replaced by the turn-local bridge
config: `--strict-mcp-config --mcp-config <0600 temp file>` (the file is the only form
accepted — it carries the turn's bearer token) plus `--allowedTools mcp__nanobot__*`, so the
per-turn loopback bridge is the only MCP server and only its tools are pre-authorized. Tool-enabled turns require Claude Code 2.1.248 or later and use `--restricted` instead of `--safe-mode`, which disables even explicitly configured MCP servers. Restricted mode ignores user/project/local settings; administrator-managed policy still applies. Nanobot also sets `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` to suppress CLAUDE.md discovery.

When the request carries a trusted system message, `--system-prompt-file <tmpfile>` is
appended: the prompt text is written to a `0600` temp file (never to the command line) and
the file is removed after the run regardless of outcome.

The environment is also scrubbed: the gateway process's Anthropic provider-routing and credential
variables (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_OAUTH_TOKEN`, and related) are
removed so they cannot steer the CLI toward a different provider, endpoint, or token source. The
CLI resolves its **own** local login.

## Errors

| `error_kind` | Meaning |
|---|---|
| `missing_cli` | The `claude` executable could not be found. Install Claude Code or set `command` to its full path. |
| `auth` | The CLI reported it is not authenticated. Sign in on the host with `claude auth login`. |
| `timeout` | The completion exceeded `timeoutS`. |
| `output_limit` | The CLI output exceeded `maxOutputBytes`. |
| `bridge_unavailable` | Tools are opted in but cannot be authorized for this call: no bridge exists in this runtime, the active turn does not carry a tool registry, the opted-in tool is not registered for the turn (for example, the file tools are disabled), or the turn's `read_file` is not the built-in workspace-scoped read tool bound to the turn's workspace. The turn was not started. |
| `bad_request` | The request was refused: tools, `tool_choice`, images, prior conversation, or a message shape other than one optional system message plus one user message. |

## Limitations (not yet supported)

The following are intentionally out of scope for this tranche and are **refused**, not silently
handled: external MCP servers of your own (only the turn-local opt-in bridge exists), images,
and warm process reuse. These are planned as follow-on work; until then this route is a text
completion surface with an optional, explicitly opted-in tool bridge.

## Troubleshooting

- **"Claude CLI executable not found"**: set `providers.claude_cli.command` to the full path of
  `claude` (run `which claude` in an interactive shell).
- **"Claude CLI is not authenticated"**: complete the local sign-in (`claude auth login`) and
  confirm `claude auth status` works; nanobot does not manage this login.
- **No reply, empty result**: some prompts end the turn with no text. This is reported as a
  completed empty turn, not an error.
