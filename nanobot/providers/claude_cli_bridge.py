"""Per-turn, localhost-authenticated MCP bridge for the Claude Code CLI route.

The Claude CLI is the MCP client; nanobot runs one MCP server per provider turn
on a loopback endpoint. No new dependencies are introduced: the endpoint is a
minimal MCP SSE/JSON-RPC server built on the standard library (the ``mcp``
package, already a project dependency, is used only by the test client). The
surface is deliberately minimal and bounded:

- Each turn owns one server bound to 127.0.0.1 on an ephemeral port, alive for
  exactly that turn. A bridge is created only when tools are explicitly opted
  in; the default route remains strictly tool-free.
- The production tool surface is an exact allowlist: zero tools, or the
  workspace-only ``read_file`` (see :data:`PRODUCTION_TOOL_ALLOWLIST`). Any
  other name is refused at turn start, before any endpoint is opened. Low-level
  tests may widen the surface explicitly via *allowed_tool_names*; the
  production configuration and provider wiring never do.
- The only opt-in name, ``read_file``, is exposed only when the name in the
  active turn's registry is positively proven to carry the required scope: it
  must be the built-in, workspace-scoped read tool (the repo's safe filesystem
  abstraction in ``nanobot.execution.files``), and the turn's immutable
  ``RequestContext`` must prove the turn's workspace. A same-named custom or
  replacement tool, a stripped clone, or a turn that cannot prove its
  workspace is refused fail-closed (no endpoint is opened). No path is
  pre-checked separately: every call is dispatched to the tool, whose scoped
  open authorizes and opens the actual path in one operation (absolute
  outside paths, relative traversal, and escaping symlinks are denied by the
  same operation).
- Every turn supplies the registry of the *active* Nanobot turn and that turn's
  immutable ``RequestContext`` at turn start: a tool is exposed only when it is
  both opted in by exact name and registered for that turn, and the turn's own
  context (channel, chat, session, workspace scope) is bound for every call it
  makes, so the scoped read resolves against the turn's workspace from the
  trusted turn context — never from tool arguments.
- Every HTTP request (the SSE session open and every JSON-RPC message) must
  carry the turn's one-time bearer token. Missing, wrong, or stale tokens are
  rejected with 401 and open nothing.
- Unknown or non-allowed names are refused with a fixed message that names no
  available tool, before any tool executes.
- Calls dispatch through the turn's shared tool dispatch boundary
  (``ToolDispatch``, the same object the agent runner uses for its own
  model-initiated calls): registration and schema validation, the turn's hook
  lifecycle (audit/progress events), the turn's policy accounting (repeated
  external-lookup blocks and workspace-violation escalation shared across all
  of the CLI's calls in the turn), read receipts, and shared error handling —
  with the turn's immutable ``RequestContext`` and workspace scope bound for
  every call, so the tools' own workspace and network policies apply exactly
  as in a normal turn.
- Every call has a wall-clock bound and a result-size bound, and the bridge
  never retries a call.
- Ending the turn (success, failure, or cancellation) revokes the token,
  cancels in-flight calls, and shuts the server down, so a stale token can
  never act.
- Real tool activity is observable exactly once per call (a start event and
  one terminal event) through the turn's own hook chain — the same
  ``on_tool_execution_event`` boundary a normal turn's tool calls use. Nothing
  is synthesized, duplicated, or replayed, and a refused call that has no
  effect produces no turn event.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import queue
import secrets
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent import futures
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from loguru import logger

from nanobot.agent.goal_permission import goal_mutation_allowed, goal_mutation_permission
from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.dispatch import ToolDispatch
from nanobot.agent.tools.file_state import bind_file_states, current_file_states, reset_file_states
from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.config.claude_cli_tools import CLAUDE_CLI_TOOL_ALLOWLIST as PRODUCTION_TOOL_ALLOWLIST
from nanobot.execution.files import WorkspaceFiles
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import bind_workspace_scope, reset_workspace_scope

#: The MCP server name the bridge registers under (also scopes ``--allowedTools``).
BRIDGE_SERVER_NAME = "nanobot"

#: Fixed refusal for ``read_file`` that cannot prove the required scope.
READ_FILE_UNTRUSTED_MESSAGE = (
    "The active turn's read_file tool is not the workspace-scoped read tool "
    "bound to this turn's workspace, so it is not exposed."
)

#: Fixed refusal for a name that is not part of this turn's surface; names nothing.
DENIED_TOOL_MESSAGE = "Tool not available in this turn."

#: Bounded size (bytes) of a single tool result the bridge forwards to the CLI.
DEFAULT_MAX_TOOL_RESULT_BYTES = 64 * 1024

#: Bounded wall-clock (seconds) for one tool call.
DEFAULT_TOOL_TIMEOUT_S = 60.0

#: Bounded size (bytes) of one JSON-RPC message accepted from the CLI.
_MAX_MESSAGE_BYTES = 1_048_576

_MCP_PROTOCOL_VERSION = "2025-06-18"

#: Sentinel pushed to a connection's event queue to make its SSE thread exit.
_CLOSE = None


class BridgeError(RuntimeError):
    """A bounded, user-safe error raised by the per-turn tool bridge."""


@runtime_checkable
class BridgeTurn(Protocol):
    """One authenticated, turn-scoped loopback MCP endpoint."""

    @property
    def url(self) -> str:
        """The ``http://127.0.0.1:<port>/sse`` URL for this turn."""

    @property
    def token(self) -> str:
        """The one-time bearer token authenticating this turn's endpoint."""

    @property
    def closed(self) -> bool:
        """Whether the turn has been closed and its access revoked."""

    @property
    def mcp_config_json(self) -> str:
        """The ``--mcp-config`` JSON for this turn (carries the bearer token)."""

    def tools(self) -> list[dict[str, Any]]:
        """The fixed tool surface exposed to the CLI for this turn."""

    async def aclose(self) -> None:
        """Revoke the turn: cancel in-flight calls and shut the server down."""


@runtime_checkable
class ToolBridge(Protocol):
    """The per-turn surface the Claude CLI provider consumes.

    The provider only knows this protocol; the concrete
    :class:`ClaudeCodeToolBridge` is constructed by the caller (e.g. the
    gateway) from the provider configuration. Each turn receives the active
    turn's own tool dispatch boundary and request context.
    """

    async def start_turn(
        self,
        *,
        tool_names: Sequence[str],
        tool_dispatch: ToolDispatch,
        request_context: RequestContext | None = None,
        model_messages: list[dict[str, Any]] | None = None,
    ) -> BridgeTurn:
        """Open one turn-scoped endpoint; raises :class:`BridgeError` on failure.

        *tool_dispatch* is the active turn's shared tool dispatch boundary
        (supplied by the agent runner via the provider call context): its
        registry is both the source of the tool snapshot and the dispatch
        boundary for every call, and its per-turn policy accounting is shared
        across all calls the CLI makes in this turn. *tool_names* must stay
        inside the bridge's authorized surface
        (:data:`PRODUCTION_TOOL_ALLOWLIST` by default), and every exposed name
        must be positively proven to carry its required scope for the active
        turn's workspace before any endpoint is opened.
        """


class ClaudeCodeToolBridge:
    """Factory for per-turn localhost MCP endpoints.

    One bridge is created per provider (from the provider configuration) and
    serves any number of turns. Each :meth:`start_turn` opens a fresh server
    (ephemeral loopback port, one-time token, fixed tool snapshot) for the
    registry of the active turn; turns are fully isolated from each other and
    from later registry changes.

    *allowed_tool_names* bounds the surface a turn may opt into. Production
    always uses the default, :data:`PRODUCTION_TOOL_ALLOWLIST` (zero tools or
    the workspace-only ``read_file``); only explicit low-level tests widen it
    with synthetic tool fixtures.
    """

    def __init__(
        self,
        *,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        max_result_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
        http_read_timeout_s: float = 15.0,
        allowed_tool_names: frozenset[str] = PRODUCTION_TOOL_ALLOWLIST,
    ) -> None:
        self._tool_timeout_s = max(1.0, float(tool_timeout_s))
        self._max_result_bytes = max(256, int(max_result_bytes))
        self._http_read_timeout_s = max(1.0, float(http_read_timeout_s))
        self._allowed_tool_names = frozenset(allowed_tool_names)

    async def start_turn(
        self,
        *,
        tool_names: Sequence[str],
        tool_dispatch: ToolDispatch,
        request_context: RequestContext | None = None,
        model_messages: list[dict[str, Any]] | None = None,
    ) -> ClaudeCodeToolTurn:
        """Open one turn-scoped endpoint for the given opted-in tool names.

        *tool_dispatch* is the active turn's shared dispatch boundary: the
        fixed snapshot is the opted-in names present in its registry, and
        every call dispatches through the same boundary (and shares its
        per-turn policy accounting). Fails closed (no server is opened) when
        a name is outside the authorized surface, when an opted-in name cannot
        prove its required scope for the turn's workspace, or when no
        opted-in name is registered. The caller owns the turn's lifecycle:
        :meth:`ClaudeCodeToolTurn.aclose` must run at turn end (success,
        failure, or cancellation).
        """
        if not tool_names:
            raise BridgeError("at least one explicitly opted-in tool name is required")
        if not isinstance(tool_dispatch, ToolDispatch):
            raise BridgeError("the active turn's tool dispatch boundary is required")
        for name in tool_names:
            if name not in self._allowed_tool_names:
                raise BridgeError(f"tool '{name}' is not in the authorized tool surface")
        allowed: tuple[str, ...] = tuple(tool_names)
        snapshot: list[Tool] = []
        for name in allowed:
            tool = tool_dispatch.tools.get(name)
            if tool is None:
                continue
            self._authorize_tool_scope(name, tool, request_context)
            snapshot.append(tool)
        if not snapshot:
            raise BridgeError("none of the opted-in tools are registered for this turn")
        turn = ClaudeCodeToolTurn(
            self,
            allowed=allowed,
            snapshot=snapshot,
            token=secrets.token_urlsafe(32),
            # The turn's view of the shared dispatch: its own model-visible
            # messages (the read-receipt index) without changing the shared
            # registry, hooks, or policy accounting.
            dispatch=dataclasses.replace(tool_dispatch, model_messages=model_messages),
            request_context=request_context,
            # The turn's own policy context, snapshotted at turn start so a
            # later turn never carries this turn's bindings into its calls.
            file_states=current_file_states(None),
            goal_mutation_allowed=goal_mutation_allowed(),
        )
        try:
            await turn.start()
        except BaseException:
            await turn.aclose()
            raise
        return turn

    # ------------------------------------------------------------------ #
    # Scope authorization (fail closed before any endpoint is opened)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _authorize_tool_scope(
        name: str,
        tool: Tool,
        request_context: RequestContext | None,
    ) -> None:
        """Refuse any name whose registered tool cannot prove its scope.

        ``read_file`` is the only production name; it may be filled only by the
        built-in, workspace-scoped read tool, and only when the turn's immutable
        context proves the workspace every scoped read will resolve against.
        The check is a proof of the tool and its binding, not a path check: the
        tool's own scoped open is the single, atomic authorization for each
        requested path.
        """
        if name != "read_file":
            return
        if type(tool) is not ReadFileTool:
            # A same-named custom or replacement tool (plugin, injected,
            # subclassed clone) is not the trusted workspace-scoped tool.
            raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE)
        files = getattr(tool, "_files", None)
        if not isinstance(files, WorkspaceFiles):
            # Not the repo's safe/scoped filesystem abstraction: nothing here
            # can prove the path boundary, so refuse rather than widen it.
            raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE)
        scope = (
            request_context.workspace_scope
            if request_context is not None
            else None
        )
        if scope is not None:
            # The turn's trusted scope is authoritative: the bridge binds it for
            # every call, so the scoped read resolves against the turn's
            # workspace (the same resolution a normal turn performs).
            if scope.project_path is None:
                raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE)
            return
        # No turn scope: the tool's own construction workspace is the effective
        # one, so it must positively equal the turn's trusted workspace.
        turn_workspace = request_context.workspace if request_context is not None else None
        if turn_workspace is None or files.workspace is None:
            raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE)
        try:
            tool_workspace = Path(files.workspace).expanduser().resolve()
            proven_workspace = Path(turn_workspace).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE) from None
        if tool_workspace != proven_workspace:
            raise BridgeError(READ_FILE_UNTRUSTED_MESSAGE)


class ClaudeCodeToolTurn:
    """One localhost MCP server bound to exactly one provider turn."""

    def __init__(
        self,
        bridge: ClaudeCodeToolBridge,
        *,
        allowed: tuple[str, ...],
        snapshot: list[Tool],
        token: str,
        dispatch: ToolDispatch,
        request_context: RequestContext | None = None,
        file_states: Any = None,
        goal_mutation_allowed: bool = False,
    ) -> None:
        self._bridge = bridge
        self._allowed = allowed
        self._snapshot = snapshot
        self._token = token
        self._dispatch = dispatch
        self._request_context = request_context
        self._file_states = file_states
        self._goal_mutation_allowed = goal_mutation_allowed
        self._tool_timeout_s = bridge._tool_timeout_s
        self._max_result_bytes = bridge._max_result_bytes
        self._http_read_timeout_s = bridge._http_read_timeout_s
        self._server: ThreadingHTTPServer | None = None
        self._serve_thread: threading.Thread | None = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None
        self._in_flight: set[futures.Future[Any]] = set()
        self._conns: set[_SseConnection] = set()
        self._closed = False

    # ------------------------------------------------------------------ #
    # BridgeTurn protocol
    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port()}/sse"

    @property
    def token(self) -> str:
        return self._token

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def mcp_config_json(self) -> str:
        return json.dumps(
            {
                "mcpServers": {
                    BRIDGE_SERVER_NAME: {
                        "type": "sse",
                        "url": self.url,
                        "headers": {"Authorization": f"Bearer {self.token}"},
                    }
                }
            }
        )

    def tools(self) -> list[dict[str, Any]]:
        """The fixed snapshot: names, descriptions, and JSON schemas."""
        return [
            {
                "name": tool.name,
                "description": tool.description or tool.name,
                "inputSchema": tool.parameters or {"type": "object", "properties": {}},
            }
            for tool in self._snapshot
        ]

    async def start(self) -> None:
        # Bind on a worker thread (server_bind is blocking) so the caller's
        # event loop is not stalled.
        try:
            self._server = await asyncio.to_thread(self._bind_server)
            self._serve_thread = threading.Thread(
                target=self._server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="claude-cli-mcp-serve",
                daemon=True,
            )
            self._serve_thread.start()
        except BaseException:
            if self._server is not None:
                with suppress(OSError):
                    self._server.server_close()
            raise
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="claude-cli-mcp-loop",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info(
            "Claude CLI tool bridge turn started: endpoint={} tools={}",
            self.url,
            [tool.name for tool in self._snapshot],
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 1. Stop accepting and serving new connections.
        if self._server is not None:
            with suppress(OSError):
                self._server.shutdown()
            with suppress(OSError):
                self._server.server_close()
        # 2. Cancel in-flight tool calls and give them a bounded drain.
        in_flight = set(self._in_flight)
        for future in in_flight:
            future.cancel()
        if in_flight:
            drain_bound = max(5.0, self._tool_timeout_s)
            try:
                await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: futures.wait(in_flight, timeout=drain_bound),
                    ),
                    timeout=drain_bound + 5.0,
                )
            except Exception:
                logger.warning("Claude CLI tool bridge: in-flight drain exceeded bound")
        # 3. Close client sessions so their SSE threads exit.
        for conn in list(self._conns):
            conn.close()
        # 4. Stop the turn's event loop.
        if self._loop_thread is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2.0)
        if not self._loop.is_closed():
            with suppress(Exception):
                self._loop.close()
        logger.info("Claude CLI tool bridge turn closed")

    # ------------------------------------------------------------------ #
    # Server plumbing
    # ------------------------------------------------------------------ #
    def _port(self) -> int:
        if self._server is None:
            raise BridgeError("turn not started")
        return self._server.server_address[1]

    def _bind_server(self) -> ThreadingHTTPServer:
        handler = self._make_handler()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        return server

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        turn = self

        class _Handler(BaseHTTPRequestHandler):
            # Bound socket operations so a stalled client cannot pin a thread.
            timeout = turn._http_read_timeout_s

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                pass  # keep the per-turn server quiet; the bridge logs lifecycle

            def _authorized(self) -> bool:
                if turn.closed:
                    return False
                header = self.headers.get("Authorization", "")
                scheme, _, value = header.partition(" ")
                if scheme.lower() != "bearer" or not value:
                    return False
                return secrets.compare_digest(value.strip(), turn._token)

            def _reply_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if urlparse(self.path).path.rstrip("/") != "/sse" or not self._authorized():
                    self._reply_json(401, {"error": "unauthorized"})
                    return
                self._handle_sse()

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path.rstrip("/") != "/messages" or not self._authorized():
                    self._reply_json(401, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._reply_json(400, {"error": "bad request"})
                    return
                if length > _MAX_MESSAGE_BYTES:
                    self._reply_json(413, {"error": "message too large"})
                    return
                raw = self.rfile.read(length) if length > 0 else b""
                self._handle_message(parsed.query, raw)

            def _handle_sse(self) -> None:
                conn = _SseConnection()
                turn._conns.add(conn)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    message_url = f"http://127.0.0.1:{turn._port()}/messages/?session_id={conn.session_id}"
                    self.wfile.write(f"event: endpoint\ndata: {message_url}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    # Frames are written in queue order until the close sentinel:
                    # a response queued before the turn is closed is always
                    # delivered before the stream ends.
                    while True:
                        try:
                            frame = conn.events.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        if frame is _CLOSE:
                            break
                        try:
                            self.wfile.write(frame.encode("utf-8"))
                            self.wfile.flush()
                        except OSError:
                            conn.close()
                            break
                finally:
                    conn.close()
                    turn._conns.discard(conn)

            def _handle_message(self, query: str, raw: bytes) -> None:
                session_id = parse_qs(query).get("session_id", [""])[0]
                conn = next((c for c in turn._conns if c.session_id == session_id), None)
                if conn is None or conn.closed:
                    self._reply_json(404, {"error": "unknown session"})
                    return
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._reply_json(
                        202,
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}},
                    )
                    return
                if not isinstance(message, dict):
                    self._reply_json(
                        202,
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}},
                    )
                    return
                turn._dispatch_rpc(conn, message)
                # Responses are delivered on the SSE stream; ack the post.
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()

        return _Handler

    def _sse_frame(self, payload: dict[str, Any]) -> str:
        return f"event: message\ndata: {json.dumps(payload)}\n\n"

    # ------------------------------------------------------------------ #
    # JSON-RPC (MCP) dispatch
    # ------------------------------------------------------------------ #
    def _dispatch_rpc(self, conn: _SseConnection, message: dict[str, Any]) -> None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        is_notification = msg_id is None

        def _respond(result: Any) -> None:
            if is_notification:
                return
            self._send(conn, {"jsonrpc": "2.0", "id": msg_id, "result": result})

        def _respond_error(code: int, text: str) -> None:
            if is_notification:
                return
            self._send(conn, {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}})

        if method == "initialize":
            protocol_version = params.get("protocolVersion")
            _respond(
                {
                    "protocolVersion": protocol_version if isinstance(protocol_version, str) else _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": BRIDGE_SERVER_NAME, "version": "1"},
                }
            )
        elif method == "tools/list":
            _respond({"tools": self.tools()})
        elif method == "tools/call":
            self._dispatch_tool_call(conn, msg_id, params)
        elif method in ("notifications/initialized", "notifications/cancelled", "ping"):
            _respond({})
        else:
            _respond_error(-32601, "method not found")

    def _send(self, conn: _SseConnection, payload: dict[str, Any]) -> None:
        conn.send(self._sse_frame(payload))

    def _dispatch_tool_call(self, conn: _SseConnection, msg_id: Any, params: dict[str, Any]) -> None:
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            self._send(conn, {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": "invalid params"}})
            return
        if name not in self._allowed:
            # Refused with a fixed message that names no available tool, before
            # any tool resolves or executes. The call has no effect, so it
            # produces no turn event.
            self._send(conn, self._result_payload(msg_id, DENIED_TOOL_MESSAGE, is_error=True))
            return
        if self._loop_thread is None or not self._loop.is_running():
            self._send(conn, self._result_payload(msg_id, "tool bridge is shutting down", is_error=True))
            return
        started = time.monotonic()
        timed_out = threading.Event()
        future = asyncio.run_coroutine_threadsafe(
            self._execute_tool(name, arguments, started), self._loop
        )
        self._in_flight.add(future)

        def _settle(fut: futures.Future[Any]) -> None:
            self._in_flight.discard(fut)
            try:
                result = fut.result()
            except (asyncio.CancelledError, futures.CancelledError):
                message = (
                    f"Tool call timed out after {self._tool_timeout_s:.1f}s."
                    if timed_out.is_set()
                    else "Tool call cancelled with the turn."
                )
                self._send(conn, self._result_payload(msg_id, message, is_error=True))
                return
            except BridgeError as exc:
                self._send(conn, self._result_payload(msg_id, str(exc), is_error=True))
                return
            except Exception:
                # Shared dispatch error handling already shaped failures into
                # tool results; anything reaching here is unexpected and must
                # not leak an internal traceback to the CLI.
                logger.exception("Claude CLI tool bridge: unexpected tool execution error")
                message = "Tool execution failed."
                self._send(conn, self._result_payload(msg_id, message, is_error=True))
                return
            text = _cap_text(str(result), self._max_result_bytes)
            self._send(conn, self._result_payload(msg_id, text, is_error=bool(getattr(result, "is_error", False))))

        def _watchdog() -> None:
            remaining = self._tool_timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                return
            done, _ = futures.wait({future}, timeout=remaining)
            if future in done or future.done():
                return  # settled within the bound
            logger.warning(
                "Claude CLI tool bridge: tool call exceeded {}s; cancelling",
                self._tool_timeout_s,
            )
            timed_out.set()
            future.cancel()

        future.add_done_callback(_settle)
        threading.Thread(target=_watchdog, name="claude-cli-tool-watchdog", daemon=True).start()

    def _result_payload(self, msg_id: Any, text: str, *, is_error: bool) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        started: float,
    ) -> ToolResult:
        """Execute one tool through the active turn's shared dispatch boundary.

        The turn's immutable ``RequestContext`` (and with it the turn's
        trusted workspace scope), the turn's session file-read state, and the
        turn's goal-mutation permission are bound for the call, so the tool's
        workspace and network policies apply exactly as in a normal turn —
        never from tool arguments. The call goes through the same
        ``ToolDispatch`` the agent runner uses for its own model-initiated
        calls: the turn's policy accounting, hooks, and error shaping apply,
        and the turn observes the call exactly once (start and one terminal
        event) through its hook chain. Cancellation propagates to the tool
        (the same semantics as turn cancellation) and the wall-clock bound is
        enforced by ``wait_for``.
        """
        if self._dispatch.tools.get(name) is None:
            raise BridgeError(f"tool '{name}' is not available for this turn")
        remaining = max(0.25, self._tool_timeout_s - (time.monotonic() - started))
        call = ToolCallRequest(id=f"cli-{uuid.uuid4().hex}", name=name, arguments=arguments)
        ctx_token = None
        scope_token = None
        files_token = None
        try:
            if self._request_context is not None:
                ctx_token = bind_request_context(self._request_context)
                # The turn's trusted workspace scope is bound with the request
                # context, so scoped file operations resolve against the turn's
                # workspace exactly as in a normal turn — never from tool args.
                scope = self._request_context.workspace_scope
                if scope is not None:
                    scope_token = bind_workspace_scope(scope)
            if self._file_states is not None:
                # The turn's own session read receipts: a read already visible
                # to this turn is not re-read through the bridge.
                files_token = bind_file_states(self._file_states)
            with goal_mutation_permission(self._goal_mutation_allowed):
                entry = await asyncio.wait_for(self._dispatch.dispatch(call), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise BridgeError(f"Tool call timed out after {self._tool_timeout_s:.1f}s.") from exc
        finally:
            if files_token is not None:
                reset_file_states(files_token)
            if scope_token is not None:
                reset_workspace_scope(scope_token)
            if ctx_token is not None:
                reset_request_context(ctx_token)
        result = entry.result
        # The boundary's terminal event is authoritative for success/failure:
        # shaped failures (tool errors, policy blocks, escalations) come back
        # as ordinary result strings, so the wire's ``isError`` follows the
        # event, exactly as the runner treats the same call in an ordinary
        # turn.
        event = entry.event if isinstance(entry.event, dict) else {}
        is_error = event.get("status") == "error"
        if isinstance(result, ToolResult):
            if is_error and not result.is_error:
                return ToolResult(result, is_error=True, error_code=result.error_code or "tool_error")
            return result
        return ToolResult(str(result), is_error=is_error, error_code="tool_error" if is_error else None)


class _SseConnection:
    """One authenticated SSE client session (one CLI connection per turn)."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.events: queue.Queue[str | None] = queue.Queue()
        self.closed = False

    def send(self, frame: str) -> None:
        if not self.closed:
            self.events.put(frame)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.events.put(_CLOSE)


def _cap_text(text: str, limit: int) -> str:
    """Cap *text* at *limit* bytes without splitting a UTF-8 sequence."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore") + "\n[truncated]"
