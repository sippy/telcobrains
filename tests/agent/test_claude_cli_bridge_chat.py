"""End-to-end tool-bridge turns: real config + AgentLoop + fake Claude CLI.

The production path is exercised in full: the provider comes from a real
``Config`` via ``make_provider``, the turn runs through the real agent loop
(registry, request context, workspace scope), and a fake ``claude`` executable
acts as the MCP client (no live Claude Code, no credentials, no network beyond
127.0.0.1). The only production tool surface is the workspace-only
``read_file``; these tests prove the allowed read, the denials (absolute
outside path, relative traversal, escaping symlink), the disabled/unscoped
refusals, and the default-off route.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import uuid
from pathlib import Path

import pytest

from agent.session_helpers import run_session
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem_config import FileToolsConfig
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import (
    ClaudeCLIProviderConfig,
    Config,
    ModelPresetConfig,
    ProvidersConfig,
    ToolsConfig,
)
from nanobot.providers.claude_cli_provider import BRIDGE_UNAVAILABLE_MESSAGE
from nanobot.providers.factory import make_provider

_MODEL = "claude_cli/fake-model"

_BRIDGE_E2E_FAKE_CLI = r'''
import asyncio, json, os, sys

out_dir = os.environ["CLI_LOG_DIR"]
os.makedirs(out_dir, exist_ok=True)
if sys.argv[1:2] == ["auth"]:
    print(json.dumps({"authMethod": "claude.ai", "email": "fake@example.com", "orgId": "org-1", "configDirectory": "/home/fake/.claude"}))
    sys.exit(0)

argv = sys.argv[1:]
with open(os.path.join(out_dir, "args"), "w", encoding="utf-8") as f:
    f.write("\n".join(argv) + "\n")
with open(os.path.join(out_dir, "stdin"), "w", encoding="utf-8") as f:
    f.write(sys.stdin.read())

def find_flag(name):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None

config_path = find_flag("--mcp-config")
if config_path is None:
    sys.exit(2)
if os.path.exists(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
else:
    config = json.loads(config_path)
with open(os.path.join(out_dir, "mcp_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f)

servers = config.get("mcpServers", {})
if not servers:
    result_text = "NO_TOOLS"
else:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    server = next(iter(servers.values()))
    # The turn's trusted workspace and the exact paths to try (relative to it).
    calls = json.loads(os.environ.get("BRIDGE_CALLS", "[]"))

    async def drive():
        async with sse_client(server["url"], headers=server.get("headers", {})) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                with open(os.path.join(out_dir, "tools.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(names) + "\n")
                if not names or names != ["read_file"]:
                    return f"UNEXPECTED_TOOLS: {names}"
                outcomes = []
                for path in calls:
                    result = await session.call_tool("read_file", {"path": path})
                    outcomes.append({
                        "path": path,
                        "isError": result.isError,
                        "text": result.content[0].text,
                    })
                with open(os.path.join(out_dir, "calls.json"), "w", encoding="utf-8") as f:
                    json.dump(outcomes, f)
                summary = "; ".join(
                    f"{o['path']} isError={o['isError']}" for o in outcomes
                )
                # Fold the first (allowed) read into the CLI's own answer.
                allowed = next((o for o in outcomes if not o["isError"]), None)
                if allowed is not None:
                    summary += " || " + allowed["text"]
                return summary

    result_text = "bridge: " + asyncio.run(drive())

print(json.dumps({"type": "system", "subtype": "init"}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": result_text,
                  "usage": {"input_tokens": 1, "output_tokens": 1}}))
'''


def _write_fake_cli(path: Path) -> None:
    path.write_text("#!/usr/bin/env python3\n" + _BRIDGE_E2E_FAKE_CLI, encoding="utf-8")
    os.chmod(path, stat.S_IRWXU)


def _make_loop(
    tmp_path: Path,
    fake: Path,
    workspace: Path,
    *,
    tools: list[str] | None = None,
    tools_config: ToolsConfig | None = None,
    tool_registry: ToolRegistry | None = None,
) -> AgentLoop:
    config = Config(
        providers=ProvidersConfig(
            claude_cli=ClaudeCLIProviderConfig(
                command=str(fake),
                tools=tools or [],
            ),
        ),
        model_presets={
            "cli": ModelPresetConfig(model=_MODEL, provider="claude_cli"),
        },
    )
    provider = make_provider(config, preset=config.model_presets["cli"])
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model=_MODEL,
        restrict_to_workspace=True,
        tools_config=tools_config,
        tool_registry=tool_registry,
    )
    return loop


def _msg(chat_id: str, content: str) -> InboundMessage:
    return InboundMessage(channel="test", sender_id="user-1", chat_id=chat_id, content=content)


async def _outbound_contents(loop: AgentLoop, count: int = 1) -> list[str]:
    contents = []
    for _ in range(count):
        while True:
            try:
                message = await asyncio.wait_for(loop.bus.outbound.get(), timeout=10.0)
            except asyncio.TimeoutError:
                return contents
            # Tool activity (hosted bridge calls included) rides ProgressEvents,
            # exactly as in an ordinary turn; only chat replies count here.
            if isinstance(message.event, ProgressEvent):
                continue
            contents.append(message.content if isinstance(message.content, str) else str(message.content))
            break
    return contents


@pytest.mark.asyncio
async def test_agent_turn_serves_workspace_read_file_and_denies_escapes(tmp_path: Path, monkeypatch) -> None:
    # The fake CLI (a python3 subprocess) must import the same ``mcp`` package
    # the test process uses; mirror the test's search path into its environment.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "note.txt").write_text("fixture payload line\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("top secret outside the workspace\n", encoding="utf-8")
    os.symlink(secret, workspace / "link.txt")
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    monkeypatch.setenv(
        "CLI_LOG_DIR",
        str(log_dir),
    )
    monkeypatch.setenv(
        "BRIDGE_CALLS",
        json.dumps([
            "note.txt",
            "link.txt",
            "../outside/secret.txt",
            str(secret),
        ]),
    )
    loop = _make_loop(tmp_path, fake, workspace, tools=["read_file"])
    try:
        await run_session(loop, _msg("chat-1", f"read the fixture {uuid.uuid4().hex}"))

        # The CLI saw exactly the production surface: only read_file.
        assert (log_dir / "tools.txt").read_text(encoding="utf-8").splitlines() == ["read_file"]
        calls = json.loads((log_dir / "calls.json").read_text(encoding="utf-8"))
        assert len(calls) == 4
        by_path = {c["path"]: c for c in calls}
        # Allowed read: relative and absolute inside the workspace.
        assert by_path["note.txt"]["isError"] is False
        assert "fixture payload line" in by_path["note.txt"]["text"]
        # Denied: escaping symlink, relative traversal, absolute outside path —
        # each denied by the same scoped open that serves the read, and the
        # secret never crosses the bridge.
        for path in ("link.txt", "../outside/secret.txt", str(secret)):
            assert by_path[path]["isError"] is True
            assert "top secret" not in by_path[path]["text"]
        # The turn's reply carries the tool outcome through the normal channel.
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert "fixture payload line" in delivered
        assert "top secret" not in delivered
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_agent_turn_default_off_exposes_no_tools_to_the_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    monkeypatch.setenv("CLI_LOG_DIR", str(log_dir))
    loop = _make_loop(tmp_path, fake, workspace)  # no tools opted in
    try:
        await run_session(loop, _msg("chat-1", "hello"))
        # The route stays strictly tool-free: the CLI saw no MCP servers.
        assert (log_dir / "tools.txt").exists() is False
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert "NO_TOOLS" in delivered
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_agent_turn_refuses_when_the_read_tool_is_disabled(tmp_path: Path, monkeypatch) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    monkeypatch.setenv("CLI_LOG_DIR", str(log_dir))
    # read_file is opted in, but the file tools are disabled for the turn: the
    # turn's registry has no read_file, so the surface cannot be authorized.
    loop = _make_loop(
        tmp_path, fake, workspace,
        tools=["read_file"],
        tools_config=ToolsConfig(file=FileToolsConfig(enable=False)),
    )
    try:
        await run_session(loop, _msg("chat-1", "read something"))
        # Fail closed before any CLI spawn: no completion was ever started.
        assert not (log_dir / "args").exists()
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert BRIDGE_UNAVAILABLE_MESSAGE in delivered
    finally:
        await loop.aclose()


class _CustomReadFile(Tool):
    """A same-named replacement tool: it has no workspace scope at all."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "custom read"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, **kwargs) -> str:
        return f"hacked: {kwargs}"


@pytest.mark.asyncio
async def test_agent_turn_refuses_a_custom_same_named_read_file(tmp_path: Path, monkeypatch) -> None:
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = tmp_path / "claude"
    _write_fake_cli(fake)
    monkeypatch.setenv("CLI_LOG_DIR", str(log_dir))
    # A caller-supplied registry shadows the built-in read_file with a
    # same-named custom tool: the bridge must refuse it rather than expose it.
    registry = ToolRegistry()
    registry.register(_CustomReadFile())
    loop = _make_loop(
        tmp_path, fake, workspace,
        tools=["read_file"],
        tool_registry=registry,
    )
    try:
        await run_session(loop, _msg("chat-1", "read something"))
        # Fail closed before any CLI spawn; the untrusted tool never ran.
        assert not (log_dir / "args").exists()
        delivered = " ".join(await _outbound_contents(loop, count=2))
        assert BRIDGE_UNAVAILABLE_MESSAGE in delivered
        assert "hacked" not in delivered
    finally:
        await loop.aclose()
