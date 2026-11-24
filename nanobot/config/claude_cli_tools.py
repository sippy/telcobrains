"""Policy constants for the Claude CLI route's production tool surface.

This module is intentionally import-safe: it carries no runtime implementation,
so both the provider configuration (``nanobot.config.schema``) and the runtime
wiring (``nanobot.providers.claude_cli_bridge`` / ``claude_cli_provider``)
enforce the same exact allowlist from one source.
"""

from __future__ import annotations

#: The exact production tool surface the Claude CLI route may expose: zero
#: tools, or the built-in, workspace-only ``read_file``. Any other name is
#: refused at configuration, at provider wiring, and at turn start.
CLAUDE_CLI_TOOL_ALLOWLIST: frozenset[str] = frozenset({"read_file"})
