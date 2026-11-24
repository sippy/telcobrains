"""Schema tests for the explicit Claude Code CLI provider configuration."""

import pytest

from nanobot.config.schema import ClaudeCLIProviderConfig, ProvidersConfig


def test_claude_cli_config_defaults() -> None:
    config = ClaudeCLIProviderConfig()
    assert config.command == "claude"
    assert config.timeout_s == 120.0
    assert config.max_output_bytes == 8 * 1024 * 1024
    # No credential is read or stored: api_key is unused (None by default).
    assert config.api_key is None


def test_claude_cli_config_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        ClaudeCLIProviderConfig(timeout_s=0)
    with pytest.raises(ValueError):
        ClaudeCLIProviderConfig(max_output_bytes=10)
    assert ClaudeCLIProviderConfig(command="/opt/bin/claude", timeout_s=30.0, max_output_bytes=1024).command == "/opt/bin/claude"


def test_providers_config_exposes_claude_cli_route() -> None:
    providers = ProvidersConfig()
    assert isinstance(providers.claude_cli, ClaudeCLIProviderConfig)
    # The route is opt-in: a default config selects no claude_cli provider.
    assert providers.claude_cli.command == "claude"
