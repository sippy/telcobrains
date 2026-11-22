from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import Config
from nanobot.webui.settings_api import update_runtime_config_settings
from nanobot.webui.settings_contracts import WebUISettingsError
from nanobot.webui.settings_runtime import (
    runtime_config_payload,
    update_runtime_config,
)
from nanobot.webui.settings_services import WebUISettingsServices


def test_shared_proxy_roundtrip_preserves_independent_feature_switches(tmp_path):
    path = tmp_path / "config.json"
    config = Config()
    config.tools.web.enable = False
    config.providers.openai.use_proxy = True
    save_config(config, path)
    for proxy in ("http://shared-proxy.example:7890", "http://shared-proxy.example:7891"):
        payload = update_runtime_config_settings({"network.proxy": proxy}, local_browser=True, config_path=path)
        assert payload["runtime_config"]["network.proxy"] == proxy
        assert payload["requires_restart"]
        saved = load_config(path)
        assert saved.network.proxy == proxy
        assert saved.tools.web.enable is False
        assert saved.tools.web.use_proxy is False
        assert saved.providers.openai.use_proxy is True
    update_runtime_config_settings({"tools.web.use_proxy": True}, config_path=path)
    assert load_config(path).tools.web.use_proxy is True


def test_legacy_sandbox_config_roundtrip_and_rejected_override_are_atomic(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tools": {"exec": {"sandbox": "bwrap", "sandboxRoBinds": ["/opt/tools"]}}}))
    config = load_config(path)
    assert config.tools.exec.sandbox_launcher == ""
    payload = update_runtime_config_settings({"tools.exec.timeout": 75}, config_path=path)
    assert payload["runtime_config"]["tools.exec.sandbox_launcher"] == ""
    assert load_config(path).tools.exec.sandbox_ro_binds == ["/opt/tools"]
    before = path.read_bytes()
    reason = "path_not_absolute" if os.name == "posix" else "platform_unsupported"
    with pytest.raises(WebUISettingsError, match=f"sandbox_launcher_{reason}"):
        update_runtime_config_settings({"tools.exec.sandbox_launcher": "bin/bwrap", "tools.exec.timeout": 99}, config_path=path)
    assert path.read_bytes() == before
    update_runtime_config_settings({"tools.exec.sandbox_launcher": ""}, config_path=path)
    assert load_config(path).tools.exec.sandbox == "bwrap"


@pytest.mark.skipif(os.name != "posix", reason="Unix protected launcher permissions")
def test_sandbox_environment_paths_are_validated_without_rewriting_references(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("SANDBOX_TEST_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("SANDBOX_TEST_LAUNCHER", "/bin/sh")
    monkeypatch.delenv("SANDBOX_TEST_MISSING_PROVIDER", raising=False)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": "${SANDBOX_TEST_WORKSPACE}"}},
        "providers": {"openai": {"apiKey": "${SANDBOX_TEST_MISSING_PROVIDER}"}},
        "tools": {"exec": {"sandbox": "bwrap"}},
    })
    save_config(config, path)
    payload = update_runtime_config_settings({
        "tools.exec.sandbox_launcher": "${SANDBOX_TEST_LAUNCHER}",
    }, config_path=path)
    saved = load_config(path)
    assert payload["runtime_config"]["tools.exec.sandbox_launcher"] == "${SANDBOX_TEST_LAUNCHER}"
    assert saved.tools.exec.sandbox_launcher == "${SANDBOX_TEST_LAUNCHER}"
    assert saved.agents.defaults.workspace == "${SANDBOX_TEST_WORKSPACE}"
    assert saved.providers.openai.api_key == "${SANDBOX_TEST_MISSING_PROVIDER}"
    before = path.read_bytes()
    monkeypatch.setenv("SANDBOX_TEST_RW_BIND", str(Path("/bin/sh").resolve().parent))
    with pytest.raises(WebUISettingsError, match="sandbox_launcher_agent_writable"):
        update_runtime_config_settings({
            "tools.exec.sandbox_rw_binds": ["${SANDBOX_TEST_RW_BIND}"],
        }, config_path=path)
    assert path.read_bytes() == before
    monkeypatch.delenv("SANDBOX_TEST_LAUNCHER")
    with pytest.raises(WebUISettingsError, match="sandbox_launcher_path_unavailable"):
        update_runtime_config_settings({
            "tools.exec.sandbox_launcher": "${SANDBOX_TEST_LAUNCHER}",
        }, config_path=path)
    assert path.read_bytes() == before


def test_enabling_web_proxy_without_an_address_preserves_config(tmp_path):
    path = tmp_path / "config.json"
    save_config(Config(), path)
    before = path.read_bytes()
    with pytest.raises(WebUISettingsError, match="Advanced"):
        update_runtime_config_settings({"tools.web.use_proxy": True}, config_path=path)
    assert path.read_bytes() == before


def test_runtime_patch_persists_and_preserves_other_domains(tmp_path):
    path = tmp_path / "config.json"
    config = Config.model_validate({
        "providers": {"openai": {"apiKey": "${PRIVATE_API_KEY}"}},
        "channels": {"telegram": {"token": "keep-me", "streaming": False}},
    })
    save_config(config, path)
    payload = update_runtime_config_settings({
        "agents.defaults.max_tool_iterations": 71,
        "agents.defaults.timezone_mode": "manual",
        "agents.defaults.timezone": "Asia/Shanghai",
        "tools.exec.timeout": 0,
        "tools.exec.allowed_env_keys": ["MY_TOKEN"],
        "tools.image_generation.save_dir": "images",
    }, local_browser=True, config_path=path)
    saved = load_config(path)
    assert saved.agents.defaults.max_tool_iterations == 71
    assert saved.agents.defaults.timezone == "Asia/Shanghai"
    assert saved.tools.exec.timeout == 0
    assert saved.tools.exec.allowed_env_keys == ["MY_TOKEN"]
    assert saved.providers.openai.api_key == "${PRIVATE_API_KEY}"
    assert saved.channels.telegram == {"token": "keep-me", "streaming": False}
    assert payload["requires_restart"]
    assert payload["runtime_config"]["tools.image_generation.save_dir"] == "images"
    assert "PRIVATE_API_KEY" not in json.dumps(runtime_config_payload(saved))


@pytest.mark.parametrize("values", [
    {"channels.telegram.token": "bad"},
    {"agents.defaults.workspace": "/not-editable-from-settings"},
    {"providers.openai.api_key": "bad"},
    {"tools.exec": {"enable": False}},
    {"agents.defaults.max_concurrent_subagents": 0},
    {"agents.defaults.max_tool_iterations": True},
    {"agents.defaults.session_ttl_minutes": -1},
    {"agents.defaults.session_ttl_minutes": 1.5},
    {"agents.defaults.session_ttl_minutes": 15.0},
    {"agents.defaults.session_ttl_minutes": "15"},
    {"agents.defaults.session_ttl_minutes": True},
    {"agents.defaults.session_ttl_minutes": False},
    {"agents.defaults.session_ttl_minutes": None},
    {"tools.exec.timeout": -1},
    {"tools.exec.timeout": 86401},
    {"tools.exec.timeout": "60"},
    {"tools.exec.enable": "false"},
    {"tools.exec.allowed_env_keys": "TOKEN"},
    {"tools.exec.allowed_env_keys": [3]},
    {"tools.exec.deny_patterns": ["["]},
    {"tools.ssrf_whitelist": ["not-a-network"]},
    {"tools.exec.sandbox": "unknown"},
    {"agents.defaults.dream.cron": "not a cron"},
    {"agents.defaults.dream.model_override": "missing"},
    {"agents.defaults.compactor.model_override": "unknown"},
    {"agents.defaults.compactor.model_override": "default"},
    {"agents.defaults.compactor.model_override": 3},
    {"agents.defaults.compactor.model_override": ["compactor"]},
    {"agents.defaults.compactor.next_turn_input_reserve_tokens": -1},
    {"agents.defaults.compactor.next_turn_input_reserve_tokens": 1.5},
    {"agents.defaults.compactor.next_turn_input_reserve_tokens": "512"},
    {"agents.defaults.compactor.next_turn_input_reserve_tokens": True},
    {"agents.defaults.compactor.min_verbatim_tail_tokens": -1},
    {"agents.defaults.compactor.min_verbatim_tail_tokens": 1.5},
    {"agents.defaults.compactor.min_verbatim_tail_tokens": "90000"},
    {"agents.defaults.compactor.min_verbatim_tail_tokens": True},
    {"agents.defaults.timezone_mode": "manual", "agents.defaults.timezone": "Mars/Olympus"},
    {"gateway.port": 65536},
    {"gateway.heartbeat.interval_s": 0},
    {"tools.image_generation.save_dir": ""},
    {"tools.image_generation.save_dir": "../outside"},
    {"tools.image_generation.save_dir": "C:/outside"},
    {"tools.image_generation.save_dir": "/outside"},
])
def test_invalid_patch_is_atomic(tmp_path, values):
    path = tmp_path / "config.json"
    save_config(Config(), path)
    before = path.read_bytes()
    with pytest.raises(WebUISettingsError):
        update_runtime_config_settings({
            "agents.defaults.max_tool_iterations": 71, **values,
        }, local_browser=True, config_path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("limit", [120, 86400, 172800])
def test_shell_runtime_editor_uses_the_configured_limit(tmp_path, limit):
    path = tmp_path / "config.json"
    config = Config.model_validate({"tools": {"exec": {"maxRuntimeLimit": limit}}})
    save_config(config, path)
    before = path.read_bytes()

    with pytest.raises(WebUISettingsError, match=f"1–{limit} seconds"):
        update_runtime_config_settings({"tools.exec.timeout": limit + 1}, config_path=path)
    assert path.read_bytes() == before

    payload = update_runtime_config_settings({"tools.exec.timeout": limit}, config_path=path)
    assert payload["advanced"]["exec_max_runtime_limit"] == limit
    assert payload["runtime_config"]["tools.exec.timeout"] == limit
    saved = load_config(path).tools.exec
    assert saved.max_runtime == limit
    assert saved.max_runtime_limit == limit


def test_other_settings_preserve_a_longer_trusted_shell_runtime(tmp_path):
    path = tmp_path / "config.json"
    save_config(Config.model_validate({"tools": {"exec": {
        "maxRuntime": 172800, "maxRuntimeLimit": 86400,
    }}}), path)

    update_runtime_config_settings({"tools.exec.path_append": "extra-bin"}, config_path=path)

    saved = load_config(path).tools.exec
    assert saved.max_runtime == 172800
    assert saved.max_runtime_limit == 86400
    assert saved.path_append == "extra-bin"


def test_remote_client_cannot_change_install_policy():
    config = Config()
    with pytest.raises(WebUISettingsError) as error:
        update_runtime_config(config, {
            "tools.webui_allow_remote_package_install": True,
        }, local_browser=False)
    assert error.value.status == 403
    assert not config.tools.webui_allow_remote_package_install
    assert update_runtime_config(config, {
        "tools.webui_allow_remote_package_install": True,
    }, local_browser=True)


def test_timezone_auto_and_nullable_fields():
    config = Config.model_validate({"agents": {"defaults": {
        "timezoneMode": "manual", "timezone": "UTC", "dream": {"cron": "0 2 * * *"},
    }}})
    assert update_runtime_config(config, {
        "agents.defaults.timezone_mode": "auto",
        "network.proxy": None,
    }, local_browser=True)
    assert config.agents.defaults.timezone_mode == "auto"
    assert config.agents.defaults.dream.cron == "0 2 * * *"
    assert not update_runtime_config(config, {}, local_browser=True)


def test_runtime_payload_matches_shared_catalogue_contract():
    fixture = Path(__file__).with_name("fixtures") / "runtime-config-keys.json"
    paths = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(paths) == len(set(paths))
    assert set(runtime_config_payload(Config())) == set(paths)


def test_disabling_memory_consolidation_preserves_other_memory_settings():
    config = Config()
    before = config.agents.defaults.model_dump()
    assert update_runtime_config(config, {"agents.defaults.dream.enabled": False}, local_browser=True)
    after = config.agents.defaults.model_dump()
    before["dream"]["enabled"] = False
    assert after == before


@pytest.mark.parametrize("backend", ["bwrap", "seatbelt"])
def test_sandbox_backend_persists_with_path_permissions(tmp_path, backend):
    path = tmp_path / "config.json"
    payload = update_runtime_config_settings({
        "tools.exec.sandbox": backend,
        "tools.exec.sandbox_ro_binds": ["shared/read"],
        "tools.exec.sandbox_rw_binds": ["shared/write"],
    }, local_browser=True, config_path=path)
    saved = load_config(path).tools.exec
    assert saved.sandbox == backend
    assert saved.sandbox_ro_binds == ["shared/read"]
    assert saved.sandbox_rw_binds == ["shared/write"]
    assert payload["runtime_config"]["tools.exec.sandbox"] == backend
    assert payload["requires_restart"]


def test_idle_compaction_settings_roundtrip_and_restart_contract(tmp_path):
    path = tmp_path / "config.json"
    config = Config()
    config.providers.openai.api_key = "${PRIVATE_API_KEY}"
    save_config(config, path)
    services = WebUISettingsServices.create(path)
    key = "agents.defaults.session_ttl_minutes"
    assert runtime_config_payload(config)[key] == 15

    for minutes in (0, 720, 721, 37, 15):
        payload = services.mutate(update_runtime_config_settings, {key: minutes})
        assert payload["runtime_config"][key] == minutes
        assert payload["requires_restart"] is True
        saved = load_config(path)
        assert saved.agents.defaults.session_ttl_minutes == minutes
        assert saved.agents.defaults.idle_compact_check_interval_seconds == 60
        assert saved.providers.openai.api_key == "${PRIVATE_API_KEY}"
        defaults = json.loads(path.read_text(encoding="utf-8"))["agents"]["defaults"]
        assert defaults["idleCompactAfterMinutes"] == minutes
        assert "sessionTtlMinutes" not in defaults
        assert "session_ttl_minutes" not in defaults

        unchanged = services.mutate(update_runtime_config_settings, {key: minutes})
        assert unchanged["runtime_config"][key] == minutes
        assert unchanged["requires_restart"] is False


def test_compactor_settings_roundtrip_unset_and_validation(tmp_path):
    path = tmp_path / "config.json"
    config = Config.model_validate({
        "modelPresets": {"compactor": {"model": "local-model", "provider": "custom"}},
    })
    save_config(config, path)
    model_key = "agents.defaults.compactor.model_override"
    reserve_key = "agents.defaults.compactor.next_turn_input_reserve_tokens"
    # Unset: session-model compaction with the default (no-reserve) sizing.
    assert runtime_config_payload(config)[model_key] is None
    assert runtime_config_payload(config)[reserve_key] == 0

    payload = update_runtime_config_settings(
        {model_key: "compactor", reserve_key: 512}, local_browser=True, config_path=path,
    )
    assert payload["runtime_config"][model_key] == "compactor"
    assert payload["runtime_config"][reserve_key] == 512
    assert payload["requires_restart"]
    saved = load_config(path)
    assert saved.agents.defaults.compactor.model_override == "compactor"
    assert saved.agents.defaults.compactor.next_turn_input_reserve_tokens == 512
    # The preset and the rest of the config survive the round trip.
    assert saved.model_presets["compactor"].model == "local-model"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["agents"]["defaults"]["compactor"]["modelOverride"] == "compactor"
    assert raw["agents"]["defaults"]["compactor"]["nextTurnInputReserveTokens"] == 512

    # Clearing the preset restores session-model compaction; the reserve stays.
    payload = update_runtime_config_settings(
        {model_key: None, reserve_key: 0}, local_browser=True, config_path=path,
    )
    assert payload["runtime_config"][model_key] is None
    assert payload["runtime_config"][reserve_key] == 0
    saved = load_config(path)
    assert saved.agents.defaults.compactor.model_override is None
    assert saved.agents.defaults.compactor.next_turn_input_reserve_tokens == 0


def test_min_verbatim_tail_floor_round_trip(tmp_path):
    path = tmp_path / "config.json"
    save_config(Config(), path)
    floor_key = "agents.defaults.compactor.min_verbatim_tail_tokens"
    # Unset: the default 90k floor applies.
    assert runtime_config_payload(Config())[floor_key] == 90_000

    payload = update_runtime_config_settings(
        {floor_key: 120_000}, local_browser=True, config_path=path,
    )
    assert payload["runtime_config"][floor_key] == 120_000
    saved = load_config(path)
    assert saved.agents.defaults.compactor.min_verbatim_tail_tokens == 120_000
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["agents"]["defaults"]["compactor"]["minVerbatimTailTokens"] == 120_000

    # 0 disables the floor (legacy full-prefix fallback).
    payload = update_runtime_config_settings(
        {floor_key: 0}, local_browser=True, config_path=path,
    )
    assert payload["runtime_config"][floor_key] == 0
    saved = load_config(path)
    assert saved.agents.defaults.compactor.min_verbatim_tail_tokens == 0
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["agents"]["defaults"]["compactor"]["minVerbatimTailTokens"] == 0


def test_idle_compaction_legacy_config_can_be_read_and_updated(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agents": {"defaults": {
        "sessionTtlMinutes": 720, "idleCompactCheckIntervalSeconds": 91,
    }}}), encoding="utf-8")
    key = "agents.defaults.session_ttl_minutes"
    assert runtime_config_payload(load_config(path))[key] == 720

    payload = update_runtime_config_settings({key: 0}, config_path=path)
    assert payload["runtime_config"][key] == 0
    assert payload["requires_restart"] is True
    saved = load_config(path)
    assert saved.agents.defaults.session_ttl_minutes == 0
    assert saved.agents.defaults.idle_compact_check_interval_seconds == 91
    defaults = json.loads(path.read_text(encoding="utf-8"))["agents"]["defaults"]
    assert defaults["idleCompactAfterMinutes"] == 0
    assert "sessionTtlMinutes" not in defaults
