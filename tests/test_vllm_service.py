import json
from pathlib import Path

import pytest

from conftest import load_module


service = load_module("vllm_service_test", "services/vllm/scripts/service.py")


def test_config_drives_argv_without_model_runtime(capsys):
    assert service.main(["plan"]) == 0
    data = json.loads(capsys.readouterr().out)
    argv = data["argv"]
    assert argv[:2] == ["vllm", "serve"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--max-num-seqs") + 1] == "8"


def test_external_environment_and_model_path_are_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("LENTICA_VLLM_EXECUTABLE", "/selected/env/bin/vllm")
    monkeypatch.setenv("LENTICA_QWEN_MODEL", str(tmp_path / "model"))
    config = service.load_config(service.DEFAULT_CONFIG)
    assert service.command(config)[:3] == ["/selected/env/bin/vllm", "serve", str(tmp_path / "model")]


def test_remote_binding_is_rejected(monkeypatch):
    monkeypatch.setenv("LENTICA_VLLM_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        service.load_config(service.DEFAULT_CONFIG)


def test_missing_weights_fail_before_start(monkeypatch):
    config = service.load_config(service.DEFAULT_CONFIG)
    config["model_path"] = "/nonexistent/lentica-test-model"
    monkeypatch.setattr(service.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not start a process"))
    with pytest.raises(ValueError, match="Provision"):
        service.start(config)


def test_incomplete_config_returns_readable_error(tmp_path, capsys):
    config = tmp_path / "incomplete.toml"
    config.write_text('host = "127.0.0.1"\n')
    assert service.main(["plan", "--config", str(config)]) == 1
    assert "Missing service configuration" in capsys.readouterr().err
