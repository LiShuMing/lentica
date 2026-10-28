import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import agent
import image_analyzer


def test_dry_run_never_creates_client_or_captures_screen(monkeypatch):
    fail = Mock(side_effect=AssertionError("No model/screen side effects in dry run"))
    monkeypatch.setattr(agent, "create_client", fail)
    monkeypatch.setattr(agent, "take_screenshot", fail)
    monkeypatch.setattr(image_analyzer, "create_client", fail)
    assert agent.main(["inspect screen"]) == 0
    assert image_analyzer.main(["screenshot"]) == 0
    fail.assert_not_called()


def test_import_and_help_work_with_no_display_or_credentials():
    directory = Path(agent.__file__).parent
    for filename in ("agent.py", "image_analyzer.py"):
        result = subprocess.run([sys.executable, str(directory / filename), "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--execute" in result.stdout


def test_action_requires_approval_and_validates_before_gui():
    gui = Mock()
    with pytest.raises(PermissionError):
        agent.execute_action({"action": "click", "params": {"x": 1, "y": 2}}, gui=gui)
    agent.execute_action({"action": "click", "params": {"x": 1, "y": 2}}, approved=True, gui=gui)
    gui.click.assert_called_once_with(1, 2)
    with pytest.raises(ValueError):
        agent.execute_action({"action": "click", "params": {"x": -1, "y": 2}}, approved=True, gui=gui)
    with pytest.raises(ValueError):
        agent.validate_action({"action": "type", "params": {"text": "中文"}})


def test_explicit_gui_run_stops_on_rejected_action(monkeypatch):
    monkeypatch.setattr(agent, "create_client", lambda env: (Mock(), "fixture"))
    monkeypatch.setattr(agent, "take_screenshot", lambda: b"synthetic")
    monkeypatch.setattr(agent, "get_action", lambda *a: {"action": "click", "params": {"x": 1, "y": 2}})
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    execute = Mock()
    monkeypatch.setattr(agent, "execute_action", execute)
    assert agent.main(["inspect screen", "--execute"]) == 0
    execute.assert_not_called()
