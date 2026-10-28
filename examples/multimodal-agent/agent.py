"""Interactive screenshot -> vision action -> confirmation -> GUI learning loop."""

from __future__ import annotations

import argparse
import base64
import json
import time

from image_analyzer import take_screenshot
from vlm_client import create_client


SYSTEM_PROMPT = """Inspect the visible screen and propose one action for the user's task.
Return JSON: {"action": "click|double_click|type|scroll|hotkey|wait|done", "params": {...}}.
Click params: x,y integer pixels. Type: text. Scroll: x,y,direction(up/down),amount.
Hotkey: keys list. Wait: seconds. Done: summary. Never claim an unobserved outcome.
"""


def validate_action(action: dict) -> dict:
    if not isinstance(action, dict):
        raise ValueError("Action must be an object")
    kind, params = action.get("action"), action.get("params", {})
    if kind not in {"click", "double_click", "type", "scroll", "hotkey", "wait", "done"} or not isinstance(
        params, dict
    ):
        raise ValueError("Unknown action or invalid params")
    if kind in {"click", "double_click", "scroll"}:
        if any(type(params.get(key)) is not int or params[key] < 0 for key in ("x", "y")):
            raise ValueError("Coordinates must be nonnegative integers")
    if kind == "type" and (not isinstance(params.get("text"), str) or not params["text"].isascii()):
        raise ValueError("This minimal example supports ASCII typing only")
    if kind == "scroll" and (
        params.get("direction") not in {"up", "down"}
        or type(params.get("amount", 3)) is not int
        or not 1 <= params.get("amount", 3) <= 50
    ):
        raise ValueError("Invalid scroll direction/amount")
    if kind == "hotkey" and (
        not isinstance(params.get("keys"), list)
        or not 1 <= len(params["keys"]) <= 4
        or any(not isinstance(key, str) or not key for key in params["keys"])
    ):
        raise ValueError("Hotkey requires 1-4 key names")
    if kind == "wait" and (
        type(params.get("seconds", 1)) not in (int, float) or not 0 <= params.get("seconds", 1) <= 10
    ):
        raise ValueError("Wait must be between 0 and 10 seconds")
    return {"action": kind, "params": params}


def get_action(screenshot: bytes, task: str, history: list[dict], client, model: str) -> dict:
    encoded = base64.b64encode(screenshot).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            *history,
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}],
            },
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return validate_action(json.loads(text))


def execute_action(action: dict, *, approved: bool = False, gui=None) -> str:
    action = validate_action(action)
    if not approved:
        raise PermissionError("Interactive approval is required for each GUI action")
    kind, params = action["action"], action["params"]
    if kind == "done":
        return str(params.get("summary", "done"))
    if kind == "wait":
        time.sleep(params.get("seconds", 1))
        return "waited"
    if gui is None:
        import pyautogui as gui
    gui.FAILSAFE = True
    if kind in {"click", "double_click"}:
        function = gui.click if kind == "click" else gui.doubleClick
        function(params["x"], params["y"])
    elif kind == "type":
        gui.write(params["text"], interval=0.05)
    elif kind == "scroll":
        count = params.get("amount", 3) * (1 if params["direction"] == "up" else -1)
        gui.scroll(count, params["x"], params["y"])
    elif kind == "hotkey":
        gui.hotkey(*params["keys"])
    return f"executed {kind}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--env-file")
    parser.add_argument("--max-steps", type=int, choices=range(1, 21), default=20)
    parser.add_argument(
        "--execute", action="store_true", help="Allow screenshots/model calls and confirmed GUI actions"
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"dry_run": True, "task": args.task, "max_steps": args.max_steps}))
        return 0
    client, model = create_client(args.env_file)
    history = []
    for _step in range(args.max_steps):
        action = get_action(take_screenshot(), args.task, history, client, model)
        print(json.dumps(action, ensure_ascii=False))
        if action["action"] == "done":
            return 0
        if input("Execute this GUI action? [y/N] ").strip().lower() != "y":
            return 0
        result = execute_action(action, approved=True)
        history.extend([{"role": "assistant", "content": json.dumps(action)}, {"role": "user", "content": result}])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
