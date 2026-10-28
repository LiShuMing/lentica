"""Explicit image/screen analysis; default commands only describe the request."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from vlm_client import create_client


def take_screenshot() -> bytes:
    import mss
    from PIL import Image

    with mss.MSS() as capture:
        screenshot = capture.grab(capture.monitors[0])
        image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def prepare_image(path: str) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1280, 1280))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def analyze_bytes(image: bytes, question: str, client, model: str) -> str:
    encoded = base64.b64encode(image).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image path, or 'screenshot' for current screen")
    parser.add_argument("question", nargs="?", default="Describe the visible content and text.")
    parser.add_argument("--env-file")
    parser.add_argument("--execute", action="store_true", help="Allow image capture/read and provider submission")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "source": args.image,
                    "question": args.question,
                    "note": "--execute explicitly permits image/screen content to reach the configured model",
                }
            )
        )
        return 0
    client, model = create_client(args.env_file)
    image = take_screenshot() if args.image == "screenshot" else prepare_image(str(Path(args.image).expanduser()))
    print(analyze_bytes(image, args.question, client, model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
