# Multimodal Agent Example

A small screenshot -> VLM action -> human confirmation -> GUI loop for learning.
It is not part of the knowledge product. Spec:
[0001](../../specs/0001-agentic-workspace/README.md).

Activate a suitable existing environment and install `requirements.txt` only when
using this example. `mss`/`pyautogui` need a working desktop/display; mock tests and
help need neither a display nor an API SDK. Imports never load a dotenv or create
a model client.

```bash
python examples/multimodal-agent/agent.py 'Open an application'  # plan only
python examples/multimodal-agent/image_analyzer.py sample.png 'Describe the visible text'
```

Actual analysis requires `--execute`; configure `LLM_BASE_URL`, `LLM_API_KEY`, and
`VL_MODEL`, or select an explicit dotenv using `--env-file /path/to/private.env`.
The selected endpoint may be local or remote. `--execute` permits the image or
screen to be sent there. The GUI loop asks for confirmation before each action;
model output is validated before execution. Non-ASCII typing is rejected by this
minimal example rather than pretending pyautogui supports it reliably.

```bash
python examples/multimodal-agent/image_analyzer.py screenshot --execute
python examples/multimodal-agent/agent.py 'Inspect the application' --execute --max-steps 3
python -m pytest examples/multimodal-agent/tests
```

Keep screenshots, API transcripts, credentials, and task outputs private. The
migration tests do not capture a screen or interact with a real desktop.
