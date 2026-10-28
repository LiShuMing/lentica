# vLLM Service

Follow root AGENTS.md and spec 0001. Source/config examples live here; environments,
weights, generated logs, and benchmarks stay private. Reuse an explicitly selected
vLLM environment. `service.py plan` must not import Torch/vLLM or start a server.
The launcher consumes TOML config and binds loopback; applications only receive
inference output, never database side effects. Test argv/health with mocks. Do not
restart or retune an existing service as part of a source migration.
