# Security

Lentica is an experimental single-user application. It exposes local file paths,
thumbnails, and task details to its UI and has no authentication or multi-tenant
authorization. Bind the API, frontend, and model services to loopback. Do not
publish these ports directly or deploy them as a public service.

Treat source files, SQLite databases, embeddings, face vectors, logs, and caches
as private data. `data/`, `.env`, and model artifacts are excluded from Git.
Model provisioning may access Hugging Face or model-provider download endpoints;
provision caches before using offline mode.

The optional media workflow and GUI example can transmit selected images/screens
to configured providers only when explicitly invoked. They are not imported or
enabled by the knowledge backend. Do not run real cloud generation or desktop
automation as a CI smoke check; use their mocked tests. Runtime artifacts in all
components, including nested data/logs/outputs directories, remain private.

Report vulnerabilities privately using GitHub's private vulnerability reporting
when available, or the maintainer's contact on their GitHub profile. Do not put
private data or live credentials in public issues.
