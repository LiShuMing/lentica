# Backend

Follow root AGENTS.md and the linked `workspace.toml` spec. Keep `backend.app.*`
imports/API routes compatible. Product runtime remains local-only; optional cloud
apps and GUI examples are not backend dependencies. Test with synthetic temporary
libraries and databases. Do not run a user's full library or change queue settings
as part of a code check. Run backend pytest plus the workspace checker.
