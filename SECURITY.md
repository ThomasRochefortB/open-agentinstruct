# Security

Security fixes are developed on `main`. The published 0.1.0 release predates the
security hardening in this branch; a new version must be built, tested and
published before those fixes reach PyPI users.

Report suspected vulnerabilities privately through the repository's security
advisory reporting feature when enabled, or contact the maintainer through their
GitHub profile to arrange private disclosure. Do not include API keys or private
datasets in public issues.

Use Python 3.11–3.14 and the committed `uv.lock` for development and reproducible
installs. CI audits all locked runtime, test and development dependencies before
installing them. A passing audit covers known package advisories for its current
platform, not every code path or deployed environment.

See [the security and publishing guide](docs/security.md) for the LiteLLM pin,
incident exposure checks, and required release configuration.
