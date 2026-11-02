# Security and release operation

## Reviewed dependency set

Use Python 3.11–3.14. Python 3.9/3.10 support is retired so the project can use
patched, supported dependencies. Install the reviewed development set with:

```sh
uv sync --locked --all-extras --group dev
uv run --locked pytest
```

Runtime security floors are included in package metadata, rather than protecting
only lockfile consumers. The current SDK pin is `litellm==1.103.8`, an official
October 24 post-incident backport listed in the maintainer's incident notice.
Later LiteLLM 1.106.6 requires `filelock<4`, which resolves to a release affected
by known FileLock advisories. This backport does not impose that cap, allowing
patched FileLock 4.2.1 or later. The pin must be reviewed together with a strict
audit and completion-adapter tests before updating it. No advisory exclusions or
resolver overrides are used.

This project calls LiteLLM's `acompletion` SDK; it does not install the proxy
extra or launch a LiteLLM proxy. SDK dependency presence alone does not prove
that proxy-specific vulnerabilities are reachable.

The compromised `litellm==1.107.0` is excluded by the exact pin. A dependency fix
cannot determine whether an earlier environment ran compromised artifacts.
Review retained build/install logs and artifact metadata for the October 21
incident window from a clean environment. If that release executed, follow the
maintainer's incident-recovery guidance; updating dependencies alone does not
undo credential theft. No existing user environment was examined or modified.

Sources:
- [LiteLLM incident notice and post-incident backports](https://docs.litellm.ai/blog/security-incident-oct-21-2026)
- [PyPI metadata for the reviewed backport](https://pypi.org/pypi/litellm/1.103.8/json)
- [Example FileLock advisory fixed in 4.2.1](https://github.com/advisories/GHSA-8mx5-7m5m-r77g)

## PyPI publishing is blocked until setup is verified

The old tag workflow used `PYPI_API_TOKEN` and did not require tests. Its
replacement validates the exact tag/version and main ancestry, runs locked audits
and tests on Python 3.11–3.14, builds wheel and source artifacts, and smoke-tests
both. Only the final job receives OIDC permission. That job downloads the tested
artifact; it does not check out source or run the project's build scripts.

Registry-side trusted-publisher configuration cannot be verified from repository
source or public PyPI metadata. Publishing therefore fails explicitly until the
maintainer completes these steps:

1. Configure a PyPI trusted publisher for project `open-agentinstruct`, owner
   `ThomasRochefortB`, repository `open-agentinstruct`, workflow `publish.yml`,
   environment `pypi`.
2. Configure the GitHub `pypi` environment to allow only release tags and require
   an eligible reviewer. For a solo-maintained project, choose reviewers and
   self-review restrictions that can actually be satisfied.
3. Protect `main` with the audit/test and distribution checks and protect version
   tags. Restrict who may create or move release tags.
4. Verify that the PyPI binding exactly matches the workflow identity and
   environment. Then set repository variable
   `PYPI_TRUSTED_PUBLISHING_READY=true`. This flag records an administrative
   prerequisite; it does not itself verify PyPI or enforce reviewer policy.
5. Bump the package version, merge the reviewed change, and create the matching
   `v<version>` tag on a commit belonging to `main`. The published 0.1.0 artifacts
   cannot be overwritten.
6. After successful trusted publication, revoke the old PyPI API token and remove
   `PYPI_API_TOKEN` from GitHub. The replacement workflow never reads that token.

No registry binding, environment rule, token revocation, repository protection or
release is changed by this source patch.

[PyPI trusted-publisher setup documentation](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)

## Remaining administrative coverage

Enable Dependabot alerts/security updates, repository secret scanning and push
protection, and private vulnerability reporting. The checked-in Dependabot and
CodeQL configurations add ongoing automation but cannot enable every
administrative setting or reconstruct historical credential exposure.
