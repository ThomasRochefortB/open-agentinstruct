"""Reject release tags that do not identify the tested package version on main."""
from pathlib import Path
import os
import subprocess
import tomllib


def validate_release(root: Path, tag: str) -> None:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if tag != f"v{version}":
        raise ValueError("Release tag must exactly match the package version")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main"],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    validate_release(Path(__file__).resolve().parents[1], os.environ["RELEASE_TAG"])
