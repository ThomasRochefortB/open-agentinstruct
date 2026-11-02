from pathlib import Path
import subprocess

import pytest

from scripts.check_release import validate_release


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Security Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    git(tmp_path, "add", "pyproject.toml")
    git(tmp_path, "commit", "-m", "initial")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return tmp_path


def test_matching_version_on_main_is_accepted(repository):
    validate_release(repository, "v0.1.0")


@pytest.mark.parametrize("tag", ["v0.2.0", "0.1.0", "v0.1.0-extra"])
def test_wrong_or_partial_tags_are_rejected(repository, tag):
    with pytest.raises(ValueError):
        validate_release(repository, tag)


def test_commit_outside_main_is_rejected(repository):
    git(repository, "switch", "-c", "other")
    (repository / "unreviewed.txt").write_text("changed")
    git(repository, "add", "unreviewed.txt")
    git(repository, "commit", "-m", "unreviewed")
    with pytest.raises(subprocess.CalledProcessError):
        validate_release(repository, "v0.1.0")
