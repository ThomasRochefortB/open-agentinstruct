"""Run from an isolated environment against the built wheel or source artifact."""
from pathlib import Path
import subprocess

from open_agentinstruct.cli import get_default_agent_config_path
from open_agentinstruct.utils.agent_utils import load_agent_configs

root = get_default_agent_config_path()
assert Path(root).is_dir()
content, instructions, _ = load_agent_configs(root, "coding")
assert content and instructions
subprocess.run(["open-agentinstruct-generate", "--help"], check=True)
