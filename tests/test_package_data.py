from pathlib import Path

from open_agentinstruct.cli import get_default_agent_config_path, setup_parser
from open_agentinstruct.utils.agent_utils import load_agent_configs


def test_bundled_agent_configs_are_available_without_pkg_resources():
    root = get_default_agent_config_path()
    assert Path(root).is_dir()
    content, instructions, example = load_agent_configs(root, "coding")
    assert content and instructions


def test_cli_help_parses_without_contacting_a_provider(capsys):
    parser = setup_parser()
    assert "dataset" in parser.format_help().lower()
