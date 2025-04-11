import json
import os


def load_agent_configs(base_path, task_type):
    # Construct paths for content and instruction files
    content_path = os.path.join(
        base_path, "content_generation", f"{task_type}_content.json"
    )
    instruction_path = os.path.join(
        base_path, "instruction_generation", f"{task_type}_instruction.json"
    )

    # Load content agents
    with open(content_path, "r") as f:
        content_agents = json.load(f)

    # Load instruction agents and one-shot example
    with open(instruction_path, "r") as f:
        instruction_config = json.load(f)
        instruction_agents = instruction_config.get("agents", [])
        one_shot_example = instruction_config.get("one_shot_example", None)

    return content_agents, instruction_agents, one_shot_example
