import json


def load_agent_configs(content_agent_path, instruction_agent_path, task_type):
    with open(content_agent_path, "r") as f:
        content_generation_configs = json.load(f)

    content_agents = content_generation_configs.get(task_type, [])

    with open(instruction_agent_path, "r") as f:
        instruction_generation_configs = json.load(f)

    # Get the task-specific configuration
    task_config = instruction_generation_configs.get(task_type, {})
    instruction_agents = task_config.get("agents", [])
    one_shot_example = task_config.get("one_shot_example", None)

    return content_agents, instruction_agents, one_shot_example
