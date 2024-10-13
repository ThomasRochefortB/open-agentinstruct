import json

def load_agent_configs(task_type):
    # Load content generation agents
    with open('agents/content_gen_agents.json', 'r') as f:
        content_generation_configs = json.load(f)

    content_agents = content_generation_configs.get(task_type, [])
    
    # Load instruction generation agents
    with open('agents/instruction_gen_agents.json', 'r') as f:
        instruction_generation_configs = json.load(f)

    instruction_agents = instruction_generation_configs.get(task_type, [])

    return content_agents, instruction_agents
