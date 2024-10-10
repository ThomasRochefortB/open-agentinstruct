
from agents.base_agent import BaseAgent

def content_transformation_flow(text, content_agents):
    transformed_contents = []
    for config in content_agents:
        agent = BaseAgent(
            name=config['name'],
            system_prompt=config['system_prompt'],
            user_prompt_template=config['user_prompt_template']
        )
        agent_output = agent.process(text)
        if agent_output:
            transformed_contents.extend(agent_output)
        else:
            print(f"{agent.name}: Skipped due to irrelevant content.")
    return transformed_contents
