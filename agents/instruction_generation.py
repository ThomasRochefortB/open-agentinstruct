import json
import openai
from utils.text_extraction import parse_instruction_answer_pairs


def generate_instructions(transformed_contents, instruction_agents, debug=False):
    instruction_answer_pairs = []
    # Limit to one agent if debug mode is enabled
    agents_to_use = instruction_agents[:1] if debug else instruction_agents

    for item in transformed_contents:
        context = item['content']  # Get the transformed content
        for agent_config in agents_to_use:
            # Extract prompts from the configuration
            system_prompt = agent_config['system_prompt']
            user_prompt_template = agent_config['user_prompt_template']
            agent_name = agent_config['name']

            # Format the user prompt with the content
            user_prompt = user_prompt_template.format(
                text=context
            )

            # Use OpenAI API to generate instruction-answer pairs
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                generated_pairs = response.choices[0].message.content
                # Parse the generated instruction-answer pair
                pairs = parse_instruction_answer_pairs(generated_pairs)

                if not pairs:
                    print(f"No instruction-answer pair found for agent {agent_name}.")
                    continue

                # Add agent name and context to each pair
                for pair in pairs:
                    pair['agent'] = agent_name
                    pair['context'] = context  # Add the context

                instruction_answer_pairs.extend(pairs)
            except Exception as e:
                print(f"Error generating instructions with {agent_name}: {e}")
                continue
    return instruction_answer_pairs



