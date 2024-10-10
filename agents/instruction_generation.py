import json
import openai
from utils.text_extraction import parse_instruction_answer_pairs


def generate_instructions(transformed_contents, instruction_agents):
    instruction_answer_pairs = []
    for item in transformed_contents:
        for agent_config in instruction_agents:
            # Extract prompts from the configuration
            system_prompt = agent_config['system_prompt']
            user_prompt_template = agent_config['user_prompt_template']
            agent_name = agent_config['name']

            # Format the user prompt with the content
            user_prompt = user_prompt_template.format(
                text=item['content']
            )

            # Use OpenAI API to generate instruction-answer pairs
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                generated_pairs = response.choices[0].message.content

                # Parse the generated instruction-answer pair
                pairs = parse_instruction_answer_pairs(generated_pairs)

                # Optionally, add agent name to each pair for tracking
                for pair in pairs:
                    pair['agent'] = agent_name

                instruction_answer_pairs.extend(pairs)
            except Exception as e:
                print(f"Error generating instructions with {agent_name}: {e}")
                continue
    return instruction_answer_pairs
