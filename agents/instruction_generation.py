import json
import openai
import asyncio
from utils.text_extraction import parse_instruction_answer_pairs
from openai import AsyncOpenAI

import json
import asyncio
from utils.text_extraction import parse_instruction_answer_pairs
from openai import AsyncOpenAI  # Ensure this is installed or use the right async client
import random

async def process_with_instruction_agent(agent_config, context):
    agent_name = agent_config['name']
    system_prompt = agent_config['system_prompt']
    user_prompt_template = agent_config['user_prompt_template']

    # Format the user prompt with the context
    user_prompt = user_prompt_template.format(text=context)

    try:
        # Use the async chat completion function
        generated_pairs = await async_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if not generated_pairs:
            print(f"No instruction-answer pair found for agent {agent_name}.")
            return []

        # Parse the generated instruction-answer pairs
        pairs = parse_instruction_answer_pairs(generated_pairs)

        if not pairs:
            print(f"No instruction-answer pair found for agent {agent_name}.")
            return []

        # Add agent name and context to each pair
        for pair in pairs:
            pair['agent'] = agent_name
            pair['context'] = context  # Add the context from the original content

        return pairs

    except Exception as e:
        print(f"Error generating instructions with {agent_name}: {e}")
        return []


async def generate_instructions(transformed_contents, instruction_agents, debug=False):
    instruction_answer_pairs = []
    
    # Limit to one agent if debug mode is enabled
    if debug:
        agents_to_use = instruction_agents[:1]
    else:
        agents_to_use = random.sample(instruction_agents, min(3, len(instruction_agents)))

    # Create a list of asyncio tasks for each transformed content and agent
    tasks = [
        process_with_instruction_agent(agent_config, item['content'])
        for item in transformed_contents
        for agent_config in agents_to_use
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten the results and collect all instruction-answer pairs
    for result in results:
        instruction_answer_pairs.extend(result)

    return instruction_answer_pairs


async def async_chat_completion(system_prompt, user_prompt, model="gpt-4o-mini", max_retries=3):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(3)  # Adjust the number based on your rate limit
    async with semaphore:
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0
                )
                content = response.choices[0].message.content
                return content.strip()
            except openai.RateLimitError as e:
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Error during API call: {e}")
                return None
        print("Max retries exceeded.")
        return None
