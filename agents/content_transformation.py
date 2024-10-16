import openai
import asyncio
from openai import AsyncOpenAI  # Ensure this is installed or use the right async client
import random
async def process_with_agent(agent_name, system_prompt, user_prompt_template, text):
    # Additional instruction to be appended
    additional_instruction = "\n\nIf the provided text has no relevant content to your task, output an empty string."
    
    # Modify the system prompt
    modified_system_prompt = system_prompt + additional_instruction
    
    # Format the user prompt with the provided text
    user_prompt = user_prompt_template.format(text=text)

    try:
        # Use the async_chat_completion function instead of directly calling the OpenAI API
        content = await async_chat_completion(
            system_prompt=modified_system_prompt,
            user_prompt=user_prompt
        )

        # Check if the output is empty
        if content == "":
            print(f"{agent_name}: No relevant content found. Skipping.")
            return None

        return {
            'type': agent_name.lower().replace(' ', '_'),
            'content': content
        }

    except Exception as e:
        print(f"Error processing with {agent_name}: {e}")
        return None


async def content_transformation_flow(text, content_agents, debug=False):
    # Limit to one agent if debug mode is enabled
    if debug:
        agents_to_use = content_agents[:1]
    else:
        agents_to_use = random.sample(content_agents, min(3, len(content_agents)))

    # Create a list of asyncio tasks for each agent
    tasks = [
        process_with_agent(config['name'], config['system_prompt'], config['user_prompt_template'], text)
        for config in agents_to_use
    ]

    # Run all tasks concurrently using asyncio.gather
    transformed_contents = await asyncio.gather(*tasks)

    # Filter out any None results (agents that returned no content)
    transformed_contents = [content for content in transformed_contents if content is not None]

    return transformed_contents



async def async_chat_completion(system_prompt, user_prompt, model="gpt-4o-mini", max_retries=3):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(10)  # Adjust the number based on your rate limit
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

