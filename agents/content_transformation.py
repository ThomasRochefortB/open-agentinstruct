import asyncio
import random


async def process_with_agent(
    agent_name, system_prompt, user_prompt_template, text, async_chat_completion
):
    # Additional instruction to be appended
    additional_instruction = "\n\nIf the provided text has no relevant content to your task, output an empty string."

    # Modify the system prompt
    modified_system_prompt = system_prompt + additional_instruction

    # Format the user prompt with the provided text
    user_prompt = user_prompt_template.format(text=text)

    try:
        # Use the async_chat_completion function instead of directly calling the OpenAI API
        content = await async_chat_completion(
            system_prompt=modified_system_prompt, user_prompt=user_prompt
        )

        # Check if the output is empty
        if content == "":
            print(f"{agent_name}: No relevant content found. Skipping.")
            return None

        return {"type": agent_name.lower().replace(" ", "_"), "content": content}

    except Exception as e:
        print(f"Error processing with {agent_name}: {e}")
        return None


async def content_transformation_flow(
    text, content_agents, async_chat_completion, debug=False
):
    # Limit to one agent if debug mode is enabled
    if debug:
        agents_to_use = content_agents[:1]
    else:
        agents_to_use = random.sample(content_agents, min(3, len(content_agents)))

    # Create a list of asyncio tasks for each agent
    tasks = [
        process_with_agent(
            config["name"],
            config["system_prompt"],
            config["user_prompt_template"],
            text,
            async_chat_completion,
        )
        for config in agents_to_use
    ]

    # Run all tasks concurrently using asyncio.gather
    transformed_contents = await asyncio.gather(*tasks)

    # Filter out any None results (agents that returned no content)
    transformed_contents = [
        content for content in transformed_contents if content is not None
    ]

    return transformed_contents
