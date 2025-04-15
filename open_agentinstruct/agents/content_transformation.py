import asyncio
import random
import re


async def process_with_agent(
    agent_name, system_prompt, user_prompt_template, text, async_chat_completion
):
    # Format the user prompt with the provided text
    user_prompt = user_prompt_template.format(text=text)

    # Add irrelevance instruction
    irrelevance_instruction = (
        "\\n\\nCritically evaluate if the provided text is relevant and suitable for your specific transformation task. "
        "If you have any doubts about the relevance of the text, classify it as IRRELEVANT and output ONLY the word 'IRRELEVANT' and nothing else."
    )
    user_prompt += irrelevance_instruction

    try:
        # Use the async_chat_completion function instead of directly calling the OpenAI API
        content = await async_chat_completion(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        # Check if the output is "IRRELEVANT" or contains just that word with whitespace
        if not content or re.match(r"^\s*IRRELEVANT\s*$", content, re.IGNORECASE):
            print(f"{agent_name}: Marked content as irrelevant.")
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
