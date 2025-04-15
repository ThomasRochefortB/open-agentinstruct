import asyncio
from open_agentinstruct.utils.text_extraction import parse_instruction_answer_pairs
import random
import re


async def process_with_instruction_agent(
    agent_config,
    transformed_content,
    original_text,
    one_shot_example,
    async_chat_completion,
):  # Add original_text parameter
    agent_name = agent_config["name"]
    system_prompt = agent_config["system_prompt"]
    user_prompt_template = agent_config["user_prompt_template"]

    # Format the user prompt with the text and one-shot example
    # The irrelevant content instruction is already added in construct_user_prompt
    user_prompt = construct_user_prompt(
        user_prompt_template, transformed_content["content"], one_shot_example
    )

    try:
        generated_pairs = await async_chat_completion(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        # Check if the output is "IRRELEVANT" or contains just that word with whitespace
        if (
            not generated_pairs
            or generated_pairs.strip() == ""
            or re.match(r"^\s*IRRELEVANT\s*$", generated_pairs, re.IGNORECASE)
        ):
            print(f"{agent_name}: No relevant content found. Skipping.")
            return []

        # Parse the generated instruction-answer pairs
        pairs = parse_instruction_answer_pairs(generated_pairs)

        if not pairs:
            print(f"No instruction-answer pair found for agent {agent_name}.")
            return []

        # Add agent name and transformed content info to each pair
        for pair in pairs:
            pair["agent"] = agent_name
            pair["instruction_agent"] = agent_name
            pair["transformed_content"] = transformed_content["content"]
            pair["transformation_type"] = transformed_content["type"]
            pair["original_text"] = original_text  # Now we have access to original_text

        return pairs

    except Exception as e:
        print(f"Error generating instructions with {agent_name}: {e}")
        return []


def construct_user_prompt(user_prompt_template, text, one_shot_example):
    """
    Constructs the user prompt by inserting the text and one-shot example into the template.
    """
    # Add instruction about making questions self-contained with proper context
    self_contained_instruction = (
        "The provided content that you see will not be passed to the next user. Therefore, it is important that the instruction is self-contained with all information needed to complete the task based on the specified format for its type."
        "Reflect on whether the task requires the content to be included directly in the instruction. "
        "For tasks like text extraction or text modification, include the relevant content directly after the main instruction, marked with the '**TEXT**' tag. "
        "For Retrieval-Augmented Generation (RAG) tasks, the instruction should clearly state the query and provide the necessary retrieved documents/contextual information separately within the instruction field. "
        "Ensure the instruction is self-contained with all information needed to complete the task based on the specified format for its type."
    )

    # Add instruction for handling irrelevant content
    irrelevant_instruction = (
        "If the content is not relevant or suitable for generating a meaningful instruction in your domain, "
        "output only the word 'IRRELEVANT' without any additional text. Skip content that doesn't fit your expertise area."
    )

    # Add instruction for conciseness
    conciseness_instruction = (
        "Aim for conciseness in the generated instruction/question."
    )

    if one_shot_example:
        # Format the one-shot example to show the expected format
        formatted_one_shot_example = (
            "Here is an example of the expected format:\n\n"
            "```\n"
            f"{one_shot_example['instruction']}\n"
            f"{one_shot_example['answer']}\n"
            "```\n\n"
            "Please follow this format strictly to generate a new instruction based on the provided content.\n\n"
            f"{self_contained_instruction}\n\n"
            f"{conciseness_instruction}\n\n"
            f"{irrelevant_instruction}"
        )

        # Insert the text and the formatted example into the template
        user_prompt = user_prompt_template.format(
            text=text,
        )
        user_prompt = user_prompt + "\n\n" + formatted_one_shot_example
    else:
        # Insert the text into the template with additional instructions
        user_prompt = (
            user_prompt_template.format(text=text)
            + f"\n\n{self_contained_instruction}"
            + f"\n\n{conciseness_instruction}"
            + f"\n\n{irrelevant_instruction}"
        )

    return user_prompt


async def generate_instructions(
    transformed_contents,
    instruction_agents,
    one_shot_example,
    async_chat_completion,
    original_text,  # Add original_text parameter
    debug=False,
):
    instruction_answer_pairs = []

    # Limit to one agent if debug mode is enabled
    if debug:
        agents_to_use = instruction_agents[:1]
    else:
        agents_to_use = random.sample(
            instruction_agents, min(3, len(instruction_agents))
        )

    # Create a list of asyncio tasks for each transformed content and agent
    tasks = [
        process_with_instruction_agent(
            agent_config,
            item,
            original_text,  # Pass original_text through
            one_shot_example,
            async_chat_completion,
        )
        for item in transformed_contents
        for agent_config in agents_to_use
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Flatten the results and collect all instruction-answer pairs
    for result in results:
        instruction_answer_pairs.extend(result)

    return instruction_answer_pairs
