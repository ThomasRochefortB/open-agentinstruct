import asyncio
from typing import List, Dict, Optional
from open_agentinstruct.utils.text_extraction import (
    parse_modified_output,
)


async def refine_instructions(
    instruction_answer_pairs: List[Dict],
    async_chat_completion,
    max_rounds: int = 2,
    min_refinements: int = 1,
    goals: Optional[List[str]] = None,
) -> List[Dict]:
    # Validate input
    if not instruction_answer_pairs:
        print(
            "No instruction-answer pairs provided for refinement. Skipping refinement process."
        )
        return []

    refined_pairs = [
        pair
        for pair in instruction_answer_pairs
        if pair.get("instruction") and pair.get("answer")
    ]

    if not refined_pairs:
        print(
            "No valid instruction-answer pairs found after filtering. Skipping refinement process."
        )
        return []

    print(f"Starting refinement process with {len(refined_pairs)} valid pairs")

    # Default goals if none provided
    if goals is None:
        goals = [
            "Make the question more challenging while maintaining its core concept",
            "Add relevant context to make the question more practical and applied",
            "Increase the precision and specificity of both the question and answer",
        ]

    for round_number in range(max_rounds):
        if len(refined_pairs) < min_refinements:
            print(
                f"Insufficient pairs ({len(refined_pairs)}) to continue refinement. Stopping."
            )
            break

        print(
            f"Refinement Round {round_number + 1} starting with {len(refined_pairs)} pairs"
        )
        new_refined_pairs = []
        tasks = []

        try:
            for pair in refined_pairs:
                tasks.append(
                    refine_pair(pair, goals, round_number, async_chat_completion)
                )

            round_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in round_results:
                if isinstance(result, Exception):
                    print(f"Error in refinement: {str(result)}")
                    continue
                if result:
                    new_refined_pairs.extend(result)

            if not new_refined_pairs:
                print(
                    f"Round {round_number + 1} produced no successful refinements. Stopping refinement process."
                )
                break

            print(
                f"Round {round_number + 1} completed with {len(new_refined_pairs)} new pairs"
            )
            refined_pairs = new_refined_pairs.copy()

        except Exception as e:
            print(f"Error in refinement round {round_number + 1}: {str(e)}")
            break

    print(f"Refinement process completed with {len(refined_pairs)} final pairs")
    return refined_pairs


async def refine_pair(pair, goals, round_number, async_chat_completion):
    if not pair.get("instruction") or not pair.get("answer"):
        print(f"Invalid pair received in refine_pair: {pair}")
        return []

    tasks = []
    for goal_number, goal in enumerate(goals, start=1):
        tasks.append(
            refine_with_goal(
                pair, goal, goal_number, round_number, async_chat_completion
            )
        )
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [
            res
            for res in results
            if res is not None
            and not isinstance(res, Exception)
            and res.get("instruction")
            and res.get("answer")
        ]
        if not valid_results:
            print(
                f"No valid refinements produced for pair: {pair.get('instruction')[:50]}..."
            )
        return valid_results
    except Exception as e:
        print(f"Error in refine_pair: {str(e)}")
        return []


async def refine_with_goal(
    pair, goal, goal_number, round_number, async_chat_completion
):
    try:
        instruction = pair["instruction"]
        answer = pair["answer"]

        system_prompt_suggester = """You are an expert in educational content refinement,
specializing in creating suggestions to achieve specific modification goals while maintaining the original format."""

        user_prompt_suggester = f"""
Based on the following instruction-answer pair, provide detailed suggestions to achieve the specified goal.
Maintain the exact same format as the original in your suggestions.

Original instruction:
{instruction}

Original answer:
{answer}

Goal:
{goal}

Provide your suggestions as a numbered list.
"""
        suggestions = await async_chat_completion(
            system_prompt_suggester, user_prompt_suggester
        )
        if not suggestions:
            print(f"No suggestions generated for goal {goal_number}")
            return None

        system_prompt_editor = """You are an expert editor specializing in applying modifications to educational content.
Your output must maintain the exact same format as the original instruction and answer pair."""

        user_prompt_editor = f"""
Using the suggestions below, modify the instruction and answer to achieve the goal.
Your response must follow this exact format:

Modified Instruction:
[Your modified instruction here]

Modified Answer:
[Your modified answer here]

Original instruction:
{instruction}

Original answer:
{answer}

Suggestions:
{suggestions}

Remember to maintain the same format as the original content (including multiple choice options if present) but prefix with "Modified Instruction:" and "Modified Answer:"."""

        modified_output = await async_chat_completion(
            system_prompt_editor, user_prompt_editor
        )
        if not modified_output:
            print(f"No modified output generated for goal {goal_number}")
            return None

        result = parse_modified_output(modified_output)

        if result:
            result["agent"] = f"Refinement Round {round_number + 1}, Goal {goal_number}"
            # Preserve the transformed content information from the original pair
            result["transformed_content"] = pair.get("transformed_content")
            result["transformation_type"] = pair.get("transformation_type")
            result["original_text"] = pair.get("original_text")
            # Preserve the original instruction agent name
            result["instruction_agent"] = pair.get("instruction_agent")
            return result

        print(f"Failed to parse modified output for goal {goal_number}")
        return None

    except Exception as e:
        print(f"Error in refine_with_goal: {str(e)}")
        return None
