import asyncio
import openai
from utils.text_extraction import parse_modified_triple
from openai import AsyncOpenAI

async def refine_instructions(instruction_answer_pairs, max_rounds=2):
    refined_pairs = instruction_answer_pairs.copy()
    
    # Define the refinement goals
    goals = [
        "Modify the passage to make the question unanswerable.",
        "Modify the passage to alter the answer, if possible, in the opposite direction.",
        "Modify the question or answer choices to make them more complex."
    ]
    
    for round_number in range(max_rounds):
        print(f"Refinement Round {round_number + 1}")
        new_refined_pairs = []
        tasks = []
        
        for pair in refined_pairs:
            # Schedule the refinement of each pair concurrently
            tasks.append(refine_pair(pair, goals, round_number))
        
        # Wait for all tasks in this round to complete
        round_results = await asyncio.gather(*tasks)
        
        # Collect the results
        for result in round_results:
            new_refined_pairs.extend(result)
        
        # Update refined_pairs for the next round
        refined_pairs = new_refined_pairs.copy()
    
    return refined_pairs


async def refine_pair(pair, goals, round_number):
    tasks = []
    for goal_number, goal in enumerate(goals, start=1):
        tasks.append(refine_with_goal(pair, goal, goal_number, round_number))
    # Run all goals concurrently for this pair
    results = await asyncio.gather(*tasks)
    # Filter out any None results due to errors
    return [res for res in results if res is not None]


async def refine_with_goal(pair, goal, goal_number, round_number):
    instruction = pair['instruction']
    answer = pair['answer']
    context = pair.get('context', '')
        
    # Step 1: Suggester Agent suggests modifications
    system_prompt_suggester = "You are an expert in educational content refinement, specializing in creating suggestions to achieve specific modification goals."
    user_prompt_suggester = f"""
Based on the following (passage, question, answer) triple, provide detailed suggestions to achieve the specified goal.

Passage:
{context}

Question:
{instruction}

Answer:
{answer}

Goal:
{goal}

Provide your suggestions as a numbered list.
"""
    suggestions = await async_chat_completion(system_prompt_suggester, user_prompt_suggester)
    if suggestions is None:
        return None
    
    # Step 2: Editor Agent modifies the content based on suggestions
    system_prompt_editor = "You are an expert editor specializing in applying modifications to educational content based on provided suggestions."
    user_prompt_editor = f"""
Using the suggestions below, modify the (passage, question, answer) triple to achieve the goal. Ensure the modified content is coherent and accurate and that it follows the original format.

Original Passage:
{context}

Original Question:
{instruction}

Original Answer:
{answer}

Suggestions:
{suggestions}

Provide the modified content in the following format:

Modified Passage:
[Your modified passage]

Modified Question:
[Your modified question]

Modified Answer:
[Your modified answer]
"""
    modified_output = await async_chat_completion(system_prompt_editor, user_prompt_editor)
    if modified_output is None:
        return None
    
    # Parse the modified output
    # After parsing the modified output
    modified_pair = parse_modified_triple(modified_output)
    if modified_pair:
        modified_pair['context'] = modified_pair['context']  # Modified passage
        modified_pair['agent'] = f"Refinement Round {round_number + 1}, Goal {goal_number}"
        return modified_pair
    else:
        print("Failed to parse modified output.")
        # print("Modified Output:")
        # print(modified_output)
        return None



async def async_chat_completion(system_prompt, user_prompt, model="gpt-4o-mini", max_retries=10):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(5)  # Adjust the number based on your rate limit
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

