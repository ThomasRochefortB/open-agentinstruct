import openai
from utils.text_extraction import parse_modified_triple, parse_instruction_answer_pairs

def refine_instructions(instruction_answer_pairs, max_rounds=2):
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
        
        for pair in refined_pairs:
            instruction = pair['instruction']
            answer = pair['answer']
            context = pair.get('context', '')  # Retrieve the context if available
            
            for goal_number, goal in enumerate(goals, start=1):
                print(f"Applying Goal {goal_number} in Round {round_number + 1}: {goal}")
                
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
                try:
                    response_suggester = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt_suggester},
                            {"role": "user", "content": user_prompt_suggester}
                        ]
                    )
                    suggestions = response_suggester.choices[0].message.content
                except Exception as e:
                    print(f"Error with Suggester Agent: {e}")
                    continue
                
                # Step 2: Editor Agent modifies the passage, question, and answer based on suggestions
                system_prompt_editor = "You are an expert editor specializing in applying modifications to educational content based on provided suggestions."
                user_prompt_editor = f"""
Using the suggestions below, modify the (passage, question, answer) triple to achieve the goal. Make sure the modified content is coherent and accurate.

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
                try:
                    response_editor = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt_editor},
                            {"role": "user", "content": user_prompt_editor}
                        ]
                    )
                    modified_output = response_editor.choices[0].message.content
                except Exception as e:
                    print(f"Error with Editor Agent: {e}")
                    continue
                
                # Parse the modified output
                modified_pair = parse_modified_triple(modified_output)
                if modified_pair:
                    # Include additional metadata if needed
                    modified_pair['context'] = modified_pair['context']  # Modified passage
                    modified_pair['agent'] = f"Refinement Round {round_number + 1}, Goal {goal_number}"
                    
                    new_refined_pairs.append(modified_pair)
                else:
                    print("Failed to parse modified output.")
                    # Optionally handle parsing failures
                    
        # Update the refined pairs for the next round
        refined_pairs = new_refined_pairs.copy()
        
    return refined_pairs


