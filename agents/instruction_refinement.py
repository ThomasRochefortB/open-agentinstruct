import openai
from utils.text_extraction import parse_refined_instruction_answer

def refine_instructions(instruction_answer_pairs, max_rounds=2):
    refined_pairs = instruction_answer_pairs.copy()
    for round_number in range(max_rounds):
        print(f"Refinement Round {round_number + 1}")
        new_refined_pairs = []
        for pair in refined_pairs:
            instruction = pair['instruction']
            answer = pair['answer']
            
            # Step 1: Suggester Agent suggests modifications
            system_prompt_suggester = "You are an expert in educational pedagogy, specializing in enhancing instructional tasks to increase complexity and challenge."
            user_prompt_suggester = f"""
                                    Analyze the following instruction and suggest specific ways to increase its complexity and challenge level. Consider aspects such as introducing additional constraints, requiring deeper analysis, integrating multiple concepts, or adding real-world applications.

                                    Instruction:
                                    {instruction}

                                    Provide your suggestions as a numbered list.
                                    """

            response_suggester = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt_suggester},
                    {"role": "user", "content": user_prompt_suggester}
                ]
            )

            suggestions = response_suggester.choices[0].message.content

            # Step 2: Editor Agent modifies the instruction and answer based on suggestions
            system_prompt_editor = "You are an expert in medicinal chemistry and education, capable of refining instructional tasks and their answers based on provided suggestions."
            user_prompt_editor = f"""
                                Based on the following instruction, answer, and suggestions, rewrite the instruction to incorporate the suggestions and enhance its complexity and challenge. Update the answer accordingly to match the refined instruction.

                                Original Instruction:
                                {instruction}

                                Original Answer:
                                {answer}

                                Suggestions:
                                {suggestions}

                                Provide the refined instruction and answer in the following format:

                                Refined Instruction:
                                [Your refined instruction]

                                Refined Answer:
                                [Your refined answer]
                                """

            response_editor = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt_editor},
                    {"role": "user", "content": user_prompt_editor}
                ]
            )

            refined_output = response_editor.choices[0].message.content

            # Parse the refined instruction and answer
            refined_pair = parse_refined_instruction_answer(refined_output)
            if refined_pair:
                new_refined_pairs.append(refined_pair)
            else:
                # If parsing fails, keep the original pair
                new_refined_pairs.append(pair)
        
        # Update the refined pairs for the next round
        refined_pairs = new_refined_pairs
    return refined_pairs
