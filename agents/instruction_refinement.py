import openai

def refine_instructions(instructions):
    refined_instructions = []
    for instr in instructions:
        # Step 1: Suggester Agent suggests modifications
        system_prompt_suggester = "You are an expert in educational pedagogy, specializing in enhancing instructional tasks to increase complexity and challenge."
        user_prompt_suggester = f"""
Analyze the following instruction and suggest specific ways to increase its complexity and challenge level. Consider aspects such as introducing additional constraints, requiring deeper analysis, or integrating multiple concepts.

Instruction:
{instr}

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

        # Step 2: Editor Agent modifies the instruction based on suggestions
        system_prompt_editor = "You are an expert in rewriting instructions to enhance their complexity and challenge level based on provided suggestions."
        user_prompt_editor = f"""
Based on the following instruction and suggestions, rewrite the instruction to incorporate the suggestions and enhance its complexity and challenge.

Original Instruction:
{instr}

Suggestions:
{suggestions}

Provide the refined instruction.
"""

        response_editor = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt_editor},
                {"role": "user", "content": user_prompt_editor}
            ]
        )

        refined_instr = response_editor.choices[0].message.content

        refined_instructions.append(refined_instr.strip())
    return refined_instructions
