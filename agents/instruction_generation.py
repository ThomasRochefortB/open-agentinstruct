import openai
import re
def generate_instructions(transformed_contents):
    instructions = []
    for item in transformed_contents:
        # Use LLM to generate instructions
        system_prompt = "You are an expert educator in medicinal chemistry, skilled at creating diverse and challenging instructional tasks for graduate-level students."
        
        # Define an instruction taxonomy
        instruction_taxonomy = """
Instruction Types:
1. Explanation tasks (e.g., "Explain the mechanism of...")
2. Analysis tasks (e.g., "Analyze the following reaction for...")
3. Problem-solving tasks (e.g., "Propose a synthetic route for...")
4. Critical thinking tasks (e.g., "Discuss the implications of...")
5. Comparison tasks (e.g., "Compare and contrast...")
6. Application tasks (e.g., "Apply the following principles to...")
7. Prediction tasks (e.g., "Predict the outcome of...")

Please use this taxonomy to generate the instructions.
"""
        user_prompt = f"""
Based on the following content, generate five diverse instructions that can be used to create problem-solving tasks for graduate-level students in medicinal chemistry. Ensure that the instructions cover different instruction types from the provided taxonomy and encourage critical thinking.

Content:
{item['content']}

{instruction_taxonomy}

Provide the instructions as a numbered list.
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                # ,temperature=0.7, max_tokens=500
            ]
        )

        generated_instructions = response.choices[0].message.content

        # Parse the generated instructions
        instruction_list = []
        lines = generated_instructions.strip().split('\n')
        for line in lines:
            match = re.match(r'\d+\.\s*(.*)', line)
            if match:
                instruction = match.group(1).strip()
                instruction_list.append(instruction)
            else:
                # Handle lines that are not numbered
                instruction_list.append(line.strip())
        
        instructions.extend(instruction_list)
    return instructions
