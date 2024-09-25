from utils.text_extraction import parse_instruction_answer_pairs
import openai

def generate_instructions(transformed_contents):
    instruction_answer_pairs = []
    for item in transformed_contents:
        # Use LLM to generate instruction-answer pairs
        system_prompt = "You are an expert educator in medicinal chemistry, skilled at creating diverse and challenging instructional tasks for graduate-level students, along with their answers."
        
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
                        Based on the following content only, generate three diverse instruction-answer pairs that can be used to create problem-solving tasks for graduate-level students in medicinal chemistry. 
                        Ensure that the instructions cover different instruction types from the provided taxonomy and encourage critical thinking.
                        Ensure that the answer is derived directly from the provided content.

                        For each pair, provide:

                        Instruction:
                        [The instruction]

                        Answer:
                        [The answer based on the content]

                        Content:
                        {item['content']}

                        {instruction_taxonomy}

                        Provide the instruction-answer pairs in the format specified.
                        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        generated_pairs = response.choices[0].message.content

        # Parse the generated instruction-answer pairs
        pairs = parse_instruction_answer_pairs(generated_pairs)
        instruction_answer_pairs.extend(pairs)
    return instruction_answer_pairs
