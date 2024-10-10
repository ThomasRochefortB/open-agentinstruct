import openai
class BaseAgent:
    def __init__(self, name, system_prompt, user_prompt_template):
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def process(self, text):
        # Additional instruction to be appended
        additional_instruction = "\n\nIf the provided text has no relevant content to your task, output an empty string."

        # Modify the system prompt or user prompt
        modified_system_prompt = self.system_prompt + additional_instruction

        # Format the user prompt with the provided text
        user_prompt = self.user_prompt_template.format(text=text)

        # Use OpenAI API to process the text
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Replace with your desired model
                messages=[
                    {"role": "system", "content": modified_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = response.choices[0].message.content.strip()

            # Check if the output is empty
            if content == "":
                print(f"{self.name}: No relevant content found. Skipping.")
                return []

            transformed_contents = [{
                'type': self.name.lower().replace(' ', '_'),
                'content': content
            }]
            return transformed_contents
        except Exception as e:
            print(f"Error processing with {self.name}: {e}")
            return []
