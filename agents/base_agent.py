import openai

class BaseAgent:
    def __init__(self, name, system_prompt, user_prompt_template):
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def process(self, text):
        # Format the user prompt with the provided text
        user_prompt = self.user_prompt_template.format(text=text)
        
        # Use OpenAI API to process the text
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        

        content  = response.choices[0].message.content

        transformed_contents = [{
            'type': self.name.lower().replace(' ', '_'),
            'content': content
        }]
        return transformed_contents
