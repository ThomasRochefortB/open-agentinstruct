import openai

def content_transformation_flow(text, content_agents):
    transformed_contents = []
    for config in content_agents:
        agent_name = config['name']
        system_prompt = config['system_prompt']
        user_prompt_template = config['user_prompt_template']

        # Additional instruction to be appended
        additional_instruction = "\n\nIf the provided text has no relevant content to your task, output an empty string."

        # Modify the system prompt
        modified_system_prompt = system_prompt + additional_instruction

        # Format the user prompt with the provided text
        user_prompt = user_prompt_template.format(text=text)

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
                print(f"{agent_name}: No relevant content found. Skipping.")
                continue

            transformed_contents.append({
                'type': agent_name.lower().replace(' ', '_'),
                'content': content
            })

        except Exception as e:
            print(f"Error processing with {agent_name}: {e}")
            continue

    return transformed_contents
