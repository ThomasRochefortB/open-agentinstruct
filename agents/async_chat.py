from litellm import acompletion
import asyncio

async def async_chat_completion(system_prompt, user_prompt, model="gpt-4o-mini", max_retries=3):
    semaphore = asyncio.Semaphore(10)  # Adjust based on your concurrency needs
    async with semaphore:
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = await acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.0
                )
                content = response['choices'][0]['message']['content']
                return content.strip()
            except Exception as e:
                wait_time = (2 ** attempt) * 1  # Exponential backoff
                print(f"Error during API call: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        print("Max retries exceeded.")
        return None
