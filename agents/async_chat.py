from litellm import acompletion
import asyncio
import random

async def async_chat_completion(
    system_prompt, user_prompt, model="gpt-4o-mini", max_retries=3
):
    semaphore = asyncio.Semaphore(10)
    async with semaphore:
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = await acompletion(
                    model=model, messages=messages, temperature=0.0
                )
                content = response["choices"][0]["message"]["content"]
                return content.strip()
            except Exception as e:
                base_wait = 2**attempt
                jitter = random.uniform(0, base_wait * 0.1)  # 10% jitter
                wait_time = base_wait + jitter
                print(f"Error during API call: {e}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        print("Max retries exceeded.")
        return None