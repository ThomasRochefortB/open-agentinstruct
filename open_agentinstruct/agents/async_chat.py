from litellm import acompletion
import asyncio
import random
import os


async def async_chat_completion(
    system_prompt, user_prompt, model="gpt-4o-mini", max_retries=3
):
    semaphore = asyncio.Semaphore(10)
    async with semaphore:
        api_base = None
        if model.startswith("hosted_vllm/"):
            api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000")
            print(f"Using VLLM API base: {api_base}")

        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.3,
                }
                if api_base:
                    completion_kwargs["api_base"] = api_base

                response = await acompletion(**completion_kwargs)
                content = response["choices"][0]["message"]["content"]
                return content.strip()
            except Exception as e:
                base_wait = 2**attempt
                jitter = random.uniform(0, base_wait * 0.1)  # 10% jitter
                wait_time = base_wait + jitter
                print(
                    f"Error during API call: {e}. Retrying in {wait_time:.2f} seconds..."
                )
                await asyncio.sleep(wait_time)
        print("Max retries exceeded.")
        return None
