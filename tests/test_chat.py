import asyncio
from unittest.mock import AsyncMock, patch

from open_agentinstruct.agents.async_chat import async_chat_completion


def test_completion_preserves_prompts_and_returns_trimmed_content():
    complete = AsyncMock(return_value={"choices": [{"message": {"content": " answer "}}]})
    with patch("open_agentinstruct.agents.async_chat.acompletion", complete):
        result = asyncio.run(async_chat_completion("system", "user"))
    assert result == "answer"
    assert complete.call_args.kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert complete.call_args.kwargs["model"] == "gpt-4o-mini"


def test_hosted_vllm_forwards_the_configured_base_url(monkeypatch):
    monkeypatch.setenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    complete = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    with patch("open_agentinstruct.agents.async_chat.acompletion", complete):
        asyncio.run(async_chat_completion("system", "user", model="hosted_vllm/model"))
    assert complete.call_args.kwargs["api_base"] == "http://127.0.0.1:8000/v1"


def test_retries_transient_errors_without_real_waits():
    complete = AsyncMock(side_effect=[RuntimeError("transient"), {"choices": [{"message": {"content": "ok"}}]}])
    with patch("open_agentinstruct.agents.async_chat.acompletion", complete), patch(
        "open_agentinstruct.agents.async_chat.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        assert asyncio.run(async_chat_completion("system", "user")) == "ok"
    assert complete.call_count == 2
    sleep.assert_awaited_once()
