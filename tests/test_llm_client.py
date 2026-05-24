import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_call_returns_stdout():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"LONG signal confirmed.\n", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt", model="claude-haiku-4-5-20251001")

    assert result == "LONG signal confirmed."


@pytest.mark.asyncio
async def test_call_returns_empty_on_nonzero_exit():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt")

    assert result == ""


@pytest.mark.asyncio
async def test_call_returns_empty_on_timeout():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient()

    async def slow_communicate():
        await asyncio.sleep(999)
        return (b"", b"")

    mock_proc = MagicMock()
    mock_proc.communicate = slow_communicate
    mock_proc.kill = MagicMock()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await client.call("test prompt", timeout=0.01)

    assert result == ""
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient(max_concurrent=2)

    active = []
    peak = []

    async def fake_communicate():
        active.append(1)
        peak.append(len(active))
        await asyncio.sleep(0.05)
        active.pop()
        return (b"ok", b"")

    mock_proc = MagicMock()
    mock_proc.communicate = fake_communicate
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        await asyncio.gather(*[client.call(f"prompt {i}") for i in range(5)])

    assert max(peak) <= 2


@pytest.mark.asyncio
async def test_system_prompt_passed_as_flag():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"result", b""))
    mock_proc.returncode = 0

    captured_args = []

    async def capture(*args, **kwargs):
        captured_args.extend(args)
        return mock_proc

    with patch("asyncio.create_subprocess_exec", capture):
        await client.call("my prompt", system="you are a trader")

    assert "--system-prompt" in captured_args
    assert "you are a trader" in captured_args


@pytest.mark.asyncio
async def test_no_system_prompt_when_empty():
    from core.llm_client import ClaudeCodeClient

    client = ClaudeCodeClient()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"result", b""))
    mock_proc.returncode = 0

    captured_args = []

    async def capture(*args, **kwargs):
        captured_args.extend(args)
        return mock_proc

    with patch("asyncio.create_subprocess_exec", capture):
        await client.call("my prompt")  # no system arg

    assert "--system-prompt" not in captured_args
