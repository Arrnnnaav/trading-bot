import asyncio
import logging

_LOG = logging.getLogger(__name__)


class ClaudeCodeClient:
    def __init__(self, max_concurrent: int = 3):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def call(
        self,
        prompt: str,
        system: str = "",
        model: str = "claude-haiku-4-5-20251001",
        timeout: float = 30.0,
    ) -> str:
        args = [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--bare",
        ]
        if system:
            args += ["--system-prompt", system]

        proc = None
        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                if proc.returncode != 0:
                    _LOG.warning(
                        "claude exited %d: %s",
                        proc.returncode,
                        stderr.decode("utf-8", errors="replace")[:200],
                    )
                    return ""
                return stdout.decode("utf-8").strip()
            except (asyncio.TimeoutError, Exception):
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                return ""
