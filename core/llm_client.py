import asyncio


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
            "--output-format",
            "text",
            "--model",
            model,
            "--bare",
        ]
        if system:
            args += ["--system-prompt", system]

        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                if proc.returncode != 0:
                    return ""
                return stdout.decode().strip()
            except (asyncio.TimeoutError, Exception):
                try:
                    proc.kill()
                except Exception:
                    pass
                return ""
