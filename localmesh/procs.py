"""Start/stop llama.cpp processes (rpc-server on workers, llama-server on the main node)."""
from __future__ import annotations

import asyncio
import collections
import re
import time

BUFFER_RE = re.compile(r"(\S+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB")


class ManagedProcess:
    def __init__(self, kind: str):
        self.kind = kind
        self.proc: asyncio.subprocess.Process | None = None
        self.cmd: list[str] = []
        self.logs: collections.deque[str] = collections.deque(maxlen=400)
        self.started: float | None = None
        self.exit_code: int | None = None
        self.meta: dict = {}
        self.buffers: dict = {}   # {"MTL0": {"model": MiB, "KV": MiB, "compute": MiB}, ...}

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, cmd: list[str], meta: dict | None = None):
        await self.stop()
        self.cmd, self.meta = cmd, meta or {}
        self.logs.clear()
        self.buffers = {}
        self.logs.append("$ " + " ".join(cmd))
        self.exit_code = None
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.started = time.time()
        asyncio.create_task(self._pump())

    async def _pump(self):
        assert self.proc and self.proc.stdout
        async for line in self.proc.stdout:
            text = line.decode(errors="replace").rstrip()
            self.logs.append(text)
            m = BUFFER_RE.search(text)
            if m:  # llama.cpp reports how much memory each device got for weights/KV/compute
                dev = m.group(1).replace("_Mapped", "").replace("_Host", "")
                self.buffers.setdefault(dev, {})
                self.buffers[dev][m.group(2)] = self.buffers[dev].get(m.group(2), 0) + float(m.group(3))
        self.exit_code = await self.proc.wait()
        self.logs.append(f"[process exited with code {self.exit_code}]")

    async def stop(self):
        if self.running:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 8)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.proc = None

    def status(self) -> dict:
        return {"kind": self.kind, "running": self.running, "cmd": self.cmd,
                "started": self.started, "exit_code": self.exit_code, "meta": self.meta,
                "pid": self.proc.pid if self.running else None, "buffers": self.buffers,
                "logs": list(self.logs)[-120:]}
