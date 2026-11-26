"""Hugging Face Hub helpers: search GGUF models, list quantizations, download with progress."""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path

import httpx

# HF_ENDPOINT lets you use a mirror (e.g. https://hf-mirror.com) — same variable huggingface_hub uses.
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

QUANT_RE = re.compile(
    r"(IQ\d_[A-Z_]+|IQ\d|Q\d_K_[SMLX]{1,2}|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32|FP16|FP8|MXFP4|UD-[A-Z0-9_]+)",
    re.IGNORECASE)
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
PARAM_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[xX]?(\d+(?:\.\d+)?)?[bB](?![a-z])")
ACTIVE_RE = re.compile(r"[-_]A(\d+(?:\.\d+)?)B", re.IGNORECASE)

# Bits per weight for estimating parameter count from file size when the name lacks it.
BPW = {"Q2": 2.7, "IQ1": 1.8, "IQ2": 2.4, "Q3": 3.6, "IQ3": 3.3, "Q4": 4.8, "IQ4": 4.5,
       "Q5": 5.6, "Q6": 6.6, "Q8": 8.5, "F16": 16, "BF16": 16, "FP16": 16, "F32": 32,
       "MXFP4": 4.3, "FP8": 8}


def _headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _client(**kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=_headers(), follow_redirects=True, timeout=30, **kw)


def guess_params(text: str) -> float | None:
    """'GLM-4.5-Air-106B' -> 106.0 ; 'Mixtral-8x7B' -> 56.0 (billions)."""
    best = None
    for m in PARAM_RE.finditer(text):
        a, b = float(m.group(1)), m.group(2)
        val = a * float(b) if b else a
        if 0.05 <= val <= 5000:
            best = val if best is None else max(best, val)
    return best


def guess_active(text: str) -> float | None:
    m = ACTIVE_RE.search(text)
    return float(m.group(1)) if m else None


async def search(query: str, limit: int = 30) -> list[dict]:
    params = {"search": query, "filter": "gguf", "sort": "downloads", "direction": "-1",
              "limit": str(limit)}
    async with _client() as c:
        r = await c.get(f"{HF}/api/models", params=params)
        r.raise_for_status()
        out = []
        for m in r.json():
            mid = m.get("id") or m.get("modelId")
            out.append({"id": mid, "downloads": m.get("downloads", 0), "likes": m.get("likes", 0),
                        "updated": m.get("lastModified"), "params_b": guess_params(mid or ""),
                        "active_b": guess_active(mid or "")})
        return out


def quant_of(path: str) -> str:
    name = path.split("/")[-1]
    found = QUANT_RE.findall(name)
    if found:
        return found[-1].upper()
    folder = path.split("/")[0] if "/" in path else ""
    found = QUANT_RE.findall(folder)
    return found[-1].upper() if found else "unknown"


def _bpw(quant: str) -> float:
    q = quant.upper().replace("UD-", "")
    for key in sorted(BPW, key=len, reverse=True):
        if q.startswith(key):
            return BPW[key]
    return 5.0


async def files(model_id: str) -> dict:
    """Return the GGUF variants of a repo, with split files grouped together."""
    async with _client() as c:
        r = await c.get(f"{HF}/api/models/{model_id}/tree/main", params={"recursive": "true"})
        r.raise_for_status()
        tree = r.json()
    groups: dict[str, dict] = {}
    for f in tree:
        if f.get("type") != "file" or not f["path"].lower().endswith(".gguf"):
            continue
        path = f["path"]
        size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
        if "mmproj" in path.lower():
            continue
        m = SPLIT_RE.search(path)
        key = SPLIT_RE.sub("", path) if m else path
        g = groups.setdefault(key, {"key": key, "quant": quant_of(path), "files": [], "size": 0})
        g["files"].append(path)
        g["size"] += size
    variants = sorted(groups.values(), key=lambda g: g["size"])
    for v in variants:
        v["files"].sort()
        v["size_gb"] = round(v["size"] / 1024 ** 3, 2)
        v["primary"] = v["files"][0]
        v["est_params_b"] = round(v["size"] * 8 / _bpw(v["quant"]) / 1e9, 1)
    return {"id": model_id, "params_b": guess_params(model_id), "active_b": guess_active(model_id),
            "variants": variants}


class Downloader:
    """Background downloads with progress, resumable via HTTP Range."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.jobs: dict[str, dict] = {}

    def local_models(self) -> list[dict]:
        out = []
        for p in sorted(self.models_dir.rglob("*.gguf")):
            m = SPLIT_RE.search(p.name)
            if m and m.group(1) != "00001":
                continue
            if p.name.endswith(".part"):
                continue
            parts = [p]
            if m:
                parts = sorted(p.parent.glob(SPLIT_RE.sub("", p.name) + "-*-of-*.gguf"))
            size = sum(x.stat().st_size for x in parts)
            out.append({"path": str(p), "name": str(p.relative_to(self.models_dir)),
                        "size_gb": round(size / 1024 ** 3, 3), "parts": len(parts),
                        "quant": quant_of(p.name)})
        return out

    def start(self, model_id: str, file_list: list[str], total: int) -> str:
        job_id = f"{model_id}:{file_list[0]}"
        if job_id in self.jobs and self.jobs[job_id]["state"] == "running":
            return job_id
        self.jobs[job_id] = {"id": job_id, "model": model_id, "files": file_list, "total": total,
                             "done": 0, "state": "running", "speed_mbs": 0, "error": None,
                             "started": time.time(), "path": None}
        asyncio.create_task(self._run(job_id))
        return job_id

    def cancel(self, job_id: str):
        if job_id in self.jobs:
            self.jobs[job_id]["state"] = "cancelled"

    async def _run(self, job_id: str):
        job = self.jobs[job_id]
        target_dir = self.models_dir / job["model"].replace("/", "__")
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(headers=_headers(), follow_redirects=True,
                                         timeout=httpx.Timeout(60, read=300)) as c:
                done_before = 0
                for rel in job["files"]:
                    dest = target_dir / rel.split("/")[-1]
                    if job["path"] is None:
                        job["path"] = str(dest)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    if dest.exists():
                        done_before += dest.stat().st_size
                        job["done"] = done_before
                        continue
                    have = tmp.stat().st_size if tmp.exists() else 0
                    headers = {"Range": f"bytes={have}-"} if have else {}
                    url = f"{HF}/{job['model']}/resolve/main/{rel}"
                    async with c.stream("GET", url, headers=headers) as r:
                        if r.status_code == 401 or r.status_code == 403:
                            raise RuntimeError("Model is gated/private: set HF_TOKEN and accept "
                                               "the licence on huggingface.co")
                        r.raise_for_status()
                        if r.status_code != 206:
                            have = 0
                        mode = "ab" if have else "wb"
                        last_t, last_b = time.time(), 0
                        with open(tmp, mode) as fh:
                            written = have
                            async for chunk in r.aiter_bytes(1 << 20):
                                if job["state"] == "cancelled":
                                    return
                                fh.write(chunk)
                                written += len(chunk)
                                job["done"] = done_before + written
                                now = time.time()
                                if now - last_t > 1:
                                    job["speed_mbs"] = round((written - last_b) / (now - last_t)
                                                             / 1e6, 1)
                                    last_t, last_b = now, written
                    tmp.rename(dest)
                    done_before += dest.stat().st_size
            job["state"] = "done"
            job["done"] = job["total"] or job["done"]
        except Exception as e:  # noqa: BLE001
            job["state"] = "error"
            job["error"] = str(e)
