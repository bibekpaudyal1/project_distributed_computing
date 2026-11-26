"""A tiny stand-in for the Hugging Face Hub API, used by tests (HF_ENDPOINT=http://127.0.0.1:9900)."""
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()
REPO = "test-org/Tiny-1B-Instruct-GGUF"
FILES = {
    "Tiny-1B-Instruct-Q4_K_M.gguf": 3 * 1024 * 1024,
    "Q8_0/Tiny-1B-Instruct-Q8_0-00001-of-00002.gguf": 2 * 1024 * 1024,
    "Q8_0/Tiny-1B-Instruct-Q8_0-00002-of-00002.gguf": 2 * 1024 * 1024,
    "mmproj-F16.gguf": 1024,
    "README.md": 100,
}


@app.get("/api/models")
def models(search: str = ""):
    return [{"id": REPO, "downloads": 1234, "likes": 5, "lastModified": "2026-11-01"}]


@app.get("/api/models/{org}/{name}/tree/main")
def tree(org: str, name: str):
    return [{"type": "file", "path": p, "size": s, "lfs": {"size": s}} for p, s in FILES.items()]


@app.get("/{org}/{name}/resolve/main/{path:path}")
def resolve(org: str, name: str, path: str, request: Request):
    size = FILES[path]
    data = (b"GGUF" + bytes(range(256)) * (size // 256 + 1))[:size]
    rng = request.headers.get("range")
    if rng:
        start = int(rng.split("=")[1].split("-")[0])
        return Response(data[start:], status_code=206, media_type="application/octet-stream")
    return Response(data, media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 9900,
                log_level="warning")
