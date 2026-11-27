"""Demo mode: simulated devices, connection tests, downloads and deployments.

Everything produced here is labelled `simulated: true` and shown with a DEMO badge in the UI,
so it can never be mistaken for a real measurement.
"""
from __future__ import annotations

import asyncio
import random
import time


def _iface(name, label, kind, ip, speed):
    bcast = ip.rsplit(".", 1)[0] + ".255"
    return {"name": name, "label": label, "type": kind, "ip": ip, "netmask": "255.255.255.0",
            "broadcast": bcast, "link_local": ip.startswith("169.254"), "speed_mbps": speed}


DEMO_NODES = [
    {"id": "demo-studio", "name": "Mac Studio (demo)", "address": "192.168.50.21:7860",
     "info": {"hostname": "mac-studio", "os": "macOS 15.5", "arch": "arm64", "cpu": "Apple M2 Ultra",
              "cores": 24, "ram_total_gb": 192, "ram_available_gb": 170,
              "gpus": [{"vendor": "apple", "name": "Apple M2 Ultra GPU (Metal, unified memory)",
                        "vram_gb": None, "bandwidth_gbs": 800}],
              "usable_memory_gb": 151.6, "memory_kind": "apple-unified", "mem_bandwidth_gbs": 800,
              "disk_free_gb": 1400,
              "interfaces": [_iface("en0", "Ethernet", "ethernet", "192.168.50.21", 10000),
                             _iface("bridge0", "Thunderbolt Bridge", "thunderbolt",
                                    "169.254.10.21", 40000),
                             _iface("en1", "Wi-Fi", "wifi", "192.168.1.21", 1200)],
              "llama_cpp": {"rpc-server": "/opt/homebrew/bin/rpc-server",
                            "llama-server": "/opt/homebrew/bin/llama-server"}}},
    {"id": "demo-rtx", "name": "RTX 4090 Box (demo)", "address": "192.168.50.31:7860",
     "info": {"hostname": "rtx-box", "os": "Linux", "arch": "x86_64", "cpu": "AMD Ryzen 9 7950X",
              "cores": 32, "ram_total_gb": 128, "ram_available_gb": 110,
              "gpus": [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24,
                        "vram_used_gb": 0.4, "bandwidth_gbs": 1000}],
              "usable_memory_gb": 22.1, "memory_kind": "nvidia-vram", "mem_bandwidth_gbs": 1000,
              "disk_free_gb": 900,
              "interfaces": [_iface("enp5s0", "enp5s0", "ethernet", "192.168.50.31", 2500),
                             _iface("wlp6s0", "wlp6s0", "wifi", "192.168.1.31", 860)],
              "llama_cpp": {"rpc-server": "/usr/local/bin/rpc-server", "llama-server": None}}},
    {"id": "demo-ws", "name": "2x3090 Workstation (demo)", "address": "192.168.50.41:7860",
     "info": {"hostname": "workstation", "os": "Linux", "arch": "x86_64",
              "cpu": "Intel Xeon W-2295", "cores": 36, "ram_total_gb": 256,
              "ram_available_gb": 230,
              "gpus": [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24,
                        "vram_used_gb": 0.3, "bandwidth_gbs": 936},
                       {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24,
                        "vram_used_gb": 0.3, "bandwidth_gbs": 936}],
              "usable_memory_gb": 44.2, "memory_kind": "nvidia-vram", "mem_bandwidth_gbs": 936,
              "disk_free_gb": 2000,
              "interfaces": [_iface("eno1", "eno1", "ethernet", "192.168.50.41", 10000)],
              "llama_cpp": {"rpc-server": "/usr/local/bin/rpc-server",
                            "llama-server": "/usr/local/bin/llama-server"}}},
    {"id": "demo-air", "name": "MacBook Air M2 (demo)", "address": "192.168.1.51:7860",
     "info": {"hostname": "macbook-air", "os": "macOS 15.5", "arch": "arm64", "cpu": "Apple M2",
              "cores": 8, "ram_total_gb": 16, "ram_available_gb": 9,
              "gpus": [{"vendor": "apple", "name": "Apple M2 GPU (Metal, unified memory)",
                        "vram_gb": None, "bandwidth_gbs": 100}],
              "usable_memory_gb": 10.0, "memory_kind": "apple-unified", "mem_bandwidth_gbs": 100,
              "disk_free_gb": 120,
              "interfaces": [_iface("en0", "Wi-Fi", "wifi", "192.168.1.51", 866)],
              "llama_cpp": {"rpc-server": None, "llama-server": None}}},
]

DEMO_CATALOG = {
    "results": [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "downloads": 0, "likes": 0, "params_b": 0.5,
         "active_b": None},
        {"id": "unsloth/GLM-4.5-Air-GGUF", "downloads": 0, "likes": 0, "params_b": 106,
         "active_b": 12},
        {"id": "unsloth/GLM-5-GGUF", "downloads": 0, "likes": 0, "params_b": 744,
         "active_b": 40},
        {"id": "bartowski/Llama-3.3-70B-Instruct-GGUF", "downloads": 0, "likes": 0,
         "params_b": 70, "active_b": None},
    ],
    "files": {
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF": [("Q4_K_M", 0.49), ("Q8_0", 0.68), ("F16", 1.27)],
        "unsloth/GLM-4.5-Air-GGUF": [("Q2_K", 44), ("Q4_K_M", 68), ("Q8_0", 117)],
        "unsloth/GLM-5-GGUF": [("IQ2_XXS", 208), ("Q2_K", 262), ("Q4_K_M", 438), ("Q8_0", 792)],
        "bartowski/Llama-3.3-70B-Instruct-GGUF": [("Q2_K", 26.4), ("Q4_K_M", 42.5),
                                                  ("Q8_0", 75)],
    },
}


def demo_files(model_id: str) -> dict | None:
    rows = DEMO_CATALOG["files"].get(model_id)
    if not rows:
        return None
    base = model_id.split("/")[-1].replace("-GGUF", "")
    params = next((r["params_b"] for r in DEMO_CATALOG["results"] if r["id"] == model_id), None)
    active = next((r["active_b"] for r in DEMO_CATALOG["results"] if r["id"] == model_id), None)
    variants = []
    for q, gb in rows:
        parts = max(1, int(gb // 48) + (1 if gb % 48 else 0)) if gb > 48 else 1
        if parts > 1:
            files = [f"{q}/{base}-{q}-{i:05d}-of-{parts:05d}.gguf" for i in range(1, parts + 1)]
        else:
            files = [f"{base}-{q}.gguf"]
        variants.append({"key": files[0], "quant": q, "files": files, "size": int(gb * 1024 ** 3),
                         "size_gb": gb, "primary": files[0], "est_params_b": params})
    return {"id": model_id, "params_b": params, "active_b": active, "variants": variants,
            "simulated": True}


def simulate_link(a_iface: dict | None, b_iface: dict | None, kind: str) -> dict:
    speeds = [i.get("speed_mbps") or 1000 for i in (a_iface, b_iface) if i]
    link = min(speeds) if speeds else 1000
    if kind == "thunderbolt":
        lat, mbps = random.uniform(0.12, 0.25), min(link, 40000) * random.uniform(0.35, 0.45)
    elif kind == "ethernet":
        lat, mbps = random.uniform(0.25, 0.6), link * random.uniform(0.88, 0.95)
    elif kind == "wifi":
        lat, mbps = random.uniform(2.5, 9), link * random.uniform(0.22, 0.45)
    else:
        lat, mbps = random.uniform(1, 3), random.uniform(200, 600)
    return {"latency_ms": round(lat, 2), "jitter_ms": round(lat * random.uniform(0.05, 0.4), 2),
            "mbps": round(mbps, 0), "loss": 0, "simulated": True}


class DemoRuntime:
    """Fake downloads / processes / chat for simulated nodes."""

    def __init__(self):
        self.downloads: dict[str, dict] = {}
        self.procs: dict[str, dict] = {}

    def start_download(self, node_id: str, model: str, files: list[str], total: int) -> str:
        job_id = f"{model}:{files[0]}"
        job = {"id": job_id, "model": model, "files": files, "total": total, "done": 0,
               "state": "running", "speed_mbs": 0, "error": None, "started": time.time(),
               "path": f"~/localmesh-models/{model.replace('/', '__')}/{files[0].split('/')[-1]}",
               "simulated": True}
        self.downloads.setdefault(node_id, {})[job_id] = job
        asyncio.create_task(self._tick_download(job))
        return job_id

    async def _tick_download(self, job):
        # Simulated speed, scaled so even a 400 GB model finishes in ~40 s on stage.
        rate = max(job["total"] / 40, 80e6)
        while job["done"] < job["total"] and job["state"] == "running":
            await asyncio.sleep(0.5)
            job["done"] = min(job["total"], job["done"] + rate * 0.5)
            job["speed_mbs"] = round(rate / 1e6, 0)
        if job["state"] == "running":
            job["state"] = "done"

    def local_models(self, node_id: str) -> list[dict]:
        out = []
        for j in self.downloads.get(node_id, {}).values():
            if j["state"] == "done":
                out.append({"path": j["path"], "name": j["path"].split("/", 2)[-1],
                            "size_gb": round(j["total"] / 1024 ** 3, 2), "parts": len(j["files"]),
                            "quant": "", "simulated": True})
        return out

    def start_proc(self, node_id: str, kind: str, cmd: list[str], meta: dict):
        logs = ["$ " + " ".join(cmd)]
        if kind == "worker":
            logs += ["create_backend: using Metal/CUDA backend", f"Starting RPC server on {meta.get('endpoint')}",
                     "Accepted client connection"]
        else:
            logs += ["llama_model_loader: loaded meta data", "load_tensors: offloading layers to RPC + local devices",
                     "llama_context: KV cache ready", "main: server is listening on http://0.0.0.0:8080"]
        self.procs.setdefault(node_id, {})[kind] = {
            "kind": kind, "running": True, "cmd": cmd, "started": time.time(), "exit_code": None,
            "meta": meta, "logs": logs, "simulated": True}

    def stop_all(self, node_id: str):
        for p in self.procs.get(node_id, {}).values():
            p["running"] = False
            p["logs"].append("[stopped]")

    def proc_status(self, node_id: str) -> dict:
        procs = self.procs.get(node_id, {})
        main = procs.get("main")
        return {"worker": procs.get("worker"), "main": main,
                "server_ready": bool(main and main["running"])}


async def demo_chat_stream(messages: list[dict]):
    import json
    last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    text = ("[DEMO — simulated reply] The cluster pipeline is working end to end: your prompt "
            f"(\"{last[:60]}\") went to the main device, layers ran across the selected "
            "devices over RPC, and tokens are streaming back here. Turn demo mode off and deploy "
            "a real model to get real answers.")
    for word in text.split(" "):
        chunk = {"choices": [{"delta": {"content": word + " "}}]}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0.04)
    yield b"data: [DONE]\n\n"


def demo_stats(node_id: str, node: dict | None) -> dict:
    """Plausible, clearly simulated live numbers for a demo device."""
    info = (node or {}).get("info") or {}
    busy = random.uniform(35, 90)
    gpus = []
    for g in info.get("gpus", []):
        total = g.get("vram_gb") or info.get("ram_total_gb", 16)
        gpus.append({"name": g["name"], "util": round(busy + random.uniform(-10, 10), 0),
                     "mem_used_gb": round(total * random.uniform(0.55, 0.8), 1),
                     "mem_total_gb": total, "kind": "vram" if g.get("vram_gb") else "unified"})
    net = [{"name": i["name"], "label": i["label"], "type": i["type"], "ip": i["ip"],
            "rx_mbps": round(random.uniform(0.2, 3), 2), "tx_mbps": round(random.uniform(0.2, 3), 2)}
           for i in info.get("interfaces", [])[:1]]
    total = info.get("ram_total_gb", 16)
    return {"cpu_percent": round(random.uniform(8, 30), 1), "ram_total_gb": total,
            "ram_used_gb": round(total * random.uniform(0.4, 0.7), 1), "gpus": gpus, "net": net,
            "procs": {}, "buffers": {}, "t": time.time(), "simulated": True}
