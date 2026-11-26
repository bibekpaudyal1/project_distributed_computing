"""LocalMesh AI node + controller.

Every device runs this same server. The one whose dashboard you open acts as the controller:
it talks to the other nodes over their HTTP API to test links, download models and deploy.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import demo, hardware, hf, optimizer, planner
from .discovery import Discovery
from .procs import ManagedProcess

VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
STATE_DIR = Path(os.environ.get("LOCALMESH_HOME", Path.home() / ".localmesh"))
TOKEN_HEADER = "X-LocalMesh-Token"
TYPE_PREF = ["thunderbolt", "ethernet", "wifi", "other", "vpn"]
BLOB = os.urandom(1 << 20)  # 1 MiB of random bytes for throughput tests


class Config:
    def __init__(self, port=7860, name=None, cluster="default", token="", models_dir=None,
                 demo_mode=False, discovery=True, rpc_port=50052, llama_port=8080):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        id_file = STATE_DIR / f"node_id_{port}"
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:10])
        self.node_id = id_file.read_text().strip()
        import socket
        self.name = name or socket.gethostname().split(".")[0]
        self.port, self.cluster, self.token = port, cluster, token
        self.models_dir = Path(models_dir or Path.home() / "localmesh-models").expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.demo = demo_mode
        self.discovery = discovery
        self.rpc_port, self.llama_port = rpc_port, llama_port


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="LocalMesh AI", version=VERSION)
    peers: dict[str, dict] = {}          # node_id -> {id,name,address,info,online,last_seen,manual}
    links: dict[str, dict] = {}          # "a|b" -> result of last connection test
    nettest_job: dict = {"state": "idle", "results": [], "total": 0, "done": 0}
    downloader = hf.Downloader(cfg.models_dir)
    worker = ManagedProcess("worker")
    main = ManagedProcess("main")
    demo_rt = demo.DemoRuntime()
    state = {"demo": cfg.demo, "deployment": None}
    discovery_ref: dict = {}

    # ------------------------------------------------------------------ helpers
    def auth_headers():
        return {TOKEN_HEADER: cfg.token} if cfg.token else {}

    def self_info() -> dict:
        info = hardware.snapshot(cfg.models_dir, PROJECT_ROOT)
        return {"id": cfg.node_id, "name": cfg.name, "version": VERSION, "cluster": cfg.cluster,
                "port": cfg.port, "rpc_port": cfg.rpc_port, "llama_port": cfg.llama_port,
                "info": info,
                "roles": {"worker": worker.running, "main": main.running}}

    def all_nodes() -> list[dict]:
        nodes = list(peers.values())
        if state["demo"]:
            for d in demo.DEMO_NODES:
                nodes.append({**d, "online": True, "simulated": True, "last_seen": time.time(),
                              "manual": False})
        return nodes

    def get_node(node_id: str) -> dict | None:
        if node_id == cfg.node_id:
            return {"id": cfg.node_id, "name": cfg.name, "address": f"127.0.0.1:{cfg.port}",
                    "info": self_info()["info"], "online": True, "self": True}
        for n in all_nodes():
            if n["id"] == node_id:
                return n
        return None

    def add_peer(node_id, name, ip, port, manual=False):
        if node_id == cfg.node_id:
            return
        p = peers.get(node_id)
        if p is None:
            peers[node_id] = {"id": node_id, "name": name, "address": f"{ip}:{port}",
                              "addresses": [f"{ip}:{port}"], "info": None, "online": True,
                              "last_seen": time.time(), "manual": manual}
            asyncio.create_task(refresh_peer(node_id))
        else:
            addr = f"{ip}:{port}"
            if addr not in p["addresses"]:
                p["addresses"].append(addr)
            p["last_seen"] = time.time()
            p["name"] = name

    async def refresh_peer(node_id):
        p = peers.get(node_id)
        if not p:
            return
        for addr in [p["address"]] + [a for a in p["addresses"] if a != p["address"]]:
            try:
                async with httpx.AsyncClient(timeout=2.5) as c:
                    r = await c.get(f"http://{addr}/api/info", headers=auth_headers())
                if r.status_code == 200:
                    data = r.json()
                    p.update({"info": data["info"], "name": data["name"], "online": True,
                              "llama_port": data.get("llama_port"),
                              "address": addr, "last_seen": time.time(), "roles": data["roles"],
                              "error": None})
                    return
                p["error"] = f"HTTP {r.status_code}" + (" (wrong token?)" if r.status_code == 401
                                                        else "")
            except Exception as e:  # noqa: BLE001
                p["error"] = type(e).__name__
        p["online"] = False

    async def refresher():
        while True:
            await asyncio.gather(*(refresh_peer(i) for i in list(peers)), return_exceptions=True)
            await asyncio.sleep(4)

    def route(a: dict, b: dict, kind: str):
        """Pick (bind_ip_on_a, target_ip_on_b, type) for a link from node a to node b."""
        a_ifs = (a.get("info") or {}).get("interfaces", [])
        b_ifs = (b.get("info") or {}).get("interfaces", [])
        kinds = TYPE_PREF if kind == "auto" else [kind]
        for k in kinds:
            for ai in [i for i in a_ifs if i["type"] == k]:
                for bi in [i for i in b_ifs if i["type"] == k]:
                    try:
                        net = ipaddress.ip_network(f"{ai['ip']}/{ai['netmask']}", strict=False)
                        if ipaddress.ip_address(bi["ip"]) in net:
                            return ai, bi, k
                    except Exception:  # noqa: BLE001
                        continue
        if kind == "auto":
            if b.get("self"):
                ips = [i for i in b_ifs if not i.get("link_local")]
                if ips:
                    return None, {"ip": ips[0]["ip"], "type": "routed"}, "routed"
            elif b.get("address"):
                ip = b["address"].split(":")[0]
                return None, {"ip": ip, "type": "routed"}, "routed"
        return None

    async def node_call(node: dict, method: str, path: str, payload=None, timeout=30):
        """Call a node's API (local node goes through loopback)."""
        if node.get("simulated"):
            raise HTTPException(500, "simulated node: handled elsewhere")
        addr = f"127.0.0.1:{cfg.port}" if node["id"] == cfg.node_id else node["address"]
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.request(method, f"http://{addr}{path}", json=payload,
                                    headers=auth_headers())
        except httpx.HTTPError as e:
            raise HTTPException(502, f"{node.get('name', addr)} is not reachable "
                                f"({type(e).__name__}). Is it still on and connected?")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:500])
        return r.json()

    # ------------------------------------------------------------------ auth
    @app.middleware("http")
    async def token_guard(request: Request, call_next):
        if cfg.token and request.url.path.startswith("/api/"):
            host = request.client.host if request.client else ""
            loopback = host in ("127.0.0.1", "::1", "localhost")
            supplied = request.headers.get(TOKEN_HEADER) or request.query_params.get("token")
            if not loopback and supplied != cfg.token:
                return JSONResponse({"detail": "token required"}, status_code=401)
        return await call_next(request)

    # ------------------------------------------------------------------ lifecycle
    @app.on_event("startup")
    async def _startup():
        if cfg.discovery:
            d = Discovery(cfg.node_id, cfg.name, cfg.port, cfg.cluster, cfg.token,
                          lambda i, n, ip, port: add_peer(i, n, ip, port))
            discovery_ref["d"] = d
            await d.start()
        asyncio.create_task(refresher())
        hardware.psutil.cpu_percent(interval=None)

    @app.on_event("shutdown")
    async def _shutdown():
        await worker.stop()
        await main.stop()

    # ------------------------------------------------------------------ UI
    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # ================================================================== NODE API
    @app.get("/api/info")
    async def api_info():
        return self_info()

    @app.get("/api/ping")
    async def api_ping():
        return {"t": time.time(), "id": cfg.node_id}

    @app.get("/api/blob")
    async def api_blob(mb: int = 20):
        mb = max(1, min(mb, 500))

        async def gen():
            for _ in range(mb):
                yield BLOB
        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"Content-Length": str(mb * len(BLOB))})

    @app.post("/api/nettest")
    async def api_nettest(body: dict):
        """Measure latency + throughput from THIS node to body.target (ip:port)."""
        target, bind_ip, mb = body["target"], body.get("bind_ip"), int(body.get("mb", 20))
        transport = httpx.AsyncHTTPTransport(local_address=bind_ip) if bind_ip else None
        lat, fails = [], 0
        try:
            async with httpx.AsyncClient(transport=transport, timeout=10,
                                         headers=auth_headers()) as c:
                for i in range(8):
                    t0 = time.perf_counter()
                    try:
                        r = await c.get(f"http://{target}/api/ping")
                        r.raise_for_status()
                        if i > 0:  # first request includes TCP setup
                            lat.append((time.perf_counter() - t0) * 1000)
                    except Exception:  # noqa: BLE001
                        fails += 1
                if not lat:
                    return {"ok": False, "error": "unreachable", "loss": 1.0}
                t0 = time.perf_counter()
                got = 0
                async with c.stream("GET", f"http://{target}/api/blob", params={"mb": mb}) as r:
                    async for chunk in r.aiter_bytes(1 << 16):
                        got += len(chunk)
                dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        return {"ok": True, "latency_ms": round(statistics.median(lat), 2),
                "jitter_ms": round(statistics.pstdev(lat), 2) if len(lat) > 1 else 0,
                "mbps": round(got * 8 / dt / 1e6, 0), "bytes": got, "seconds": round(dt, 2),
                "loss": round(fails / 8, 2)}

    # ---- processes on this node
    @app.post("/api/worker/start")
    async def api_worker_start(body: dict):
        bins = hardware.find_llama_binaries(PROJECT_ROOT)
        if not bins["rpc-server"]:
            raise HTTPException(400, "rpc-server not found on this device. Run "
                                "scripts/install_llama_cpp.sh (or set LLAMA_CPP_DIR).")
        host, port = body.get("host", "0.0.0.0"), int(body.get("port", cfg.rpc_port))
        cmd = [bins["rpc-server"], "-H", host, "-p", str(port)]
        if body.get("cache"):
            cmd.append("-c")
        await worker.start(cmd, {"endpoint": f"{host}:{port}"})
        await asyncio.sleep(1.0)
        if not worker.running:
            raise HTTPException(500, "rpc-server exited: " + " | ".join(list(worker.logs)[-5:]))
        return worker.status()

    @app.post("/api/main/start")
    async def api_main_start(body: dict):
        bins = hardware.find_llama_binaries(PROJECT_ROOT)
        if not bins["llama-server"]:
            raise HTTPException(400, "llama-server not found on this device. Run "
                                "scripts/install_llama_cpp.sh (or set LLAMA_CPP_DIR).")
        model = body["model_path"]
        if not Path(model).exists():
            raise HTTPException(400, f"model file not found on this device: {model}")
        cmd = [bins["llama-server"], "-m", model, "-ngl", "999", "-c", str(body.get("ctx", 4096)),
               "--host", "0.0.0.0", "--port", str(cfg.llama_port)]
        if body.get("rpc"):
            cmd += ["--rpc", ",".join(body["rpc"])]
        notes = []
        split = body.get("split")
        if split:
            dev_args, note = await device_split_args(bins["llama-server"], body.get("rpc", []),
                                                     split)
            cmd += dev_args
            notes.append(note)
        cmd += [a for a in str(body.get("extra_args", "")).split() if a]
        await main.start(cmd, {"model": model, "rpc": body.get("rpc", []), "notes": notes})
        for n in notes:
            main.logs.append("[localmesh] " + n)
        await asyncio.sleep(1.0)
        if not main.running:
            raise HTTPException(500, "llama-server exited: " + " | ".join(list(main.logs)[-4:]))
        return main.status()

    async def device_split_args(binary: str, rpc: list[str], split: dict):
        """Ask llama.cpp which devices it sees, then pin our layer split with --device and
        --tensor-split (same order in both, so the split is exact). Falls back to llama.cpp's
        default memory-proportional split if the device list can't be read."""
        cmd = [binary, "--list-devices"] + (["--rpc", ",".join(rpc)] if rpc else [])
        try:
            p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(p.communicate(), 30)
        except Exception as e:  # noqa: BLE001
            return [], f"could not list devices ({e}); using llama.cpp default split"
        names, seen = [], False
        for line in out.decode(errors="replace").splitlines():
            if "available devices" in line.lower():
                seen = True
                continue
            if seen and ":" in line and line.startswith((" ", "\t")):
                names.append(line.strip().split(":", 1)[0].strip())
        if not names:
            return [], "device list not understood; using llama.cpp default split"
        local = [n for n in names if "rpc" not in n.lower() and n.upper() != "CPU"]
        order, weights = [], []
        if split.get("local_layers", 0) > 0:
            if not local:
                return [], "main has no GPU device; using llama.cpp default split"
            for n in local:
                order.append(n)
                weights.append(split["local_layers"] / len(local))
        for ep, layers in split.get("rpc", []):
            match = next((n for n in names if ep in n), None)
            if not match:
                idx = rpc.index(ep) if ep in rpc else -1
                rpc_names = [n for n in names if "rpc" in n.lower()]
                match = rpc_names[idx] if 0 <= idx < len(rpc_names) else None
            if not match:
                return [], f"RPC device {ep} not in device list; using default split"
            order.append(match)
            weights.append(layers)
        ts = ",".join(f"{w:g}" for w in weights)
        return (["--device", ",".join(order), "--tensor-split", ts],
                f"layer split pinned: {', '.join(f'{o}={w:g}' for o, w in zip(order, weights))}")

    @app.post("/api/procs/stop")
    async def api_procs_stop():
        await main.stop()
        await worker.stop()
        return {"ok": True}

    @app.get("/api/procs")
    async def api_procs():
        ready = False
        if main.running:
            try:
                async with httpx.AsyncClient(timeout=1.5) as c:
                    r = await c.get(f"http://127.0.0.1:{cfg.llama_port}/health")
                    ready = r.status_code == 200
            except Exception:  # noqa: BLE001
                ready = False
        return {"worker": worker.status(), "main": main.status(), "server_ready": ready,
                "api_port": cfg.llama_port}

    # ---- models on this node
    @app.get("/api/local-models")
    async def api_local_models():
        return {"models_dir": str(cfg.models_dir), "models": downloader.local_models()}

    @app.post("/api/download")
    async def api_download(body: dict):
        job = downloader.start(body["model"], body["files"], int(body.get("total", 0)))
        return {"job": job}

    @app.get("/api/downloads")
    async def api_downloads():
        return {"jobs": list(downloader.jobs.values())}

    @app.post("/api/download/cancel")
    async def api_download_cancel(body: dict):
        downloader.cancel(body["job"])
        return {"ok": True}

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()
        body["stream"] = True

        async def gen():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=600)) as c:
                    async with c.stream("POST", f"http://127.0.0.1:{cfg.llama_port}"
                                        "/v1/chat/completions", json=body) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
            except Exception as e:  # noqa: BLE001
                err = {"error": f"main server not reachable: {e}"}
                yield f"data: {json.dumps(err)}\n\n".encode()
        return StreamingResponse(gen(), media_type="text/event-stream")

    # ================================================================== CONTROLLER API
    @app.get("/api/cluster")
    async def api_cluster():
        me = self_info()
        nodes = [{"id": me["id"], "name": me["name"], "address": f"this device:{cfg.port}",
                  "info": me["info"], "online": True, "self": True, "roles": me["roles"]}]
        for n in all_nodes():
            nodes.append({k: n.get(k) for k in ("id", "name", "address", "info", "online",
                                                 "simulated", "manual", "last_seen", "error",
                                                 "roles")})
        dep = state["deployment"]
        if dep and dep.get("state") == "running" and not dep.get("simulated"):
            gone = [get_node(i) for i in [dep["main_id"]] + dep.get("worker_ids", [])]
            off = [n["name"] if n else "?" for n in gone if not n or not n.get("online")]
            dep["health"] = {"ok": not off, "offline": off}
        d = discovery_ref.get("d")
        return {"self_id": cfg.node_id, "cluster": cfg.cluster, "token_set": bool(cfg.token),
                "demo": state["demo"], "nodes": nodes, "links": links,
                "deployment": state["deployment"],
                "discovery_errors": d.errors if d else ["discovery disabled"]}

    @app.post("/api/demo")
    async def api_demo(body: dict):
        state["demo"] = bool(body.get("on"))
        if not state["demo"]:
            for k in [k for k in links if "demo-" in k]:
                links.pop(k)
        return {"demo": state["demo"]}

    @app.post("/api/peers/add")
    async def api_peers_add(body: dict):
        addr = body["address"].strip().replace("http://", "").rstrip("/")
        if ":" not in addr:
            addr += ":7860"
        try:
            async with httpx.AsyncClient(timeout=4) as c:
                r = await c.get(f"http://{addr}/api/info", headers=auth_headers())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Could not reach {addr}: {type(e).__name__}. Is LocalMesh "
                                "running there and is port 7860 allowed by its firewall?")
        if r.status_code == 401:
            raise HTTPException(400, "That device uses a different cluster token.")
        if r.status_code != 200:
            raise HTTPException(400, f"{addr} answered HTTP {r.status_code}")
        data = r.json()
        if data["id"] == cfg.node_id:
            raise HTTPException(400, "That address is this device.")
        ip, port = addr.rsplit(":", 1)
        add_peer(data["id"], data["name"], ip, int(port), manual=True)
        peers[data["id"]].update({"info": data["info"], "online": True})
        return {"ok": True, "id": data["id"], "name": data["name"]}

    @app.post("/api/peers/remove")
    async def api_peers_remove(body: dict):
        peers.pop(body["id"], None)
        return {"ok": True}

    # ---- connection tests across the cluster
    async def run_nettest(node_ids: list[str], kind: str, mb: int, both: bool):
        nodes = [get_node(i) for i in node_ids]
        nodes = [n for n in nodes if n and n.get("online")]
        pairs = []
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if a["id"] != b["id"] and (both or i < j):
                    pairs.append((a, b))
        nettest_job.update({"state": "running", "results": [], "total": len(pairs), "done": 0,
                            "kind": kind, "started": time.time()})
        for a, b in pairs:
            key = f"{a['id']}|{b['id']}"
            r = route(a, b, kind)
            res = {"from": a["id"], "to": b["id"], "requested": kind}
            if r is None:
                res.update({"ok": False, "error": f"no shared {kind} network between these "
                            "devices", "type": kind})
            else:
                a_if, b_if, used = r
                res.update({"type": used, "from_ip": a_if["ip"] if a_if else None,
                            "to_ip": b_if["ip"]})
                if a.get("simulated") or b.get("simulated"):
                    await asyncio.sleep(0.25)
                    res.update({"ok": True, **demo.simulate_link(a_if, b_if, used)})
                else:
                    port = int(b["address"].split(":")[-1]) if not b.get("self") else cfg.port
                    try:
                        out = await node_call(a, "POST", "/api/nettest",
                                              {"target": f"{b_if['ip']}:{port}",
                                               "bind_ip": a_if["ip"] if a_if else None,
                                               "mb": mb}, timeout=120)
                        res.update(out)
                    except Exception as e:  # noqa: BLE001
                        res.update({"ok": False, "error": str(e)[:200]})
            res["tested_at"] = time.time()
            links[key] = res
            nettest_job["results"].append(res)
            nettest_job["done"] += 1
        nettest_job["state"] = "done"

    @app.post("/api/cluster/nettest")
    async def api_cluster_nettest(body: dict):
        if nettest_job.get("state") == "running":
            raise HTTPException(409, "A test is already running")
        ids = body.get("node_ids") or [n["id"] for n in (await api_cluster())["nodes"]
                                       if n.get("online")]
        asyncio.create_task(run_nettest(ids, body.get("type", "auto"), int(body.get("mb", 20)),
                                        bool(body.get("both_directions", True))))
        return {"ok": True}

    @app.get("/api/cluster/nettest")
    async def api_cluster_nettest_status():
        return nettest_job

    # ---- Hugging Face
    @app.get("/api/hf/search")
    async def api_hf_search(q: str = "", limit: int = 30):
        try:
            return {"results": await hf.search(q or "GGUF", limit), "source": "huggingface"}
        except Exception as e:  # noqa: BLE001
            if state["demo"]:
                ql = q.lower()
                res = [r for r in demo.DEMO_CATALOG["results"] if ql in r["id"].lower()] \
                    or demo.DEMO_CATALOG["results"]
                return {"results": res, "source": "offline-sample", "error": str(e)[:120]}
            raise HTTPException(502, f"Hugging Face not reachable: {type(e).__name__}. "
                                "Check the internet connection (or enable demo mode).")

    @app.get("/api/hf/files")
    async def api_hf_files(id: str):  # noqa: A002
        try:
            return await hf.files(id)
        except Exception as e:  # noqa: BLE001
            d = demo.demo_files(id) if state["demo"] else None
            if d:
                return d
            raise HTTPException(502, f"Could not list files for {id}: {e}"[:300])

    # ---- planning
    @app.post("/api/plan")
    async def api_plan(body: dict):
        nodes = []
        for nid in body.get("node_ids", []):
            n = get_node(nid)
            if n and n.get("info"):
                nodes.append({"id": n["id"], "name": n["name"],
                              "usable_memory_gb": n["info"]["usable_memory_gb"],
                              "mem_bandwidth_gbs": n["info"].get("mem_bandwidth_gbs", 50)})
        return planner.plan(float(body["size_gb"]), nodes, int(body.get("ctx", 4096)),
                            body.get("params_b"), body.get("active_b"), links,
                            body.get("main_id"))

    # ---- proxy to any node (downloads, local models, process status...)
    @app.api_route("/api/node/{node_id}/{path:path}", methods=["GET", "POST"])
    async def api_node_proxy(node_id: str, path: str, request: Request):
        node = get_node(node_id)
        if not node:
            raise HTTPException(404, "unknown device")
        payload = await request.json() if request.method == "POST" else None
        if node.get("simulated"):
            return await demo_node_api(node_id, path, payload)
        return await node_call(node, request.method, f"/api/{path}", payload, timeout=60)

    async def demo_node_api(node_id: str, path: str, payload):
        if path == "download":
            return {"job": demo_rt.start_download(node_id, payload["model"], payload["files"],
                                                  int(payload.get("total", 0)))}
        if path == "downloads":
            return {"jobs": list(demo_rt.downloads.get(node_id, {}).values())}
        if path == "download/cancel":
            j = demo_rt.downloads.get(node_id, {}).get(payload["job"])
            if j:
                j["state"] = "cancelled"
            return {"ok": True}
        if path == "local-models":
            return {"models_dir": "~/localmesh-models (simulated)",
                    "models": demo_rt.local_models(node_id)}
        if path == "procs":
            return demo_rt.proc_status(node_id)
        if path == "procs/stop":
            demo_rt.stop_all(node_id)
            return {"ok": True}
        raise HTTPException(400, f"not available on a simulated device: {path}")

    # ---- optimization (see localmesh/optimizer.py and docs/RESEARCH.md)
    def opt_devices(node_ids):
        out = []
        for nid in node_ids:
            n = get_node(nid)
            if n and n.get("online") and n.get("info"):
                i = n["info"]
                out.append({"id": n["id"], "name": n["name"],
                            "usable_memory_gb": i.get("usable_memory_gb", 0),
                            "mem_bandwidth_gbs": i.get("mem_bandwidth_gbs", 50),
                            "memory_kind": i.get("memory_kind"),
                            "ram_total_gb": i.get("ram_total_gb")})
        return out

    def opt_model(size_gb, name="", params_b=None, active_b=None, ctx=4096):
        params_b = params_b or hf.guess_params(name)
        active_b = active_b or hf.guess_active(name)
        return optimizer.build_model(float(size_gb), params_b, active_b, int(ctx))

    @app.post("/api/optimize")
    async def api_optimize(body: dict):
        devs = opt_devices(body.get("node_ids", []))
        if not devs:
            raise HTTPException(400, "Select at least one online device.")
        main_id = body.get("main_id") or devs[0]["id"]
        if main_id not in [d["id"] for d in devs]:
            main_id = devs[0]["id"]
        m = opt_model(body["size_gb"], body.get("name", ""), body.get("params_b"),
                      body.get("active_b"), body.get("ctx", 4096))
        kw = {"draft_gb": body.get("draft_gb"), "draft_accept": float(body.get("draft_accept", 0.6))}
        res = optimizer.compare(devs, m, main_id, links, **kw)
        bm = optimizer.best_main(devs, m, links, **kw)
        res["best_main"] = ({"id": bm["main_id"], "name": get_node(bm["main_id"])["name"],
                             "tps": bm["plan"]["tps"]} if bm else None)
        res["main_id"] = main_id
        res["names"] = {d["id"]: d["name"] for d in devs}
        return res

    # ---- deployment
    @app.post("/api/cluster/deploy")
    async def api_deploy(body: dict):
        state["last_deploy"] = dict(body)
        main_node = get_node(body["main_id"])
        worker_ids = [i for i in body.get("worker_ids", []) if i != body["main_id"]]
        workers = [get_node(i) for i in worker_ids]
        if not main_node or any(w is None for w in workers):
            raise HTTPException(400, "unknown device in selection")
        offline = [w["name"] for w in workers if not w.get("online")]
        if offline:
            raise HTTPException(400, "Offline: " + ", ".join(offline) + ". Re-plan without them.")
        kind = body.get("type", "auto")
        steps, rpc, plan = [], [], None
        simulated = main_node.get("simulated") or any(w.get("simulated") for w in workers)
        extra = str(body.get("extra_args", ""))

        # ---- optimizer: choose devices, layers, KV cache type, MoE offload, speculation
        if body.get("optimize", True) and body.get("model_size_gb") is not None:
            devs = opt_devices([main_node["id"]] + worker_ids)
            m = opt_model(max(float(body["model_size_gb"]), 0.001), body["model_path"],
                          ctx=body.get("ctx", 4096))
            draft_gb = body.get("draft_size_gb") if body.get("draft_path") else None
            plan = optimizer.optimize(devs, m, main_node["id"], links, draft_gb=draft_gb)
            if not plan.get("fits"):
                state["deployment"] = {"state": "failed", "steps": [{
                    "node": "planner", "step": "optimize placement", "ok": False,
                    "error": "Model does not fit on the selected devices even with KV cache "
                             "quantization and MoE offload. Add devices or pick a smaller "
                             "version."}]}
                return state["deployment"]
            dropped = [w for w in workers if w["id"] not in plan["used_ids"]]
            for w in dropped:
                why = next((e["reason"] for e in plan["excluded"] if e["id"] == w["id"]), "")
                steps.append({"node": w["name"], "step": "not used — " + why, "ok": True,
                              "skipped": True})
            workers = [w for w in workers if w["id"] in plan["used_ids"]]
            extra = " ".join(plan["flags"]) + " " + extra
            if plan.get("speculative") and body.get("draft_path"):
                extra = f"-md {body['draft_path']} " + extra
            steps.append({"node": "planner", "ok": True, "step":
                          f"{len(plan['used_ids'])} device(s), KV {plan['kv_type']}"
                          + (", MoE experts in CPU RAM" if plan["moe_hybrid"] else "")
                          + (f", speculative k={plan['speculative']['k']}"
                             if plan.get("speculative") and body.get("draft_path") else "")
                          + f" → est. {plan['tps']} tok/s"})

        endpoints = {}
        for w in workers:
            r = route(main_node, w, kind)
            if r is None:
                raise HTTPException(400, f"No shared {kind} network between {main_node['name']} "
                                    f"and {w['name']}. Pick another network or run the test.")
            _, w_if, used = r
            endpoint = f"{w_if['ip']}:{cfg.rpc_port}"
            if simulated:
                demo_rt.start_proc(w["id"], "worker", ["rpc-server", "-H", w_if["ip"], "-p",
                                                       str(cfg.rpc_port)],
                                   {"endpoint": endpoint})
                steps.append({"node": w["name"], "step": f"rpc-server on {endpoint} ({used})",
                              "ok": True, "simulated": True})
            else:
                try:
                    st = await node_call(w, "POST", "/api/worker/start",
                                         {"host": w_if["ip"],
                                          "cache": bool(body.get("rpc_cache"))})
                    endpoint = st["meta"]["endpoint"]
                    steps.append({"node": w["name"], "step": f"rpc-server on {endpoint} ({used})",
                                  "ok": True})
                except HTTPException as e:
                    steps.append({"node": w["name"], "step": "start rpc-server", "ok": False,
                                  "error": e.detail})
                    state["deployment"] = {"state": "failed", "steps": steps}
                    return state["deployment"]
            rpc.append(endpoint)
            endpoints[w["id"]] = endpoint
        main_body = {"model_path": body["model_path"], "rpc": rpc, "ctx": body.get("ctx", 4096),
                     "extra_args": extra}
        if plan:
            main_body["split"] = {"local_layers": plan["assign"].get(main_node["id"], 0),
                                  "rpc": [[endpoints[i], n] for i, n in plan["assign"].items()
                                          if i != main_node["id"] and i in endpoints]}
        if simulated:
            demo_rt.start_proc(main_node["id"] if main_node.get("simulated") else "self-demo",
                               "main", ["llama-server", "-m", body["model_path"], "--rpc",
                                        ",".join(rpc), "-ngl", "999"] + extra.split(), main_body)
            steps.append({"node": main_node["name"], "step": "llama-server (main)", "ok": True,
                          "simulated": True})
        else:
            try:
                await node_call(main_node, "POST", "/api/main/start", main_body, timeout=90)
                steps.append({"node": main_node["name"], "step": "llama-server (main)", "ok": True})
            except HTTPException as e:
                steps.append({"node": main_node["name"], "step": "start llama-server",
                              "ok": False, "error": e.detail})
                state["deployment"] = {"state": "failed", "steps": steps}
                return state["deployment"]
        main_ip = "127.0.0.1"
        main_llama_port = cfg.llama_port
        if not main_node.get("self"):
            main_ip = main_node["address"].split(":")[0]
            main_llama_port = (main_node.get("llama_port") or cfg.llama_port)
        state["deployment"] = {"state": "running", "steps": steps, "main_id": main_node["id"],
                               "worker_ids": [w["id"] for w in workers], "rpc": rpc,
                               "model_path": body["model_path"], "simulated": bool(simulated),
                               "api": f"http://{main_ip}:{main_llama_port}/v1",
                               "plan": plan, "started": time.time()}
        return state["deployment"]

    @app.post("/api/cluster/redeploy")
    async def api_redeploy():
        """Re-plan and redeploy the last deployment without devices that went offline."""
        last = state.get("last_deploy")
        if not last:
            raise HTTPException(400, "Nothing was deployed yet.")
        await api_stop()
        body = dict(last)
        body["worker_ids"] = [i for i in last.get("worker_ids", [])
                              if (get_node(i) or {}).get("online")]
        main = get_node(body["main_id"])
        if not main or not main.get("online"):
            raise HTTPException(400, "The main device itself is offline; pick a new main device "
                                "that has the model file.")
        return await api_deploy(body)

    @app.post("/api/cluster/stop")
    async def api_stop():
        dep = state["deployment"] or {}
        ids = [dep.get("main_id")] + dep.get("worker_ids", [])
        for i in [x for x in ids if x]:
            n = get_node(i)
            if not n:
                continue
            if n.get("simulated"):
                demo_rt.stop_all(i)
            else:
                try:
                    await node_call(n, "POST", "/api/procs/stop")
                except Exception:  # noqa: BLE001
                    pass
        demo_rt.stop_all("self-demo")
        if state["deployment"]:
            state["deployment"]["state"] = "stopped"
        return {"ok": True}

    @app.post("/api/cluster/chat")
    async def api_cluster_chat(request: Request):
        body = await request.json()
        dep = state["deployment"]
        if not dep or dep.get("state") != "running":
            raise HTTPException(400, "Nothing deployed yet.")
        if dep.get("simulated"):
            return StreamingResponse(demo.demo_chat_stream(body.get("messages", [])),
                                     media_type="text/event-stream")
        node = get_node(dep["main_id"])
        addr = f"127.0.0.1:{cfg.port}" if node.get("self") else node["address"]

        async def gen():
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=600)) as c:
                async with c.stream("POST", f"http://{addr}/api/chat", json=body,
                                    headers=auth_headers()) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
