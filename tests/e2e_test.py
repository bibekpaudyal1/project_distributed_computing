"""End-to-end test: 3 nodes on this machine + fake Hugging Face + fake llama.cpp binaries.

    python tests/e2e_test.py

Covers discovery, connection tests, HF search/variants, split-file download, planning,
deployment across workers, streaming chat, stop, manual add, token protection.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "fakes"
procs, results, by_name = [], [], {}


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("  PASS " if cond else "  FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))


def spawn(args, env_extra=None, log=None):
    env = {**os.environ, **(env_extra or {})}
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log or subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)
    procs.append(p)
    return p


def wait_for(fn, timeout=30, every=0.5):
    end = time.time() + timeout
    while time.time() < end:
        try:
            v = fn()
            if v:
                return v
        except Exception:  # noqa: BLE001
            pass
        time.sleep(every)
    return None


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lm-e2e-"))
    for b in (FAKES / "bin").iterdir():
        b.chmod(0o755)
    spawn([sys.executable, str(FAKES / "fake_hf.py"), "9900"])
    common = {"HF_ENDPOINT": "http://127.0.0.1:9900", "LLAMA_CPP_DIR": str(FAKES / "bin")}
    nodes = [("alpha", 7960, 51052, 8180), ("beta", 7970, 51062, 8190), ("gamma", 7980, 51072, 8200)]
    for name, port, rpc, llama in nodes:
        by_name[name] = spawn([sys.executable, "-m", "localmesh", "--port", str(port), "--name", name,
               "--cluster", "e2e", "--rpc-port", str(rpc), "--llama-port", str(llama),
               "--models-dir", str(tmp / name)],
              {**common, "LOCALMESH_HOME": str(tmp / f"state-{name}")},
              log=open(tmp / f"{name}.log", "w"))
    A = "http://127.0.0.1:7960"
    c = httpx.Client(timeout=30)

    print("Discovery")
    cl = wait_for(lambda: (lambda d: d if sum(n["online"] and n["info"] is not None
                                              for n in d["nodes"]) == 3 else None)(
        c.get(f"{A}/api/cluster").json()), timeout=25)
    check("3 nodes discovered automatically", cl, "discovery timeout")
    if not cl:
        return
    ids = {n["name"]: n["id"] for n in cl["nodes"]}

    print("Connection tests")
    c.post(f"{A}/api/cluster/nettest", json={"type": "auto", "mb": 5})
    job = wait_for(lambda: (lambda j: j if j["state"] == "done" else None)(
        c.get(f"{A}/api/cluster/nettest").json()), timeout=60)
    check("6 directed links tested", job and job["total"] == 6)
    check("all links ok with latency + throughput",
          job and all(r["ok"] and r["latency_ms"] > 0 and r["mbps"] > 0 for r in job["results"]),
          json.dumps(job["results"][:1]) if job else "")
    r = c.post(f"{A}/api/cluster/nettest", json={"type": "thunderbolt", "mb": 5})
    job = wait_for(lambda: (lambda j: j if j["state"] == "done" else None)(
        c.get(f"{A}/api/cluster/nettest").json()), timeout=30)
    check("missing network type reported clearly",
          job and all((not x["ok"]) and "no shared thunderbolt" in x["error"]
                      for x in job["results"]))

    print("Hugging Face")
    s = c.get(f"{A}/api/hf/search", params={"q": "tiny"}).json()
    check("search returns GGUF repo", s["results"] and s["results"][0]["id"].endswith("GGUF"))
    check("param count parsed from name", s["results"][0]["params_b"] == 1)
    f = c.get(f"{A}/api/hf/files", params={"id": s["results"][0]["id"]}).json()
    quants = {v["quant"]: v for v in f["variants"]}
    check("variants grouped, mmproj skipped", set(quants) == {"Q4_K_M", "Q8_0"}, str(quants.keys()))
    check("split files merged into one variant", len(quants["Q8_0"]["files"]) == 2
          and quants["Q8_0"]["size"] == 4 * 1024 * 1024)

    print("Planner")
    p = c.post(f"{A}/api/plan", json={"size_gb": quants["Q8_0"]["size_gb"],
                                      "node_ids": list(ids.values()),
                                      "main_id": ids["alpha"]}).json()
    check("plan fits and has 3 shares summing to 100%", p["fits"] and len(p["shares"]) == 3
          and abs(sum(x["fraction"] for x in p["shares"]) - 1) < 0.01)
    big = c.post(f"{A}/api/plan", json={"size_gb": 99999, "node_ids": list(ids.values())}).json()
    check("oversized model rejected with reason", (not big["fits"]) and big["reason"])

    print("Download (split model) to alpha")
    v = quants["Q8_0"]
    c.post(f"{A}/api/node/{ids['alpha']}/download",
           json={"model": f["id"], "files": v["files"], "total": v["size"]})
    jobs = wait_for(lambda: (lambda j: j if j and all(x["state"] == "done" for x in j) else None)(
        c.get(f"{A}/api/node/{ids['alpha']}/downloads").json()["jobs"]), timeout=30)
    check("download completed", jobs, "timeout")
    lm = c.get(f"{A}/api/node/{ids['alpha']}/local-models").json()["models"]
    check("local model listed once with 2 parts", len(lm) == 1 and lm[0]["parts"] == 2, str(lm))

    print("Deploy main=alpha, workers=beta,gamma")
    d = c.post(f"{A}/api/cluster/deploy", json={"main_id": ids["alpha"],
                                                "worker_ids": [ids["beta"], ids["gamma"]],
                                                "model_path": lm[0]["path"], "ctx": 2048}).json()
    check("deploy started", d["state"] == "running", json.dumps(d)[:300])
    ready = wait_for(lambda: c.get(f"{A}/api/node/{ids['alpha']}/procs").json()["server_ready"],
                     timeout=25)
    check("main server ready", ready,
          "\n".join(c.get(f"{A}/api/node/{ids['alpha']}/procs").json()["main"]["logs"][-5:]))
    wk = c.get(f"{A}/api/node/{ids['beta']}/procs").json()["worker"]
    check("worker rpc-server running on beta", wk["running"])

    print("Chat through the cluster")
    text = ""
    with c.stream("POST", f"{A}/api/cluster/chat",
                  json={"messages": [{"role": "user", "content": "hi"}]}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                j = json.loads(line[5:])
                text += j["choices"][0]["delta"].get("content", "")
    check("streamed reply used 2 RPC workers", "2 RPC worker" in text, text)

    print("Stop")
    c.post(f"{A}/api/cluster/stop", json={})
    time.sleep(1.5)
    pr = c.get(f"{A}/api/node/{ids['alpha']}/procs").json()
    check("processes stopped", not pr["main"]["running"]
          and not c.get(f"{A}/api/node/{ids['beta']}/procs").json()["worker"]["running"])

    print("Optimizer-driven deploy (model bigger than one device)")
    size = 8.0  # GB: does not fit on one test node (~5 GB each) -> optimizer must use 2+
    opt_r = c.post(f"{A}/api/optimize", json={"size_gb": size, "node_ids": list(ids.values()),
                                              "main_id": ids["alpha"], "name": "Tiny-8B"}).json()
    check("optimize endpoint returns naive vs optimized", opt_r["optimized"]["fits"]
          and opt_r["naive"]["fits"] and opt_r["speedup"] >= 1, json.dumps(opt_r)[:300])
    d = c.post(f"{A}/api/cluster/deploy", json={
        "main_id": ids["alpha"], "worker_ids": [ids["beta"], ids["gamma"]],
        "model_path": lm[0]["path"], "model_size_gb": size, "optimize": True}).json()
    check("optimized deploy running", d["state"] == "running", json.dumps(d)[:400])
    used = d.get("plan", {}).get("used_ids", [])
    check("plan uses >= 2 devices", len(used) >= 2, str(used))
    wait_for(lambda: c.get(f"{A}/api/node/{ids['alpha']}/procs").json()["server_ready"], 25)
    text = ""
    with c.stream("POST", f"{A}/api/cluster/chat",
                  json={"messages": [{"role": "user", "content": "hi"}]}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                text += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
    check("layer split pinned with --device/--tensor-split", "devices=[FAKEGPU0,RPC[" in text
          and "split=[" in text and "split=[]" not in text, text)

    st = c.get(f"{A}/api/node/{ids['alpha']}/stats").json()
    check("live stats: CPU/RAM/net + per-device model memory from llama.cpp",
          "cpu_percent" in st and st["net"] and any(k.startswith("RPC") for k in st["buffers"])
          and st["procs"]["main"] and st["procs"]["main"]["rss_gb"] >= 0, json.dumps(st)[:300])
    wst = c.get(f"{A}/api/node/{ids['beta']}/stats").json()
    check("live stats proxied from a worker device", "ram_total_gb" in wst, json.dumps(wst)[:200])

    B = "http://127.0.0.1:7970"
    rm = wait_for(lambda: c.get(B + "/api/cluster").json().get("remote_main"), timeout=15)
    check("another device sees the served model (remote_main)", rm and rm["id"] == ids["alpha"],
          str(rm))
    text_b = ""
    with c.stream("POST", f"{B}/api/cluster/chat",
                  json={"messages": [{"role": "user", "content": "hi"}]}) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                text_b += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
    check("chat from another device without downloading the model", "fake cluster" in text_b,
          text_b[:200])

    print("Failure drill: a worker dies mid-session")
    victim = next(n for n, i in ids.items() if i in used and n != "alpha")
    by_name[victim].terminate()
    health = wait_for(lambda: (lambda dep: dep["health"] if dep and dep.get("health") and
                               not dep["health"]["ok"] else None)(
        c.get(f"{A}/api/cluster").json()["deployment"]), timeout=20)
    check(f"deployment marked degraded ({victim} offline)", health and victim in health["offline"],
          str(health))
    d = c.post(f"{A}/api/cluster/redeploy").json()
    check("one-click re-plan redeploys without the dead device",
          d.get("state") == "running" and ids[victim] not in d.get("worker_ids", []),
          json.dumps(d)[:400])
    ready = wait_for(lambda: c.get(f"{A}/api/node/{ids['alpha']}/procs").json()["server_ready"],
                     timeout=25)
    check("serving again after re-plan", ready)
    c.post(f"{A}/api/cluster/stop", json={})

    print("Errors are explained")
    r = c.post(f"{A}/api/node/{ids['alpha']}/main/start",
               json={"model_path": lm[0]["path"], "extra_args": "--fake-crash"})
    check("main server crash -> error with last log lines", r.status_code == 500
          and "out of memory" in r.text, r.text[:200])
    r = c.post(f"{A}/api/peers/add", json={"address": "127.0.0.1:1"})
    check("unreachable manual add -> helpful message", r.status_code == 400
          and "firewall" in r.json()["detail"])
    d = c.post(f"{A}/api/cluster/deploy", json={"main_id": ids["alpha"], "worker_ids": [],
                                                "model_path": "/nope.gguf"}).json()
    check("missing model file -> failed step with reason", d["state"] == "failed"
          and "not found" in d["steps"][-1]["error"])

    spawn([sys.executable, "-m", "localmesh", "--port", "7995", "--no-discovery",
           "--models-dir", str(tmp / "nobin")],
          {"LOCALMESH_HOME": str(tmp / "state-nobin"), "LLAMA_CPP_DIR": "/nonexistent",
           "PATH": "/usr/bin:/bin"})
    wait_for(lambda: httpx.get("http://127.0.0.1:7995/api/ping").status_code == 200, 15)
    r = httpx.post("http://127.0.0.1:7995/api/worker/start", json={})
    check("llama.cpp missing -> clear install hint", r.status_code == 400
          and "install_llama_cpp" in r.text, r.text[:200])

    print("Token protection")
    spawn([sys.executable, "-m", "localmesh", "--port", "7990", "--token", "s3cret",
           "--no-discovery", "--models-dir", str(tmp / "tok")],
          {**common, "LOCALMESH_HOME": str(tmp / "state-tok")})
    wait_for(lambda: httpx.get("http://127.0.0.1:7990/api/ping").status_code == 200, 15)
    lan_ip = [i["ip"] for i in cl["nodes"][0]["info"]["interfaces"] if not i["link_local"]]
    if lan_ip:
        r1 = httpx.get(f"http://{lan_ip[0]}:7990/api/info")
        r2 = httpx.get(f"http://{lan_ip[0]}:7990/api/info", headers={"X-LocalMesh-Token": "s3cret"})
        check("LAN request without token -> 401, with token -> 200",
              r1.status_code == 401 and r2.status_code == 200)
        r = c.post(f"{A}/api/peers/add", json={"address": f"{lan_ip[0]}:7990"})
        check("joining a token-protected node from another cluster is refused",
              r.status_code == 400 and "token" in r.json()["detail"])


if __name__ == "__main__":
    try:
        main()
    finally:
        for p in procs:
            p.terminate()
        time.sleep(1)
        for p in procs:
            if p.poll() is None:
                p.kill()
    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) and results else 1)
