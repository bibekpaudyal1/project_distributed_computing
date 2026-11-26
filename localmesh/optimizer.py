"""Placement optimizer: picks devices, splits layers and chooses runtime flags.

Ideas taken from recent research (see docs/RESEARCH.md for the full list and what was used):

* prima.cpp / Halda (ICLR'26), EdgeShard, Helix (ASPLOS'25):
  heterogeneity-aware layer assignment *and* device selection. A device is only used if
  the speed it adds outweighs the network cost it brings, unless it is needed for memory.
* TPI-LLM: on home/office networks link *latency* (not bandwidth) dominates per-token cost,
  so measured round-trip times drive the decision and slow links are pruned first.
* KIVI (ICML'24): KV cache quantization frees memory with little quality loss -> q8_0 KV
  cache when memory is tight (or when it lets us drop a slow device).
* EAGLE-3 (NeurIPS'25) and speculative decoding: a small draft model on the main device
  proposes k tokens; the cluster verifies them in one pass, so one network round trip pays
  for several tokens. "Speculative decoding: performance or illusion?" (2025) warns that
  gains depend on acceptance rate, so k is chosen from a model of acceptance, not fixed.
* KTransformers (SOSP'25), OSDI'26 local-MoE hybrid, ExpertFlow / DALI (2026):
  for MoE models, routed experts can live in CPU RAM while attention/shared weights stay
  on the GPU -> more capacity on an NVIDIA main device with lots of system RAM.

The cost model is intentionally simple and transparent (all units are seconds per token):
    T = sum_i layers_i * t_layer_i  +  sum_{workers used} (2 * latency_i + hop_overhead)
It reproduces llama.cpp RPC's star topology: the main device talks to every worker once
per token. For a fixed device subset the optimal split is a greedy fill of the fastest
devices (linear cost with capacities); we enumerate subsets (n <= ~10 devices).
"""
from __future__ import annotations

import itertools
import math

EFF = 0.6                 # achievable fraction of memory bandwidth (dense reads)
MOE_EFF = 0.6             # extra penalty: routed-expert reads are scattered, not streamed
LAYER_OVERHEAD_S = 0.00006  # kernel launch / sync per layer per token
HOP_OVERHEAD_S = 0.0015   # llama.cpp RPC: serialisation, graph split sync per worker per token
PER_DEVICE_OVERHEAD_GB = 0.8
KV_FACTOR = {"f16": 1.0, "q8_0": 0.53, "q4_0": 0.28}
CPU_BW_DEFAULT = 60.0     # GB/s, typical desktop DDR5 dual/quad channel
DEFAULT_LAN_LAT_MS = 4.0  # used when a link was not measured


def guess_layers(params_b: float | None) -> int:
    if not params_b:
        return 64
    table = [(2, 24), (5, 32), (10, 36), (20, 48), (40, 64), (80, 80), (150, 92), (1000, 78),
             (10_000, 92)]
    for limit, layers in table:
        if params_b <= limit:
            return layers
    return 80


def kv_gb_f16(params_b: float, ctx: int) -> float:
    """Conservative KV cache size in GB at f16 for one sequence (GQA-era models)."""
    return max(0.25, 0.0048 * params_b * ctx / 1024)


def _link_latency(links: dict, a: str, b: str) -> tuple[float, bool]:
    l = links.get(f"{a}|{b}") or links.get(f"{b}|{a}")
    if l and l.get("ok", True) and l.get("latency_ms") is not None:
        return l["latency_ms"] / 1000, True
    return DEFAULT_LAN_LAT_MS / 1000, False


def _device_model(d: dict, model: dict, kv_type: str, moe_hybrid: bool, is_main: bool) -> dict:
    """Per-layer memory and time for a device."""
    L = model["layers"]
    w = model["size_gb"] / L                       # weight GB per layer
    kv = model["kv_f16_gb"] * KV_FACTOR[kv_type] / L
    active = model["active_frac"]
    a = w * active                                 # GB read per token per layer
    bw = max(d.get("mem_bandwidth_gbs") or 50, 1) * EFF * (MOE_EFF if model["is_moe"] else 1)
    cap = max(d["usable_memory_gb"] - PER_DEVICE_OVERHEAD_GB, 0)
    hybrid = bool(moe_hybrid and is_main and model["is_moe"] and d.get("memory_kind") == "nvidia-vram"
                  and d.get("ram_total_gb"))
    if hybrid:
        e = model["expert_frac"]
        ram_cap = max(d["ram_total_gb"] * 0.75 - 4, 0)
        gpu_per_layer = w * (1 - e) + kv
        cpu_per_layer = w * e
        max_layers = min(cap / gpu_per_layer if gpu_per_layer else 1e9,
                         ram_cap / cpu_per_layer if cpu_per_layer else 1e9)
        cpu_bw = CPU_BW_DEFAULT * EFF * MOE_EFF
        t = 0.4 * a / bw + 0.6 * a / cpu_bw + 2 * LAYER_OVERHEAD_S
    else:
        per_layer = w + kv
        max_layers = cap / per_layer if per_layer else 0
        t = a / bw + LAYER_OVERHEAD_S
    return {"max_layers": int(max_layers), "t_layer": t, "hybrid": hybrid,
            "mem_per_layer": w + kv}


def evaluate(devices: list[dict], model: dict, main_id: str, links: dict, kv_type: str,
             moe_hybrid: bool, layers_override: dict | None = None) -> dict:
    """Cost of a given assignment (layers_override) or greedy-optimal assignment for this set."""
    dm = {d["id"]: _device_model(d, model, kv_type, moe_hybrid, d["id"] == main_id)
          for d in devices}
    L = model["layers"]
    if layers_override is not None:
        assign = dict(layers_override)
    else:
        assign, left = {d["id"]: 0 for d in devices}, L
        for d in sorted(devices, key=lambda d: dm[d["id"]]["t_layer"]):
            take = min(left, dm[d["id"]]["max_layers"])
            assign[d["id"]] = take
            left -= take
        if left > 0:
            return {"fits": False}
    if any(assign[i] > dm[i]["max_layers"] for i in assign):
        return {"fits": False, "assign": assign}
    compute = sum(assign[i] * dm[i]["t_layer"] for i in assign)
    net, unmeasured, per = 0.0, 0, {}
    for d in devices:
        i = d["id"]
        lat, measured = (0.0, True) if i == main_id else _link_latency(links, main_id, i)
        n_cost = 0.0
        if i != main_id and assign[i] > 0:
            n_cost = 2 * lat + HOP_OVERHEAD_S
            unmeasured += 0 if measured else 1
        net += n_cost
        per[i] = {"layers": assign[i], "compute_ms": round(assign[i] * dm[i]["t_layer"] * 1000, 2),
                  "network_ms": round(n_cost * 1000, 2), "hybrid": dm[i]["hybrid"],
                  "mem_gb": round(assign[i] * dm[i]["mem_per_layer"] + PER_DEVICE_OVERHEAD_GB, 1),
                  "latency_measured": measured}
    T = compute + net
    return {"fits": True, "assign": assign, "per_device": per, "token_s": T,
            "compute_s": compute, "network_s": net, "tps": 1 / T, "unmeasured_links": unmeasured}


def speculative(T_target: float, main: dict, draft_gb: float | None, accept: float,
                is_moe: bool = False) -> dict | None:
    """Best draft length k for speculative decoding; returns speed multiplier."""
    if not draft_gb:
        return None
    t_draft = draft_gb / (max(main.get("mem_bandwidth_gbs") or 50, 1) * EFF)
    best = None
    for k in range(1, 17):
        expected = (1 - accept ** (k + 1)) / (1 - accept) if accept < 1 else k + 1
        # Verifying k+1 tokens: nearly free for dense models (memory-bound), but MoE tokens
        # route to different experts, so verification reads more weights.
        cost = k * t_draft + T_target * (1 + (0.15 if is_moe else 0.03) * k)
        tps = expected / cost
        if best is None or tps > best["tps"]:
            best = {"k": k, "tps": tps, "expected_tokens": round(expected, 2)}
    best["speedup"] = round(best["tps"] * T_target, 2)
    return best if best["speedup"] > 1.05 else None


def build_model(size_gb: float, params_b: float | None, active_b: float | None, ctx: int,
                layers: int | None = None) -> dict:
    if not params_b:
        params_b = size_gb * 8 / 4.8  # assume ~Q4
    is_moe = bool(active_b and active_b < params_b * 0.6)
    active_frac = (active_b / params_b) if is_moe else 1.0
    return {"size_gb": size_gb, "params_b": params_b, "active_b": active_b,
            "layers": layers or guess_layers(params_b), "kv_f16_gb": kv_gb_f16(params_b, ctx),
            "is_moe": is_moe, "active_frac": active_frac,
            "expert_frac": 0.9 if is_moe else 0.0, "ctx": ctx}


def naive_plan(devices: list[dict], model: dict, main_id: str, links: dict) -> dict:
    """What llama.cpp does by default: every selected device, split by free memory, f16 KV."""
    L = model["layers"]
    total = sum(max(d["usable_memory_gb"], 0) for d in devices) or 1
    raw = {d["id"]: L * max(d["usable_memory_gb"], 0) / total for d in devices}
    assign = {k: int(v) for k, v in raw.items()}
    rest = L - sum(assign.values())
    for k in sorted(raw, key=lambda k: raw[k] - assign[k], reverse=True)[:rest]:
        assign[k] += 1
    r = evaluate(devices, model, main_id, links, "f16", False, assign)
    r["kv_type"] = "f16"
    return r


def optimize(devices: list[dict], model: dict, main_id: str, links: dict,
             draft_gb: float | None = None, draft_accept: float = 0.6,
             allow_kv_quant: bool = True, allow_moe_hybrid: bool = True) -> dict:
    main = next(d for d in devices if d["id"] == main_id)
    others = [d for d in devices if d["id"] != main_id]
    kv_types = ["f16", "q8_0", "q4_0"] if allow_kv_quant else ["f16"]
    hybrid_opts = [False, True] if (allow_moe_hybrid and model["is_moe"]) else [False]
    candidates = []
    for r in range(len(others) + 1):
        for subset in itertools.combinations(others, r):
            devs = [main, *subset]
            for kv in kv_types:
                for hy in hybrid_opts:
                    res = evaluate(devs, model, main_id, links, kv, hy)
                    if res["fits"]:
                        # quality preference: f16 > q8_0 (tiny loss) > q4_0 (noticeable)
                        penalty = {"f16": 1.0, "q8_0": 1.03, "q4_0": 1.25}[kv]
                        candidates.append((res["token_s"] * penalty, kv, hy, devs, res))
    if not candidates:
        return {"fits": False}
    _, kv, hy, devs, best = min(candidates, key=lambda c: c[0])
    used = [d for d in devs if best["assign"][d["id"]] > 0 or d["id"] == main_id]
    excluded = []
    for d in others:
        if d not in used:
            lat, measured = _link_latency(links, main_id, d["id"])
            excluded.append({"id": d["id"], "name": d.get("name"),
                             "reason": (f"left out: it would add ~{2 * lat * 1000 + HOP_OVERHEAD_S * 1000:.1f}"
                                        " ms of network per token, more than the compute it "
                                        "saves, and the model fits without it"
                                        + ("" if measured else " (link not measured; 4 ms assumed)"))})
    spec = speculative(best["token_s"], main, draft_gb, draft_accept, model["is_moe"])
    flags = []
    if kv != "f16":
        flags += ["--cache-type-k", kv, "--cache-type-v", kv, "-fa", "on"]
    if hy:
        flags += ["-ot", r"\.ffn_(up|down|gate)_exps\.=CPU"]
    if spec:
        flags += ["--draft-max", str(spec["k"]), "--draft-min", "1"]
    tps = best["tps"] * (spec["speedup"] if spec else 1)
    return {"fits": True, "kv_type": kv, "moe_hybrid": hy, "used_ids": [d["id"] for d in used],
            "excluded": excluded, "per_device": {k: v for k, v in best["per_device"].items()
                                                  if k in {d["id"] for d in used}},
            "assign": {d["id"]: best["assign"][d["id"]] for d in used},
            "token_ms": round(best["token_s"] * 1000, 2), "tps_base": round(best["tps"], 2),
            "speculative": spec, "tps": round(tps, 2), "flags": flags,
            "layers": model["layers"], "unmeasured_links": best["unmeasured_links"]}


def best_main(devices: list[dict], model: dict, links: dict, **kw) -> dict | None:
    """Try every device as the main one (it holds the file and serves the API)."""
    best = None
    for d in devices:
        r = optimize(devices, model, d["id"], links, **kw)
        if r.get("fits") and (best is None or r["tps"] > best[1]["tps"]):
            best = (d["id"], r)
    return {"main_id": best[0], "plan": best[1]} if best else None


def compare(devices: list[dict], model: dict, main_id: str, links: dict, **kw) -> dict:
    naive = naive_plan(devices, model, main_id, links)
    opt = optimize(devices, model, main_id, links, **kw)
    out = {"model": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in model.items()},
           "naive": {"fits": naive.get("fits", False),
                     "tps": round(naive["tps"], 2) if naive.get("fits") else None,
                     "assign": naive.get("assign"), "per_device": naive.get("per_device")},
           "optimized": opt}
    if naive.get("fits") and opt.get("fits"):
        out["speedup"] = round(opt["tps"] / naive["tps"], 2)
    return out


def tensor_split(plan: dict, main_id: str, endpoints: dict[str, str], local_gpus: int) -> dict:
    """Order + weights for llama.cpp `--device` / `--tensor-split`.

    endpoints: worker_id -> "ip:port" of its rpc-server. Main's layers are spread evenly over
    its local GPUs. Returns {"order": [("local", i) | ("rpc", endpoint)], "weights": [...]}
    """
    order, weights = [], []
    main_layers = plan["assign"].get(main_id, 0)
    for g in range(max(local_gpus, 1)):
        order.append(("local", g))
        weights.append(main_layers / max(local_gpus, 1))
    for wid, n in plan["assign"].items():
        if wid == main_id or n <= 0:
            continue
        order.append(("rpc", endpoints[wid]))
        weights.append(n)
    return {"order": order, "weights": [round(w, 3) for w in weights]}


def _self_test():
    devs = [
        {"id": "m", "name": "main", "usable_memory_gb": 22, "mem_bandwidth_gbs": 1000,
         "memory_kind": "nvidia-vram", "ram_total_gb": 128},
        {"id": "s", "name": "studio", "usable_memory_gb": 150, "mem_bandwidth_gbs": 800},
        {"id": "a", "name": "air", "usable_memory_gb": 10, "mem_bandwidth_gbs": 100},
    ]
    links = {"m|s": {"latency_ms": 0.4}, "m|a": {"latency_ms": 6}}
    m = build_model(42.5, 70, None, 4096)
    c = compare(devs, m, "m", links)
    assert c["optimized"]["fits"] and "a" not in c["optimized"]["used_ids"], c
    assert c["speedup"] >= 1, c
    big = build_model(262, 744, 40, 4096)
    assert not compare(devs, big, "m", links)["optimized"]["fits"]   # 262 GB > pool: refused
    moe = build_model(208, 744, 40, 4096)
    c2 = compare(devs, moe, "m", links)
    assert c2["optimized"]["fits"] and c2["optimized"]["moe_hybrid"], c2  # fits only via hybrid
    assert not c2["naive"]["fits"]
    return c, c2


if __name__ == "__main__":
    import json
    for r in _self_test():
        print("naive fits:", r["naive"]["fits"], "speedup:", r.get("speedup"))
        print(json.dumps(r["optimized"], indent=1)[:1200])
