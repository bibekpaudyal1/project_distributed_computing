"""Decide whether a model fits on the selected devices and how to split it.

The numbers are deliberately simple and explained in the UI. llama.cpp splits layers
across devices in proportion to free memory, which is what we model here.
"""
from __future__ import annotations


def plan(model_size_gb: float, nodes: list[dict], ctx: int = 4096,
         params_b: float | None = None, active_b: float | None = None,
         links: dict | None = None, main_id: str | None = None) -> dict:
    """
    nodes: [{id, name, usable_memory_gb, mem_bandwidth_gbs}]
    links: {"<a>|<b>": {"latency_ms":..., "mbps":...}} measured connection tests (optional)
    """
    links = links or {}
    nodes = [n for n in nodes if n.get("usable_memory_gb", 0) > 0.5]
    if not nodes:
        return {"fits": False, "reason": "No devices with usable memory selected.", "shares": []}

    # Overheads: KV cache + compute buffers. ~0.5 GB per 4k ctx per 10 GB of model, min 0.5 GB,
    # plus ~0.8 GB runtime per device.
    kv_gb = max(0.5, model_size_gb / 10 * 0.5 * (ctx / 4096))
    per_node_overhead = 0.8
    pool = sum(n["usable_memory_gb"] for n in nodes)
    needed = model_size_gb + kv_gb + per_node_overhead * len(nodes)

    # Main node first (it runs llama-server), then biggest-memory first.
    ordered = sorted(nodes, key=lambda n: (n["id"] != main_id, -n["usable_memory_gb"]))
    shares = []
    for n in ordered:
        frac = n["usable_memory_gb"] / pool
        load = (model_size_gb + kv_gb) * frac + per_node_overhead
        shares.append({"id": n["id"], "name": n.get("name"), "fraction": round(frac, 3),
                       "weights_gb": round(model_size_gb * frac, 2),
                       "load_gb": round(load, 2), "capacity_gb": n["usable_memory_gb"],
                       "utilisation": round(load / n["usable_memory_gb"], 2)})

    fits = needed <= pool
    # ---- Speed estimate (decode tokens/sec), rough.
    # Each token reads the *active* weights once. Dense models: all weights.
    active_frac = 1.0
    if params_b and active_b and active_b < params_b:
        active_frac = active_b / params_b
    compute_s = sum(model_size_gb * s["fraction"] * active_frac /
                    max(next(n for n in nodes if n["id"] == s["id"]).get("mem_bandwidth_gbs", 50)
                        * 0.6, 1)
                    for s in shares)
    # Network: llama.cpp RPC does a round trip per device per token (activations are small,
    # so latency dominates). Use measured links to the main node when known.
    net_s, missing_links = 0.0, 0
    if main_id:
        for s in shares:
            if s["id"] == main_id:
                continue
            link = links.get(f"{main_id}|{s['id']}") or links.get(f"{s['id']}|{main_id}")
            if link and link.get("latency_ms") is not None:
                net_s += 2 * link["latency_ms"] / 1000 + 0.002
            else:
                missing_links += 1
                net_s += 2 * 0.004 + 0.002  # assume ~4 ms LAN
    tps = 1 / (compute_s + net_s) if (compute_s + net_s) > 0 else None

    warnings = []
    if len(nodes) > 1 and missing_links:
        warnings.append(f"{missing_links} link(s) to the main device not tested yet: speed uses "
                        "a 4 ms guess. Run the connection test for a real number.")
    for s in shares:
        if s["utilisation"] > 1:
            warnings.append(f"{s['name']} would be over its memory ({s['load_gb']} GB of "
                            f"{s['capacity_gb']} GB).")
    for k, l in links.items():
        if l.get("mbps") and l["mbps"] < 100 and main_id and main_id in k:
            warnings.append("A link to the main device is under 100 Mbps: loading weights over "
                            "it will be slow (Wi-Fi?). Prefer Ethernet/Thunderbolt.")
            break
    return {
        "fits": fits,
        "pool_gb": round(pool, 1),
        "needed_gb": round(needed, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "shares": shares,
        "est_tokens_per_s": round(tps, 1) if tps else None,
        "warnings": warnings,
        "reason": None if fits else (f"Needs ~{needed:.0f} GB but the selected devices offer "
                                     f"{pool:.0f} GB. Pick a smaller quant or add devices."),
    }
