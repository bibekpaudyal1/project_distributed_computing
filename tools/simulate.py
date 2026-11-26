"""Simulation: llama.cpp default split vs LocalMesh optimizer, plus failure scenarios.

    python tools/simulate.py            # writes docs/sim/*.png and docs/sim/results.json

All numbers come from the same transparent cost model the app uses (localmesh/optimizer.py).
They are estimates for comparing strategies, not benchmarks of real hardware.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from localmesh import optimizer as opt  # noqa: E402

OUT = ROOT / "docs" / "sim"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the simulated office
DEVICES = [
    {"id": "rtx", "name": "RTX 4090 box", "usable_memory_gb": 22.1, "mem_bandwidth_gbs": 1000,
     "memory_kind": "nvidia-vram", "ram_total_gb": 128},
    {"id": "studio", "name": "Mac Studio M2 Ultra", "usable_memory_gb": 151.6,
     "mem_bandwidth_gbs": 800, "memory_kind": "apple-unified", "ram_total_gb": 192},
    {"id": "ws", "name": "2x3090 workstation", "usable_memory_gb": 44.2, "mem_bandwidth_gbs": 936,
     "memory_kind": "nvidia-vram", "ram_total_gb": 256},
    {"id": "mbp", "name": "MacBook Pro M2 Max", "usable_memory_gb": 49.2,
     "mem_bandwidth_gbs": 400, "memory_kind": "apple-unified", "ram_total_gb": 64},
    {"id": "air", "name": "MacBook Air M2 (Wi-Fi)", "usable_memory_gb": 10.0,
     "mem_bandwidth_gbs": 100, "memory_kind": "apple-unified", "ram_total_gb": 16},
]
ETH, WIFI = 0.45, 6.0  # ms


def links_with(wifi_ms=WIFI, eth_ms=ETH, mbp_wifi=False):
    lat = {}
    ids = [d["id"] for d in DEVICES]
    for a in ids:
        for b in ids:
            if a == b:
                continue
            slow = "air" in (a, b) or (mbp_wifi and "mbp" in (a, b))
            lat[f"{a}|{b}"] = {"latency_ms": wifi_ms if slow else eth_ms, "ok": True}
    return lat


MODELS = [
    ("Llama 3.3 70B · Q4_K_M", 42.5, 70, None),
    ("Qwen3 235B-A22B · Q4_K_M", 142.0, 235, 22),
    ("GLM-4.5-Air 106B · Q4_K_M", 68.0, 106, 12),
    ("GLM-5 744B · IQ2_XXS", 208.0, 744, 40),
    ("GLM-5 744B · Q2_K", 262.0, 744, 40),
    ("GLM-5 744B · Q4_K_M", 438.0, 744, 40),
]
MAIN = "rtx"
DRAFT_GB, ACCEPT = 0.6, 0.6

plt.rcParams.update({"figure.facecolor": "#0b0e13", "axes.facecolor": "#121720",
                     "axes.edgecolor": "#2a3342", "axes.labelcolor": "#e6ebf2",
                     "text.color": "#e6ebf2", "xtick.color": "#a9b4c4", "ytick.color": "#a9b4c4",
                     "grid.color": "#232b38", "font.size": 10.5, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "savefig.dpi": 160})
C_NAIVE, C_OPT, C_ACC = "#7c879a", "#5eead4", "#7c9cff"
STEP_COLORS = ["#7c879a", "#7c9cff", "#c084fc", "#f472b6", "#5eead4"]


def run_models():
    rows = []
    L = links_with()
    for name, size, params, active in MODELS:
        m = opt.build_model(size, params, active, 4096)
        naive = opt.naive_plan(DEVICES, m, MAIN, L)
        steps = {
            "llama.cpp default": naive.get("tps") if naive.get("fits") else None,
        }
        kw = dict(allow_kv_quant=False, allow_moe_hybrid=False)
        a = opt.optimize(DEVICES, m, MAIN, L, **kw)
        steps["+ device selection & placement"] = a.get("tps") if a.get("fits") else None
        b = opt.optimize(DEVICES, m, MAIN, L, allow_kv_quant=True, allow_moe_hybrid=False)
        steps["+ KV cache quantization"] = b.get("tps") if b.get("fits") else None
        c = opt.optimize(DEVICES, m, MAIN, L, allow_kv_quant=True, allow_moe_hybrid=True)
        steps["+ MoE experts in CPU RAM"] = c.get("tps") if c.get("fits") else None
        d = opt.optimize(DEVICES, m, MAIN, L, draft_gb=DRAFT_GB, draft_accept=ACCEPT)
        steps["+ speculative decoding"] = d.get("tps") if d.get("fits") else None
        bm = opt.best_main(DEVICES, m, L, draft_gb=DRAFT_GB, draft_accept=ACCEPT)
        rows.append({"model": name, "size_gb": size, "steps": steps,
                     "final_plan": d, "naive_assign": naive.get("assign"),
                     "naive_fits": naive.get("fits", False),
                     "best_main": bm["main_id"] if bm else None,
                     "best_main_tps": round(bm["plan"]["tps"], 2) if bm else None})
    return rows


def chart_ablation(rows):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    labels = list(rows[0]["steps"].keys())
    n, width = len(labels), 0.16
    for i, lab in enumerate(labels):
        xs = [k + (i - (n - 1) / 2) * width for k in range(len(rows))]
        vals = [r["steps"][lab] or 0 for r in rows]
        bars = ax.bar(xs, vals, width * 0.92, color=STEP_COLORS[i], label=lab)
        for x, v, r in zip(xs, vals, rows):
            if r["steps"][lab] is None:
                ax.text(x, 0.3, "✗", ha="center", color="#f87171", fontsize=11, fontweight="bold")
    for k, r in enumerate(rows):
        base, final = r["steps"]["llama.cpp default"], r["steps"]["+ speculative decoding"]
        if base and final:
            ax.text(k, max(v or 0 for v in r["steps"].values()) + 0.8, f"{final / base:.1f}×",
                    ha="center", color=C_OPT, fontweight="bold")
        elif final and not base:
            ax.text(k, max(v or 0 for v in r["steps"].values()) + 0.8, "now fits",
                    ha="center", color=C_OPT, fontweight="bold")
        elif not final:
            ax.text(k, 1.2, "does not fit\n(needs more devices)", ha="center", color="#f87171",
                    fontsize=9)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["model"].replace(" · ", "\n") for r in rows], fontsize=9.5)
    ax.set_ylabel("estimated tokens / second (higher is better)")
    ax.set_title("What each optimization adds — same 5 office devices (✗ = does not fit)")
    ax.set_ylim(0, max(max(v or 0 for v in r["steps"].values()) for r in rows) * 1.18)
    fig.text(0.01, 0.01, "Speculative step assumes a small same-family draft model (0.6 GB, 60% "
             "acceptance). Estimates from LocalMesh's cost model, not hardware benchmarks.",
             fontsize=8.5, color="#a9b4c4")
    ax.grid(axis="y", alpha=.6)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "1_optimizations_by_model.png")
    plt.close(fig)


def chart_placement(rows):
    pick = [r for r in rows if r["naive_fits"] and r["final_plan"].get("fits")][:3]
    fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.0 + 1.6 * len(pick)), squeeze=False)
    colors = {"rtx": "#5eead4", "studio": "#7c9cff", "ws": "#f472b6", "mbp": "#facc15",
              "air": "#fb923c"}
    names = {d["id"]: d["name"] for d in DEVICES}
    for ax, r in zip(axes[:, 0], pick):
        for row_i, (label, assign) in enumerate([("optimized", r["final_plan"]["assign"]),
                                                 ("default", r["naive_assign"])]):
            left = 0
            total = sum(assign.values())
            for did in [d["id"] for d in DEVICES]:
                n = assign.get(did, 0)
                if n:
                    ax.barh(row_i, n / total * 100, left=left, color=colors[did],
                            edgecolor="#0b0e13")
                    if n / total > 0.06:
                        ax.text(left + n / total * 50, row_i, f"{n}", ha="center", va="center",
                                color="#06110f", fontweight="bold", fontsize=9)
                    left += n / total * 100
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["optimized", "llama.cpp default"])
        ax.set_xlim(0, 100)
        ax.set_title(r["model"] + " — layers per device", fontsize=11, loc="left")
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    fig.legend(handles, [names[k] for k in colors], loc="lower center", ncol=5, frameon=False,
               fontsize=9)
    fig.suptitle("Where the layers go: slow Wi-Fi and low-bandwidth devices get dropped",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "2_layer_placement.png")
    plt.close(fig)


def chart_latency():
    m = opt.build_model(42.5, 70, None, 4096)
    xs = [0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30]
    naive, best, dev_count = [], [], []
    for lat in xs:
        L = links_with(wifi_ms=lat, eth_ms=lat)
        n = opt.naive_plan(DEVICES, m, MAIN, L)
        o = opt.optimize(DEVICES, m, MAIN, L, allow_moe_hybrid=False)
        naive.append(n["tps"])
        best.append(o["tps"])
        dev_count.append(len(o["used_ids"]))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(xs, naive, "o-", color=C_NAIVE, label="llama.cpp default (all 5 devices)")
    ax.plot(xs, best, "o-", color=C_OPT, lw=2.4, label="LocalMesh optimizer")
    for x, y, c in zip(xs, best, dev_count):
        ax.annotate(f"{c} dev", (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=C_OPT)
    ax.axvspan(0.1, 1, color="#14532d", alpha=.25)
    ax.axvspan(2, 30, color="#713f12", alpha=.18)
    ax.text(0.12, min(naive) * 1.05, "Ethernet / Thunderbolt", color="#86efac", fontsize=9)
    ax.text(2.3, min(naive) * 1.05, "typical office Wi-Fi", color="#fde68a", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.minorticks_off()
    ax.set_xlabel("round-trip latency between devices (ms, log scale)")
    ax.set_ylabel("tokens / second")
    ax.set_title("Llama 70B: latency, not bandwidth, decides speed (TPI-LLM finding)")
    ax.grid(alpha=.6)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    fig.tight_layout()
    fig.savefig(OUT / "3_latency_sensitivity.png")
    plt.close(fig)
    return list(zip(xs, naive, best, dev_count))


def chart_speculative():
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    alphas = [a / 100 for a in range(30, 92, 4)]
    main = DEVICES[0]
    for label, T, col in [("single machine (40 ms/token)", 0.040, C_NAIVE),
                          ("cluster over Ethernet (80 ms/token)", 0.080, C_ACC),
                          ("cluster with a Wi-Fi hop (130 ms/token)", 0.130, C_OPT)]:
        ys = []
        for a in alphas:
            s = opt.speculative(T, main, DRAFT_GB, a, False)
            ys.append(s["speedup"] if s else 1.0)
        ax.plot([a * 100 for a in alphas], ys, "o-", color=col, label=label)
    ax.axhline(1, color="#f87171", lw=1, ls="--")
    ax.text(31, 1.04, "no gain → optimizer turns it off", color="#f87171", fontsize=9)
    ax.set_xlabel("draft acceptance rate (%)")
    ax.set_ylabel("speed-up vs no speculation")
    ax.set_title("Speculative decoding pays off more on a cluster (one round trip, many tokens)")
    ax.grid(alpha=.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "4_speculative_decoding.png")
    plt.close(fig)


# ---------------------------------------------------------------- failure simulation
def failure_timeline():
    """Simulate: worker dies at t=60s, LocalMesh detects (<=4 s heartbeat), re-plans, reloads."""
    m = opt.build_model(68.0, 106, 12, 4096)
    L = links_with()
    full = opt.optimize(DEVICES, m, MAIN, L, draft_gb=DRAFT_GB, draft_accept=ACCEPT)
    victim = max((i for i in full["assign"] if i != MAIN), key=lambda i: full["assign"][i])
    survivors = [d for d in DEVICES if d["id"] != victim]
    after = opt.optimize(survivors, m, MAIN, L, draft_gb=DRAFT_GB, draft_accept=ACCEPT)
    # reload time: weights for layers that moved, pushed at ~110 MB/s effective over 1 GbE
    moved_gb = sum(max(after["assign"].get(i, 0) - full["assign"].get(i, 0), 0)
                   for i in after["assign"]) * m["size_gb"] / m["layers"]
    reload_s = moved_gb * 1024 / 280 + 6   # 2.5 GbE effective
    t, ys = [], []
    for s in range(0, int(64 + reload_s + 80)):
        t.append(s)
        if s < 60:
            ys.append(full["tps"])
        elif s < 64:
            ys.append(0)                      # requests fail until heartbeat notices
        elif s < 64 + reload_s:
            ys.append(0)                      # re-deploying without the dead device
        else:
            ys.append(after["tps"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(t, ys, step="post", color=C_OPT, alpha=.35)
    ax.plot(t, ys, color=C_OPT, drawstyle="steps-post")
    ax.axvline(60, color="#f87171", ls="--")
    ax.text(61, full["tps"] * 1.02, f"{dict((d['id'], d['name']) for d in DEVICES)[victim]}\ngoes offline",
            color="#f87171", fontsize=9)
    ax.axvline(64, color="#facc15", ls=":")
    ax.text(65, full["tps"] * 0.6, "heartbeat detects\n(≤ 4 s) → re-plan", color="#fde68a",
            fontsize=9)
    ax.text(64 + reload_s + 2, after["tps"] * 1.05,
            f"serving again: {after['tps']:.1f} tok/s\non {len(after['used_ids'])} devices",
            color=C_OPT, fontsize=9)
    ax.set_xlabel("seconds")
    ax.set_ylabel("tokens / second")
    ax.set_title("Failure drill — GLM-4.5-Air: a worker drops out mid-session")
    ax.set_ylim(0, full["tps"] * 1.3)
    ax.grid(alpha=.5)
    fig.tight_layout()
    fig.savefig(OUT / "5_failure_recovery.png")
    plt.close(fig)
    return {"victim": victim, "before_tps": full["tps"], "after_tps": after["tps"],
            "reload_s": round(reload_s, 1), "moved_gb": round(moved_gb, 1)}


ERROR_DRILLS = [
    ("Worker goes offline", "heartbeat every 4 s marks it offline; deployment shows 'degraded'",
     "one click: re-plan & redeploy without it", "e2e: failure drill"),
    ("Office Wi-Fi blocks discovery", "device list stays empty / partial",
     "add the device by IP on the Devices page", "e2e: manual add"),
    ("Firewall blocks port 7860", "manual add returns 'Could not reach … firewall?'",
     "allow Python in the firewall prompt", "e2e: unreachable add"),
    ("Different cluster token", "peer answers 401 → 'different cluster token'",
     "start all devices with the same --token", "e2e: token"),
    ("No shared network of the chosen type", "test cell: 'no shared thunderbolt network'",
     "pick Auto or Ethernet", "e2e: thunderbolt test"),
    ("llama.cpp missing on a device", "pre-flight ✗ + deploy step error",
     "run scripts/install_llama_cpp.sh", "e2e: missing binary"),
    ("Model file missing on main", "deploy step 'model file not found'",
     "download it to the main device first", "e2e: missing model"),
    ("Model too big for the pool", "plan says DOES NOT FIT with needed vs available GB",
     "smaller quant, KV q8_0, MoE→CPU RAM, or add devices", "unit: optimizer"),
    ("Gated / private Hugging Face model", "download error: 'set HF_TOKEN and accept licence'",
     "export HF_TOKEN=… before starting", "code path"),
    ("Not enough disk", "download refused before starting", "free disk or pick another device",
     "UI check"),
    ("Main server crashes (OOM etc.)", "status: main not running + last log lines",
     "lower context, KV q8_0, or add a device", "e2e: main exit"),
]


def chart_errors():
    fig, ax = plt.subplots(figsize=(13, 0.36 * len(ERROR_DRILLS) + 0.9))
    ax.axis("off")
    cols = ["Failure injected", "How LocalMesh detects / shows it", "Recovery", "Covered by"]
    table = ax.table(cellText=[list(r) for r in ERROR_DRILLS], colLabels=cols, loc="center",
                     cellLoc="left", colWidths=[0.2, 0.37, 0.3, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#232b38")
        cell.set_facecolor("#171d28" if r == 0 else ("#121720" if r % 2 else "#0f141c"))
        cell.get_text().set_color("#5eead4" if r == 0 else "#e6ebf2")
        if r == 0:
            cell.get_text().set_fontweight("bold")
    ax.set_title("Error drills: every failure has a visible message and a fix", pad=4)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "6_error_drills.png")
    plt.close(fig)


def main():
    rows = run_models()
    chart_ablation(rows)
    chart_placement(rows)
    lat = chart_latency()
    chart_speculative()
    fail = failure_timeline()
    chart_errors()
    summary = []
    for r in rows:
        b, f = r["steps"]["llama.cpp default"], r["steps"]["+ speculative decoding"]
        summary.append({"model": r["model"], "size_gb": r["size_gb"],
                        "default_tps": round(b, 2) if b else None,
                        "optimized_tps": round(f, 2) if f else None,
                        "speedup": round(f / b, 2) if b and f else None,
                        "devices_used": r["final_plan"].get("used_ids"),
                        "kv": r["final_plan"].get("kv_type"),
                        "moe_hybrid": r["final_plan"].get("moe_hybrid"),
                        "best_main": r["best_main"], "best_main_tps": r["best_main_tps"],
                        "steps": {k: (round(v, 2) if v else None) for k, v in r["steps"].items()}})
    (OUT / "results.json").write_text(json.dumps({"models": summary, "latency_sweep": lat,
                                                  "failure": fail}, indent=1))
    for s in summary:
        print(f"{s['model']:<30} default={s['default_tps']!s:>6}  optimized={s['optimized_tps']!s:>6}"
              f"  x{s['speedup']!s:<5} devices={s['devices_used']} kv={s['kv']} moe_cpu={s['moe_hybrid']}"
              f" best_main={s['best_main']}({s['best_main_tps']})")
    print("failure drill:", fail)


if __name__ == "__main__":
    main()
