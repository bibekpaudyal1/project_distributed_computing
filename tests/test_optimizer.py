"""Unit tests for the placement optimizer:  python tests/test_optimizer.py  (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localmesh import optimizer as opt  # noqa: E402

DEVS = [
    {"id": "gpu", "name": "4090", "usable_memory_gb": 22, "mem_bandwidth_gbs": 1000,
     "memory_kind": "nvidia-vram", "ram_total_gb": 128},
    {"id": "mac", "name": "studio", "usable_memory_gb": 150, "mem_bandwidth_gbs": 800},
    {"id": "air", "name": "air", "usable_memory_gb": 10, "mem_bandwidth_gbs": 100},
]
LINKS = {"gpu|mac": {"latency_ms": 0.4}, "gpu|air": {"latency_ms": 6}, "mac|air": {"latency_ms": 6}}


def test_drops_slow_wifi_device():
    r = opt.optimize(DEVS, opt.build_model(42.5, 70, None, 4096), "gpu", LINKS)
    assert r["fits"] and "air" not in r["used_ids"]
    assert any(e["id"] == "air" for e in r["excluded"])


def test_never_slower_than_default():
    for size, p, a in [(4, 7, None), (42.5, 70, None), (68, 106, 12), (142, 235, 22)]:
        m = opt.build_model(size, p, a, 4096)
        c = opt.compare(DEVS, m, "gpu", LINKS)
        if c["naive"]["fits"]:
            assert c["optimized"]["tps"] >= c["naive"]["tps"] * 0.999, (size, c)


def test_moe_offload_makes_glm5_fit():
    m = opt.build_model(208, 744, 40, 4096)
    c = opt.compare(DEVS, m, "gpu", LINKS)
    assert not c["naive"]["fits"] and c["optimized"]["fits"] and c["optimized"]["moe_hybrid"]
    assert "-ot" in c["optimized"]["flags"]


def test_refuses_impossible():
    assert not opt.optimize(DEVS, opt.build_model(800, 744, 40, 4096), "gpu", LINKS)["fits"]


def test_layers_sum_and_capacity():
    m = opt.build_model(142, 235, 22, 8192)
    r = opt.optimize(DEVS, m, "gpu", LINKS)
    assert sum(r["assign"].values()) == m["layers"]
    for i, pd in r["per_device"].items():
        cap = next(d for d in DEVS if d["id"] == i)["usable_memory_gb"]
        assert pd["hybrid"] or pd["mem_gb"] <= cap + 0.01


def test_speculation_only_when_it_helps():
    assert opt.speculative(0.08, DEVS[0], 0.6, 0.7)["speedup"] > 1.3
    assert opt.speculative(0.08, DEVS[0], 0.6, 0.05) is None  # bad draft -> off


def test_kv_quant_used_when_tight():
    m = opt.build_model(150, 235, 22, 32768)  # long context -> big KV cache
    r = opt.optimize(DEVS, m, "gpu", LINKS, allow_moe_hybrid=False)
    assert r["fits"] and r["kv_type"] != "f16" and "--cache-type-k" in r["flags"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("  PASS", name)
            except AssertionError as e:
                fails += 1
                print("  FAIL", name, str(e)[:300])
    sys.exit(1 if fails else 0)
