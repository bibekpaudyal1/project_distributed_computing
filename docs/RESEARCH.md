# Research behind the optimizer

Ten recent papers on running large models across mixed, memory-limited devices, what each
contributes, and exactly what LocalMesh does with it. "Implemented" means it is in
`localmesh/optimizer.py` / `server.py` and covered by tests. "Not implemented" means it would
need changes inside the inference engine (llama.cpp) itself.

| # | Paper | Venue | Key idea | In LocalMesh |
|---|-------|-------|----------|--------------|
| 1 | **Prima.cpp**: fast 30–70B inference on heterogeneous, low-resource home clusters (Li et al.) | ICLR 2026 | *Halda* scheduler co-optimizes per-device layer assignment **and device selection** under RAM/VRAM limits; Wi-Fi tolerant; mixed OS | **Implemented:** heterogeneity-aware layer split + automatic device selection (`optimize()`). *Not implemented:* piped-ring parallelism and mmap prefetching (engine-level). |
| 2 | **TPI-LLM**: serving 70B-scale LLMs on low-resource edge devices (Li et al.) | arXiv 2024 / journal 2025 | On edge networks, **link latency, not bandwidth**, is the bottleneck | **Implemented:** measured round-trip latency drives the cost model; slow (Wi-Fi) links are pruned first; UI recommends Ethernet/Thunderbolt. |
| 3 | **Helix**: serving LLMs over heterogeneous GPUs and network via max-flow (Mei et al.) | ASPLOS 2025 | Model placement as a graph/MILP problem over heterogeneous compute + links | **Implemented (small-scale exact version):** exhaustive subset search + greedy-optimal fill — exact for our linear cost model with ≤ ~10 devices. *Not implemented:* multi-request scheduling across replicas. |
| 4 | **EdgeShard**: collaborative LLM inference at the edge (Zhang et al.) | arXiv 2024 | Pick a subset of devices and partition layers to minimize latency | **Implemented:** same objective; also recommends the best *main* device (`best_main()`). |
| 5 | **LLM partitioning for low-latency inference at the edge** (Kafetzis et al.) | arXiv 2025 | Re-partition as conditions change; migrate blocks when cheaper | **Implemented (coarse):** health monitor marks deployments *degraded* when a device drops; one-click **re-plan & redeploy** on the survivors. |
| 6 | **KTransformers**: CPU/GPU hybrid inference for MoE models (Chen et al.) | SOSP 2025 | Keep attention/shared weights on GPU, routed experts in large CPU RAM | **Implemented:** for MoE models on an NVIDIA main device, the optimizer can place experts in system RAM (`-ot "\.ffn_(up|down|gate)_exps\.=CPU"`) — this is what makes GLM-5 IQ2 fit in the test pool. |
| 7 | **Cloud-grade SLOs for local MoE inference via CPU–GPU hybrid design** (Wang et al.) | OSDI 2026 | Local MoE needs careful CPU–GPU orchestration to stay responsive | **Used in the cost model:** expert reads on CPU are charged at CPU bandwidth with a scatter penalty, so hybrid is only chosen when it is needed or faster. |
| 8 | **EAGLE-3**: scaling up speculative decoding via training-time test (Li et al.) | NeurIPS 2025 | Draft many tokens cheaply, verify in one pass — up to 6.5× | **Implemented (vanilla draft model):** pick a small same-family GGUF as *Draft model*; the cluster verifies k tokens per network round trip (`-md`, `--draft-max`). *Not implemented:* EAGLE heads (need trained heads + engine support). |
| 9 | **Speculative decoding: performance or illusion?** (Liu et al.) | arXiv 2025/26 | Real gains depend on acceptance rate and verification cost | **Implemented:** k is chosen from an acceptance model; verification of MoE models is charged extra; speculation is switched off when the gain is < 5 %. |
| 10 | **KIVI**: tuning-free asymmetric 2-bit KV cache quantization (Liu et al.) | ICML 2024 | KV cache can be quantized aggressively with little quality loss | **Implemented (llama.cpp-supported form):** q8_0 / q4_0 KV cache (`--cache-type-k/v`, `-fa on`) when memory is tight or it lets a slow device be dropped; f16 preferred otherwise. |

Further reading (not used directly): ExpertFlow (DAC 2026) and DALI (2026) on predictive
expert caching for MoE offloading; MoBiLE (ASP-DAC 2026); Hyperion (2025) multi-tier
partitioning; Voltron (2026) elastic multi-device execution.

## The cost model (what the numbers mean)

Per generated token:

```
T = Σ_devices  layers_i × (active_bytes_per_layer / (bandwidth_i × 0.6) + 0.06 ms)
  + Σ_workers  (2 × measured_latency_i + 1.5 ms RPC overhead)
```

* MoE models only read their *active* experts per token, at a scatter penalty (×0.6).
* llama.cpp RPC is a star: the main device talks to each worker every token.
* For a fixed device set, filling the fastest devices first is optimal for this cost;
  all device subsets are enumerated (2ⁿ⁻¹, fine for office-sized clusters).
* Speculative decoding: `E[tokens] = (1-α^(k+1))/(1-α)` per verify pass;
  cost `k·t_draft + T·(1 + c·k)` with `c = 0.03` dense, `0.15` MoE.

The model was sanity-checked against public llama.cpp numbers (e.g. 70B Q4 ≈ 11–13 tok/s on
an M2 Ultra; GLM-4.5-Air ≈ 30–40 tok/s). It is for **comparing strategies**, not a benchmark.
Run the connection test before planning, otherwise a 4 ms link latency is assumed.

## Sources

- Prima.cpp — https://arxiv.org/abs/2504.08791
- TPI-LLM — https://arxiv.org/abs/2410.00531
- Helix — https://arxiv.org/abs/2406.01566
- EdgeShard — https://arxiv.org/abs/2405.14371
- Kafetzis et al. — https://arxiv.org/abs/2505.02533
- KTransformers — https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
- Local MoE CPU–GPU hybrid (OSDI'26) — https://arxiv.org/abs/2606.10493
- EAGLE-3 — https://arxiv.org/abs/2503.01840
- Speculative decoding: performance or illusion? — https://arxiv.org/abs/2601.11580
- KIVI — https://arxiv.org/abs/2402.02750
