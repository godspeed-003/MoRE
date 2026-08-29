"""
Capacity-vs-wall-clock microbenchmark.  NOT an experiment, NOT a canonical run.

Question: the canonical matrix ran with the GPU at ~0-30% utilization and ~2 W.
Would a BIGGER model use the GPU properly and therefore finish FASTER?

Method: build the real MoREModel and time forward+backward+step on synthetic
batches of the canonical shape. Synthetic input is the point -- it removes the
dataloader, so what is left is model compute plus Python/kernel-launch overhead.

Two sweeps, because they answer different halves of the question:

  A. BATCH sweep at canonical d_model=256. If ms/step is flat as the batch grows,
     the step is launch-bound and there is free GPU headroom. Where it starts
     rising linearly, the GPU has become the bottleneck.
  B. WIDTH sweep at canonical batch=768. Tells you what a bigger model costs.

Measurement hygiene, learned from a first pass that produced non-monotonic
numbers: a laptop GPU idling at 2 W is heavily downclocked, so a short burst
measures clock ramp rather than throughput. Every configuration therefore gets a
long warmup, and each timing is the MIN of several repeats -- min is the robust
estimator for benchmarks, since noise only ever adds time.

Reads nothing from runs/, writes nothing but stdout. Deletable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from more.model import MoREModel  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, FEAT = 7, 12
CANON_BATCH, CANON_D = 768, 256
WARMUP, ITERS, REPEATS = 12, 20, 3


def build(d_model: int, ffn_mult: int = 4):
    return MoREModel(
        step_feat_dim=FEAT, d_model=d_model, num_experts=6, max_depth=7,
        num_blocks=1, dropout=0.1, fixed_depth=False,
        routing_mode="top1_sparse", router_noise="none",
        ffn_mult=ffn_mult, num_families=6,
    ).to(DEV)


def bench(d_model: int, batch: int, ffn_mult: int = 4) -> dict:
    torch.manual_seed(0)
    model = build(d_model, ffn_mult)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params = sum(p.numel() for p in model.parameters())

    x = torch.randn(batch, STEPS, FEAT, device=DEV)
    mask = torch.ones(batch, STEPS, dtype=torch.bool, device=DEV)
    ops = torch.randint(0, 6, (batch, STEPS), device=DEV)
    experts = torch.randint(0, 6, (batch, STEPS), device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(x, mask, step_experts=experts, step_ops=ops)
        loss = 0.0
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if torch.is_tensor(t) and t.is_floating_point():
                loss = loss + t.float().sum()
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        step()
    if DEV == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    best = float("inf")
    for _ in range(REPEATS):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / ITERS)

    mem = torch.cuda.max_memory_allocated() / 2**20 if DEV == "cuda" else 0.0
    del model, opt, x, mask, ops, experts
    if DEV == "cuda":
        torch.cuda.empty_cache()
    tokens = batch * STEPS
    return {"d_model": d_model, "batch": batch, "params": params,
            "ms": best * 1e3, "tok_s": tokens / best, "peak_mib": mem}


def table(rows: list[dict], vary: str, base_ms: float) -> None:
    print(f"{vary:>8s} {'params':>12s} {'ms/step':>9s} {'tok/s':>10s} "
          f"{'peak MiB':>9s} {'x canon ms':>11s} {'ms/token':>10s}")
    for r in rows:
        print(f"{r[vary]:>8d} {r['params']:>12,d} {r['ms']:>9.2f} "
              f"{r['tok_s']:>10,.0f} {r['peak_mib']:>9.0f} "
              f"{r['ms']/base_ms:>11.2f} "
              f"{r['ms']*1e3/(r['batch']*STEPS):>10.4f}")


def main() -> int:
    print(f"device={DEV}  steps={STEPS}  iters={ITERS}  repeats={REPEATS} (min taken)")
    if DEV == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"canonical point: d_model={CANON_D}, batch={CANON_BATCH}")

    print("\n== A. BATCH sweep at canonical d_model=256 ==")
    print("   flat ms/step => launch-bound, free GPU headroom."
          "  rising linearly => GPU-bound.")
    rows = []
    for b in (96, 192, 384, 768, 1536, 3072, 6144):
        try:
            rows.append(bench(CANON_D, b))
        except RuntimeError as e:
            print(f"{b:>8d}  FAILED (likely OOM): {str(e)[:50]}")
            break
    canon = next(r for r in rows if r["batch"] == CANON_BATCH)
    table(rows, "batch", canon["ms"])

    print("\n== B. WIDTH sweep at canonical batch=768 ==")
    rows_w = []
    for d in (256, 384, 512, 768, 1024):
        try:
            rows_w.append(bench(d, CANON_BATCH))
        except RuntimeError as e:
            print(f"{d:>8d}  FAILED (likely OOM): {str(e)[:50]}")
            break
    table(rows_w, "d_model", canon["ms"])

    print("\nReading:")
    print("  * 'ms/token' falling as batch grows is unused GPU capacity being")
    print("    absorbed at no extra wall-clock cost. Once it flattens, the GPU is")
    print("    saturated and more work costs proportional time.")
    print("  * In sweep B, 'x canon ms' is what a wider model COSTS. A wider model")
    print("    can never make a run finish sooner -- it adds work to the same loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
