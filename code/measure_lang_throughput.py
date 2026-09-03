"""measure_lang_throughput.py - T-L7.0: what the 6 GB card actually allows.

Run:
    D:/res/git/MoRE/.venv_cuda/Scripts/python.exe code/measure_lang_throughput.py

Sweeps `(batch_size, seq_len)` for the frozen language shape and records tokens/s, peak
VRAM and whether the step OOMs. `batch_size` and `seq_len` are `null` in
`code/canonical_spec_language.json` precisely so the proxy guard refuses every canonical
claim until they are measured -- that refusal is the mechanism working, and this is the
measurement that lifts it.

WHY THE ARITHMETIC PROTOCOL'S NUMBER IS USELESS HERE. `batch_size = 768` was chosen for
records whose largest activation is `[768, 7, 12]`. A language step's largest tensor is the
LOGITS, `[B, S, V]` -- at `B=32, S=256, V=8192` that is 67.1 M floats, 268 MB in fp32, and
another 268 MB for its gradient. Batch size and sequence length multiply into that, so the
constraint has to be measured rather than carried.

SYNTHETIC INPUT IDS, DELIBERATELY. The grid needs `seq_len` values the corpus is not packed
at, and repacking wikitext-2 four times to measure VRAM would be measuring the dataloader,
not the card. Random ids in `[0, V)` produce exactly the same tensor shapes and the same
kernel launches; what they do not reproduce is host-to-device copy and page-cache
behaviour, so the tokens/s here is an UPPER BOUND on the end-to-end rate. The real rate is
measured separately by an actual run and both numbers go in the ledger -- quoting only the
synthetic one would overstate throughput.

PEAK MEMORY IS `max_memory_allocated`, NOT `memory_allocated`. The transient peak inside the
backward pass is what OOMs a run, and the steady-state figure misses it entirely. Reset
between configurations so one config's peak cannot be attributed to the next.
"""

import argparse
import json
import os
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch                                              # noqa: E402
import torch.nn.functional as F                           # noqa: E402

from more.model import MoREModel, TASK_LANGUAGE           # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL           # noqa: E402
from more.config import CANONICAL_FFN_MULT_LANGUAGE       # noqa: E402

# The frozen language shape, from canonical_spec_language.json:enforced_fields.
D_MODEL, N_HEADS, NUM_BLOCKS, VOCAB = 256, 4, 1, 8192
MAX_DEPTH = 7


def build(arch, seq_len, device):
    """The arm at its budget-matched width, exactly as `engine.train` builds it."""
    e = 1 if arch == "mor" else NUM_EXPERTS_CANONICAL
    md = 1 if arch == "moe" else MAX_DEPTH
    return MoREModel(
        step_feat_dim=8, d_model=D_MODEL, num_experts=e, max_depth=md,
        num_blocks=NUM_BLOCKS, dropout=0.1,
        ffn_mult=CANONICAL_FFN_MULT_LANGUAGE[arch],
        attention=True, n_heads=N_HEADS, max_seq_len=seq_len,
        task=TASK_LANGUAGE, vocab_size=VOCAB,
        num_families=NUM_EXPERTS_CANONICAL,
    ).to(device)


def one_config(arch, batch, seq_len, device, warmup=2, timed=5):
    """`(tokens_per_sec, peak_mib, oom, note)` for one point of the grid."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(42)
        model = build(arch, seq_len, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        ids = torch.randint(0, VOCAB, (batch, seq_len), device=device)
        sm = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        se = torch.randint(0, NUM_EXPERTS_CANONICAL, (batch, seq_len), device=device)
        so = torch.full((batch, seq_len), -1, dtype=torch.long, device=device)

        def step():
            opt.zero_grad(set_to_none=True)
            out = model(ids, sm, se, so)
            loss = F.cross_entropy(
                out[0][:, :-1, :].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
            # The ponder cost is in the real objective, so it is in the timing: it adds a
            # backward path through every halt head and leaving it out would understate
            # both time and memory.
            (loss + 0.001 * out[4]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        tps = (timed * batch * seq_len) / dt
        del model, opt, ids, sm, se, so
        return tps, peak, False, ""
    except torch.cuda.OutOfMemoryError as exc:
        # Cleared before returning, or the next configuration inherits the fragmentation
        # and reports an OOM that is really this one's.
        torch.cuda.empty_cache()
        return None, None, True, str(exc).split("\n")[0][:110]
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if "out of memory" in str(exc).lower():
            return None, None, True, str(exc).split("\n")[0][:110]
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", default="more", choices=["moe", "mor", "more"],
                    help="MoRE is the reference: it is the widest arm, so a "
                         "(batch, seq_len) that fits MoRE fits all three")
    ap.add_argument("--batches", default="8,16,32,64,128")
    ap.add_argument("--seq_lens", default="128,256,512,1024")
    ap.add_argument("--timed", type=int, default=5)
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. T-L7.0 is a measurement of the real card and there "
            "is nothing to report from a CPU: use "
            "D:/res/git/MoRE/.venv_cuda/Scripts/python.exe."
        )
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[sweep] {name}  {total:.2f} GiB  torch {torch.__version__}")
    print(f"[sweep] arch={args.arch} d_model={D_MODEL} V={VOCAB} "
          f"ffn_mult={CANONICAL_FFN_MULT_LANGUAGE[args.arch]} "
          f"max_depth={1 if args.arch == 'moe' else MAX_DEPTH} "
          f"timed_steps={args.timed}")
    print()
    print(f"{'seq_len':>8s} {'batch':>6s} {'tokens/batch':>13s} "
          f"{'tokens/s':>10s} {'peak MiB':>10s} {'% of VRAM':>10s} {'OOM':>5s}")

    rows = []
    for sl in [int(x) for x in args.seq_lens.split(",")]:
        for b in [int(x) for x in args.batches.split(",")]:
            tps, peak, oom, note = one_config(args.arch, b, sl, device,
                                              timed=args.timed)
            rows.append({"arch": args.arch, "seq_len": sl, "batch_size": b,
                         "tokens_per_batch": b * sl,
                         "tokens_per_sec": tps, "peak_mib": peak, "oom": oom,
                         "note": note})
            if oom:
                print(f"{sl:>8d} {b:>6d} {b * sl:>13,d} {'--':>10s} {'--':>10s} "
                      f"{'--':>10s} {'YES':>5s}")
                # Larger batches at this seq_len cannot fit either; skip them rather
                # than spending a minute proving it four more times.
                break
            print(f"{sl:>8d} {b:>6d} {b * sl:>13,d} {tps:>10,.0f} {peak:>10,.0f} "
                  f"{100 * peak / (total * 1024):>9.1f}% {'no':>5s}")

    ok = [r for r in rows if not r["oom"]]
    best = max(ok, key=lambda r: r["tokens_per_sec"]) if ok else None
    print()
    if best:
        print(f"[sweep] fastest fitting config: seq_len={best['seq_len']} "
              f"batch={best['batch_size']} -> {best['tokens_per_sec']:,.0f} tok/s at "
              f"{best['peak_mib']:,.0f} MiB "
              f"({100 * best['peak_mib'] / (total * 1024):.1f}% of VRAM)")
        print(f"[sweep] NOTE: synthetic ids, so tokens/s is an UPPER BOUND on the "
              f"end-to-end rate -- no host-to-device copy and no page-cache effects. "
              f"Compare against a real run before freezing anything.")
    else:
        print("[sweep] every configuration OOMed, which is itself the finding")

    dest = os.path.join(CODE, f"lang_throughput_{args.arch}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"device": name, "total_vram_gib": total,
                   "torch": torch.__version__, "arch": args.arch,
                   "shape": {"d_model": D_MODEL, "n_heads": N_HEADS,
                             "num_blocks": NUM_BLOCKS, "vocab_size": VOCAB,
                             "max_depth": 1 if args.arch == "moe" else MAX_DEPTH,
                             "ffn_mult": CANONICAL_FFN_MULT_LANGUAGE[args.arch]},
                   "timed_steps": args.timed,
                   "synthetic_ids": True,
                   "tokens_per_sec_is_upper_bound": True,
                   "rows": rows}, fh, indent=2)
        fh.write("\n")
    print(f"[sweep] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
