"""derive_lang_ffn_mult.py - T-L6.3: the parameter-matched MoR width for language.

Run:
    C:/Users/vedan/anaconda3/python.exe code/derive_lang_ffn_mult.py

WHY THIS CANNOT BE COPIED FROM ARITHMETIC. `config.CANONICAL_FFN_MULT` is
`{moe: 4, mor: 24, more: 4}`, and 24 was solved for the ARITHMETIC model: MoR runs one
expert instead of six, so its single FFN has to be widened until the parameter counts
match, and the multiplier that achieves that depends on everything else in the model.
The language arms differ in three ways that all move the answer:

  + attention, ~4*d^2 + 4*d per block, added EQUALLY to all three arms;
  + a tied token embedding, V*d_model, also equal across arms;
  - no `step_proj`, no `op_embed`, no `regression_head`, no `cls_head`.

The two equal-across-arms additions are the reason the total-count criterion alone is
too easy to satisfy on language: a 2,097,152-element embedding shared by both arms sits
in both numerators and both denominators and shrinks the RELATIVE gap for free. So
T-L6.3 requires the match on BOTH totals and non-embedding counts, and the second is
the one that is actually about the expert stack.

WHAT THIS SCRIPT DOES NOT DO. It does not write `config.py`. It prints the derivation
so the number entering `CANONICAL_FFN_MULT_LANGUAGE` is one a human accepted from a
table, not one a script chose silently -- the same discipline `canonical_spec*.json`
applies to every other frozen field.
"""

import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from more.model import MoREModel, TASK_LANGUAGE      # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL      # noqa: E402

# The frozen language shape, from canonical_spec_language.json:enforced_fields.
D_MODEL, N_HEADS, NUM_BLOCKS, VOCAB = 256, 4, 1, 8192
# seq_len is Gate L5's to freeze; the positional table is [seq_len, d_model] and so
# does enter the count. 256 is config.LANGUAGE_SEQ_LEN_DEFAULT, the provisional value,
# and the sensitivity of the answer to it is printed below.
SEQ_LEN = 256
MAX_DEPTH = 7
TOLERANCE = 0.05


def build(num_experts, ffn_mult, seq_len=SEQ_LEN, max_depth=MAX_DEPTH):
    return MoREModel(
        step_feat_dim=8, d_model=D_MODEL, num_experts=num_experts,
        max_depth=max_depth, num_blocks=NUM_BLOCKS, dropout=0.1,
        ffn_mult=ffn_mult, attention=True, n_heads=N_HEADS,
        max_seq_len=seq_len, task=TASK_LANGUAGE, vocab_size=VOCAB,
    )


def counts(m):
    """`(total, non_embedding)` parameter counts.

    Non-embedding excludes `tok_embed` (which IS `lm_head`, tied) and `pos_embed`.
    Those two are identical across arms by construction, so including them in the
    denominator of a relative gap makes the gap look smaller than the expert stack's
    actually is -- which is precisely why T-L6.3 asks for both numbers.
    """
    total, emb = 0, 0
    for n, p in m.named_parameters():
        total += p.numel()
        if n.startswith("tok_embed") or n.startswith("pos_embed") \
           or n.startswith("lm_head"):
            emb += p.numel()
    return total, total - emb


def main():
    ref = build(NUM_EXPERTS_CANONICAL, 4)
    t_more, n_more = counts(ref)
    print(f"MoRE reference (E={NUM_EXPERTS_CANONICAL}, ffn_mult=4, attention=True, "
          f"V={VOCAB}, d_model={D_MODEL}, seq_len={SEQ_LEN}):")
    print(f"  total          {t_more:>10,d}")
    print(f"  non-embedding  {n_more:>10,d}")
    print(f"  embedding      {t_more - n_more:>10,d}  (tied, shared by every arm)")
    print()

    rows, best = [], None
    print(f"{'ffn_mult':>8s} {'MoR total':>12s} {'rel(total)':>11s} "
          f"{'MoR non-emb':>12s} {'rel(non-emb)':>13s} {'both<5%':>8s}")
    for fm in range(1, 41):
        m = build(1, fm)
        t, n = counts(m)
        rt = abs(t - t_more) / t_more
        rn = abs(n - n_more) / n_more
        ok = rt < TOLERANCE and rn < TOLERANCE
        rows.append({"ffn_mult": fm, "mor_total": t, "rel_total": rt,
                     "mor_non_embedding": n, "rel_non_embedding": rn,
                     "within_tolerance": ok})
        flag = "  YES" if ok else ""
        if 14 <= fm <= 30 or ok:
            print(f"{fm:>8d} {t:>12,d} {rt:>10.3%} {n:>12,d} {rn:>12.3%} {flag:>8s}")
        # The best candidate is chosen on the NON-EMBEDDING gap, because that is the
        # number about the expert stack. Ties broken toward the smaller multiplier.
        if best is None or rn < best["rel_non_embedding"]:
            best = rows[-1]

    print()
    ok_rows = [r for r in rows if r["within_tolerance"]]
    print(f"within {TOLERANCE:.0%} on BOTH criteria: "
          f"{[r['ffn_mult'] for r in ok_rows] or 'none'}")
    print(f"best on the non-embedding gap: ffn_mult={best['ffn_mult']}  "
          f"rel(non-emb)={best['rel_non_embedding']:.4%}  "
          f"rel(total)={best['rel_total']:.4%}")
    print()
    print(f"arithmetic value for comparison: ffn_mult.mor = 24 "
          f"(config.CANONICAL_FFN_MULT)")

    # How much does the still-unfrozen seq_len move the answer? If it does not, Gate
    # L5's decision and this one are independent and can be made in either order.
    print()
    print("sensitivity to the unfrozen seq_len (best ffn_mult on non-embedding gap):")
    for sl in (128, 256, 512, 1024):
        r2 = build(NUM_EXPERTS_CANONICAL, 4, seq_len=sl)
        tm, nm = counts(r2)
        cand = min(
            ((fm, abs(counts(build(1, fm, seq_len=sl))[1] - nm) / nm)
             for fm in range(1, 41)), key=lambda kv: kv[1])
        print(f"  seq_len {sl:>5d} -> ffn_mult {cand[0]:>3d}  "
              f"rel(non-emb) {cand[1]:.4%}")

    out = {
        "reference": {"total": t_more, "non_embedding": n_more,
                      "embedding": t_more - n_more},
        "shape": {"d_model": D_MODEL, "n_heads": N_HEADS,
                  "num_blocks": NUM_BLOCKS, "vocab_size": VOCAB,
                  "seq_len": SEQ_LEN, "max_depth": MAX_DEPTH,
                  "attention": True},
        "tolerance": TOLERANCE,
        "rows": rows,
        "within_tolerance": [r["ffn_mult"] for r in ok_rows],
        "best_on_non_embedding": best["ffn_mult"],
    }
    dest = os.path.join(CODE, "lang_ffn_mult_derivation.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print()
    print(f"derivation -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
