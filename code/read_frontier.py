"""Read the frontier calibration and propose the final k lists.

The chance level is EXACTLY computable, which is what makes this a
frontier rather than a vibe: an F row's answer is 8 hex digits plus the EOS,
so a model that has learned NOTHING about the operation scores
(8 * 1/16 + 1) / 9 = 0.1667 per scored position. The old F measured
0.161-0.172 -- i.e. exactly chance, at every k.

So the frontier is the largest k whose measured accuracy is meaningfully above
0.1667, and the final lists are chosen from it:
  * the trained k values are the ones the model CAN learn (otherwise the
    training rows are noise);
  * one or more k values inside the trained RANGE but never seen (the
    interpolation test);
  * the k values beyond the trained range and beyond the frontier (the
    extrapolation test -- a stretch, as designed).
"""
import glob
import json
import os
import sys

CHANCE = (8 * (1 / 16) + 1) / 9      # 0.16666...
MARGIN = 0.02                          # "meaningfully above chance"


def per_k(metrics, prefix="val/accuracy_by_op/"):
    out = {}
    for k, v in metrics.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def main(root="runs", pattern="opdiffB_calib_*"):
    dirs = sorted(glob.glob(os.path.join(root, pattern + "*")))
    if not dirs:
        print("no calibration run found")
        return 1
    for d in dirs:
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            print(f"{os.path.basename(d)}: no metrics.json (interrupted)")
            continue
        m = json.load(open(mp, encoding="utf-8"))
        acc = per_k(m)
        ex = per_k(m, "val/exact_match_by_op/")
        dep = per_k(m, "recursion/avg_depth_by_op/")
        print("=" * 78)
        print(os.path.basename(d))
        print(f"  overall val loss {m.get('val/task_loss')}  "
              f"accuracy {m.get('val/accuracy')}  "
              f"exact match {m.get('val/exact_match')}")
        print(f"  chance level for an F row: {CHANCE:.4f}")
        print(f"  {'k':>4} {'accuracy':>10} {'exact':>8} {'depth':>8}  verdict")
        learnable = []
        for k in sorted(acc, key=lambda s: int(s[1:])):
            a = acc[k]
            e = ex.get(k)
            dp = dep.get(k)
            v = "LEARNED" if a > CHANCE + MARGIN else (
                "above chance" if a > CHANCE else "at chance")
            if a > CHANCE + MARGIN:
                learnable.append(int(k[1:]))
            print(f"  {k:>4} {a:>10.4f} "
                  f"{(('%8.4f' % e) if e is not None else '     N/A')} "
                  f"{(('%8.2f' % dp) if dp is not None else '     N/A')}  {v}")
        print(f"  learnable k (above chance + {MARGIN}): {learnable}")
        for sp in ("ood_interp", "ood_extrap"):
            if sp in m:
                print(f"  {sp}: loss {m.get(sp + '/task_loss')} "
                      f"acc {m.get(sp + '/accuracy')} exact {m.get(sp + '/exact_match')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
