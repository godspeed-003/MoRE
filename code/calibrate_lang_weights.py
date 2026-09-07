"""calibrate_lang_weights.py - T-L7.1: measure the two loss weights, do not carry them.

Run:
    D:/res/git/MoRE/.venv_cuda/Scripts/python.exe code/calibrate_lang_weights.py

`canonical_spec_language.json` leaves `routing_balance_weight` and `halting_weight` NULL
with a stated reason: both were selected at T4.2 against ARITHMETIC magnitudes, where the
task loss is an MSE near 0.06. The language task loss is a per-token cross-entropy that
starts at ln(8192) = 9.01 and was still 4.27 after eight epochs -- **two orders of magnitude
larger**. Carrying 0.001 would not be reuse, it would be an untested assumption wearing a
frozen number's authority.

MEASURED, NOT ARGUED. On the 8-epoch GPU run at `routing_balance = 0.001`:

    task_loss            4.2695
    routing_balance_loss 0.3462   -> weighted contribution 8.1e-05 x task loss
    ponder_cost          0.6770   -> weighted contribution 1.6e-04 x task loss

Arithmetic's T4.2 acceptance target was 0.017 x the task loss. Matching that ratio needs
`0.017 * 4.2695 / 0.3462 = 0.21` for balance and `0.017 * 4.2695 / 0.6770 = 0.107` for
ponder -- roughly 200x and 100x the arithmetic values.

WHY A RATIO MATCH IS NOT ENOUGH, AND WHY THIS SCRIPT RUNS MODELS. The ponder cost pushes
depth DOWN, and the balance term pushes the router toward uniform load. A 100x increase is
not a rescaling, it is a different objective: at `halting = 0.001` the 8-epoch run reached
`avg_depth = 5.16` of 7, and there is no way to know from a ratio what it reaches at 0.107.
So each candidate is RUN, and the numbers that decide are behavioural -- `avg_depth`,
`expert_load_entropy_normalized` against the POS partition's own 0.8884, and whether
`val_loss` still beats the corpus floor.

DEV CORPUS, SHORT RUNS, AND THAT IS A STATED LIMITATION. wikitext-2 at 3 epochs is ~4 min
per point on the 3050, which is what makes a grid affordable. It is a CALIBRATION corpus
here, and T-L2.0's finding applies: wikitext-2 shares its val split byte-for-byte with
wikitext-103, so these runs may pick a weight but may never be quoted as a result.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# The target from arithmetic's T4.2: the weighted auxiliary term should sit at this
# fraction of the task loss. Carried as a TARGET RATIO rather than as a weight, which is
# the whole point -- the ratio is the thing T4.2 justified, the weight was its solution on
# a different loss scale.
TARGET_RATIO = 0.017

# Recommended by T-L7.0's sweep on the 6 GB card: 16,384 tokens/step at ~67% VRAM, at the
# length the corpus is already packed at.
SEQ_LEN, BATCH = 256, 64
EPOCHS = 3
CORPUS = "wikitext-2"
DEV_FLOOR = 5.398304166059712      # wikitext-2 bigram floor, T-L2.5


def write_config(dest, routing_balance, halting, lr, dropout, weight_decay):
    """A config file per grid point, because `--routing_balance` is the only weight with a
    CLI flag and inventing three more flags for a calibration script would put scaffolding
    into the run interface. `--config` is the supported path."""
    with open(os.path.join(CODE, "config.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["loss_weights"]["routing_balance"] = routing_balance
    cfg["loss_weights"]["halting"] = halting
    # apply_task refuses a declared non-zero family_cls on language (§7.3), and it is the
    # base config's arithmetic 0.5 that would otherwise be declared.
    cfg["loss_weights"]["family_cls"] = 0.0
    cfg["training"]["lr"] = lr
    cfg["training"]["weight_decay"] = weight_decay
    cfg["model"]["dropout"] = dropout
    # Validate every epoch: three epochs with the default interval of 2 would give two
    # validation points and the last-epoch number is what decides.
    cfg["logging"]["log_interval"] = 1
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return dest


def run_point(tag, cfg_path, seed, env):
    """One calibration run. Returns its `metrics.json`, or None with the reason printed."""
    cmd = [os.environ.get("LANG_PY", sys.executable), "train.py",
           "--architecture", "more", "--task", "language", "--corpus", CORPUS,
           "--config", cfg_path, "--epochs", str(EPOCHS),
           "--batch_size", str(BATCH), "--seq_len", str(SEQ_LEN),
           "--seed", str(seed), "--experiment_group", f"lang_calib_{tag}"]
    res = subprocess.run(cmd, cwd=CODE, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  [{tag}] FAILED rc={res.returncode}: "
              f"{res.stderr.strip().splitlines()[-1][:150] if res.stderr.strip() else ''}")
        return None
    # The run directory is named from the config hash, so find it by mtime rather than by
    # reconstructing the name -- reconstructing it would duplicate run_context's logic.
    runs = os.path.join(os.path.dirname(CODE), "runs")
    cands = [os.path.join(runs, d) for d in os.listdir(runs) if d.startswith("langB_")]
    newest = max(cands, key=os.path.getmtime)
    mp = os.path.join(newest, "metrics.json")
    if not os.path.exists(mp):
        print(f"  [{tag}] no metrics.json in {os.path.basename(newest)}")
        return None
    with open(mp, "r", encoding="utf-8") as fh:
        return json.load(fh), os.path.basename(newest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--balance", default="0.001,0.05,0.21,1.0",
                    help="routing_balance candidates; 0.001 is the arithmetic value kept "
                         "as the reference point, 0.21 is the T4.2 ratio match")
    ap.add_argument("--halting", default="0.001,0.03,0.107,0.5",
                    help="ponder weight candidates; 0.107 is the T4.2 ratio match")
    ap.add_argument("--lr", default="0.001")
    ap.add_argument("--dropout", default="0.0,0.1")
    ap.add_argument("--weight_decay", default="0.0001")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--stage", default="weights", choices=["weights", "reg"],
                    help="weights = the (balance, halting) grid; reg = (lr, dropout)")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("WANDB_DIR", os.environ.get("TEMP", "."))

    tmp = os.path.join(os.environ.get("TEMP", "."), "more_calib")
    os.makedirs(tmp, exist_ok=True)

    rows = []
    if args.stage == "weights":
        # ONE AXIS AT A TIME against the arithmetic reference, not a full cross product.
        # The two terms act on different things -- balance on the router, ponder on depth --
        # and a 4x4 grid would cost 16 runs to answer two separable questions.
        grid = ([(b, 0.001) for b in [float(x) for x in args.balance.split(",")]]
                + [(0.001, h) for h in [float(x) for x in args.halting.split(",")]
                   if h != 0.001])
        print(f"[calib] {len(grid)} points, {EPOCHS} epochs each, {CORPUS}, "
              f"seq_len={SEQ_LEN} batch={BATCH} seed={args.seed}")
        print(f"{'balance':>8s} {'halting':>8s} {'val_loss':>9s} {'vs floor':>9s} "
              f"{'depth':>6s} {'entropy':>8s} {'bal/task':>9s} {'ami':>7s}")
        for bal, halt in grid:
            tag = f"b{bal}_h{halt}".replace(".", "p")
            cfg = write_config(os.path.join(tmp, f"cfg_{tag}.json"), bal, halt,
                               float(args.lr.split(",")[0]), 0.0,
                               float(args.weight_decay.split(",")[0]))
            got = run_point(tag, cfg, args.seed, env)
            if not got:
                continue
            m, rd = got
            row = {"routing_balance": bal, "halting": halt,
                   "val_loss": m.get("val/task_loss"),
                   "nats_below_floor": m.get("val/nats_below_bigram_floor"),
                   "avg_depth": m.get("train/avg_recursion_steps"),
                   "load_entropy": m.get("train/expert_load_entropy_normalized"),
                   "balance_to_task_ratio": m.get("train/balance_to_task_ratio"),
                   "ponder_cost": m.get("train/ponder_cost"),
                   "task_loss": m.get("train/task_loss"),
                   "ami": m.get("val/routing_ami"), "run": rd}
            rows.append(row)
            print(f"{bal:>8.4g} {halt:>8.4g} {row['val_loss']:>9.4f} "
                  f"{row['nats_below_floor']:>+9.4f} {row['avg_depth']:>6.2f} "
                  f"{row['load_entropy']:>8.4f} "
                  f"{(row['balance_to_task_ratio'] or 0):>9.2e} "
                  f"{(row['ami'] if isinstance(row['ami'], float) else float('nan')):>7.4f}")
    else:
        combos = list(itertools.product([float(x) for x in args.lr.split(",")],
                                        [float(x) for x in args.dropout.split(",")]))
        print(f"[calib] {len(combos)} (lr, dropout) points, {EPOCHS} epochs each")
        print(f"{'lr':>8s} {'dropout':>8s} {'val_loss':>9s} {'vs floor':>9s} {'depth':>6s}")
        for lr, dp in combos:
            tag = f"lr{lr}_d{dp}".replace(".", "p")
            cfg = write_config(os.path.join(tmp, f"cfg_{tag}.json"), 0.001, 0.001,
                               lr, dp, float(args.weight_decay.split(",")[0]))
            got = run_point(tag, cfg, args.seed, env)
            if not got:
                continue
            m, rd = got
            row = {"lr": lr, "dropout": dp, "val_loss": m.get("val/task_loss"),
                   "nats_below_floor": m.get("val/nats_below_bigram_floor"),
                   "avg_depth": m.get("train/avg_recursion_steps"), "run": rd}
            rows.append(row)
            print(f"{lr:>8.4g} {dp:>8.3g} {row['val_loss']:>9.4f} "
                  f"{row['nats_below_floor']:>+9.4f} {row['avg_depth']:>6.2f}")

    dest = os.path.join(CODE, f"lang_calibration_{args.stage}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"stage": args.stage, "corpus": CORPUS, "epochs": EPOCHS,
                   "seq_len": SEQ_LEN, "batch_size": BATCH, "seed": args.seed,
                   "dev_bigram_floor": DEV_FLOOR,
                   "target_ratio_from_T4_2": TARGET_RATIO,
                   "pos_partition_own_load_entropy": 0.8884,
                   "caveat": ("CALIBRATION on the dev corpus. wikitext-2 shares its val "
                              "split byte-for-byte with wikitext-103 (T-L2.0), so these "
                              "runs may PICK a weight and may never be quoted as a result."),
                   "rows": rows}, fh, indent=2)
        fh.write("\n")
    print()
    print(f"[calib] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
