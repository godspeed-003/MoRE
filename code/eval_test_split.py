"""
T-LX.16 -- score a trained checkpoint on the HELD-OUT TEST SPLIT, which no
training run in this repository has ever read.

WHY THIS EXISTS
---------------
NeurIPS-workshop reviewer 652J, concern 4:

    headline results are reported on VALIDATION despite a declared TEST split,
    and the paper does not say whether the same held-out data both selected and
    evaluated the constant c = 2.

That is a model-selection leak if the same split did both, and it is the most
serious methodological objection in either review, because unlike the others it
can invalidate a number rather than merely under-describe it.

THE GOOD NEWS, ESTABLISHED BEFORE THIS FILE WAS WRITTEN
-------------------------------------------------------
`data/test.jsonl` (5,250 records, 2.4 MB) exists and is genuinely untouched:

  * `engine.py` reads `dc["train_path"]`/`dc["jsonl_path"]` (engine.py:179) and
    `dc["val_path"]` (engine.py:201) and NOTHING ELSE. There is no code path in
    the training engine that can open a test file, so no run could have touched
    it even by accident. Verified by grep: `test` does not appear as a split
    name anywhere in the arithmetic path -- `SPLITS = ("train","val","test")`
    exists only in `lang_data.py:68`.
  * It is byte-disjoint from both other splits. Measured on whole-line SHA-1:
    train 59,500 / val 5,250 / test 5,250 records, all unique within a split,
    train&test = 0, val&test = 0, train&val = 0 overlapping records.

So the fix for 652J concern 4 needs NO retraining and NO GPU rental. It needs a
forward pass over 5,250 records per checkpoint. This file is that forward pass.

WHAT IT REPORTS, AND WHY EACH ONE
---------------------------------
  test/task_loss     the headline. Arithmetic: `F.mse_loss(reg.squeeze(-1), tgt)`
                     sample-weighted exactly as engine.py:999-1002, so the number
                     is comparable to the published `val/task_loss` rather than a
                     differently-reduced near-miss. Language: next-token CE in
                     nats, engine.py:994-997.
  val_minus_test     the delta against the run's own published validation number.
                     This is the quantity the reviewer is really asking about: if
                     conclusions hold on data that never influenced any choice,
                     the delta is small and the paper's ranking survives.
  test/routing_acc   `mean(predicted == oracle)` over exactly the tokens the
                     confusion matrix uses (CLAUDE.md 4), with the confusion
                     diagonal cross-checked against it -- see `_ROUTING_TOL`.
  test/depth_mean, early/forced exit rate
                     the depth story has to survive the split change too, since
                     the near-cap histogram is the paper's central negative claim.

INTEGRITY GUARDS -- all hard failures, none silent
--------------------------------------------------
  1. The resolved test path must differ from BOTH the train and val paths the run
     actually used. A config that points them at one file would make this script
     manufacture a fake "test" number, which is the exact failure the reviewer is
     worried about.
  2. Record-level disjointness from train and val is RE-VERIFIED on every
     invocation, not trusted from a prior session. Cheap (~2 s) against the cost
     of publishing a leaked number.
  3. The test file's SHA-256 goes in the sidecar, so a later run proves it scored
     the same bytes.
  4. Routing accuracy and the confusion diagonal must agree to `_ROUTING_TOL`,
     the invariant CLAUDE.md 4 requires; disagreement raises rather than
     reporting the more flattering of the two.
  5. Missing quantities are `"N/A"`, never 0.0 (CLAUDE.md 4).

This file NEVER writes `metrics.json`, `results.tsv` or the checkpoint. Its only
output is the `test_split_metrics.json` sidecar, so a canonical run's published
artifacts keep their hashes and `config_hash` does not move.

USAGE
    python eval_test_split.py <run_dir> [--device cpu|cuda] [--test-path P]
    python eval_test_split.py --group canonical_phase_b      # every cell
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from more.data import MoREDataset                                     # noqa: E402
from more.families import NUM_EXPERTS_CANONICAL                       # noqa: E402
from more.model import (MoREModel, CANONICAL_ROUTING_MODE,            # noqa: E402
                        CANONICAL_ROUTER_NOISE,
                        ROUTER_NOISE_INIT_SCALE_DEFAULT,
                        ROUTER_NOISE_ANNEAL_STEPS_DEFAULT,
                        ROUTING_PERSISTENCE_LEGACY)
from more.metrics import compute_token_exit_depths                    # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent
SIDECAR = "test_split_metrics.json"
TASK_LANGUAGE = "language"
DEFAULT_TEST = REPO / "data" / "test.jsonl"
# Routing accuracy and the confusion diagonal are the same count computed two
# ways over the same token set, so they agree EXACTLY in float64 unless one of
# them drifted onto a different mask. The tolerance is for float64 summation
# order only, not for genuine population differences.
_ROUTING_TOL = 1e-9


def _out(s: str) -> None:
    sys.stdout.buffer.write(s.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.flush()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def line_hashes(p: Path) -> set[str]:
    """Whole-line SHA-1 set. Whole-line because two records differing only in
    `id` are genuinely different training examples, and hashing a canonicalized
    subset of fields would call them duplicates and overstate the overlap."""
    out = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(hashlib.sha1(line.encode("utf-8")).hexdigest())
    return out


def verify_disjoint(test_p: Path, train_p: Path | None,
                    val_p: Path | None) -> dict:
    """
    Guard 2. Raises on ANY overlap: a single shared record means the "held-out"
    number is partly a training number, and there is no honest way to report it.
    """
    th = line_hashes(test_p)
    info = {"test_records": len(th)}
    for name, p in (("train", train_p), ("val", val_p)):
        if p is None or not p.exists():
            info[f"{name}_overlap"] = "N/A"
            continue
        oh = line_hashes(p)
        n = len(th & oh)
        info[f"{name}_records"] = len(oh)
        info[f"{name}_overlap"] = n
        if n:
            raise RuntimeError(
                f"LEAK: {n} of {len(th)} test records also appear in {name} "
                f"({p}). Refusing to report a test metric computed on data the "
                f"model may have trained on."
            )
    return info


def build_model(mc: dict, dc: dict, task: str, step_feat_dim: int,
                device: torch.device) -> MoREModel:
    """Mirror of engine.py:265, including the T-LX.12 legacy `routing_persistence`
    fallback: a checkpoint written before that axis existed must rebuild as the
    per-step architecture it WAS, or the state_dict would load cleanly while the
    forward pass differed and the test number would describe a model that never
    trained."""
    return MoREModel(
        step_feat_dim=step_feat_dim,
        d_model=mc["d_model"],
        num_experts=mc["num_experts"],
        max_depth=mc["max_depth"],
        num_blocks=mc["num_blocks"],
        dropout=mc["dropout"],
        fixed_depth=mc.get("fixed_depth", False),
        routing_mode=mc.get("routing_mode", CANONICAL_ROUTING_MODE),
        routing_persistence=mc.get("routing_persistence",
                                   ROUTING_PERSISTENCE_LEGACY),
        router_noise=mc.get("router_noise", CANONICAL_ROUTER_NOISE),
        router_noise_init=mc.get("router_noise_init",
                                 ROUTER_NOISE_INIT_SCALE_DEFAULT),
        router_noise_anneal_steps=mc.get("router_noise_anneal_steps",
                                         ROUTER_NOISE_ANNEAL_STEPS_DEFAULT),
        ffn_mult=mc.get("ffn_mult", 4),
        num_families=NUM_EXPERTS_CANONICAL,
        attention=bool(mc.get("attention", False)),
        n_heads=int(mc.get("n_heads", 4)),
        max_seq_len=(int(dc["seq_len"]) if task == TASK_LANGUAGE else None),
        task=task,
        vocab_size=(int(dc["vocab_size"]) if task == TASK_LANGUAGE else None),
    ).to(device)


def evaluate(run_dir: Path, device: torch.device,
             test_path: Path | None) -> dict:
    rc_p, ck_p = run_dir / "resolved_config.json", run_dir / "checkpoint.pt"
    for p in (rc_p, ck_p):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p.name} in {run_dir.name}. The 15 canonical "
                f"phase_b cells were produced on Ayan's machine and "
                f"`runs/**/checkpoint.pt` is gitignored (.gitignore:21), so the "
                f"checkpoint is not in this worktree -- fetch it from that "
                f"machine rather than retraining, which would produce a "
                f"different model than the one the paper reports."
            )

    rc = json.loads(rc_p.read_text(encoding="utf-8"))
    mc, dc = rc.get("model", {}), rc.get("data", {})
    task = rc.get("task") or rc.get("provenance", {}).get("task") or "arithmetic"
    # `step_feat_dim` is a MODEL field (engine.py:137 reads mc["step_feat_dim"]),
    # not a data field. Every config in runs/ carries model.step_feat_dim=12 and
    # no data.step_feat_dim, so a `dc.get(..., 8)` default builds `step_proj` with
    # 8 input columns instead of 12 and the state_dict load fails with a size
    # mismatch -- which is how this was caught. Reading mc first, and raising on
    # absence rather than defaulting, because a wrong width here would silently
    # evaluate a DIFFERENT model than the one that trained.
    if "step_feat_dim" in mc:
        step_feat_dim = int(mc["step_feat_dim"])
    elif "step_feat_dim" in dc:
        step_feat_dim = int(dc["step_feat_dim"])
    else:
        raise KeyError(
            f"{run_dir.name}: no step_feat_dim in model or data config. Refusing "
            f"to guess an input width -- a wrong one either fails the load or, "
            f"worse, loads a model that never trained.")
    max_steps = int(mc.get("max_steps", dc.get("max_steps", 8)))

    guard: dict = {}
    if task == TASK_LANGUAGE:
        from more.lang_data import MoRELanguageDataset
        corpus = dc.get("corpus") or dc.get("dataset_version")
        ds = MoRELanguageDataset(corpus, "test")
        guard["source"] = f"{corpus}:test"
    else:
        tp = test_path or DEFAULT_TEST
        train_p = dc.get("train_path") or dc.get("jsonl_path")
        val_p = dc.get("val_path")
        train_p = Path(train_p) if train_p else None
        val_p = Path(val_p) if val_p else None
        # Guard 1: a config whose test path coincides with train or val would
        # make everything below a training number wearing a test label.
        for nm, other in (("train", train_p), ("val", val_p)):
            if other is not None and other.exists() and \
                    tp.resolve() == other.resolve():
                raise RuntimeError(
                    f"test path == {nm} path ({tp}). Refusing to fabricate a "
                    f"held-out metric from data the run used.")
        guard = verify_disjoint(tp, train_p, val_p)
        guard["source"] = str(tp)
        guard["sha256"] = sha256_file(tp)          # Guard 3
        ds = MoREDataset(
            jsonl_path=str(tp),
            max_steps=max_steps,
            step_feat_dim=step_feat_dim,
            max_val=dc["max_val"],
            pad_value=dc["pad_value"],
            num_experts=mc["num_experts"],
        )

    bs = int(rc.get("training", {}).get("batch_size", 64))
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    model = build_model(mc, dc, task, step_feat_dim, device)
    model.load_state_dict(torch.load(ck_p, map_location=device))
    model.eval()

    E = int(mc["num_experts"])
    md = int(mc["max_depth"])
    total, n = 0.0, 0
    confusion = torch.zeros(E, E, dtype=torch.float64)
    route_hit, route_n = 0.0, 0
    depth_sum, depth_n = 0.0, 0
    exits = torch.zeros(md, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            x, sm, se, so, fam, _, tgt = [
                b.to(device) if torch.is_tensor(b) else b for b in batch]
            (reg, _c, _sc, _b, _p, depth_exits, _ad, _bi, _oc,
             first_route, _rs, _ed, _hs) = model(x, sm, se, so)

            # engine.py:993-1002 exactly, including the sample weighting.
            loss = (F.cross_entropy(reg[:, :-1, :].reshape(-1, reg.shape[-1]),
                                    x[:, 1:].reshape(-1))
                    if task == TASK_LANGUAGE
                    else F.mse_loss(reg.squeeze(-1), tgt))
            total += float(loss) * x.shape[0]
            n += x.shape[0]

            if first_route is None or depth_exits is None:
                continue

            flat_mask = sm.reshape(-1)
            flat_oracle = se.reshape(-1)
            flat_route = first_route.reshape(-1)
            ed = compute_token_exit_depths(depth_exits, md).reshape(-1)

            # engine.py:1072 -- the authoritative token set, used for BOTH the
            # scalar accuracy and the confusion matrix so guard 4 is meaningful.
            valid = flat_mask & (flat_oracle >= 0) & (flat_route >= 0)
            if valid.any():
                o = flat_oracle[valid].long().cpu()
                r = flat_route[valid].long().cpu()
                route_hit += float((o == r).sum())
                route_n += int(o.numel())
                confusion.index_put_((o, r),
                                     torch.ones(o.shape[0], dtype=torch.float64),
                                     accumulate=True)

            dv = flat_mask.bool()
            if dv.any():
                d = ed[dv].double().cpu()
                depth_sum += float(d.sum())
                depth_n += int(d.numel())
                for k in range(md):
                    exits[k] += float((d == (k + 1)).sum())

    task_loss = total / max(n, 1)
    acc = (route_hit / route_n) if route_n else None
    diag = (float(confusion.diag().sum() / confusion.sum())
            if float(confusion.sum()) > 0 else None)
    # Guard 4.
    if acc is not None and diag is not None and abs(acc - diag) > _ROUTING_TOL:
        raise RuntimeError(
            f"routing accuracy {acc:.12f} != confusion diagonal fraction "
            f"{diag:.12f} (delta {abs(acc-diag):.2e} > {_ROUTING_TOL:.0e}). "
            f"CLAUDE.md 4 requires these to be the same measurement over the "
            f"same tokens; one of the two masks drifted.")

    tot_exit = float(exits.sum())
    res = {
        "run": run_dir.name,
        "task": task,
        "architecture": rc.get("architecture") or mc.get("architecture"),
        "seed": rc.get("provenance", {}).get("seed") or rc.get("seed"),
        "experiment_group": (rc.get("provenance", {}).get("experiment_group")
                             or rc.get("experiment_group")),
        "test_split": guard,
        "n_test_items": n,
        "test/task_loss": task_loss,
        "test/routing_accuracy": acc if acc is not None else "N/A",
        "test/confusion_diagonal_fraction": diag if diag is not None else "N/A",
        "test/depth_mean": (depth_sum / depth_n) if depth_n else "N/A",
        "test/early_exit_rate": (float(exits[:md - 1].sum()) / tot_exit
                                 if tot_exit and md > 1 else "N/A"),
        "test/forced_exit_rate": (float(exits[md - 1]) / tot_exit
                                  if tot_exit else "N/A"),
        "test/exit_histogram": {f"step_{i+1}": int(exits[i]) for i in range(md)},
        "device": device.type,
        "torch": torch.__version__,
    }

    # The delta the reviewer is actually asking about. `"N/A"` rather than 0.0
    # when the published number is absent, so a missing comparison can never be
    # mistaken for a perfect agreement.
    mp = run_dir / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        pv = m.get("val/task_loss", m.get("best_val_loss"))
        res["published_val_task_loss"] = pv if pv is not None else "N/A"
        res["val_minus_test"] = (pv - task_loss) if pv is not None else "N/A"
    return res


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        _out(__doc__ or "")
        return 2

    device = torch.device(
        argv[argv.index("--device") + 1] if "--device" in argv
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    tp = (Path(argv[argv.index("--test-path") + 1])
          if "--test-path" in argv else None)

    targets: list[Path] = []
    if "--group" in argv:
        want = argv[argv.index("--group") + 1]
        for d in sorted((REPO / "runs").iterdir()):
            rc_p = d / "resolved_config.json"
            if not rc_p.exists():
                continue
            rc = json.loads(rc_p.read_text(encoding="utf-8"))
            g = (rc.get("provenance", {}).get("experiment_group")
                 or rc.get("experiment_group"))
            if g == want:
                targets.append(d)
        if not targets:
            _out(f"[FAIL] no runs with experiment_group={want!r}")
            return 1
    else:
        d = Path(argv[0])
        targets = [d if d.is_absolute() else (REPO / d).resolve()]

    rows, failed = [], []
    for d in targets:
        try:
            r = evaluate(d, device, tp)
        except FileNotFoundError as e:
            failed.append((d.name, str(e).split(".")[0]))
            continue
        rows.append(r)
        (d / SIDECAR).write_text(json.dumps(r, indent=2), encoding="utf-8")

    if rows:
        g = rows[0]["test_split"]
        _out("")
        _out(f"Held-out TEST split evaluation -- device={device.type}, "
             f"torch {torch.__version__}")
        _out(f"  source={g.get('source')}  records={g.get('test_records', 'N/A')}")
        _out(f"  disjointness: train_overlap={g.get('train_overlap')}  "
             f"val_overlap={g.get('val_overlap')}")
        if "sha256" in g:
            _out(f"  test sha256={g['sha256'][:16]}...")
        _out("")
        _out(f"  {'run':<34} {'arch':<5} {'test_loss':>11} {'pub_val':>11} "
             f"{'val-test':>10} {'route_acc':>10} {'depth':>7}")

        def cell(v: object, width: int, spec: str) -> str:
            """Format a float, or right-align 'N/A'. A non-float is always a
            genuinely absent quantity here (CLAUDE.md 4), so it must never be
            coerced to 0.0 to satisfy the format spec."""
            return format(v, spec) if isinstance(v, float) else f"{'N/A':>{width}}"

        for r in rows:
            _out(f"  {r['run']:<34} {str(r['architecture']):<5} "
                 f"{r['test/task_loss']:>11.6f} "
                 f"{cell(r.get('published_val_task_loss'), 11, '11.6f')} "
                 f"{cell(r.get('val_minus_test'), 10, '+10.6f')} "
                 f"{cell(r['test/routing_accuracy'], 10, '10.4f')} "
                 f"{cell(r['test/depth_mean'], 7, '7.3f')}")
        _out("")
        _out(f"  wrote {SIDECAR} into {len(rows)} run director"
             f"{'y' if len(rows) == 1 else 'ies'}")

    if failed:
        _out("")
        _out(f"  {len(failed)} run(s) skipped -- no checkpoint in this worktree:")
        for nm, why in failed:
            _out(f"    {nm}: {why}")
    _out("")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
