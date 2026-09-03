"""
T6.6 + T6.7 verification -- run naming and provenance completeness
(plan.md 8.6 / 8.7, updated_rules.md 9).

Why this file exists
--------------------
A number in a paper is worthless if you cannot say which code, config, data
split and seed produced it. updated_rules.md 9 lists the fields that make a run
reconstructible; this file checks that the list is actually satisfied by a real
run rather than by intention. Two failure modes it is aimed at:

  * a field present but None -- reported, unusable, and easy to miss in a wide
    W&B table;
  * an ablation whose provenance is indistinguishable from canonical, which is
    how a variant ends up in a headline table.

The run itself is a 1-epoch smoke run, launched through the real CLI, because
the fields are assembled across cli.py -> run_context.py -> engine.py and a
unit test of any one of them would not catch a field lost between them.
"""

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPO, "code")
RUNS = os.path.join(REPO, "runs")
PY = sys.executable

sys.path.insert(0, CODE)

from more.config import (canonical_run_name, stamp_seed_into_run_name,
                         resolve_variant, apply_architecture, load_config,
                         ARCH_DISPLAY, CANONICAL_VARIANT)
from more.run_context import config_hash

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f"   ({detail})" if detail else ""))


print("=" * 74)
print("T6.6 / T6.7  run naming and provenance")
print("=" * 74)

# ---------------------------------------------------------------- T6.6
print("\n-- T6.6  stable run names ------------------------------------------")

check("T6.6 canonical names match updated_rules.md 9 exactly",
      [canonical_run_name(a, 42) for a in ("moe", "mor", "more")]
      == ["phaseB_MoE_seed42", "phaseB_MoR_seed42", "phaseB_MoRE_seed42"],
      str([canonical_run_name(a, 42) for a in ("moe", "mor", "more")]))
check("T6.6 the display spelling is the paper spelling, not the config enum",
      ARCH_DISPLAY == {"moe": "MoE", "mor": "MoR", "more": "MoRE"},
      str(ARCH_DISPLAY))
check("T6.6 an undeclared seed produces a name that does not claim one",
      canonical_run_name("more", None) == "phaseB_MoRE",
      canonical_run_name("more", None))

bad_arch = False
try:
    canonical_run_name("moree", 42)
except ValueError:
    bad_arch = True
check("T6.6 an unknown architecture raises rather than producing a name",
      bad_arch)

cfg = {"logging": {"run_name": "phaseB_MoRE"}, "provenance": {"seed": 43}}
check("T6.6 the seed is stamped into the label once the seed is known",
      stamp_seed_into_run_name(cfg) == "phaseB_MoRE_seed43",
      cfg["logging"]["run_name"])

cfg2 = {"logging": {"run_name": "phaseB_MoRE_seed43"}, "provenance": {"seed": 43}}
check("T6.6 stamping is idempotent -- no phaseB_MoRE_seed43_seed43",
      stamp_seed_into_run_name(cfg2) == "phaseB_MoRE_seed43",
      cfg2["logging"]["run_name"])

cfg3 = {"logging": {"run_name": "t63_pim_check"}, "provenance": {"seed": 42}}
check("T6.6 a diagnostic name is kept and extended, not overwritten",
      stamp_seed_into_run_name(cfg3) == "t63_pim_check_seed42",
      cfg3["logging"]["run_name"])

cfg4 = {"logging": {"run_name": "phaseB_MoRE"}, "provenance": {}}
check("T6.6 no seed declared -> no seed in the label (T6.1 honesty)",
      stamp_seed_into_run_name(cfg4) == "phaseB_MoRE",
      cfg4["logging"]["run_name"])

spec = json.load(open(os.path.join(CODE, "canonical_spec.json"),
                     encoding="utf-8"))
tmpl = spec["run_name_template"]
check("T6.6 the produced name matches canonical_spec.json's template",
      tmpl.format(architecture=ARCH_DISPLAY["more"], seed=42)
      == canonical_run_name("more", 42),
      f"{tmpl} -> {canonical_run_name('more', 42)}")

# ---------------------------------------------------------------- T6.7a
print("\n-- T6.7a  variant is derived, never declared -----------------------")

base = load_config(os.path.join(CODE, "config.json"))
apply_architecture(base, "more")
check("T6.7a a canonical MoRE config reports variant 'canonical'",
      resolve_variant(base) == CANONICAL_VARIANT, resolve_variant(base))

dense = json.loads(json.dumps(base))
dense["model"]["routing_mode"] = "dense_blend"
check("T6.7a the dense routing ablation cannot be reported as canonical",
      resolve_variant(dense) != CANONICAL_VARIANT, resolve_variant(dense))

noisy = json.loads(json.dumps(base))
noisy["model"]["router_noise"] = "gaussian"
check("T6.7a a router-noise ablation is named in the variant",
      "router_noise_gaussian" in resolve_variant(noisy), resolve_variant(noisy))

sup = json.loads(json.dumps(base))
sup["model"]["halting_supervision"] = True
sup["loss_weights"]["halting_supervision"] = 0.1
check("T6.7a the supervised depth curriculum is named in the variant",
      resolve_variant(sup) == "supervised_curriculum", resolve_variant(sup))

sup_off = json.loads(json.dumps(base))
sup_off["model"]["halting_supervision"] = True
sup_off["loss_weights"]["halting_supervision"] = 0.0
check("T6.7a a curriculum flag with zero weight is NOT a variant -- the term "
      "contributes nothing", resolve_variant(sup_off) == CANONICAL_VARIANT,
      resolve_variant(sup_off))

both = json.loads(json.dumps(base))
both["model"]["fixed_depth"] = True
both["model"]["router_noise"] = "gaussian"
check("T6.7a two deviations compose into one stable, sorted label",
      resolve_variant(both) == "fixed_depth+router_noise_gaussian",
      resolve_variant(both))

# ---------------------------------------------------------------- T6.7b
print("\n-- T6.7b  config_hash identifies the configuration, not the label --")

a = json.loads(json.dumps(base))
b = json.loads(json.dumps(base))
a["logging"]["run_name"] = "phaseB_MoRE_seed42"
b["logging"]["run_name"] = "phaseB_MoRE_seed43"
check("T6.7b the run LABEL is outside the hash -- two names, one hash",
      config_hash(a) == config_hash(b), f"{config_hash(a)[:8]} vs {config_hash(b)[:8]}")

# T9.1b: the assertion above used to be worded "two seeds of one experiment share
# a config_hash, so the exporter can average them instead of refusing to mix
# hashes" -- while varying only `logging.run_name`. It therefore never exercised
# the claim it made, and passed. It is false: --seed lands in
# `data.subset_seed` / `training.subset_seed`, which are INSIDE the hashed config.
# Discovered when the T9.1 matrix reported 5 distinct hashes per architecture on
# 15 otherwise-clean canonical runs.
#
# `config_hash()` is deliberately left alone: subset_seed really does select a
# different data subset when subset_fraction < 1.0, so excluding it would let two
# different datasets share one identity. The correct predicate for "these five
# runs may be averaged into one row" is seed-BLIND config equality, which is what
# run_phase9_matrix.py:seed_blind_diff() checks by naming the differing field.
sa = json.loads(json.dumps(base))
sb = json.loads(json.dumps(base))
for c, s in ((sa, 42), (sb, 43)):
    c.setdefault("data", {})["subset_seed"] = s
    c.setdefault("training", {})["subset_seed"] = s
check("T6.7b/T9.1b changing the SEED does change config_hash -- the seed is "
      "inside the hashed config, so the matrix may not verify seeds by hash",
      config_hash(sa) != config_hash(sb),
      f"{config_hash(sa)[:8]} vs {config_hash(sb)[:8]}")


def _seed_blind(cfg: dict) -> dict:
    """Same reduction as run_phase9_matrix.scientific_config(), inlined so this
    gate does not import the driver."""
    def flat(d, pre=""):
        o = {}
        for k, v in d.items():
            kk = pre + k
            o.update(flat(v, kk + ".")) if isinstance(v, dict) else o.setdefault(kk, v)
        return o
    f = flat({k: v for k, v in cfg.items() if k not in ("provenance", "logging")})
    return {k: v for k, v in f.items()
            if k not in ("data.subset_seed", "training.subset_seed")}


check("T6.7b/T9.1b two seeds of one experiment are identical once the seed is "
      "factored out -- this is what lets five seeds share one row",
      _seed_blind(sa) == _seed_blind(sb),
      f"{[k for k in set(_seed_blind(sa)) | set(_seed_blind(sb)) if _seed_blind(sa).get(k) != _seed_blind(sb).get(k)]}")

sc = json.loads(json.dumps(sa))
sc["training"]["lr"] = 0.002
check("T6.7b/T9.1b the seed-blind comparison still catches a real config change",
      _seed_blind(sa) != _seed_blind(sc), "lr 0.001 vs 0.002")

c = json.loads(json.dumps(base))
c["training"]["lr"] = 0.002
check("T6.7b a real configuration change still changes the hash",
      config_hash(a) != config_hash(c),
      f"{config_hash(a)[:8]} vs {config_hash(c)[:8]}")

d = json.loads(json.dumps(base))
d["model"]["ffn_mult"] = 2
check("T6.7b ffn_mult is inside the hash -- it changes the parameter count",
      config_hash(a) != config_hash(d),
      f"{config_hash(a)[:8]} vs {config_hash(d)[:8]}")

# ---------------------------------------------------------------- T6.7c
def _required_fields(rc: dict) -> dict:
    """
    The updated_rules.md 9 provenance list, resolved from the blocks a reader
    would actually open, as {name: value}. A value of None means the field is
    missing; `variant` is allowed to be the string "canonical".

    T-L0.3: extracted from the T6.7c block so the identical field list can be
    asserted on more than one architecture. It was checked on a `more` run only,
    which leaves any E=1 or max_depth=1 specific provenance gap invisible -- the
    same blind spot class as T8.3, where a head sized by num_experts passed at
    E=6 and hit a CUDA device-side assert at E=1.
    """
    prov = rc.get("provenance", {})
    mc, tc, lw = (rc.get("model", {}), rc.get("training", {}),
                  rc.get("loss_weights", {}))
    return {
        "experiment_id":       prov.get("experiment_id"),
        "experiment_group":    prov.get("experiment_group"),
        "architecture":        prov.get("architecture"),
        "variant":             prov.get("variant"),
        "seed":                prov.get("seed"),
        "dataset_version":     prov.get("dataset_version"),
        "train_split_version": prov.get("train_split_version"),
        "code_git_commit":     prov.get("code_git_commit"),
        "config_hash":         prov.get("config_hash"),
        "num_experts":         mc.get("num_experts"),
        "max_depth":           mc.get("max_depth"),
        "num_blocks":          mc.get("num_blocks"),
        "d_model":             mc.get("d_model"),
        "ffn_mult":            prov.get("ffn_mult"),
        "batch_size":          tc.get("batch_size"),
        "lr":                  tc.get("lr"),
        "weight_decay":        tc.get("weight_decay"),
        "routing_balance":     lw.get("routing_balance"),
        "halting_weight":      prov.get("ponder_weight"),
        "routing_supervision_enabled": prov.get("routing_supervision_enabled"),
        "routing_supervision_weight":  prov.get("routing_supervision_weight"),
        "router_noise_mode":   prov.get("resolved_router_noise"),
        "router_noise_scale":  prov.get("router_noise_scale"),
        "halt_target_mode":    prov.get("halt_target_mode"),
        "resolved_epochs":     prov.get("resolved_epochs"),
        "resolved_subset_fraction": prov.get("resolved_subset_fraction"),
    }


def _smoke_run(architecture: str, run_name: str, seed: int,
               tag: str = "T6.7c") -> str | None:
    """Run 1 epoch of `architecture` and return its run directory, or None."""
    proc = subprocess.run(
        [PY, "train.py", "--architecture", architecture, "--epochs", "1",
         "--seed", str(seed), "--run_name", run_name],
        cwd=CODE, env=dict(os.environ, WANDB_MODE="disabled"),
        capture_output=True, text=True, timeout=3600,
    )
    check(f"{tag} the {architecture} smoke run exits 0", proc.returncode == 0,
          (proc.stderr or "")[-400:])
    dirs = sorted((d for d in os.listdir(RUNS) if d.startswith(run_name)),
                  key=lambda d: os.path.getmtime(os.path.join(RUNS, d)))
    return os.path.join(RUNS, dirs[-1]) if dirs else None


print("\n-- T6.7c  a real run carries every required field -------------------")

RUN_NAME = "t67_provenance_check"
env = dict(os.environ, WANDB_MODE="disabled")
proc = subprocess.run(
    [PY, "train.py", "--architecture", "more", "--epochs", "1", "--seed", "44",
     "--run_name", RUN_NAME],
    cwd=CODE, env=env, capture_output=True, text=True, timeout=3600,
)
check("T6.7c the smoke run exits 0", proc.returncode == 0,
      (proc.stderr or "")[-400:])

run_dirs = sorted((d for d in os.listdir(RUNS) if d.startswith(RUN_NAME)),
                  key=lambda d: os.path.getmtime(os.path.join(RUNS, d)))
check("T6.7c the run directory carries the stamped seed in its name",
      bool(run_dirs) and "seed44" in run_dirs[-1],
      str(run_dirs[-1:] or "none"))
check("T6.7c the seed appears once in the directory name, not twice",
      bool(run_dirs) and run_dirs[-1].count("seed44") == 1,
      str(run_dirs[-1:] or "none"))

if run_dirs:
    rd = os.path.join(RUNS, run_dirs[-1])
    rc = json.load(open(os.path.join(rd, "resolved_config.json"),
                        encoding="utf-8"))
    prov = rc.get("provenance", {})
    mc, tc, lw, dc = (rc.get("model", {}), rc.get("training", {}),
                      rc.get("loss_weights", {}), rc.get("data", {}))

    check("T6.7c --run_name is extended with the seed, not replaced",
          rc["logging"]["run_name"] == f"{RUN_NAME}_seed44",
          rc["logging"]["run_name"])

    # updated_rules.md 9, field by field, resolved from the block a reader would
    # actually open. `variant` is allowed to be the string "canonical"; nothing
    # here is allowed to be None.
    missing = sorted(k for k, v in _required_fields(rc).items() if v is None)
    check("T6.7c every field in updated_rules.md 9 is present and not None",
          not missing, f"missing/None: {missing}")

    check("T6.7c architecture is the one that ran",
          prov.get("architecture") == "more", str(prov.get("architecture")))
    check("T6.7c variant of an unmodified run is 'canonical'",
          prov.get("variant") == CANONICAL_VARIANT, str(prov.get("variant")))
    check("T6.7c halt_target_mode names the halting objective",
          prov.get("halt_target_mode") in ("pure_act", "supervised_curriculum"),
          str(prov.get("halt_target_mode")))
    check("T6.7c router_noise_scale is 0.0 under the canonical 'none' mode, "
          "not an inherited init value",
          prov.get("resolved_router_noise") == "none"
          and prov.get("router_noise_scale") == 0.0,
          f"{prov.get('resolved_router_noise')} / {prov.get('router_noise_scale')}")
    check("T6.7c ffn_mult defaults to 4, reproducing every earlier run",
          prov.get("ffn_mult") == 4 and mc.get("ffn_mult") == 4,
          str(prov.get("ffn_mult")))
    # T8.2: was pinned to 6358553, the TWO-block count. That literal was written
    # while load_config_defaults still defaulted num_blocks to 2, disagreeing with
    # both config.json and canonical_spec.json (which freeze 1). The canonical
    # MoRE model is 3,201,555 parameters at num_blocks=1, d_model=256,
    # num_experts=6, ffn_mult=4. The number is pinned rather than recomputed
    # because this check reads a FINISHED run's provenance, and the point is that
    # the recorded value is the one Phase 9's parameter-matching table consumes.
    check("T6.7c total_params is recorded (Phase 9 reads it from here)",
          prov.get("total_params") == 3201555, str(prov.get("total_params")))
    check("T8.2 the run really used the canonical single-block configuration",
          mc.get("num_blocks") == 1 and tc.get("batch_size") == 768,
          f"num_blocks={mc.get('num_blocks')}, batch_size={tc.get('batch_size')}")
    check("T6.7c the declared seed reached the RNGs",
          prov.get("seed") == 44 and prov.get("resolved_seed") == 44,
          f"seed={prov.get('seed')} resolved={prov.get('resolved_seed')}")

    m = json.load(open(os.path.join(rd, "metrics.json"), encoding="utf-8"))
    check("T6.7c metrics.json carries the permutation-invariant block (T6.3)",
          "routing_permutation_invariant" in m,
          str(sorted(m.keys())[:6]))
    check("T6.7c no provenance value is the string 'None'",
          not [k for k, v in prov.items() if v == "None"],
          str([k for k, v in prov.items() if v == "None"]))

# ---------------------------------------------------------------- T6.7d
# T-L0.3: a HEAD-commit SINGLE-EXPERT run, for two reasons.
#
# 1. Gate 5's G5.2b block asserts the mirror-image metric contract -- that a MoR
#    run reports every routing and expert-diversity quantity as the STRING "N/A",
#    present rather than absent. It had no producer, so it fell back to whatever
#    MoR directory sorted last by mtime. `runs/` is tracked, so a `git worktree
#    add` rewrote all 132 mtimes into checkout order and the block graded
#    `t83_mor_headfix_..._r2` from commit e5b0f6df -- from before the T8.1 fix
#    that made those keys present-and-"N/A". It reported a defect the current
#    writers do not have. Producing the run here, ahead of gate 5 in the same
#    gate, removes the dependence on filesystem timestamps entirely.
# 2. Provenance itself was only ever verified on a `more` run. E=1 and
#    max_depth=1 are exactly where architecture-specific gaps hide (T8.3).
print("\n-- T6.7d  provenance holds at E=1 too, not only on MoRE ------------")

RUN_NAME_E1 = "t67d_provenance_check_mor"
rd1 = _smoke_run("mor", RUN_NAME_E1, 44, tag="T6.7d")
check("T6.7d the single-expert run produced a directory", rd1 is not None,
      str(rd1))

if rd1:
    rc1 = json.load(open(os.path.join(rd1, "resolved_config.json"),
                         encoding="utf-8"))
    prov1 = _required_fields(rc1)
    missing1 = sorted(k for k, v in prov1.items() if v is None)
    check("T6.7d every updated_rules.md 9 field is present and not None at E=1",
          not missing1, f"missing/None: {missing1}")
    check("T6.7d the recorded architecture and expert count are MoR's",
          prov1["architecture"] == "mor" and prov1["num_experts"] == 1,
          f"{prov1['architecture']} / E={prov1['num_experts']}")
    # MoR routes nothing, so apply_architecture zeroes the balance weight. Zero is
    # a real resolved value here and must survive as 0.0, not be read as missing --
    # `if not lw.get("routing_balance")` would treat it as absent, which is the
    # falsy-zero variant of reporting a sentinel as a measurement.
    check("T6.7d a legitimately zero weight is recorded as 0.0, not dropped",
          rc1.get("loss_weights", {}).get("routing_balance") == 0.0,
          str(rc1.get("loss_weights", {}).get("routing_balance")))
    check("T6.7d ffn_mult is MoR's parameter-matched 24, not the default 4",
          prov1["ffn_mult"] == 24, str(prov1["ffn_mult"]))
    check("T6.7d the run is recognisably NOT canonical_phase_b (1 epoch)",
          prov1["experiment_group"] != "canonical_phase_b",
          str(prov1["experiment_group"]))

print("\n" + "=" * 74)
print(f"T6.6 / T6.7:  {PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
