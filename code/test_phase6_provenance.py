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
check("T6.7b two seeds of one experiment share a config_hash, so the exporter "
      "can average them instead of refusing to mix hashes",
      config_hash(a) == config_hash(b), f"{config_hash(a)[:8]} vs {config_hash(b)[:8]}")

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
    REQUIRED = {
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
    missing = sorted(k for k, v in REQUIRED.items() if v is None)
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

print("\n" + "=" * 74)
print(f"T6.6 / T6.7:  {PASS} passed, {FAIL} failed")
print("=" * 74)
sys.exit(1 if FAIL else 0)
