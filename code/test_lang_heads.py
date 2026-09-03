"""test_lang_heads.py - Phase L-5 verification for the heads and the language loss.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_lang_heads.py

Scope: T-L5.0 .. T-L5.4 (`plan_language.md` §7). Kept out of
`run_correctness_suite.py` for the same reason as the other language suites --
Gate L0's entire content is that the arithmetic total is still 356.

THE TWO CHECKS THAT CARRY THE PHASE.

TLH.1a asserts `task_loss ~ ln(V)` at initialisation. That single number
simultaneously validates the shift (`logits[:, :-1]` against `input_ids[:, 1:]`),
the reshape, the vocabulary size, and the initialisation scale of a WEIGHT-TIED
head -- and it is not decoration: with `nn.Embedding`'s default N(0, 1) the
measured value was **167.6 nats** against a floor of 9.011, because tying makes the
embedding's init scale an output-logit scale. Nothing else about that run looks
wrong; the loss curve is smooth and the gradients are finite.

TLH.3b asserts that the probe's loss weight cannot change a single trunk gradient,
BITWISE, across weights 0.0 / 0.5 / 1.0. `assert not h.requires_grad` at the probe
input is the naive form and it is insufficient -- a detached tensor that still
routed gradient to the trunk through a second path would pass it. Comparing the
actual gradients is what makes §7.3's "the probe cannot create the decodability it
reports" a measured fact.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from more.model import MoREModel, TASK_ARITHMETIC, TASK_LANGUAGE   # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {reason}")


V, DM, S, B = 8192, 256, 16, 4


def lang(seed=42, d_model=DM, vocab=V, seq=S, max_depth=4, num_experts=6):
    torch.manual_seed(seed)
    m = MoREModel(step_feat_dim=8, d_model=d_model, num_experts=num_experts,
                  max_depth=max_depth, num_blocks=1, dropout=0.0,
                  attention=True, n_heads=4, max_seq_len=seq,
                  task=TASK_LANGUAGE, vocab_size=vocab)
    return m


def arith(seed=42):
    torch.manual_seed(seed)
    return MoREModel(step_feat_dim=8, d_model=DM, num_experts=6, max_depth=4,
                     num_blocks=1, dropout=0.0)


def lang_batch(seed=1, vocab=V, seq=S, b=B):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (b, seq), generator=g)
    return (ids,
            torch.ones(b, seq, dtype=torch.bool),
            torch.randint(0, 6, (b, seq), generator=g),
            torch.full((b, seq), -1, dtype=torch.long))

# ===========================================================================
print("\n=== T-L5.0  Token embedding and a weight-TIED LM head ===")
# ===========================================================================

M = lang()
M.eval()

check("TLH.0a lm_head.weight IS tok_embed.weight -- the same tensor object",
      M.lm_head.weight is M.tok_embed.weight,
      "identity, not equality: a copy kept in sync would drift the moment either "
      "was updated in place")

check("TLH.0b the tied head has NO bias",
      M.lm_head.bias is None,
      "a bias would be a per-token logit offset with no counterpart in the "
      "embedding, so the head would stop being the transpose of the input map")

_emb = V * DM
_tot = sum(p.numel() for p in M.parameters())
_uniq = sum(p.numel() for p in {id(p): p for p in M.parameters()}.values())
# Measured against an UNTIED copy of the same model rather than against a formula, so
# the saving is a number this file produced.
_untied = lang()
_untied.lm_head.weight = torch.nn.Parameter(_untied.tok_embed.weight.detach().clone())
_tot_untied = sum(p.numel() for p in _untied.parameters())
check("TLH.0c the V x d_model block is counted ONCE, so the budget is about the "
      "expert stack",
      _tot == _uniq and _tot_untied - _tot == _emb,
      f"{_tot:,} tied vs {_tot_untied:,} untied = exactly {_emb:,} saved. The "
      f"expert stack and attention are the other {_tot - _emb:,}, so a "
      f"MoR/MoRE budget comparison is about them rather than about lookup tables")

# The tie has to be real in the BACKWARD direction too, which is the property that
# "tying keeps the embedding under gradient from both directions" actually names.
_ids, _sm, _se, _so = lang_batch()
_M = lang()
_out = _M(_ids, _sm, _se, _so)
F.cross_entropy(_out[0][:, :-1].reshape(-1, V), _ids[:, 1:].reshape(-1)).backward()
_g_tied = _M.tok_embed.weight.grad
_rows_used = torch.unique(_ids)
_nz_rows = int((_g_tied.abs().sum(dim=1) > 0).sum())
check("TLH.0d the tied weight accumulates gradient from BOTH paths",
      _g_tied is not None and _nz_rows > int(_rows_used.numel()),
      f"{_nz_rows:,} of {V:,} embedding rows have gradient, against only "
      f"{_rows_used.numel():,} distinct ids in the batch -- the excess is the "
      f"output path, which touches every row of the vocabulary")

_no_vocab_raised = False
try:
    MoREModel(step_feat_dim=8, d_model=DM, attention=True, max_seq_len=S,
              task=TASK_LANGUAGE)
except ValueError:
    _no_vocab_raised = True
check("TLH.0e task='language' without vocab_size RAISES rather than defaulting",
      _no_vocab_raised,
      "a default vocabulary would silently disagree with the corpus the manifest "
      "was built for")

_float_raised = False
try:
    M(torch.randn(2, S), torch.ones(2, S, dtype=torch.bool), None,
      torch.full((2, S), -1, dtype=torch.long))
except TypeError:
    _float_raised = True
check("TLH.0f a FLOAT input to the language path raises, not truncates",
      _float_raised,
      "an arithmetic feature matrix handed to a token embedding would be silently "
      "indexed by truncation")

# ===========================================================================
print()
print("=== T-L5.1  Causal-LM cross-entropy in nats; init sits at ln(V) ===")
# ===========================================================================

_UNIFORM = math.log(V)
_losses, _ppls = [], []
for _seed in (42, 43, 44, 45, 46):
    _m = lang(seed=_seed)
    _m.eval()
    _i, _s, _e, _o = lang_batch(seed=_seed)
    with torch.no_grad():
        _lg = _m(_i, _s, _e, _o)[0]
    _l = float(F.cross_entropy(_lg[:, :-1].reshape(-1, V), _i[:, 1:].reshape(-1)))
    _losses.append(_l)
    _ppls.append(math.exp(_l))

_worst = max(abs(l - _UNIFORM) / _UNIFORM for l in _losses)
check("TLH.1a task_loss at initialisation is the UNIFORM floor ln(V)",
      _worst < 0.02,
      f"ln({V}) = {_UNIFORM:.4f}; measured {min(_losses):.4f}..{max(_losses):.4f} "
      f"over the 5 frozen seeds, worst relative error {_worst:.2%}. This one number "
      f"validates the shift, the reshape, the vocabulary AND the tied head's init "
      f"scale -- with nn.Embedding's default N(0,1) it read 167.6")

check("TLH.1b perplexity equals exp(task_loss) to floating-point tolerance",
      all(abs(p - math.exp(l)) < 1e-9 * max(1.0, p)
          for p, l in zip(_ppls, _losses)),
      f"ppl {min(_ppls):,.0f}..{max(_ppls):,.0f} against V = {V:,} -- a model at "
      f"the uniform floor has perplexity ~V, which is the sanity form of TLH.1a")

# NATS, not bits. The unit is checked against the closed form rather than against a
# comment, because a bits/nats mix inside one weighted sum is a silent 0.693 scaling
# bug on whichever term is the odd one out.
check("TLH.1c the loss is in NATS, not bits",
      abs(_losses[0] - _UNIFORM) < abs(_losses[0] - _UNIFORM / math.log(2)),
      f"measured {_losses[0]:.4f} is near ln(V) = {_UNIFORM:.4f}, not near "
      f"log2(V) = {_UNIFORM / math.log(2):.4f}; every other term in the assembly "
      f"is in nats")

# The shift is checked by its CONSEQUENCE. A tied head over a residual trunk retains
# a large component of `tok_embed(x_i)` at position i, so `logits_i` already peaks at
# x_i AT INITIALISATION -- the classic copy shortcut. Scoring the UNSHIFTED target
# therefore reads far below the uniform floor before any training, which is precisely
# the flattering number a wrongly-shifted run would report.
_m = lang(seed=7)
_m.eval()
_rep = torch.arange(S).repeat(B, 1) % V
with torch.no_grad():
    _lg = _m(_rep, _sm, _se, _so)[0]
_shifted = float(F.cross_entropy(_lg[:, :-1].reshape(-1, V), _rep[:, 1:].reshape(-1)))
_unshifted = float(F.cross_entropy(_lg.reshape(-1, V), _rep.reshape(-1)))
check("TLH.1d the NEXT-token target sits at the floor while the current-token "
      "target does not",
      abs(_shifted - _UNIFORM) < 0.3 and _unshifted < _UNIFORM - 2.0,
      f"shifted {_shifted:.4f} vs ln(V) {_UNIFORM:.4f}; UNSHIFTED {_unshifted:.4f}, "
      f"{_UNIFORM - _unshifted:.2f} nats below the uniform floor at INIT. That gap is "
      f"the copy shortcut a tied head gives for free -- so a mis-shifted language run "
      f"would report ~{_unshifted:.1f} at epoch 0 and look like it had learned "
      f"instantly. GATE L1 is what detects the general case; this pins the magnitude")

# ===========================================================================
print()
print("=== T-L5.2  The arithmetic heads are ABSENT on the language path ===")
# ===========================================================================

A = arith()
_lang_keys, _arith_keys = set(M.state_dict()), set(A.state_dict())

check("TLH.2a a language state_dict has NO regression_head.* key",
      not [k for k in _lang_keys if k.startswith("regression_head")],
      "and not a zero-weighted one: an unused head still contributes parameters to "
      "the budget comparison and still invites a future reader to weight it")

check("TLH.2b an arithmetic state_dict still DOES",
      len([k for k in _arith_keys if k.startswith("regression_head")]) > 0,
      f"{sorted(k for k in _arith_keys if k.startswith('regression_head'))} -- the "
      f"published arithmetic checkpoints are unaffected")

_gone = ("regression_head", "cls_head", "step_cls_head", "step_proj", "op_embed")
check("TLH.2c every arithmetic-only module is absent as an ATTRIBUTE too",
      not any(hasattr(M, n) for n in _gone),
      f"{list(_gone)} -- op_embed would otherwise be nn.Embedding(0, d_model) on "
      f"language, which builds fine and raises only when indexed")

_new = ("tok_embed", "lm_head", "family_probe", "pos_embed")
check("TLH.2d the language-only modules are present, and absent on arithmetic",
      all(hasattr(M, n) for n in _new)
      and not any(hasattr(A, n) for n in _new),
      f"{list(_new)}")

check("TLH.2e slot 1 is None on language: there is no whole-sequence family head",
      _out[1] is None and _out[0].shape == (B, S, V)
      and _out[2].shape == (B, S, M.num_families),
      f"slot 0 logits {tuple(_out[0].shape)}, slot 1 None, slot 2 probe "
      f"{tuple(_out[2].shape)} -- arity 13 unchanged, so engine.py needs no second "
      f"training loop")

check("TLH.2f there is NO pooling on the language path",
      _out[0].shape[1] == S,
      "every position is a prediction, so collapsing the sequence would throw the "
      "task away")


# ===========================================================================
print()
print("=== T-L5.3  family_probe is READ-ONLY: the trunk cannot feel it ===")
# ===========================================================================

_seen = []
_hk = M.family_probe.register_forward_pre_hook(
    lambda _m, inp: _seen.append(bool(inp[0].requires_grad)))
_ = M(_ids, _sm, _se, _so)
_hk.remove()
check("TLH.3a the probe's INPUT has requires_grad False",
      _seen == [False],
      "h.detach() -- the naive check, kept because it is cheap, but TLH.3b is the "
      "one that would catch a second gradient path")


def _grads(w_probe, seed=7):
    m = lang(seed=seed)
    m.train()
    out = m(_ids, _sm, _se, _so)
    task = F.cross_entropy(out[0][:, :-1].reshape(-1, V), _ids[:, 1:].reshape(-1))
    probe = F.cross_entropy(out[2].reshape(-1, m.num_families), _se.reshape(-1))
    (task + w_probe * probe).backward()
    trunk = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in m.named_parameters() if not n.startswith("family_probe")}
    head = {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in m.named_parameters() if n.startswith("family_probe")}
    return trunk, head


_t0, _h0 = _grads(0.0)
_t5, _h5 = _grads(0.5)
_t1, _h1 = _grads(1.0)


def _differ(a, b):
    return [k for k in set(a) | set(b)
            if (a[k] is None) != (b[k] is None)
            or (a[k] is not None and not torch.equal(a[k], b[k]))]


check("TLH.3b trunk gradients are BITWISE identical at probe weight 0.0 / 0.5 / 1.0",
      not _differ(_t0, _t5) and not _differ(_t0, _t1),
      f"{len(_t0)} trunk tensors compared with torch.equal -- so the probe's loss "
      f"weight is scientifically inert and every routing/AMI/purity number measures "
      f"emergent structure with no family signal in the trunk's objective")

check("TLH.3c the probe itself DOES learn, and linearly in its weight",
      all(v is None or bool((v == 0).all()) for v in _h0.values())
      and any(v is not None and bool((v != 0).any()) for v in _h1.values())
      and all(torch.allclose(_h5[k], 0.5 * _h1[k], rtol=1e-6, atol=0)
              for k in _h1 if _h1[k] is not None),
      "zero gradient at weight 0, non-zero at 1, and exactly half at 0.5 -- so "
      "TLH.3b is not passing because the probe is inert")

check("TLH.3d the arithmetic head is NOT detached, which is the confound "
      "language refuses to inherit",
      "self.step_cls_head(h)" in open(
          os.path.join(CODE, "more", "model.py"), encoding="utf-8").read()
      and "self.family_probe(h.detach())" in open(
          os.path.join(CODE, "more", "model.py"), encoding="utf-8").read(),
      "two names for two scientific roles: on arithmetic the family CE is in the "
      "objective at weight 0.5 and shapes the trunk, which is why that study needs "
      "the no_family_supervision ablation")

# ===========================================================================
print()
print("=== T-L5.4  results.tsv columns are APPENDED, never reordered ===")
# ===========================================================================

ARITH_COLUMNS_BEFORE_L5 = [
    "epoch", "train_task_loss", "val_loss", "expert_entropy_normalized",
    "avg_depth", "mean_cos_sim", "max_cos_sim", "routing_accuracy",
    "routing_hungarian_acc", "routing_ami",
]
NEW_COLUMNS = ["val_perplexity", "nats_below_bigram_floor", "depth_rho_model_loss"]

import glob  # noqa: E402

_tsvs = sorted(glob.glob(os.path.join(REPO, "runs", "t67_provenance_check*",
                                      "results.tsv")),
               key=os.path.getmtime, reverse=True)
if not _tsvs:
    skip("TL5.4a-c", "no gate-produced arithmetic results.tsv on disk; run "
                     "code/run_correctness_suite.py first")
else:
    with open(_tsvs[0], "r", encoding="utf-8") as fh:
        _hdr = fh.readline().rstrip("\r\n").split("\t")
        _row = fh.readline().rstrip("\r\n").split("\t")

    check("TL5.4a the ten pre-L5 arithmetic columns are unchanged IN POSITION",
          _hdr[:len(ARITH_COLUMNS_BEFORE_L5)] == ARITH_COLUMNS_BEFORE_L5,
          f"positions 0..9 identical -- the archived arithmetic files and every "
          f"existing reader index by position, so an insertion would silently "
          f"reinterpret published columns")

    check("TL5.4b the three new columns are APPENDED at the end",
          _hdr[len(ARITH_COLUMNS_BEFORE_L5):] == NEW_COLUMNS,
          " ".join(NEW_COLUMNS))

    check("TL5.4c on an arithmetic run all three read N/A, never 0.0",
          len(_row) == len(_hdr)
          and all(_row[len(ARITH_COLUMNS_BEFORE_L5) + i] == "N/A"
                  for i in range(len(NEW_COLUMNS))),
          f"exp() of an MSE is not a perplexity, and a 0.0 in a perplexity column "
          f"is a fabricated measurement (CLAUDE.md §4). Source: "
          f"{os.path.basename(os.path.dirname(_tsvs[0]))}")

# The header is built from `routing_agreement_key`, so a language run's routing
# column is named for what it measures. Checked on the source because the header is
# written by a real run.
with open(os.path.join(CODE, "more", "engine.py"), "r", encoding="utf-8") as fh:
    _eng = fh.read()
check("TL5.4d the language columns and the routing name both come from one place",
      "val_perplexity\\tnats_below_bigram_floor\\tdepth_rho_model_loss" in _eng
      and "_agree_key.split('/', 1)[1]" in _eng,
      "so a language results.tsv reads routing_agreement_with_pos and carries the "
      "three language columns, without a second writer to keep in step")

check("TL5.4e depth_rho_model_loss exists now and reads N/A until Phase L-6",
      "depth_rho_model_loss" in _eng
      and "val/depth_rho_model_loss" in _eng,
      "the column is reserved so the header never has to be reordered later; N/A is "
      "the honest value for 'not measured yet'")


# ===========================================================================
print()
print("=== T-L6.8  Every depth key states its pass and its token population ===")
# ===========================================================================

from more.metrics import (DEPTH_KEY_PROVENANCE, depth_key_provenance,   # noqa: E402
                          uncovered_depth_keys)
import json as _json  # noqa: E402
import glob as _glob  # noqa: E402

check("TL6.8a longest-prefix matching, so a specific key is not claimed by a general one",
      depth_key_provenance("depth/mean_by_family/L2_NOUN")["prefix"]
      == "depth/mean_by_family/"
      and depth_key_provenance("depth/hist/step_2")["prefix"] == "depth/hist/step_",
      "depth/mean_by_family/ must win over any shorter depth/ prefix, or the note would "
      "describe the wrong measurement")

_passes = {k: v["pass"] for k, v in DEPTH_KEY_PROVENANCE.items()}
check("TL6.8b the train-pass and validation-pass keys are DISTINGUISHED",
      _passes["depth_dist/step_"] == "train"
      and _passes["depth/hist/step_"] == "validation",
      "depth_dist/step_7_pct matched halt/forced_exit_rate exactly while "
      "depth/hist/step_7 was 0 -- both correct, over different passes, and that is "
      "now stated rather than discovered")

_norm = lambda t: t.lower().replace("-", " ")
check("TL6.8c the two mean-depth-by-family keys name the difference between them",
      "last position" in _norm(DEPTH_KEY_PROVENANCE["depth/mean_by_family/"]["note"])
      and "last position" in
      _norm(DEPTH_KEY_PROVENANCE["recursion/avg_depth_by_family/"]["note"])
      and "N/A" in DEPTH_KEY_PROVENANCE["depth/allocation_error_"]["note"],
      "2.0034746 vs 2.0034485 is the last-position exclusion and nothing else; a reader "
      "comparing them without that would report a discrepancy")

check("TL6.8d the spearman entry warns that exceeds_null is not an effect size",
      "EFFECT SIZE" in DEPTH_KEY_PROVENANCE["depth/spearman_"]["note"],
      "at n ~ 2.8e5 the permutation band is about +-0.004, so significance there says "
      "almost nothing -- the note carries that with the number")

_lang_runs = sorted(_glob.glob(os.path.join(REPO, "runs", "langB_*", "metrics.json")),
                    key=os.path.getmtime, reverse=True)
if not _lang_runs:
    skip("TL6.8e-f", "no completed language run on disk")
else:
    _bad = {}
    for _r in _lang_runs:
        with open(_r, "r", encoding="utf-8") as fh:
            _mj = _json.load(fh)
        _u = uncovered_depth_keys(_mj.keys())
        if _u:
            _bad[os.path.basename(os.path.dirname(_r))] = _u
    check("TL6.8e every depth key in every completed language run is covered",
          not _bad,
          f"{len(_lang_runs)} run(s) checked, 0 uncovered keys"
          if not _bad else f"UNCOVERED {_bad}")

    # The writer's claim is "the provenance ships in the run directory". ONE run carrying
    # it demonstrates that; runs made before the writer existed cannot, and reporting
    # those as a failure would be reporting the wrong thing. So: pass if any run has it,
    # SKIP with the reason if none does, and fail only if a run has the uncovered-keys
    # flag set -- which would mean the writer ran and found a gap.
    _with_prov = []
    _flagged = []
    for _r in _lang_runs:
        with open(_r, "r", encoding="utf-8") as fh:
            _mj = _json.load(fh)
        if isinstance(_mj.get("depth_key_provenance"), dict):
            _with_prov.append((_r, len(_mj["depth_key_provenance"])))
        if "depth_key_provenance_uncovered" in _mj:
            _flagged.append(os.path.basename(os.path.dirname(_r)))

    if _flagged:
        check("TL6.8f no completed run reports an uncovered depth key", False,
              f"runs flagging uncovered keys: {_flagged} -- add them to "
              f"metrics.DEPTH_KEY_PROVENANCE")
    elif _with_prov:
        check("TL6.8f the provenance SHIPS in the run directory, not only in the source",
              _with_prov[0][1] == len(DEPTH_KEY_PROVENANCE),
              f"{_with_prov[0][1]} entries in "
              f"{os.path.basename(os.path.dirname(_with_prov[0][0]))}/metrics.json, and "
              f"no run flags an uncovered key -- a note that lives only in a module is "
              f"one a reader of the artifact never sees")
    else:
        skip("TL6.8f provenance in a run directory",
             f"{len(_lang_runs)} language run(s) on disk, all predating the writer; "
             f"none flags an uncovered key. Re-run a language run to confirm.")


# ===========================================================================
print()
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("\nFAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
