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
# In append order. T-L6.10h added the fourth, and this list must GROW at the end rather than
# be rewritten -- the whole rule being checked is that columns are only ever appended.
NEW_COLUMNS = ["val_perplexity", "nats_below_bigram_floor", "depth_rho_model_loss",
               "depth_logfreq_partial_norm"]

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

    _tail = _hdr[len(ARITH_COLUMNS_BEFORE_L5):]
    check("TL5.4b the language columns are APPENDED in order, nothing inserted",
          _tail == NEW_COLUMNS[:len(_tail)] and len(_tail) >= 3,
          f"{len(_tail)} appended: {' '.join(_tail)}. A file written by an older commit "
          f"legitimately has fewer, so this checks the PREFIX -- what must never happen is "
          f"a column appearing before position {len(ARITH_COLUMNS_BEFORE_L5)} or the order "
          f"changing")

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

# -- T-LX.6: the load-entropy comparison target is PUBLISHED BY THE RUN -----------
# The defect: `0.8884` is the wikitext-2 partition entropy and was quoted in five
# documents beside canonical wikitext-103 numbers, whose value is 0.8942. It is a
# threshold with a direction -- a run at 0.890 reads below the partition on canonical
# and above it on dev -- so the wrong constant inverts the claim that the balance term
# is matching the data's own imbalance rather than overriding it. The fix is structural:
# the run publishes its own corpus's value, so no document can go stale.
check("TLX.6a the engine publishes the corpus's own partition load entropy",
      "val/routing_pos_partition_load_entropy" in _eng
      and 'getattr(_ds, "pos_partition_load_entropy", None)' in _eng,
      "read off the dataset beside primary_metric_floor and logged per validation, so "
      "every language metrics.json carries the number its own entropy must be read "
      "against")

check("TLX.6b and it is ABSENT rather than substituted when the corpus has no "
      "shuffled-control stage",
      "if _pos_entropy is not None:" in _eng,
      "a default would put another corpus's threshold into a run's own record, which "
      "is the defect being closed, not a smaller version of it")

# The engine must not USE either corpus's value: a literal in an expression is the whole
# bug. Comments naming both values are the opposite -- they are why the fix exists -- so
# the check is on code, with `#` tails stripped, rather than on the file's text. A
# whole-file substring search fails on the explanatory comment and would push the next
# person to delete the explanation to make a test pass.
_code_lines = [ln.split("#", 1)[0] for ln in _eng.splitlines()]
_bad_consts = sorted({c for c in ("0.888", "0.894")
                      for ln in _code_lines if c in ln})
check("TLX.6c no corpus-specific partition-entropy literal is USED in engine.py",
      not _bad_consts,
      "the engine computes and reports; the constant belongs to the corpus manifest "
      "(both values do appear in a comment, which is the fix's rationale, not a value)"
      if not _bad_consts else f"FOUND LITERALS IN CODE: {_bad_consts}")


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
        # The flag is a HISTORICAL record: a run made against an older, smaller registry
        # legitimately carries keys that were uncovered THEN and are covered now. What must
        # be empty is the set the CURRENT registry still misses -- re-derived here rather
        # than read from the run, so an old flag cannot fail a fixed registry, and a real
        # gap still cannot hide.
        _still = uncovered_depth_keys(_mj.keys())
        if _still:
            _flagged.append((os.path.basename(os.path.dirname(_r)), _still))

    if _flagged:
        check("TL6.8f no completed run has a depth key the CURRENT registry misses",
              False,
              f"{_flagged} -- add them to metrics.DEPTH_KEY_PROVENANCE")
    elif _with_prov:
        _pr, _n = _with_prov[0]
        with open(_pr, "r", encoding="utf-8") as fh:
            _shipped = _json.load(fh)["depth_key_provenance"]
        _wellformed = all(
            isinstance(v, dict) and {"pass", "population", "note"} <= set(v)
            for v in _shipped.values())
        # NOT an exact count match against the current source: the registry grows, and a
        # run written by an older commit legitimately shipped fewer entries. What matters is
        # that the block is there and every entry is a real provenance record; whether the
        # CURRENT registry covers everything is TL6.8e's job.
        check("TL6.8f the provenance SHIPS in the run directory, not only in the source",
              _n >= 7 and _wellformed,
              f"{_n} well-formed entries in "
              f"{os.path.basename(os.path.dirname(_pr))}/metrics.json "
              f"(current source has {len(DEPTH_KEY_PROVENANCE)}; a run written by an older "
              f"commit legitimately has fewer, and TL6.8e is what checks current coverage)")
    else:
        skip("TL6.8f provenance in a run directory",
             f"{len(_lang_runs)} language run(s) on disk, all predating the writer; "
             f"none flags an uncovered key. Re-run a language run to confirm.")


# ===========================================================================
print()
print("=== T-L6.7  Span-level depth: within- vs between-document variance ===")
# ===========================================================================

from more.metrics import (depth_variance_decomposition,          # noqa: E402
                          document_index_from_ids,
                          depth_by_document_to_wandb)
import numpy as _np  # noqa: E402

# The identity Var = within + between must hold EXACTLY, including with unequal document
# sizes -- which is where an unweighted average of per-document variances breaks, and the
# discrepancy would then look like a bug in whichever term was quoted second.
_cases = {
    "distinct means": (_np.array([1.] * 50 + [4.] * 50 + [7.] * 50),
                       _np.array([0] * 50 + [1] * 50 + [2] * 50)),
    "identical means": (_np.tile([1., 7.], 75),
                        _np.array([0] * 50 + [1] * 50 + [2] * 50)),
    "unequal sizes": (_np.array([1.] * 10 + [5.] * 190),
                      _np.array([0] * 10 + [1] * 190)),
}
_dvs = {k: depth_variance_decomposition(d, g) for k, (d, g) in _cases.items()}
check("TL6.7a the decomposition sums to the total variance EXACTLY, even with "
      "unequal document sizes",
      all(v["sum_matches_total"] for v in _dvs.values()),
      "  ".join(f"{k}: {v['total']:.4f} = {v['within']:.4f} + {v['between']:.4f}"
                for k, v in _dvs.items()))

check("TL6.7b between-share is 1.0 when documents differ only in mean, 0.0 when they "
      "differ only within",
      abs(_dvs["distinct means"]["between_share"] - 1.0) < 1e-12
      and _dvs["identical means"]["between_share"] < 1e-12,
      f"distinct-mean {_dvs['distinct means']['between_share']:.4f}, "
      f"identical-mean {_dvs['identical means']['between_share']:.2e} -- so the statistic "
      f"actually separates passage-level allocation from within-passage variation")

check("TL6.7c the document index comes from EOT positions in the STORED stream",
      document_index_from_ids(_np.array([5, 6, 0, 7, 8, 0, 9]), 0).tolist()
      == [0, 0, 0, 1, 1, 1, 2],
      "the separator is counted as closing its own document -- the one arbitrary half of "
      "an otherwise exact definition, ~0.4% of a 256-token block, and stated rather "
      "than hidden. No re-segmentation of the raw text, which could drift from the "
      "packing")

_mis_raised = False
try:
    depth_variance_decomposition(_np.zeros(5), _np.zeros(6))
except ValueError:
    _mis_raised = True
check("TL6.7d a length mismatch RAISES rather than silently truncating",
      _mis_raised,
      "a decomposition over two different token sets would look like a measurement")

check("TL6.7e per-document MEANS are summarised, not logged one key per document",
      set(depth_by_document_to_wandb(_dvs["distinct means"])) ==
      {f"depth/by_document/{k}" for k in
       ("n_documents", "total", "within", "between", "between_share",
        "per_document_mean_std", "per_document_mean_min", "per_document_mean_max",
        "sum_matches_total")},
      "61 validation documents and 29,445 training ones; individual keys no plot reads "
      "would bury the three numbers the passage-level question is answered with")

_span_runs = [r for r in _lang_runs
              if "depth/by_document/total" in _json.load(open(r, encoding="utf-8"))]
if not _span_runs:
    skip("TL6.7f span-level depth in a run directory",
         f"{len(_lang_runs)} language run(s) on disk, all predating the writer; "
         f"a fresh run is needed to confirm the keys land")
else:
    with open(_span_runs[0], "r", encoding="utf-8") as fh:
        _sm = _json.load(fh)
    check("TL6.7f the decomposition SHIPS in a run directory and sums there too",
          _sm.get("depth/by_document/sum_matches_total") == 1
          and _sm.get("depth/by_document/n_documents", 0) > 1,
          f"{_sm['depth/by_document/n_documents']} documents, total "
          f"{_sm['depth/by_document/total']:.6f} = within "
          f"{_sm['depth/by_document/within']:.6f} + between "
          f"{_sm['depth/by_document/between']:.6f}, between-share "
          f"{_sm['depth/by_document/between_share']:.4f} "
          f"({os.path.basename(os.path.dirname(_span_runs[0]))})")


# ===========================================================================
print()
print("=== T-L6.10  The embedding-norm confound: partialled out, and measured ===")
# ===========================================================================

from more.metrics import (partial_spearman, spearman_within_bins,      # noqa: E402
                          spearman_rho, WITHIN_BIN_DEFAULT,
                          language_depth_metrics, language_depth_to_wandb)

_rng = _np.random.default_rng(0)
_n = 4000
# Fully mediated: depth and logfreq are both ~2*norm, so ALL of their correlation runs
# through the norm. This is the shape the confound would take.
_nm = _rng.normal(size=_n)
_dp = _nm * 2 + _rng.normal(scale=.05, size=_n)
_lfq = _nm * 2 + _rng.normal(scale=.05, size=_n)
_raw_a, _par_a = spearman_rho(_dp, _lfq), partial_spearman(_dp, _lfq, _nm)
# Genuine: depth tracks logfreq, norm is unrelated noise.
_nm2, _lfq2 = _rng.normal(size=_n), _rng.normal(size=_n)
_dp2 = _lfq2 * 1.5 + _rng.normal(scale=.3, size=_n)
_raw_b, _par_b = spearman_rho(_dp2, _lfq2), partial_spearman(_dp2, _lfq2, _nm2)

check("TL6.10a partialling KILLS a fully mediated correlation and PRESERVES a genuine one",
      abs(_par_a) < 0.05 and abs(_par_b - _raw_b) < 0.01,
      f"mediated: raw {_raw_a:+.4f} -> partial {_par_a:+.4f};  "
      f"genuine: raw {_raw_b:+.4f} -> partial {_par_b:+.4f}. So a positive partial is "
      f"evidence the frequency-depth relation is not just the tied embedding's row norm")

check("TL6.10b a constant depth vector gives an UNDEFINED partial, not 1.0",
      partial_spearman(_np.full(_n, 2.0), _lfq2, _nm2) is None,
      "the collapsed-depth case reaches this branch: 99.5% of tokens at one depth in the "
      "seed-42 run, and a partial correlation with a zero denominator is undefined")

# The ledger says within-norm-deciles is "equivalently" the same test. It is not, and the
# gap is measured -- this check exists so the claim cannot quietly be treated as true.
_bins = {nb: spearman_within_bins(_dp, _lfq, _nm, n_bins=nb)["mean"]
         for nb in (10, 50, 200)}
check("TL6.10c within-bin is NOT equivalent to partialling, and under-corrects",
      _bins[10] > 0.8 and _bins[200] < _bins[50] < _bins[10]
      and WITHIN_BIN_DEFAULT == 50,
      f"same fully-mediated case: raw {_raw_a:+.4f}, true partial {_par_a:+.4f}, "
      f"within-bin {_bins[10]:+.3f} at 10 bins ('norm deciles', which would have passed "
      f"the confound straight through), {_bins[50]:+.3f} at 50, {_bins[200]:+.3f} at 200. "
      f"Binning removes only BETWEEN-bin variation, so partial_spearman is primary and "
      f"this is a secondary assumption-free cross-check")

_wb_b = spearman_within_bins(_dp2, _lfq2, _nm2, n_bins=50)["mean"]
check("TL6.10d within-bin does not OVER-correct a genuine correlation either",
      abs(_wb_b - _raw_b) < 0.05,
      f"genuine case: raw {_raw_b:+.4f}, within-bin {_wb_b:+.4f} -- so a low value from "
      f"it is informative even though a high one is not conclusive")

# The report carries the diagnostic, and it is gathered over token OCCURRENCES.
_V, _nn = 512, 3000
_ids_s = _rng.integers(0, _V, _nn)
_norm_s = _rng.random(_V) + 0.5
_lf_s = _np.log1p(_rng.integers(1, 10000, _V).astype(float))
_us_s = -_np.log(_np.ones(_V) / _V)
_tf_s = _rng.integers(-1, 6, _V).astype(_np.int8)
_dep_s = _np.clip((_norm_s[_ids_s] * 6).astype(int) + 1, 1, 7).astype(float)
_dm_s = language_depth_metrics(_dep_s, _ids_s, None, _tf_s, _lf_s, _us_s, 7, 6,
                               [f"L{i + 1}" for i in range(6)], embed_norm=_norm_s)
check("TL6.10e the depth report carries the norm diagnostic when a norm is supplied",
      _dm_s["embed_norm"] is not None
      and _dm_s["embed_norm"]["logfreq_partial_norm"] is not None
      and abs(_dm_s["embed_norm"]["depth_vs_norm"]) > 0.8,
      f"depth~norm {_dm_s['embed_norm']['depth_vs_norm']:+.4f} on a synthetic "
      f"norm-driven depth, so the mediator is detected when it is present")

check("TL6.10f the report OMITS the diagnostic rather than faking it when no norm exists",
      language_depth_metrics(_dep_s, _ids_s, None, _tf_s, _lf_s, _us_s, 7, 6,
                             [f"L{i + 1}" for i in range(6)])["embed_norm"] is None,
      "MoE/MoR arithmetic models have no tok_embed; absent is the honest report")

_lg_s = language_depth_to_wandb(_dm_s)
check("TL6.10g the flattened keys include the partial and the binned cross-check",
      "depth/embed_norm_logfreq_partial_norm" in _lg_s
      and "depth/embed_norm_logfreq_within_bins_mean" in _lg_s
      and "depth/embed_norm_depth_vs_norm" in _lg_s,
      f"{len([k for k in _lg_s if 'embed_norm' in k])} embed_norm keys, so a reader of "
      f"metrics.json sees the confound's strength beside the corrected number")

skip("TL6.10h across-epoch trend",
     "needs a MULTI-EPOCH language run: an artifact should weaken as training equalises "
     "the row norms, a real behaviour should not. One epoch on CPU takes tens of minutes, "
     "so this is a GPU task -- see HANDOFF.md")


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
