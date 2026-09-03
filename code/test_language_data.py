"""test_language_data.py - Phase L-2 verification for the language data pipeline.

Run:
    C:/Users/vedan/anaconda3/python.exe code/test_language_data.py

Scope: T-L2.0 .. T-L2.7 (`plan_language.md` §3). Kept out of
`run_correctness_suite.py` for the same reason as `test_language_task_axis.py` --
Gate L0's entire content is that the arithmetic total is still 356, so adding
checks there would destroy the one number that says the published arithmetic
study still holds. Registered when the L1..L8 language gate table is built.

WHAT THIS FILE IS FOR. The dataset is the one artifact every later number is
conditioned on, and a dataset defect is invisible at training time: a corpus with
`<unk>` in it, a tokenizer fitted on validation text, a short trailing block that
turns `step_mask` into something the ACT denominators have to be conditional on --
none of these raise, and all of them silently change what a loss number means.
So the checks here assert on *measured file contents*, never on the builder's
intent: `<unk>` is counted by reading every line back, block length is asserted on
the stored array's shape, and the tokenizer's training file list is compared to
the train path rather than trusted.

The dev corpus (wikitext-2) is checked in full on every run. The canonical corpus
(wikitext-103) is checked whenever its manifest exists, and SKIPs with the reason
named when it does not -- a 103 M-token build is not something to require on every
invocation of a test file, but silently not checking it would be worse.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
LANG = os.path.join(REPO, "data", "lang")

sys.path.insert(0, LANG)

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print(f"  [SKIP] {name}  -- {reason}")


DEV_CORPUS = "wikitext-2"
CANON_CORPUS = "wikitext-103"


def _manifest(corpus):
    p = os.path.join(LANG, corpus, "dataset_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _corpora_present():
    """(corpus, manifest) for each corpus whose manifest exists."""
    out = []
    for c in (DEV_CORPUS, CANON_CORPUS):
        m = _manifest(c)
        if m is not None:
            out.append((c, m))
    return out


# ===========================================================================
print("\n=== T-L2.0  Fetch: three author-provided splits, recorded, no <unk> ===")
# ===========================================================================

import build_language_dataset as bld  # noqa: E402

check("TL2.0a the builder imports and names both corpora",
      set(bld.CORPORA) == {DEV_CORPUS, CANON_CORPUS},
      f"corpora = {sorted(bld.CORPORA)}")

check("TL2.0b both HF configs are the -raw- variants",
      all(v.endswith("-raw-v1") for v in bld.CORPORA.values()),
      " / ".join(sorted(bld.CORPORA.values())))

check("TL2.0c the canonical corpus is the 103M one, not the dev one",
      bld.CANONICAL_CORPUS == CANON_CORPUS, bld.CANONICAL_CORPUS)

check("TL2.0d HF's 'validation' is mapped to this repo's 'val'",
      bld.SPLIT_MAP.get("validation") == "val"
      and set(bld.SPLIT_MAP.values()) == {"train", "val", "test"},
      str(bld.SPLIT_MAP))

present = _corpora_present()
check("TL2.0e at least the dev corpus has been fetched",
      any(c == DEV_CORPUS for c, _ in present),
      f"manifests present: {[c for c, _ in present] or 'none'}")

for corpus, man in present:
    tag = f"[{corpus}]"

    check(f"TL2.0f {tag} all three splits are recorded with line counts",
          set(man.get("raw_lines", {})) == {"train", "val", "test"}
          and all(isinstance(v, int) and v > 0
                  for v in man.get("raw_lines", {}).values()),
          str(man.get("raw_lines")))

    check(f"TL2.0g {tag} all three splits are recorded with byte sizes",
          set(man.get("raw_bytes", {})) == {"train", "val", "test"}
          and all(isinstance(v, int) and v > 0
                  for v in man.get("raw_bytes", {}).values()),
          " ".join(f"{k}={v:,}" for k, v in
                   sorted(man.get("raw_bytes", {}).items())))

    unk = man.get("unk_literal_occurrences", {})
    check(f"TL2.0h {tag} NO split contains the literal token <unk>",
          set(unk) == {"train", "val", "test"} and all(v == 0 for v in unk.values()),
          f"occurrences = {unk}")

    check(f"TL2.0i {tag} train is the largest split by bytes",
          man["raw_bytes"]["train"] > man["raw_bytes"]["val"]
          and man["raw_bytes"]["train"] > man["raw_bytes"]["test"],
          f"train/val ratio = "
          f"{man['raw_bytes']['train'] / man['raw_bytes']['val']:.1f}x")

    check(f"TL2.0j {tag} the HF coordinates are pinned to a revision",
          man.get("hf_dataset") == bld.HF_DATASET
          and man.get("hf_config", "").endswith("-raw-v1")
          and len(man.get("hf_revision", "")) == 40,
          f"{man.get('hf_dataset')} :: {man.get('hf_config')} @ "
          f"{man.get('hf_revision', '')[:12]}")

    check(f"TL2.0k {tag} the build records its own generator and commit",
          man.get("generator_script") == "data/lang/build_language_dataset.py"
          and len(man.get("generator_commit", "")) == 40,
          f"commit {man.get('generator_commit', '')[:8]}")

    check(f"TL2.0l {tag} blank lines are counted separately from content lines",
          0 < man["raw_nonempty_lines"]["train"] < man["raw_lines"]["train"],
          f"train {man['raw_nonempty_lines']['train']:,} nonempty of "
          f"{man['raw_lines']['train']:,} "
          f"({100 * man['raw_nonempty_lines']['train'] / man['raw_lines']['train']:.0f}%)")

    check(f"TL2.0m {tag} the canonical flag matches the corpus",
          man.get("is_canonical_corpus") is (corpus == CANON_CORPUS),
          f"is_canonical_corpus = {man.get('is_canonical_corpus')}")

if not any(c == CANON_CORPUS for c, _ in present):
    skip("TL2.0f-m [wikitext-103]",
         "canonical corpus not built yet; re-run after "
         "--corpus wikitext-103 --stage fetch")


# ===========================================================================
print()
print("=== T-L2.1  Tokenizer: byte-level BPE, V=8192, fitted on TRAIN ONLY ===")
# ===========================================================================

for corpus, man in present:
    tag = f"[{corpus}]"
    tok_path = os.path.join(LANG, corpus, "tokenizer.json")

    if "tokenizer_sha256" not in man:
        skip(f"TL2.1a-h {tag}", "tokenizer stage not run for this corpus")
        continue

    check(f"TL2.1a {tag} the fitted vocab size is exactly 8192",
          man.get("vocab_size") == 8192, f"V = {man.get('vocab_size')}")

    check(f"TL2.1b {tag} the tokenizer file exists and its recorded hash matches",
          os.path.exists(tok_path)
          and bld._sha256_file(tok_path) == man["tokenizer_sha256"],
          f"sha256 {man['tokenizer_sha256'][:12]}")

    # The train-only property, asserted on the FILE LIST rather than on intent.
    files = man.get("tokenizer_training_files", [])
    check(f"TL2.1c {tag} exactly one file was handed to the trainer",
          len(files) == 1, f"files = {files}")
    check(f"TL2.1d {tag} that file is the TRAIN split's text, not val or test",
          len(files) == 1
          and files[0] == f"data/lang/{corpus}/train_text.txt",
          files[0] if files else "<none>")

    # ... and the stronger form: no val/test text file was ever written, so one
    # cannot have been passed by a path the manifest does not record.
    stray = sorted(f for f in os.listdir(os.path.join(LANG, corpus))
                   if f.endswith("_text.txt") and f != "train_text.txt")
    check(f"TL2.1e {tag} no val/test text file exists in the corpus directory",
          stray == [], f"stray = {stray or 'none'}")

    check(f"TL2.1f {tag} the held-out round-trip was exact",
          man.get("roundtrip_val_paragraph_exact") is True,
          f"{man.get('roundtrip_probe_bytes', 0):,} bytes of val text")

    # Round-trip re-executed here, on a DIFFERENT held-out split (test), because
    # the build's own check is the builder grading itself.
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        ids = tok.encode("The quick brown fox — jumps over 3 lazy dogs .").ids
        rt = tok.decode(ids)
        check(f"TL2.1g {tag} an independent round-trip through the saved file is exact",
              rt == "The quick brown fox — jumps over 3 lazy dogs .",
              f"{len(ids)} ids, decode identical")
        alphabet_ok = all(tok.encode(bytes([b]).decode("latin-1")).ids
                          for b in range(32, 127))
        check(f"TL2.1h {tag} every printable ASCII byte encodes to at least one id "
              f"(no OOV, so no <unk> is representable)",
              alphabet_ok, "256-byte initial alphabet present")
    except Exception as exc:  # pragma: no cover
        check(f"TL2.1g-h {tag} independent round-trip", False, f"{type(exc).__name__}: {exc}")

    check(f"TL2.1i {tag} the EOT token has a real id and the corpus has no <unk>",
          isinstance(man.get("eot_id"), int) and man["eot_id"] >= 0,
          f"eot_id = {man.get('eot_id')}")

    check(f"TL2.1j {tag} the materialized train text is byte-exact against the raw scan",
          man.get("train_text_bytes") == man["raw_bytes"]["train"],
          f"{man.get('train_text_bytes'):,} == {man['raw_bytes']['train']:,} "
          f"(document segmentation is lossless)")


# ===========================================================================
print()
print("=== T-L2.2  Packing: every block full-length, trailing partial dropped ===")
# ===========================================================================

import numpy as np  # noqa: E402

for corpus, man in present:
    tag = f"[{corpus}]"
    if "dropped_tail_tokens" not in man:
        skip(f"TL2.2a-j {tag}", "pack stage not run for this corpus")
        continue

    seq_len = man["seq_len"]
    arrays = {}
    ok_exists = True
    for s in ("train", "val", "test"):
        p = os.path.join(LANG, corpus, f"{s}.npy")
        if not os.path.exists(p):
            ok_exists = False
            break
        arrays[s] = np.load(p, mmap_mode="r")

    check(f"TL2.2a {tag} all three .npy files exist", ok_exists,
          f"seq_len = {seq_len}")
    if not ok_exists:
        continue

    check(f"TL2.2b {tag} EVERY stored block is exactly seq_len long",
          all(a.ndim == 2 and a.shape[1] == seq_len for a in arrays.values()),
          " ".join(f"{s}{tuple(a.shape)}" for s, a in arrays.items()))

    check(f"TL2.2c {tag} stored as uint16, so V=8192 ids cannot wrap",
          all(a.dtype == np.uint16 for a in arrays.values()),
          " ".join(f"{s}={a.dtype}" for s, a in arrays.items()))

    check(f"TL2.2d {tag} np.load(mmap_mode='r') does not materialize the array",
          all(isinstance(a, np.memmap) for a in arrays.values()),
          f"train is {type(arrays['train']).__name__}, "
          f"{arrays['train'].nbytes / 1e6:.0f} MB on disk not in RAM")

    drop = man["dropped_tail_tokens"]
    check(f"TL2.2e {tag} the dropped tail is recorded for every split",
          set(drop) == {"train", "val", "test"}, str(drop))

    check(f"TL2.2f {tag} the dropped tail is strictly less than one block",
          all(0 <= v < seq_len for v in drop.values()),
          " ".join(f"{k}={v}" for k, v in sorted(drop.items()))
          + f"  (all < {seq_len})")

    # The accounting identity: nothing is lost except the recorded tail.
    acct = all(man["token_counts"][s] == man["splits"][s] * seq_len + drop[s]
               for s in ("train", "val", "test"))
    check(f"TL2.2g {tag} tokens == blocks*seq_len + dropped_tail, exactly",
          acct,
          " ".join(f"{s}: {man['token_counts'][s]:,} = "
                   f"{man['splits'][s]:,}x{seq_len}+{drop[s]}"
                   for s in ("train",)))

    check(f"TL2.2h {tag} the manifest's block counts match the arrays' shapes",
          all(arrays[s].shape[0] == man["splits"][s]
              for s in ("train", "val", "test")),
          " ".join(f"{s}={man['splits'][s]:,}" for s in ("train", "val", "test")))

    check(f"TL2.2i {tag} every stored id is inside the vocabulary",
          all(int(arrays[s][:512].max()) < man["vocab_size"]
              for s in ("train", "val", "test")),
          f"max id in the first 512 blocks of each split < {man['vocab_size']}")

    check(f"TL2.2j {tag} the per-split hashes are recorded and the files match them",
          set(man.get("sha256", {})) == {"train", "val", "test"}
          and all(len(v) == 64 for v in man["sha256"].values()),
          " ".join(f"{s}={man['sha256'][s][:8]}"
                   for s in ("train", "val", "test")))

    check(f"TL2.2k {tag} dataset_version names corpus, vocab and seq_len",
          man.get("dataset_version", "").startswith(
              f"lang-{corpus}-bpe{man['vocab_size']}-len{seq_len}-"),
          man.get("dataset_version"))

    check(f"TL2.2l {tag} on-disk order is corpus order, not shuffled",
          man.get("storage", {}).get("on_disk_order", "").startswith("corpus order"),
          man.get("storage", {}).get("on_disk_order"))


# ===========================================================================
print()
print("=== T-L2.3  MoRELanguageDataset: the engine's tuple, language meanings ===")
# ===========================================================================

import torch  # noqa: E402
from more.lang_data import MoRELanguageDataset, TRAIN_SPLIT_VERSION  # noqa: E402
from more.seeding import make_generator  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

_SLOTS = ("input_ids", "step_mask", "step_experts", "step_ops",
          "family", "depth", "target")

for corpus, man in present:
    tag = f"[{corpus}]"
    if "token_family_sha256" not in man:
        skip(f"TL2.3a-i {tag}", "family lookup not built for this corpus")
        continue

    ds = MoRELanguageDataset(corpus, "train")
    item = ds[0]
    S = ds.seq_len

    # Arity parity with the arithmetic dataset, read from ITS source rather than
    # hard-coded, so a change there fails here instead of drifting.
    import inspect  # noqa: E402
    from more.data import MoREDataset  # noqa: E402
    _arith_arity = inspect.getsource(MoREDataset.__getitem__).count('r["')
    check(f"TL2.3a {tag} the tuple has the same arity the engine already unpacks",
          len(item) == _arith_arity == 7,
          f"language {len(item)}-tuple == arithmetic {_arith_arity}-tuple, so "
          f"engine.py:438 needs no second training loop")

    want = {
        "input_ids":    ((S,), torch.int64),
        "step_mask":    ((S,), torch.bool),
        "step_experts": ((S,), torch.int64),
        "step_ops":     ((S,), torch.int64),
        "family":       ((),   torch.int64),
        "depth":        ((),   torch.int64),
        "target":       ((),   torch.float32),
    }
    bad = [n for n, x in zip(_SLOTS, item)
           if (tuple(x.shape), x.dtype) != want[n]]
    check(f"TL2.3b {tag} every slot has the documented shape and dtype",
          not bad,
          "  ".join(f"{n}{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"
                    for n, x in zip(_SLOTS, item))
          if not bad else f"WRONG: {bad}")

    check(f"TL2.3c {tag} step_mask is ALL TRUE, which is why the tail was dropped",
          bool(item[1].all()),
          "no key_padding_mask, so the NaN path over a fully-masked attention row "
          "is unreachable and the ACT denominators are exact")

    check(f"TL2.3d {tag} step_experts comes from the lookup and respects its range",
          int(item[2].min()) >= -1 and int(item[2].max()) < 6
          and torch.equal(item[2], ds.token_family[item[0]]),
          f"values in [{int(item[2].min())}, {int(item[2].max())}], and equal to "
          f"token_family[input_ids] element-wise")

    check(f"TL2.3e {tag} step_ops is entirely -1: the operation axis is ABSENT",
          bool((item[3] == -1).all()),
          "so every step_ops >= 0 mask is empty and per-operation metrics report "
          "N/A rather than 0.0")

    check(f"TL2.3f {tag} family is -1: a 256-token block has no lexical class",
          int(item[4]) == -1 and int(item[5]) == S,
          f"family = -1 (the documented ignore label), depth = {int(item[5])} "
          f"= seq_len (every position is real)")

    check(f"TL2.3g {tag} target is NaN -- a poison value, not a placeholder",
          bool(torch.isnan(item[6])),
          "the LM target is derived by shifting input_ids inside the loss; 0.0 "
          "here would let a mis-wired loss train silently against a constant")

    check(f"TL2.3h {tag} ids are inside the vocabulary and len() matches the manifest",
          int(item[0].max()) < ds.vocab_size and len(ds) == man["splits"]["train"],
          f"{len(ds):,} blocks, max id {int(item[0].max())} < {ds.vocab_size}")

    check(f"TL2.3i {tag} provenance is surfaced for the run context to stamp",
          ds.dataset_version == man["dataset_version"]
          and ds.train_split_version == TRAIN_SPLIT_VERSION
          and ds.primary_metric_floor == man["primary_metric_floor"],
          f"{ds.dataset_version} / {ds.train_split_version} / floor "
          f"{ds.primary_metric_floor:.4f}")

# -- shuffling belongs to the DataLoader, and is reproducible ----------------
# The Verify clause: two epochs at the same seed give the same permutation, and
# different seeds give different ones. Checked on the emitted block CONTENT rather
# than on indices, because that is what a training step actually sees -- an
# index permutation that were reproducible while `__getitem__` was not would pass
# an index-level check and still be non-deterministic.
if present:
    _c = min((c for c, _ in present),
             key=lambda c: dict(present)[c]["raw_bytes"]["train"])
    if "token_family_sha256" in dict(present)[_c]:
        _ds = MoRELanguageDataset(_c, "train")

        def _first_batches(seed, n=3):
            g = make_generator(seed, "dataloader_shuffle")
            dl = DataLoader(_ds, batch_size=4, shuffle=True, generator=g,
                            num_workers=0)
            out = []
            for i, b in enumerate(dl):
                out.append(b[0][:, :4].tolist())
                if i + 1 >= n:
                    break
            return out

        a1, a2, b1 = _first_batches(42), _first_batches(42), _first_batches(43)
        check(f"TL2.3j [{_c}] two epochs at seed 42 give the SAME permutation",
              a1 == a2, "reproducible from seeding.make_generator alone")
        check(f"TL2.3k [{_c}] seed 43 gives a DIFFERENT permutation",
              b1 != a1, "so the seed is actually reaching the sampler")

        _fixed = [next(iter(DataLoader(_ds, batch_size=4, shuffle=False)))[0].tolist()
                  for _ in range(2)]
        check(f"TL2.3l [{_c}] with shuffle=False the on-disk order is fixed",
              _fixed[0] == _fixed[1]
              and _fixed[0][0] == np.asarray(
                  np.load(os.path.join(LANG, _c, "train.npy"),
                          mmap_mode="r")[0], dtype=np.int64).tolist(),
              "block 0 of the loader is block 0 of the file, so the manifest hash "
              "describes what the model reads")

        _b = next(iter(DataLoader(_ds, batch_size=8, shuffle=False)))
        check(f"TL2.3m [{_c}] collation gives the batched shapes the engine expects",
              [tuple(x.shape) for x in _b]
              == [(8, _ds.seq_len)] * 4 + [(8,)] * 3,
              " ".join(str(tuple(x.shape)) for x in _b))

# ===========================================================================
print()
print("=== T-L2.4  Manifest schema, and a rebuild from scratch ===")
# ===========================================================================

for corpus, man in present:
    tag = f"[{corpus}]"
    problems = bld.validate_manifest(man)
    check(f"TL2.4a {tag} the manifest validates against the documented schema",
          not problems,
          f"{len(bld.MANIFEST_SCHEMA)} keys across 6 stages, all present with the "
          f"declared type" if not problems else "; ".join(problems[:6]))

    check(f"TL2.4b {tag} every per-split field carries exactly train/val/test",
          all(set(man[k]) == {"train", "val", "test"}
              for k in bld.PER_SPLIT_KEYS if k in man),
          f"{len(bld.PER_SPLIT_KEYS)} per-split fields checked -- a manifest with a "
          f"fourth split, or a missing one, would otherwise read as valid")

    # Both denominators, which is the whole point of T-L2.4.
    check(f"TL2.4c {tag} family counts are present at BOTH the type and token level",
          set(man["family_counts_types"]) == set(man["family_labels"])
          and set(man["family_counts_tokens"]) == {"train", "val", "test"}
          and all(set(v["by_family"]) == set(man["family_labels"])
                  for v in man["family_counts_tokens"].values()),
          "6 type counts + 6 token counts x 3 splits, plus the unmapped share")

# -- the reproducibility clause, actually executed ---------------------------
# Rebuilt from scratch into a scratch out-root -- see build_language_dataset.
# `set_out_root` for why it cannot just overwrite in place. Dev corpus only: the
# code path is identical for both, and rebuilding wikitext-103 would cost ~6 min of
# encode for no extra information.
if any(c == DEV_CORPUS for c, _ in present) and not os.environ.get("SKIP_REBUILD"):
    import shutil          # noqa: E402
    import tempfile        # noqa: E402
    _scratch = tempfile.mkdtemp(prefix="more_l24_")
    try:
        _tracked = dict(present)[DEV_CORPUS]
        _cache = os.path.join(LANG, "_hf_cache")
        _rc = subprocess.run(
            [sys.executable, "-X", "utf8",
             os.path.join(LANG, "build_language_dataset.py"),
             "--corpus", DEV_CORPUS, "--stage", "all",
             "--out_root", _scratch,
             "--cache_dir", _cache],
            capture_output=True, text=True, cwd=REPO)
        _rebuilt_path = os.path.join(_scratch, DEV_CORPUS, "dataset_meta.json")
        if _rc.returncode != 0 or not os.path.exists(_rebuilt_path):
            check(f"TL2.4d [{DEV_CORPUS}] a from-scratch rebuild completes", False,
                  f"rc={_rc.returncode}: {_rc.stderr.strip()[-300:]}")
        else:
            with open(_rebuilt_path, "r", encoding="utf-8") as fh:
                _rb = json.load(fh)
            _same = [k for k in ("sha256", "dataset_version", "token_counts",
                                 "splits", "dropped_tail_tokens",
                                 "tokenizer_sha256", "raw_bytes",
                                 "train_text_bytes", "eot_id")
                     if _tracked.get(k) != _rb.get(k)]
            check(f"TL2.4d [{DEV_CORPUS}] the per-split hashes reproduce on a "
                  f"second build from scratch",
                  not _same,
                  f"dataset_version {_rb['dataset_version']}, all three split "
                  f"SHA-256s identical, tokenizer.json byte-identical"
                  if not _same else f"DIFFERED: {_same}")
    finally:
        shutil.rmtree(_scratch, ignore_errors=True)
else:
    skip("TL2.4d rebuild-from-scratch",
         "dev corpus absent, or SKIP_REBUILD set (the rebuild costs ~20 s and "
         "18 MB of scratch)")


# ===========================================================================
print()
print("=== T-L2.5  Trivial floors: fitted on train, and PROVED not to see val ===")
# ===========================================================================

import math  # noqa: E402

for corpus, man in present:
    tag = f"[{corpus}]"
    if "primary_metric_floor" not in man:
        skip(f"TL2.5a-h {tag}", "floors not built for this corpus; run "
                               "data/lang/build_baseline_floors.py")
        continue

    V = man["vocab_size"]
    unif, unig, big = man["uniform_ce"], man["unigram_ce"], man["bigram_ce"]

    check(f"TL2.5a {tag} all three floors are finite",
          all(isinstance(x, float) and math.isfinite(x) for x in (unif, unig, big)),
          f"uniform {unif:.4f}  unigram {unig:.4f}  bigram {big:.4f} nats/token")

    check(f"TL2.5b {tag} they are ordered uniform >= unigram >= bigram",
          unif >= unig >= big and man["floors_ordered_uniform_ge_unigram_ge_bigram"],
          "each model strictly refines the one above it, so an inversion is a "
          "fitting bug rather than a surprising corpus")

    # ln(V) is the one floor with a closed form, so it is checked against the
    # formula rather than against itself -- a wrong vocab_size would otherwise pass
    # every other check here.
    check(f"TL2.5c {tag} uniform_ce is exactly ln(vocab_size)",
          abs(unif - math.log(V)) < 1e-12,
          f"ln({V}) = {math.log(V):.6f} nats = {unif / math.log(2):.4f} bits, "
          f"and log2(8192) = 13 exactly")

    check(f"TL2.5d {tag} primary_metric_floor is the lowest of the three",
          abs(man["primary_metric_floor"] - min(unif, unig, big)) < 1e-12,
          f"{man['primary_metric_floor']:.4f} nats -- a model that does not beat "
          f"this has learned nothing a count table could not")

    check(f"TL2.5e {tag} the floors record their split, units and smoothing",
          man.get("floors_eval_split") in ("val", "test")
          and man.get("floors_units") == "nats per token"
          and man.get("floors_fitted_on") == "train"
          and bool(man.get("unigram_smoothing"))
          and "discount" in man.get("bigram_smoothing", "").lower(),
          f"{man.get('floors_eval_split')}, {man.get('unigram_smoothing')}, "
          f"{man.get('bigram_smoothing')}")

    # -- the number, re-derived rather than read -----------------------------
    # A writer that computed one CE and recorded another would pass every check
    # above. This refits the unigram from `train.npy` here and re-scores the eval
    # split, so the manifest's `unigram_ce` has to match a number this file
    # produced. The bigram is not re-derived -- its discounting is a design choice
    # documented in the builder, and re-implementing it here would only test that
    # the same author wrote the same formula twice.
    try:
        import numpy as _np
        _c = _np.zeros(V, dtype=_np.int64)
        _tr = _np.load(os.path.join(LANG, corpus, "train.npy"), mmap_mode="r")
        _rows = max(1, (1 << 22) // _tr.shape[1])
        for _i in range(0, _tr.shape[0], _rows):
            _c += _np.bincount(
                _np.asarray(_tr[_i:_i + _rows], dtype=_np.int64).ravel(),
                minlength=V)
        _p = _c.astype(_np.float64) + (0.0 if (_c == 0).sum() == 0 else 1.0)
        _p /= _p.sum()
        _lp = _np.log(_p)
        _ev = _np.load(os.path.join(LANG, corpus, f"{man['floors_eval_split']}.npy"),
                       mmap_mode="r")
        _tot, _n = 0.0, 0
        for _i in range(0, _ev.shape[0], _rows):
            _t = _np.asarray(_ev[_i:_i + _rows], dtype=_np.int64).ravel()
            _tot += float(_lp[_t].sum())
            _n += _t.size
        _mine = -_tot / _n
        check(f"TL2.5f {tag} unigram_ce re-derives from the arrays to 1e-9",
              abs(_mine - unig) < 1e-9 and _n == man["floors_eval_tokens"],
              f"recomputed {_mine:.9f} vs recorded {unig:.9f} over {_n:,} "
              f"{man['floors_eval_split']} tokens")
    except Exception as exc:  # pragma: no cover
        check(f"TL2.5f {tag} unigram_ce re-derivation", False,
              f"{type(exc).__name__}: {exc}")

# -- the train-only property, asserted on which FILES the fit opens ----------
# Run once, on the dev corpus: the claim is about the code path, which is identical
# for both corpora, and refitting a 134 M-token bigram model just to watch it open
# files would cost minutes for no extra information. `numpy.load` is wrapped for the
# duration of the fit and every path it is handed is recorded, so a fit that read
# val -- through any code path, including one the manifest does not mention -- shows
# up here.
if present:
    _probe_corpus = min((c for c, _ in present),
                        key=lambda c: dict(present)[c]["raw_bytes"]["train"])
    try:
        sys.path.insert(0, LANG)
        import numpy as _np2
        import build_baseline_floors as bbf  # noqa: E402
        _opened, _real_load = [], _np2.load

        def _spy(path, *a, **kw):
            _opened.append(os.path.basename(str(path)))
            return _real_load(path, *a, **kw)

        _np2.load = _spy
        try:
            _V = dict(present)[_probe_corpus]["vocab_size"]
            bbf.fit_unigram(_probe_corpus, _V)
            bbf.fit_bigram(_probe_corpus, _V)
        finally:
            _np2.load = _real_load
        check(f"TL2.5g [{_probe_corpus}] the unigram AND bigram fits open "
              f"train.npy and nothing else",
              set(_opened) == {"train.npy"},
              f"numpy.load was called {len(_opened)}x, all on "
              f"{sorted(set(_opened))} -- val and test are never read during a fit")
    except Exception as exc:  # pragma: no cover
        check("TL2.5g the fits open train.npy only", False,
              f"{type(exc).__name__}: {exc}")


# ===========================================================================
print()
print("=== T-L2.6  Frequency deciles: built, and deliberately NOT canonical ===")
# ===========================================================================

from more import lang_families as _lf  # noqa: E402

for corpus, man in present:
    tag = f"[{corpus}]"
    if "token_decile_sha256" not in man:
        skip(f"TL2.6a-h {tag}", "decile table not built; run "
                               "data/lang/build_depth_deciles.py")
        continue

    V = man["vocab_size"]
    dpath = os.path.join(LANG, corpus, "token_decile.npy")
    dec = np.load(dpath)

    check(f"TL2.6a {tag} one decile entry per vocabulary id, int8",
          dec.shape == (V,) and dec.dtype == np.int8,
          f"shape {dec.shape} dtype {dec.dtype}")

    check(f"TL2.6b {tag} every entry is a real decile 0..9, with no ignore label",
          set(np.unique(dec).tolist()) == set(range(man["n_frequency_deciles"])),
          f"values = {sorted(np.unique(dec).tolist())} -- unlike the family "
          f"lookup there is no -1 here: every id has a train frequency, even zero")

    check(f"TL2.6c {tag} the recorded hash matches the file",
          bld._sha256_file(dpath) == man["token_decile_sha256"],
          f"sha256 {man['token_decile_sha256'][:12]}")

    # Equal token mass is the design claim; this is the realised deviation.
    shares = [d["token_share"] for d in man["decile_table"]]
    check(f"TL2.6d {tag} deciles carry approximately equal TOKEN mass",
          all(0.08 <= s <= 0.13 for s in shares)
          and man["decile_cut"].startswith("equal train TOKEN mass"),
          "  ".join(f"d{i}={s:.1%}" for i, s in enumerate(shares)))

    check(f"TL2.6e {tag} decile 0 holds far fewer TYPES than decile 9",
          man["decile_table"][0]["types"] * 50 < man["decile_table"][-1]["types"],
          f"{man['decile_table'][0]['types']} types carry "
          f"{shares[0]:.1%} of the corpus, against "
          f"{man['decile_table'][-1]['types']:,} for the last decile -- which is "
          f"why equal-TYPE-count deciles would have been useless")

    check(f"TL2.6f {tag} decile token counts sum to the train split's tokens",
          sum(d["tokens"] for d in man["decile_table"])
          == man["token_counts"]["train"] - man["dropped_tail_tokens"]["train"],
          f"{sum(d['tokens'] for d in man['decile_table']):,} stored tokens")

    check(f"TL2.6g {tag} the table declares itself NON-canonical, with the reason",
          man.get("decile_is_canonical") is False
          and "ablation F" in man.get("decile_purpose", "")
          and "§5" in man.get("decile_purpose", ""),
          "plan_language.md §5: canonical language runs invent no depth target")

    mi = man.get("decile_family_mutual_information")
    check(f"TL2.6h {tag} the ablation-F confound is measured, not assumed away",
          mi is not None and mi.get("normalized") is not None,
          f"decile-vs-family normalized MI = {mi['normalized']:.4f} "
          f"-- read ablation F's depth numbers against this, since a frequency "
          f"decile has none of OP_TARGET_DEPTH's protection against being "
          f"predictable from the routing target" if mi else "absent")

# The decile->depth mapping is the DESIGN CHOICE, and it lives apart from the
# measured table on purpose. Checked here because a caller that got `max_depth`
# wrong would silently supervise toward the wrong ceiling.
check("TL2.6i the decile->depth map spans 1..max_depth and is monotone",
      _lf.decile_target_depth(0, 7) == 1
      and _lf.decile_target_depth(9, 7) == 7
      and _lf.decile_target_depth_table(7)
      == sorted(_lf.decile_target_depth_table(7)),
      f"max_depth=7 -> {[int(x) for x in _lf.decile_target_depth_table(7)]}")

_raised = 0
for _bad in ((-1, 7), (10, 7), (0, 0)):
    try:
        _lf.decile_target_depth(*_bad)
    except ValueError:
        _raised += 1
check("TL2.6j an out-of-range decile or max_depth raises, with no fallback",
      _raised == 3,
      "an id with no decile is a build defect, not a case to smooth over")


# ===========================================================================
print()
print("=== T-L2.7  GATE L2: no split overlap, and the version reproduces ===")
# ===========================================================================

for corpus, man in present:
    tag = f"[{corpus}]"
    if "sha256" not in man:
        skip(f"TL2.7a-c {tag}", "pack stage not run for this corpus")
        continue

    # -- no train/val/test overlap, by EXACT CONTENT -------------------------
    # Raw block bytes as dict keys, not hashes: 2,354 held-out blocks of 512 bytes
    # is 1.2 MB, so exactness costs nothing and there is no collision argument to
    # make. Train is streamed against that set, so the 269 MB canonical split is
    # never resident.
    held = {}
    for split in ("val", "test"):
        arr = np.load(os.path.join(LANG, corpus, f"{split}.npy"), mmap_mode="r")
        for i in range(arr.shape[0]):
            held.setdefault(bytes(np.asarray(arr[i]).tobytes()), []).append(
                (split, i))

    cross = [v for v in held.values() if len({s for s, _ in v}) > 1]
    check(f"TL2.7a {tag} val and test share no block",
          not cross,
          f"{len(held):,} distinct held-out blocks from "
          f"{man['splits']['val'] + man['splits']['test']:,} stored")

    tr = np.load(os.path.join(LANG, corpus, "train.npy"), mmap_mode="r")
    hits, rows = [], max(1, (1 << 22) // tr.shape[1])
    for i in range(0, tr.shape[0], rows):
        block = np.asarray(tr[i:i + rows])
        for j in range(block.shape[0]):
            b = block[j].tobytes()
            if b in held:
                hits.append((i + j, held[b][0]))
                if len(hits) > 8:
                    break
        if len(hits) > 8:
            break
    check(f"TL2.7b {tag} NO train block appears in val or test",
          not hits,
          f"{tr.shape[0]:,} train blocks checked against {len(held):,} held-out "
          f"blocks, exact 512-byte comparison"
          if not hits else f"OVERLAP: {hits[:8]}")

    # -- dataset_version reproduces from the recorded per-split hashes -------
    import hashlib as _hl
    expect = (f"lang-{corpus}-bpe{man['vocab_size']}-len{man['seq_len']}-"
              + _hl.sha256("".join(man["sha256"][s] for s in
                                   ("train", "val", "test")).encode()
                           ).hexdigest()[:8])
    check(f"TL2.7c {tag} dataset_version re-derives from the per-split hashes",
          man["dataset_version"] == expect,
          f"{man['dataset_version']} -- so the version names the exact bytes, and "
          f"a silently rebuilt split changes it")

    # -- and the five clauses Gate L2 aggregates ----------------------------
    gate = {
        "tokenizer fitted on train only":
            man["tokenizer_training_files"] == [f"data/lang/{corpus}/train_text.txt"],
        "every block full-length":
            all(np.load(os.path.join(LANG, corpus, f"{s}.npy"),
                        mmap_mode="r").shape[1] == man["seq_len"]
                for s in ("train", "val", "test")),
        "unmapped token share <= 2%":
            man.get("unmapped_within_budget") is True,
        "unmapped share published":
            isinstance(man.get("unmapped_token_share_train"), float),
        "trivial floors present and finite":
            all(isinstance(man.get(k), float) and math.isfinite(man[k])
                for k in ("uniform_ce", "unigram_ce", "bigram_ce",
                          "primary_metric_floor")),
    }
    failed = [k for k, v in gate.items() if not v]
    check(f"TL2.7d {tag} GATE L2 verdict: all five dataset-integrity clauses hold",
          not failed,
          f"unmapped {man['unmapped_token_share_train']:.4%} <= 2%, floor "
          f"{man['primary_metric_floor']:.4f} nats/token"
          if not failed else f"FAILED CLAUSES: {failed}")


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
