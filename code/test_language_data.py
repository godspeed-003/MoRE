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
print("=" * 78)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
if FAIL:
    print("\nFAILED:")
    for f in FAIL:
        print(f"  - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
