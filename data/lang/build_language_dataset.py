"""
data/lang/build_language_dataset.py -- WikiText -> packed uint16 blocks + manifest.

The language counterpart of `data/script.py`. It produces, per corpus, exactly
what `code/more/lang_data.py` memory-maps at train time:

    data/lang/<corpus>/train.npy  val.npy  test.npy    uint16 token ids
    data/lang/<corpus>/tokenizer.json                  byte-level BPE, V=8192
    data/lang/<corpus>/dataset_meta.json               the manifest

Design decisions, all load-bearing and all recorded here because the reason is
not visible from the code (plan_language.md 3):

* **`-raw-` variants only.** The non-raw WikiText configs are pre-tokenized with
  rare words replaced by the literal string `<unk>`. That would insert an
  artificial high-frequency type into the middle of the frequency distribution
  that every depth/frequency correlation in Phase L-9 is measured against, and
  it would be a vocabulary truncation made by someone else at a threshold we do
  not control. T-L2.0 asserts the absence of `<unk>` rather than assuming it.

* **Author-provided splits, used as given.** WikiText's train/validation/test are
  document-disjoint by construction. Re-splitting it ourselves would make split
  cleanliness something we manufacture instead of something Gate L1 can audit.

* **Tokenizer fitted on the train split alone** (T-L2.1). Fitting on the full
  corpus leaks held-out token statistics into the input representation. Small
  leak, routinely ignored in the LM literature, not permitted by
  `updated_rules.md` 10.

* **V = 8192, not GPT-2's 50257** (T-L2.1). At `d_model = 256` a 50257-entry
  embedding is 12.9 M parameters against a ~3.2 M expert stack, so
  `updated_rules.md` 6's `|P_MoR - P_MoRE| / P_MoRE < 0.05` bound would be
  satisfied by the embedding alone -- matched on the part nobody is studying.
  At 8192 the tied embedding is 2.1 M, comparable to the expert stack, so the
  bound keeps constraining the thing it is about. Consequence to state in the
  write-up: absolute perplexities are NOT comparable to published WikiText
  numbers.

* **The trailing partial block is dropped** (T-L2.2). Every stored sequence is
  then exactly `seq_len` long, so `step_mask` is all-True for language, so no
  `key_padding_mask` is needed, so the all-padded-row NaN in
  `nn.MultiheadAttention` is unreachable by construction rather than guarded,
  and the ACT loop's `real_tokens` / `avg_depth` / `forced_exit_rate` /
  `mean_remainder` denominators are exact rather than mask-conditional. The
  number of dropped tokens is `< seq_len` per split and is recorded as a
  measurement.

* **`.npy` `uint16`, memory-mapped** (T-L2.2). At V=8192 every id fits in
  `uint16`; wikitext-103 is ~206 MB on disk and a `__getitem__` is a slice with
  no per-record Python object. The arithmetic path keeps JSONL because 70 000
  records with nested step structure is what JSONL is good at and 10^8 integers
  is not.

* **Shuffling is the DataLoader's, never the disk's.** On-disk token order is
  corpus order, so the split hashes are reproducible from the generator inputs
  alone and shuffling stays a function of the run seed.

Usage:

    python data/lang/build_language_dataset.py --corpus wikitext-2   --stage fetch
    python data/lang/build_language_dataset.py --corpus wikitext-103 --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# --- Coordinates ------------------------------------------------------------
# Pinned by revision, not by "latest". `Salesforce/wikitext` is the canonical
# home of the dataset since the bare `wikitext` id became a redirect; the
# revision below is what was resolved at build time and is re-recorded on every
# build, so a silent upstream change shows up as a manifest diff.
HF_DATASET = "Salesforce/wikitext"

CORPORA = {
    # short name  -> HF config name
    "wikitext-2":   "wikitext-2-raw-v1",
    "wikitext-103": "wikitext-103-raw-v1",
}

# wikitext-2 is the development and gate corpus (fast enough for a CPU smoke
# run); wikitext-103 is canonical. Same loader, same manifest schema, same code
# path -- the only difference is which one a canonical run may name.
CANONICAL_CORPUS = "wikitext-103"

# HF calls it "validation"; every other file in this repo calls it "val".
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
SPLITS = ("train", "val", "test")

DEFAULT_VOCAB_SIZE = 8192
DEFAULT_SEQ_LEN = 256
EOT_TOKEN = "<|endoftext|>"

# The string whose absence is the whole point of choosing the -raw- configs.
UNK_LITERAL = "<unk>"

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256. Chunked because train.npy is ~200 MB and
    reading it whole would double peak memory for no reason."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    """The commit the build ran at, or the string "unknown".

    Never raises: a build in a tarball with no .git is legitimate, and a
    provenance field that says "unknown" is honest, whereas a crash here would
    lose a 40-minute tokenization.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO,
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def corpus_dir(corpus: str) -> str:
    return os.path.join(_HERE, corpus)


def manifest_path(corpus: str) -> str:
    return os.path.join(corpus_dir(corpus), "dataset_meta.json")


def load_manifest(corpus: str) -> dict:
    """Read the manifest if it exists, else an empty dict.

    The build is staged (T-L2.0 fetch, T-L2.1 tokenizer, T-L2.2 pack, ...) and
    each stage APPENDS to the manifest rather than rewriting it, so a later
    stage cannot silently drop an earlier stage's measurement.
    """
    p = manifest_path(corpus)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(corpus: str, manifest: dict) -> str:
    os.makedirs(corpus_dir(corpus), exist_ok=True)
    p = manifest_path(corpus)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p


# ===========================================================================
# T-L2.0  Fetch
# ===========================================================================

def fetch_splits(corpus: str, cache_dir: str | None = None):
    """Load the three author-provided splits and return {val_name: Dataset}.

    Returns HuggingFace `Dataset` objects, not lists: wikitext-103's train split
    is 1.8 M rows and materializing it as Python strings costs ~1 GB before a
    single token is produced. The rows are consumed as a stream in `raw_stats`
    and again in the tokenizer/packing stages.
    """
    from datasets import load_dataset  # imported late: 2.5 s of import cost

    config = CORPORA[corpus]
    out = {}
    for hf_split, our_split in SPLIT_MAP.items():
        out[our_split] = load_dataset(
            HF_DATASET, config, split=hf_split, cache_dir=cache_dir,
        )
    return out


def hf_revision() -> str:
    """The resolved commit sha of the dataset repo, or "unknown" offline.

    Recorded so that "we built from WikiText" is a checkable claim. Not fatal if
    unavailable -- the parquet files themselves are hashed downstream.
    """
    try:
        from huggingface_hub import dataset_info
        return dataset_info(HF_DATASET).sha
    except Exception:
        return "unknown"


def raw_stats(splits: dict) -> dict:
    """Line counts, byte sizes and `<unk>` occurrences per split.

    `<unk>` is counted, not asserted, here -- the assertion belongs to the
    verifier so that the measured number reaches the manifest either way. A
    non-zero count means the wrong config was fetched (a `-v1` instead of a
    `-raw-v1`), which is a fetch bug, not a data property.
    """
    stats = {"lines": {}, "bytes": {}, "unk_occurrences": {},
             "nonempty_lines": {}}
    for split, ds in splits.items():
        n_lines = 0
        n_nonempty = 0
        n_bytes = 0
        n_unk = 0
        for row in ds:
            text = row["text"]
            n_lines += 1
            n_bytes += len(text.encode("utf-8"))
            if text.strip():
                n_nonempty += 1
            if UNK_LITERAL in text:
                n_unk += text.count(UNK_LITERAL)
        stats["lines"][split] = n_lines
        stats["bytes"][split] = n_bytes
        stats["unk_occurrences"][split] = n_unk
        stats["nonempty_lines"][split] = n_nonempty
    return stats


def stage_fetch(corpus: str, cache_dir: str | None = None) -> dict:
    """T-L2.0. Download, scan, record. Writes no .npy -- that is T-L2.2."""
    t0 = time.time()
    print(f"[fetch] {HF_DATASET} :: {CORPORA[corpus]}")
    rev = hf_revision()
    splits = fetch_splits(corpus, cache_dir=cache_dir)
    for s in SPLITS:
        if s not in splits:
            raise RuntimeError(f"split {s!r} missing from the fetched dataset")
    print(f"[fetch] loaded in {time.time() - t0:.1f}s   revision={rev[:12]}")

    t1 = time.time()
    stats = raw_stats(splits)
    print(f"[fetch] scanned in {time.time() - t1:.1f}s")

    manifest = load_manifest(corpus)
    manifest.update({
        "corpus": corpus,
        "hf_dataset": HF_DATASET,
        "hf_config": CORPORA[corpus],
        "hf_revision": rev,
        "generator_script": "data/lang/build_language_dataset.py",
        "generator_commit": _git_commit(),
        "is_canonical_corpus": corpus == CANONICAL_CORPUS,
        "raw_lines": stats["lines"],
        "raw_nonempty_lines": stats["nonempty_lines"],
        "raw_bytes": stats["bytes"],
        "unk_literal_occurrences": stats["unk_occurrences"],
        "splits_are_author_provided": True,
    })
    p = save_manifest(corpus, manifest)

    for s in SPLITS:
        print(f"[fetch] {s:5s} lines={stats['lines'][s]:>9,} "
              f"nonempty={stats['nonempty_lines'][s]:>9,} "
              f"bytes={stats['bytes'][s]:>12,} "
              f"unk={stats['unk_occurrences'][s]}")
    print(f"[fetch] manifest -> {p}")
    return manifest


# ===========================================================================
# Document segmentation
# ===========================================================================
# WikiText rows are either '' or a single line ending in '\n', so joining a
# split's rows with '' reproduces the original raw text byte-for-byte. Verified
# on wikitext-2 validation: 2461 of 3760 rows end in '\n' and the remaining 1299
# are empty strings.
#
# A document starts at a TOP-LEVEL heading, ' = Title = \n'. Subsection headings
# are ' = = Description = = \n', so the discriminator is the second character
# after the opening ' = ' -- a '=' means a subsection. This is the same rule the
# WikiText authors' own preprocessing uses and it is what makes the splits
# document-disjoint, so getting it wrong would not corrupt the text but WOULD put
# the <|endoftext|> separators in the wrong places.

def _is_doc_start(text: str) -> bool:
    return (text.startswith(" = ")
            and not text.startswith(" = = ")
            and text.strip().endswith(" ="))


def iter_documents(ds):
    """Yield one string per document, in dataset order.

    Leading rows before the first heading (wikitext-2 validation starts with an
    empty row) are emitted as a leading document rather than dropped: dropping
    corpus content to make the segmentation tidy would be a silent edit to the
    dataset.
    """
    buf = []
    for row in ds:
        text = row["text"]
        if _is_doc_start(text) and buf:
            yield "".join(buf)
            buf = []
        buf.append(text)
    if buf:
        yield "".join(buf)


def split_text_path(corpus: str, split: str) -> str:
    return os.path.join(corpus_dir(corpus), f"{split}_text.txt")


# ===========================================================================
# T-L2.1  Tokenizer -- byte-level BPE, V=8192, fitted on train ONLY
# ===========================================================================

def tokenizer_path(corpus: str) -> str:
    return os.path.join(corpus_dir(corpus), "tokenizer.json")


def write_train_text(corpus: str, splits: dict) -> tuple[str, int, int]:
    """Materialize the TRAIN split's raw text, and only the train split's.

    `tokenizers` trains from files, and T-L2.1's Verify is "assert on the file
    list, not on intent" -- so the train-only property is made checkable by
    never writing a val or test text file at all. If one ever appears in this
    directory it did not come from here.

    Returns (path, n_documents, n_bytes).
    """
    path = split_text_path(corpus, "train")
    os.makedirs(corpus_dir(corpus), exist_ok=True)
    n_docs = 0
    n_bytes = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for doc in iter_documents(splits["train"]):
            fh.write(doc)
            n_docs += 1
            n_bytes += len(doc.encode("utf-8"))
    return path, n_docs, n_bytes


def fit_tokenizer(train_text: str, vocab_size: int, out_path: str) -> dict:
    """Fit a byte-level BPE and save it. Returns the fields for the manifest.

    Byte-level with the full 256-byte initial alphabet, so there is no OOV and
    no `<unk>` token can exist -- the property T-L2.0 measured in the corpus is
    then also true of the tokenizer, and a decode is always exact.
    `add_prefix_space=False` because WikiText lines already begin with a space
    and adding another would make the round-trip inexact.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([train_text], trainer)
    tok.save(out_path)
    return {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": tok.token_to_id(EOT_TOKEN),
        "tokenizer_sha256": _sha256_file(out_path),
        "tokenizer_training_files": [
            os.path.relpath(train_text, _REPO).replace("\\", "/")],
    }


def stage_tokenizer(corpus: str, vocab_size: int,
                    cache_dir: str | None = None) -> dict:
    """T-L2.1. Fit on train only, record the file list, round-trip a held-out
    paragraph as part of the build rather than only in the test file."""
    t0 = time.time()
    splits = fetch_splits(corpus, cache_dir=cache_dir)

    train_text, n_docs, n_bytes = write_train_text(corpus, splits)
    print(f"[tok] train text: {n_docs:,} documents, {n_bytes:,} bytes "
          f"-> {os.path.basename(train_text)}")

    out = tokenizer_path(corpus)
    fields = fit_tokenizer(train_text, vocab_size, out)
    print(f"[tok] fitted V={fields['vocab_size']} "
          f"eot_id={fields['eot_id']} in {time.time() - t0:.1f}s")

    # Round-trip a HELD-OUT paragraph -- val, not train. A tokenizer that only
    # round-trips text it was fitted on would pass a train-side check and still
    # be lossy on the data every reported number is computed from.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(out)
    probe = next(d for d in iter_documents(splits["val"]) if len(d) > 2000)
    rt = tok.decode(tok.encode(probe).ids)
    fields["roundtrip_val_paragraph_exact"] = bool(rt == probe)
    fields["roundtrip_probe_bytes"] = len(probe.encode("utf-8"))
    print(f"[tok] held-out round-trip exact: "
          f"{fields['roundtrip_val_paragraph_exact']} "
          f"({fields['roundtrip_probe_bytes']:,} bytes)")
    if not fields["roundtrip_val_paragraph_exact"]:
        raise RuntimeError(
            "byte-level BPE failed to round-trip a held-out paragraph; the "
            "corpus would be silently altered by tokenization")

    fields["train_documents"] = n_docs
    fields["train_text_bytes"] = n_bytes
    manifest = load_manifest(corpus)
    manifest.update(fields)
    print(f"[tok] manifest -> {save_manifest(corpus, manifest)}")
    return manifest


# ===========================================================================
# T-L2.2  Pack into fixed-length blocks, drop the trailing partial
# ===========================================================================

def split_npy_path(corpus: str, split: str) -> str:
    return os.path.join(corpus_dir(corpus), f"{split}.npy")


def _encode_stream_to_bin(tok, ds, bin_path: str, eot_id: int,
                          batch_docs: int = 256) -> int:
    """Tokenize a split document-by-document straight to a raw uint16 file.

    Written incrementally rather than accumulated in memory: wikitext-103's train
    split is ~130 M tokens, and a Python list of that many ints is ~3.6 GB before
    any array exists. Streaming keeps peak memory at one batch.

    One `<|endoftext|>` is inserted BETWEEN documents (not after the last), which
    is what makes a block boundary crossing a document boundary a signal the
    model can condition on rather than a silent concatenation.

    Returns the total number of tokens written.
    """
    import numpy as np

    total = 0
    batch: list[str] = []
    first = True
    with open(bin_path, "wb") as fh:
        def flush(batch):
            nonlocal total, first
            if not batch:
                return
            for enc in tok.encode_batch(batch):
                ids = enc.ids
                if not first:
                    ids = [eot_id] + ids
                first = False
                arr = np.asarray(ids, dtype=np.uint16)
                fh.write(arr.tobytes())
                total += arr.size

        for doc in iter_documents(ds):
            batch.append(doc)
            if len(batch) >= batch_docs:
                flush(batch)
                batch = []
        flush(batch)
    return total


def pack_split(tok, ds, corpus: str, split: str, seq_len: int,
               eot_id: int) -> dict:
    """Encode, chunk to exactly `seq_len`, drop the trailing partial, store .npy."""
    import numpy as np

    bin_path = os.path.join(corpus_dir(corpus), f".{split}.tokens.bin")
    t0 = time.time()
    total = _encode_stream_to_bin(tok, ds, bin_path, eot_id)

    n_blocks = total // seq_len
    dropped = total - n_blocks * seq_len
    if n_blocks == 0:
        os.remove(bin_path)
        raise RuntimeError(
            f"{split}: {total} tokens is fewer than one seq_len={seq_len} block")

    out = split_npy_path(corpus, split)
    dst = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.uint16, shape=(n_blocks, seq_len))
    src = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(total,))
    # Copied in row batches so neither array is materialized whole.
    step = max(1, (1 << 22) // seq_len)
    for i in range(0, n_blocks, step):
        j = min(i + step, n_blocks)
        dst[i:j] = src[i * seq_len:j * seq_len].reshape(j - i, seq_len)
    dst.flush()
    del dst, src
    os.remove(bin_path)

    return {
        "split": split,
        "tokens": total,
        "blocks": n_blocks,
        "dropped_tail_tokens": dropped,
        "sha256": _sha256_file(out),
        "seconds": round(time.time() - t0, 1),
    }


def stage_pack(corpus: str, seq_len: int,
               cache_dir: str | None = None) -> dict:
    """T-L2.2. Requires the tokenizer stage to have run."""
    from tokenizers import Tokenizer

    man = load_manifest(corpus)
    tok_path = tokenizer_path(corpus)
    if not os.path.exists(tok_path) or "eot_id" not in man:
        raise RuntimeError(
            f"{corpus}: run --stage tokenizer first (no tokenizer.json / eot_id)")
    tok = Tokenizer.from_file(tok_path)
    eot_id = man["eot_id"]

    splits = fetch_splits(corpus, cache_dir=cache_dir)
    per = {}
    for split in SPLITS:
        r = pack_split(tok, splits[split], corpus, split, seq_len, eot_id)
        per[split] = r
        print(f"[pack] {split:5s} tokens={r['tokens']:>11,} "
              f"blocks={r['blocks']:>8,} dropped_tail={r['dropped_tail_tokens']:>4} "
              f"({r['seconds']}s)")

    man["seq_len"] = seq_len
    man["splits"] = {s: per[s]["blocks"] for s in SPLITS}
    man["token_counts"] = {s: per[s]["tokens"] for s in SPLITS}
    man["dropped_tail_tokens"] = {s: per[s]["dropped_tail_tokens"] for s in SPLITS}
    man["sha256"] = {s: per[s]["sha256"] for s in SPLITS}
    man["storage"] = {
        "format": "npy", "dtype": "uint16", "load": "np.load(mmap_mode='r')",
        "shape": "(n_blocks, seq_len)",
        "on_disk_order": "corpus order; all shuffling is the DataLoader's",
    }
    man["dataset_version"] = (
        f"lang-{corpus}-bpe{man['vocab_size']}-len{seq_len}-"
        f"{_sha256_bytes(''.join(per[s]['sha256'] for s in SPLITS).encode())[:8]}")
    print(f"[pack] dataset_version = {man['dataset_version']}")
    print(f"[pack] manifest -> {save_manifest(corpus, man)}")
    return man


# ===========================================================================
# CLI
# ===========================================================================

STAGES = ("fetch", "tokenizer", "pack")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the packed WikiText language dataset + manifest",
    )
    p.add_argument("--corpus", choices=sorted(CORPORA), required=True,
                   help=f"wikitext-2 = dev/gate corpus; "
                        f"{CANONICAL_CORPUS} = canonical")
    p.add_argument("--stage", choices=STAGES + ("all",), default="all")
    p.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--cache_dir", default=None,
                   help="HF cache override; default is the user HF cache")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "fetch":
            stage_fetch(args.corpus, cache_dir=args.cache_dir)
        elif stage == "tokenizer":
            stage_tokenizer(args.corpus, args.vocab_size,
                            cache_dir=args.cache_dir)
        elif stage == "pack":
            stage_pack(args.corpus, args.seq_len, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())



