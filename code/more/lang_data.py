"""lang_data.py - `MoRELanguageDataset`: packed LM blocks in the engine's tuple shape.

T-L2.3. Reads the artifacts `data/lang/build_language_dataset.py` produced and hands
the engine the SAME 7-TUPLE `more/data.py:MoREDataset` does, with the language
meanings:

    slot  arithmetic                          language
    ----  ----------------------------------  --------------------------------------
    0     x            [S, F] float           input_ids    [seq_len] long
    1     step_mask    [S]    bool            step_mask    [seq_len] bool, ALL TRUE
    2     step_experts [S]    long            step_experts [seq_len] long, from
                                              token_family[input_ids]; -1 = ignore
    3     step_ops     [S]    long            step_ops     [seq_len] long, ALL -1
    4     family       scalar long            family       scalar long, -1
    5     depth        scalar long            depth        scalar long = seq_len
    6     target       scalar float           target       scalar float = NaN (poison)

WHY THE ARITY IS PRESERVED RATHER THAN A NEW SIGNATURE INTRODUCED. `engine.py:438`
unpacks a 7-tuple positionally, and so does the validation loop at 686. A language
dataset with its own arity would force a second training loop, which is exactly what
`plan.md` §9 forbids (ONE training system with an explicit mode). Slots whose
language meaning is "nothing" are filled with the documented ignore label rather than
dropped.

SLOT 3 IS ALL -1, AND THAT IS THE OPERATION AXIS BEING ABSENT, NOT EMPTY.
`lang_families.ALL_OP_NAMES` is `[]`, so every `step_ops >= 0` mask is empty and
every per-operation aggregation produces nothing -- which the metric layer reports as
`N/A` rather than as `0.0`. See `lang_families` §2.

SLOT 4 IS -1 BECAUSE A 256-TOKEN BLOCK HAS NO LEXICAL CLASS. Arithmetic's `family` is
the whole-program family, and a multi-operation program already carries -1 there;
`canonical_spec_language.json` sets `family_cls_weight = 0.0` for the same reason.

SLOT 6 IS NaN ON PURPOSE -- IT IS A POISON VALUE, NOT A PLACEHOLDER. The LM target is
`input_ids[1:]` predicted from `input_ids[:-1]`, derived by shifting inside the loss
so the ids are not stored twice (`plan_language.md` §3). But the slot has to hold
something collatable, and `0.0` or `-1.0` would let a mis-wired language loss train
silently against a constant and produce a plausible-looking curve. NaN makes that
defect surface on the first optimizer step. It is never logged and never reported, so
this is not a sentinel presented as a measurement (CLAUDE.md §4) -- it is the opposite:
a value that cannot be mistaken for one.

SHUFFLING IS THE DATALOADER'S, ALWAYS. On-disk order is corpus order and `__getitem__`
is a pure function of the index, so the block order is fixed and the manifest hash is
stable. Reproducibility of the permutation comes from `seeding.make_generator(seed,
"dataloader")`, unchanged from arithmetic.

THE ARRAY IS MEMORY-MAPPED, AND OPENED PER PROCESS. `train.npy` is 269 MB for the
canonical corpus. A `np.memmap` held as an attribute would be pickled by
materializing it, so with `num_workers > 0` on Windows (spawn) every worker would get
its own 269 MB copy. The handle is therefore opened lazily on first access and keyed
on the current pid.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .lang_families import NO_FAMILY_IDX, NUM_FAMILIES

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
LANG_ROOT = os.path.join(_REPO, "data", "lang")

SPLITS = ("train", "val", "test")

# Stamped into provenance so a run records that it used WikiText's author-provided,
# document-disjoint splits rather than a re-split of the corpus. T-L2.7 verified the
# consequence directly: zero exact-content overlap between train and val/test across
# all 526,320 canonical train blocks.
TRAIN_SPLIT_VERSION = "wikitext-author-splits-v1"


class MoRELanguageDataset(Dataset):
    """One packed `seq_len`-token block per item, in the engine's 7-tuple shape.

    See the module docstring for what each slot means and why the empty ones hold
    `-1` or NaN rather than being dropped.
    """

    def __init__(self, corpus, split, lang_root=None, require_families=True):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.corpus = corpus
        self.split = split
        self.lang_root = os.path.abspath(lang_root or LANG_ROOT)
        self.dir = os.path.join(self.lang_root, corpus)
        self.array_path = os.path.join(self.dir, f"{split}.npy")

        man_path = os.path.join(self.dir, "dataset_meta.json")
        if not os.path.exists(man_path):
            raise FileNotFoundError(
                f"{man_path} is missing. Build the corpus first: "
                f"python data/lang/build_language_dataset.py --corpus {corpus} "
                f"--stage all"
            )
        with open(man_path, "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        for key in ("seq_len", "vocab_size", "splits", "dataset_version"):
            if key not in self.manifest:
                raise KeyError(
                    f"{man_path} has no {key!r}: the pack stage has not run for "
                    f"{corpus}. A dataset that guessed seq_len would silently "
                    f"disagree with the stored blocks."
                )
        self.seq_len = int(self.manifest["seq_len"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.n_blocks = int(self.manifest["splits"][split])

        # Provenance surfaced as attributes so `RunContext` stamps measured values
        # rather than a config author's recollection of them.
        self.dataset_version = self.manifest["dataset_version"]
        self.train_split_version = TRAIN_SPLIT_VERSION
        self.primary_metric_floor = self.manifest.get("primary_metric_floor")

        self.token_family = self._load_family_lookup(require_families)
        self._array = None
        self._array_pid = None

    # -- construction helpers ------------------------------------------------

    def _load_family_lookup(self, require):
        """`token_family[V]` as a long tensor, or None when families are not needed.

        Validated against the manifest rather than trusted: a lookup of the wrong
        length would index out of range on the first batch, and a lookup with a
        family index >= NUM_FAMILIES would silently widen the oracle label space --
        which is the defect T8.3 fixed on the arithmetic side.
        """
        path = os.path.join(self.dir, "token_family.npy")
        if not os.path.exists(path):
            if require:
                raise FileNotFoundError(
                    f"{path} is missing. Build it with "
                    f"python data/lang/build_family_lookup.py --corpus "
                    f"{self.corpus}, or pass require_families=False to get "
                    f"step_experts filled with the ignore label."
                )
            return None
        table = np.load(path)
        if table.shape != (self.vocab_size,):
            raise ValueError(
                f"{path} has shape {table.shape}, expected ({self.vocab_size},). "
                f"The lookup is indexed by token id, so a length mismatch means it "
                f"was built against a different tokenizer."
            )
        lo, hi = int(table.min()), int(table.max())
        if lo < NO_FAMILY_IDX or hi >= NUM_FAMILIES:
            raise ValueError(
                f"{path} holds family indices in [{lo}, {hi}], outside "
                f"[{NO_FAMILY_IDX}, {NUM_FAMILIES - 1}]. Only -1 (ignore) and "
                f"0..{NUM_FAMILIES - 1} are families; anything else widens the "
                f"oracle label space by accident."
            )
        return torch.from_numpy(table.astype(np.int64))

    def _blocks(self):
        """The memory-mapped array, opened once per process.

        Keyed on pid because a `np.memmap` attribute would be pickled by
        materializing it -- 269 MB per DataLoader worker on the canonical corpus,
        under Windows spawn.
        """
        pid = os.getpid()
        if self._array is None or self._array_pid != pid:
            self._array = np.load(self.array_path, mmap_mode="r")
            self._array_pid = pid
            if self._array.shape != (self.n_blocks, self.seq_len):
                raise ValueError(
                    f"{self.array_path} has shape {self._array.shape}, but the "
                    f"manifest says ({self.n_blocks}, {self.seq_len}). One of the "
                    f"two is stale; the manifest records a SHA-256 per split, so "
                    f"run code/test_language_data.py to find out which."
                )
        return self._array

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        ids = torch.from_numpy(
            np.asarray(self._blocks()[idx], dtype=np.int64))

        # All True, and that is the whole reason the trailing partial block was
        # dropped at pack time: no `key_padding_mask` means the NaN path in
        # attention over a fully-masked row is unreachable and the ACT denominators
        # are exact rather than mask-conditional (plan_language.md §3).
        step_mask = torch.ones(self.seq_len, dtype=torch.bool)

        if self.token_family is None:
            step_experts = torch.full((self.seq_len,), NO_FAMILY_IDX,
                                      dtype=torch.long)
        else:
            step_experts = self.token_family[ids]

        return (
            ids,                                                     # 0
            step_mask,                                               # 1
            step_experts,                                            # 2
            torch.full((self.seq_len,), -1, dtype=torch.long),       # 3 step_ops
            torch.tensor(NO_FAMILY_IDX, dtype=torch.long),           # 4 family
            torch.tensor(self.seq_len, dtype=torch.long),            # 5 depth
            torch.tensor(float("nan"), dtype=torch.float32),         # 6 poison
        )

    def __repr__(self):
        return (f"MoRELanguageDataset({self.corpus}/{self.split}: "
                f"{self.n_blocks:,} blocks x {self.seq_len} tokens, "
                f"V={self.vocab_size}, {self.dataset_version})")

    # -- convenience ---------------------------------------------------------

    def family_token_counts(self):
        """Per-family token counts for THIS split, from the manifest.

        Read rather than recomputed: `build_family_lookup.py` already counted every
        stored token, and a second implementation here would be a second thing to
        keep correct. `code/test_language_data.py` TL3.1h is what checks the
        manifest's numbers against the arrays.
        """
        rec = self.manifest.get("family_counts_tokens", {}).get(self.split)
        return dict(rec["by_family"]) if rec else None
