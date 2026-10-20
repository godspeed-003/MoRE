"""opdiff_data.py - `MoREOpdiffDataset`: op-difficulty rows in the engine's tuple shape.

The opdiff counterpart of `more/data.py:MoREDataset` (arithmetic) and
`more/lang_data.py:MoRELanguageDataset` (language), and it hands the engine the
SAME 7-TUPLE, with these meanings:

    slot  arithmetic                          opdiff
    ----  ----------------------------------  ----------------------------------
    0     x            [S, F] float           input_ids    [seq_len] long
    1     step_mask    [S]    bool            step_mask    [seq_len] bool, ALL TRUE
    2     step_experts [S]    long            step_experts [seq_len] long, the
                                              ROW's oracle group (0..6) at every
                                              position
    3     step_ops     [S]    long            step_ops     [seq_len] long, the
                                              owning example's operation id
                                              (0..24) at every position
    4     family       scalar long            family       scalar long, the row's
                                              group (0..6) -- DIAGNOSTIC only
    5     depth        scalar long            depth        scalar long = seq_len
    6     target       scalar float           target       scalar float = NaN (poison)

WHY THE ARITY IS PRESERVED RATHER THAN A NEW SIGNATURE INTRODUCED. `engine.py`
unpacks a 7-tuple positionally, in the training loop and in the validation
loop. A third dataset with its own arity would force a second training loop,
which is exactly what `plan.md` §9 forbids (ONE training system with an
explicit mode). Slots whose opdiff meaning is "nothing" are filled with the
documented ignore label rather than dropped.

SLOT 3 IS NOT EMPTY, AND THAT IS THE POINT. Language has no operation axis
(`lang_families.ALL_OP_NAMES` is `[]`), so its `step_ops` is all -1. opdiff HAS
one: 25 operation ids, and it is what the per-operation depth breakdown
(`recursion/avg_depth_by_op/<OP>`) is reported on. It is derived from the OP
TOKEN that is already in the input row -- position `t` belongs to the example
starting at `t - t % 19`, whose first token IS the operation. Nothing is read
from a side file, so the operation a token is attributed to is the operation
the model actually SAW.

SLOT 2 IS THE ORACLE GROUP, AND IT IS NOT AN INPUT FEATURE. The group label
lives in `{split}_group.npy` and is used for exactly two things: the confusion
matrix's row axis (the oracle's own indexing) and the supervision target of
the labelled `oracle_routing` ablation. It is never handed to the model: the
model sees token ids only, and the router has to infer the operation
difficulty from the operation token that the generator put there. That is the
whole experimental point of this dataset -- `updated_rules.md` §3 prohibits the
oracle routing class from being an input feature, and this is the task where
that prohibition is testable.

SLOT 4 IS THE GROUP, DIAGNOSTICALLY. There is no `cls_head` on an LM-shaped
task (`model.py` does not construct one), so this value reaches no loss. It is
carried because the engine's tuple has the slot and because a reader of a
metrics file may want to know the row's group; it is not a supervision signal
and `loss_weights.family_cls` is refused on this task.

SLOT 5 IS `seq_len` AND IS UNUSED. Arithmetic's `depth` slot carries the number
of steps in the program; an opdiff row has no step count, because the whole
point of the format is that there is NO chain of thought -- the answer digits
follow the operands directly. The engine does not read this slot on an
LM-shaped task, and the value is a placeholder rather than a measurement.

SLOT 6 IS NaN ON PURPOSE -- IT IS A POISON VALUE, NOT A PLACEHOLDER. The LM
target is `input_ids[1:]` predicted from `input_ids[:-1]`, derived by shifting
inside the loss so the ids are not stored twice. The slot has to hold
something collatable, and `0.0` or `-1.0` would let a mis-wired opdiff loss
train silently against a constant and produce a plausible-looking curve. NaN
makes that defect surface on the first optimizer step. It is never logged and
never reported, so this is not a sentinel presented as a measurement
(CLAUDE.md §4) -- it is the opposite: a value that cannot be mistaken for one.

THE LOSS MASK COMES FROM THE MANIFEST, NOT FROM A RECOMPUTATION. `meta.json`
carries `loss_mask` -- the positions the generator intends to be scored -- and
the engine consumes `mask[1:]`, because the TARGETS are `x[:, 1:]` and the
mask is indexed by the position whose value is predicted. Recomputing the mask
from the layout instead of reading it would be a second implementation of a
rule the generator owns, and a divergence between the two would be a silent
change of what is being trained on.

SHUFFLING IS THE DATALOADER'S, ALWAYS. On-disk order is generation order and
`__getitem__` is a pure function of the index, so the row order is fixed and
the dataset fingerprint is stable. Reproducibility of the permutation comes
from `seeding.make_generator(seed, "dataloader")`, unchanged from arithmetic.

THE ARRAY IS MEMORY-MAPPED, AND OPENED PER PROCESS. `train_x.npy` is 113 MB
for the canonical dataset. A `np.memmap` held as an attribute would be
pickled by materializing it, so with `num_workers > 0` on Windows (spawn)
every worker would get its own copy. The handle is therefore opened lazily on
first access and keyed on the current pid -- the same shape as
`lang_data.MoRELanguageDataset`.
"""

import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .opdiff_families import (NUM_FAMILIES, NUM_OP_TYPES, OP_NAME_TO_IDX,
                               OP_TOKEN_OFFSET)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# The dataset root is declared RELATIVE TO THE REPO ROOT, not to the config
# file's directory and not to the working directory. The config file lives in
# `code/`, and its other relative paths resolve against `code/`; opdiff_root
# deliberately does not, because the same config file has to mean the same
# thing on two different machines (Ayan's 4060 and this one) and an absolute
# path baked into the config would move the `config_hash` per machine and
# embed a machine-specific string in the provenance record.
OPDIFF_ROOT_DEFAULT = "opdiff_v2"

# The generator's splits. `ood` is a FOURTH split, not a fifth row of `test`:
# it holds F3/F6/F12 rows -- iteration counts never seen in training -- and
# keeping it separate is what makes "does the model generalise past the
# trained iteration counts" a measurable question rather than a remark.
SPLITS = ("train", "val", "test", "ood")


def resolve_opdiff_root(value=None) -> str:
    """The absolute dataset root for a declared `opdiff_root`.

    ONE resolution point, called by the dataset constructor and by the version
    stamper, so a relative root cannot mean two different directories in two
    different places. An absolute path is taken as given (that is how a
    caller points at a dataset on another drive); a relative one is resolved
    against the REPO ROOT, not against the process's working directory --
    a training run must not depend on where it was launched from.
    """
    root = str(value if value is not None else OPDIFF_ROOT_DEFAULT)
    if os.path.isabs(root):
        return os.path.normpath(root)
    return os.path.normpath(os.path.join(_REPO, root))


def opdiff_versions(root=None) -> tuple[str | None, str | None]:
    """`(dataset_version, train_split_version)` for the dataset at `root`.

    DELIBERATELY NON-FATAL when the dataset is absent, and this is the
    load-bearing choice. The opdiff dataset is NOT tracked in git (113 MB), so
    a fresh clone -- Ayan's machine included -- has no dataset until the
    generator has been run. Raising here would make `--task opdiff` unusable
    before the build, and defaulting to a guess would put an unmeasured
    string into provenance. So a missing dataset yields `None`, which the
    proxy guard already refuses a canonical claim on -- and
    `MoREOpdiffDataset` raises the real, actionable error with the build
    command in it when the run actually reaches for the data.

    WHAT THE STRING IS. `meta.json` is the generator's own description of the
    dataset, and (since the generator was extended) it carries a
    `dataset_version` computed from the SHA-256 of every split it wrote. That
    is a CONTENT identity: two generations of the same configuration with
    different sampled rows produce different versions, which is exactly what a
    provenance field is for.

    THE FALLBACK, for a dataset generated before the generator carried
    hashes: the SHA-256 of `meta.json` plus a 4 MiB edge fingerprint of the
    train split. It is weaker than a full content hash -- it can in principle
    miss a change confined to the middle of the train split -- and it is
    labelled by construction: the two strings are not interchangeable, because
    they have different lengths and different prefixes of the same shape but
    a different provenance story. The fallback exists so that a dataset
    generated by an older generator is still RECORDED, rather than reported as
    unversioned.

    WHY `train_split_version` IS THE SAME STRING. On the language task the
    two are different facts: the corpus identity (content hashes of every
    split) and the SPLIT PROTOCOL (WikiText's author-provided, document-disjoint
    splits), which is a fact about the corpus provider rather than about the
    bytes. On opdiff there is no provider and no author split: the splits are
    carved by the generator itself, deterministically, from one generation.
    The train split's identity IS a property of the same generation, so
    publishing one measured string twice is honest, whereas inventing a second
    string ("opdiff-splits-v1") would be a claim about a protocol that is not
    written down anywhere but in this paragraph.
    """
    r = resolve_opdiff_root(root)
    meta_path = os.path.join(r, "meta.json")
    train_x = os.path.join(r, "train_x.npy")
    if not (os.path.exists(meta_path) and os.path.exists(train_x)):
        return None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return None, None

    dsv = man.get("dataset_version")
    if dsv:
        return str(dsv), str(dsv)

    try:
        with open(meta_path, "rb") as fh:
            meta_bytes = fh.read()
        with open(train_x, "rb") as fh:
            head = fh.read(4 * 1024 * 1024)
    except OSError:
        return None, None
    h = hashlib.sha256()
    h.update(meta_bytes)
    h.update(b"|")
    h.update(head)
    return f"opdiff-v2-fallback-{h.hexdigest()[:8]}", f"opdiff-v2-fallback-{h.hexdigest()[:8]}"


class MoREOpdiffDataset(Dataset):
    """One packed `seq_len`-token row per item, in the engine's 7-tuple shape.

    See the module docstring for what each slot means and why the empty ones
    hold `-1` or NaN rather than being dropped.
    """

    def __init__(self, split: str, root=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.root = resolve_opdiff_root(root)
        self.dir = self.root

        meta_path = os.path.join(self.root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} is missing. Build the dataset first: "
                f"python make_opdiff_dataset_v2.py --out {self.root}. For the "
                "smoke subset: --out opdiff_v2_smoke --train_rows_per_group 512 "
                "--eval_rows_per_group 128."
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)

        for key in ("vocab_size", "row_len", "example_len", "examples_per_row",
                    "tokens", "op_tokens", "loss_mask", "rows"):
            if key not in self.manifest:
                raise KeyError(
                    f"{meta_path} has no {key!r}. The dataset was not generated "
                    "completely, and a dataset that guessed its own layout would "
                    "silently disagree with the rows on disk."
                )
        self.seq_len = int(self.manifest["row_len"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.example_len = int(self.manifest["example_len"])
        self.examples_per_row = int(self.manifest["examples_per_row"])
        if self.seq_len != self.example_len * self.examples_per_row:
            raise ValueError(
                f"meta.json is internally inconsistent: row_len={self.seq_len} "
                f"but example_len ({self.example_len}) x examples_per_row "
                f"({self.examples_per_row}) = "
                f"{self.example_len * self.examples_per_row}. The row length is "
                "derived, so a mismatch means the manifest mixes generations."
            )
        if split not in self.manifest["rows"]:
            raise KeyError(
                f"meta.json:rows has no {split!r}. It has "
                f"{sorted(self.manifest['rows'])}."
            )
        self.n_rows = int(self.manifest["rows"][split])

        # THE LOSS MASK, READ FROM THE MANIFEST. `mask[1:]` is the one the
        # engine uses, because the targets are `x[:, 1:]`.
        lm = np.asarray(self.manifest["loss_mask"], dtype=np.uint8)
        if lm.shape != (self.seq_len,):
            raise ValueError(
                f"meta.json:loss_mask has shape {lm.shape}, expected "
                f"({self.seq_len},). The mask is indexed by position, so a "
                "length mismatch would score the wrong positions."
            )
        self.loss_mask = lm.astype(bool)
        self.loss_mask_shifted = self.loss_mask[1:]

        # THE OPERATION TOKEN LAYOUT, CHECKED AGAINST THE MANIFEST. The dense
        # op axis the code uses is `token - op0`; if the generator ever moved
        # the op tokens, every operation id in every metric would shift by the
        # same amount and nothing would look wrong.
        self.op0 = int(self.manifest["tokens"]["op0"])
        if self.op0 != OP_TOKEN_OFFSET:
            raise ValueError(
                f"meta.json:tokens.op0 is {self.op0}, but the operation axis in "
                f"opdiff_families assumes {OP_TOKEN_OFFSET}. The op id is "
                "`token - op0`, so the two must agree or every operation id is "
                "silently offset."
            )
        _mt = self.manifest["op_tokens"]
        if len(_mt) != NUM_OP_TYPES:
            raise ValueError(
                f"meta.json:op_tokens has {len(_mt)} operations, expected "
                f"{NUM_OP_TYPES}. The op axis width comes from the manifest and "
                "the manifest and the code must agree."
            )
        for _name, _idx in OP_NAME_TO_IDX.items():
            if _name not in _mt:
                raise KeyError(
                    f"meta.json:op_tokens has no {_name!r}. The generator's "
                    "operation set and the code's operation axis have drifted "
                    "apart."
                )
            if int(_mt[_name]) != _idx + OP_TOKEN_OFFSET:
                raise ValueError(
                    f"meta.json:op_tokens[{_name!r}] is {_mt[_name]}, expected "
                    f"{_idx + OP_TOKEN_OFFSET}. The op id is derived from the "
                    "token id, so a shifted token table silently relabels the "
                    "whole operation axis."
                )
        if self.vocab_size != OP_TOKEN_OFFSET + NUM_OP_TYPES:
            raise ValueError(
                f"meta.json:vocab_size is {self.vocab_size}, expected "
                f"{OP_TOKEN_OFFSET + NUM_OP_TYPES} "
                f"({OP_TOKEN_OFFSET} fixed + {NUM_OP_TYPES} operation tokens). "
                "The tied LM head is [vocab_size, d_model], so this is the size "
                "of the output layer."
            )

        # WHICH EXAMPLE EACH POSITION BELONGS TO. Position `t` belongs to the
        # example that starts at `t - t % example_len`, and that example's
        # FIRST token is its operation token.
        pos = np.arange(self.seq_len, dtype=np.int64)
        self._bases = (pos - pos % self.example_len)
        self._example_starts = (np.arange(self.examples_per_row, dtype=np.int64)
                                * self.example_len)

        # THE FILES THIS SPLIT NEEDS. Checked up front so a half-generated
        # dataset fails with the file name in the message rather than at the
        # first batch.
        self.x_path = os.path.join(self.root, f"{split}_x.npy")
        self.group_path = os.path.join(self.root, f"{split}_group.npy")
        self.diff_path = os.path.join(self.root, f"{split}_diff.npy")
        self.sub_path = os.path.join(self.root, f"{split}_sub.npy")
        for _p in (self.x_path, self.group_path, self.diff_path, self.sub_path):
            if not os.path.exists(_p):
                raise FileNotFoundError(
                    f"{_p} is missing. The dataset is incomplete -- regenerate "
                    f"it with: python make_opdiff_dataset_v2.py --out {self.root}"
                )

        # Provenance surfaced as attributes so `RunContext` stamps measured
        # values rather than a config author's recollection of them.
        self.dataset_version, self.train_split_version = opdiff_versions(self.root)
        self._x = None
        self._g = None
        self._pid = None

    # -- construction helpers ------------------------------------------------

    def _arrays(self):
        """The memory-mapped row and group arrays, opened once per process.

        Keyed on pid because a `np.memmap` attribute would be pickled by
        materializing it -- 113 MB per DataLoader worker on the canonical
        dataset, under Windows spawn.
        """
        pid = os.getpid()
        if self._x is None or self._pid != pid:
            self._x = np.load(self.x_path, mmap_mode="r")
            self._g = np.load(self.group_path, mmap_mode="r")
            self._pid = pid
            if self._x.shape != (self.n_rows, self.seq_len):
                raise ValueError(
                    f"{self.x_path} has shape {self._x.shape}, but meta.json "
                    f"says ({self.n_rows}, {self.seq_len}). One of the two is "
                    "stale; regenerate the dataset rather than guessing which."
                )
            if self._g.shape != (self.n_rows,):
                raise ValueError(
                    f"{self.group_path} has shape {self._g.shape}, expected "
                    f"({self.n_rows},). The group label is one per ROW."
                )
        return self._x, self._g

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        x_arr, g_arr = self._arrays()
        tok = np.asarray(x_arr[idx], dtype=np.int64)
        grp = int(g_arr[idx])
        if not (1 <= grp <= NUM_FAMILIES):
            raise ValueError(
                f"row {idx} of split {self.split!r} carries group {grp}, which "
                f"is outside 1..{NUM_FAMILIES}. The generator writes the group "
                "as a uint8 in 1..7; anything else means the file and the "
                "manifest are from different generations."
            )

        ids = torch.from_numpy(tok)

        # All True: an opdiff row is 16 complete examples with no padding, so
        # there is no `key_padding_mask` case and the ACT depth accounting is
        # exact rather than mask-conditional.
        step_mask = torch.ones(self.seq_len, dtype=torch.bool)

        # The oracle group, at every position. It is the ROW's group because
        # a row never mixes groups -- that is the generator's defining
        # constraint -- so the label is exact for every token in the row and
        # not a per-example approximation.
        group_idx = grp - 1
        step_experts = torch.full((self.seq_len,), group_idx, dtype=torch.long)

        # The operation of the owning example, at every position. Taken from
        # the OP TOKEN the model itself is shown, not from a side file.
        op_tokens = tok[self._example_starts]
        if op_tokens.min() < self.op0 or op_tokens.max() >= self.op0 + NUM_OP_TYPES:
            raise ValueError(
                f"row {idx} of split {self.split!r} has operation tokens in "
                f"[{int(op_tokens.min())}, {int(op_tokens.max())}], outside the "
                f"operation range [{self.op0}, {self.op0 + NUM_OP_TYPES - 1}]. "
                "Each example must START with its operation token; a row that "
                "does not was generated against a different layout."
            )
        step_ops = torch.from_numpy(tok[self._bases] - self.op0)

        return (
            ids,                                                     # 0
            step_mask,                                               # 1
            step_experts,                                            # 2
            step_ops,                                               # 3
            torch.tensor(group_idx, dtype=torch.long),               # 4 family
            torch.tensor(self.seq_len, dtype=torch.long),            # 5 depth
            torch.tensor(float("nan"), dtype=torch.float32),         # 6 poison
        )

    def __repr__(self):
        return (f"MoREOpdiffDataset({self.root}/{self.split}: "
                f"{self.n_rows:,} rows x {self.seq_len} tokens, "
                f"V={self.vocab_size}, {self.dataset_version})")

    # -- convenience, for the aggregator rather than the engine ---------------

    def nominal_difficulty(self):
        """`{split}_diff.npy` -- the generator's NOMINAL difficulty per row.

        A PLACEHOLDER, not a measurement: the generator writes the group
        number, or `7 + log2(k)` for the F operations, and says so. It is
        exposed so the aggregator can correlate measured depth against it
        WITHOUT re-implementing the nominal rule in a second place, which
        would be a formula that could drift from the generator's.
        """
        return np.load(self.diff_path, mmap_mode="r")

    def sub_op_tokens(self):
        """`{split}_sub.npy` -- the operation token of a single-op row, else -1.

        Set for the E7 rows (one sub-operation per row) and for the OOD rows.
        -1 for the groups whose rows mix operations, which is the whole point
        of those groups: a row there has no single operation.
        """
        return np.load(self.sub_path, mmap_mode="r")
