# plan_language.md — MoRE on English text (branch `claude/english-language-dataset-migration`)

## 0. Authority and scope

This document is the execution plan for **one** thing: making the existing MoRE
engine train and evaluate on **English natural language** instead of the
synthetic arithmetic micro-POC, with the same scientific discipline.

It does **not** supersede anything. The authority chain is unchanged:

| Document | Role | Applies to language work? |
|---|---|---|
| [updated_rules.md](updated_rules.md) | architectural + scientific constraints | **yes, in full** |
| [updated_objective.md](updated_objective.md) | what is tested, what counts as success | **yes**, re-instantiated in §2 |
| [plan.md](plan.md) | execution plan for the arithmetic POC | closed; §20 failure protocol still binds |
| [CLAUDE.md](CLAUDE.md) | working contract | **yes, in full** |
| **this file** | execution plan for the language POC | — |

Where `updated_rules.md` names something arithmetic-specific (`OP_TO_EXPERT`,
`E1 ADD/SUB`, "operation identity"), §4 and §5 below give the language
instantiation of the *same* rule. The rule itself is never relaxed. Two rules
are made **stricter** for language (§4.4, §7.3) because the language task
removes the ground truth that made the looser version safe.

The arithmetic POC is **finished and published as Outcome C** (see
[README.md](README.md)). Its result is evidence and is preserved bit-for-bit:

- no arithmetic run, metric, table or test is deleted, edited, or re-interpreted;
- `data/{train,val,test}.jsonl`, `data/dataset_meta.json`, `code/config.json`,
  `code/canonical_spec.json` and all seven `test_phase*.py` suites keep their
  current meaning and keep passing;
- `python code/run_correctness_suite.py` must still print
  `TOTAL 356 356 0 0` / `CORRECTNESS SUITE: ALL GATES PASS`
  **after every language commit**. This is Gate L0 and it is checked first.
  The reference was written as 350 from the arithmetic-era notes; the measured
  baseline on this machine is **356**, established in T-L0.3 after two
  environment-dependent defects in the suite itself were fixed (a cross-drive
  `os.path.relpath` that killed Gate 4 mid-run, and a Gate 5 run-selection key
  that graded pre-fix history). 356 is the number every later comparison uses.

---

## 1. What changes: a `task` axis orthogonal to `architecture`

CLAUDE.md §9 forbids parallel implementations: *"One training system with an
explicit `architecture = moe | mor | more` mode. Do not create parallel
`run_moe.py` / `run_mor.py` implementations."* The same reasoning forbids a
parallel `train_language.py`. If the language and arithmetic paths were two
programs, the MoE/MoR/MoRE comparison would no longer be guaranteed to share one
optimizer setup, one seeding order, one provenance stamp and one exporter — which
is the entire reason the arithmetic result is trustworthy.

So language is added as a **second value on a new axis**:

```text
task         = arithmetic | language      <-- NEW
architecture = moe | mor | more           <-- unchanged
```

The two axes are independent. `task` selects the input adapter, the output head,
the loss on the task path, the oracle-label source and the metric vocabulary.
`architecture` continues to select *only* `num_experts`, `max_depth`,
`adaptive_halting` and `ffn_mult`, exactly as `config.apply_architecture` does
today. Nothing in the recursion, routing, halting, balance-normalisation,
provenance, guard, seeding or export layers is duplicated.

**The default is `arithmetic`.** `load_config_defaults` sets
`cfg["task"] = "arithmetic"` if absent, so every existing config file, every
hand-built dict in the test suites, and every archived resolved config keeps its
current meaning without being edited. A language run must say so explicitly
(`--task language`, or `"task": "language"` in the config). This direction of
default is deliberate: the dangerous error is a language change silently altering
what an arithmetic run does, not the reverse.

### 1.1 Where the axis is read

| Location | Behaviour under `task = language` |
|---|---|
| `config.py:load_config_defaults` | applies the `language` default block (§10) after the shared defaults |
| `config.py:apply_task` (new) | stamps task-definitional fields; refuses combinations that cannot mean anything (§7.3) |
| `config.py:resolve_variant` | unchanged mechanism; language-only deviations get their own tags |
| `cli.py` | `--task {arithmetic,language}`, plus `--seq_len`, `--vocab_size` |
| `run_context.py:load_canonical_spec` | selects `canonical_spec_language.json` when `task == "language"` (§10.2) |
| `engine.py` | dataset construction, task-loss assembly, metric namespace |
| `model.py:MoREWrapper` | constructs the shared causal-attention sublayer (§6) |
| `model.py:MoREModel` | token-embedding input adapter + tied LM head instead of `step_proj`/`regression_head` |
| `metrics.py` | perplexity, difficulty-correlation metrics, family-ordered depth figure |

Everything else — `MoEBlock`, the ACT loop, balance normalisation, the
rectangular confusion matrix, Hungarian/AMI/purity, `RunContext`, `seeding.py`,
`seed_stats.py`, `export_results.py` — is used unmodified.

---

## 2. The scientific question, re-instantiated

`updated_objective.md` §0 asks whether expert specialization and adaptive
recursive computation can be learned **jointly**. On arithmetic the answer was
Outcome C: MoR alone was best, recursion did not pay for itself once bugs were
removed, and both recursive arms collapsed to ~2 steps for 93–97% of tokens.

Language is a materially different test of the same question, for three reasons
that are worth stating before any run, so the result cannot be re-narrated
afterwards:

1. **Token difficulty is genuinely heterogeneous and is not designed by us.**
   In arithmetic the "hard" tokens were hard because a human wrote
   `OP_TARGET_DEPTH`. In English, `the` after `on` is nearly free and a rare
   proper noun is expensive, and that spread is a measurable property of the
   corpus (§5.2). Adaptive depth therefore has something real to allocate
   against for the first time in this project.
2. **The expert partition is no longer ground truth.** `OP_TO_EXPERT` was a
   *functional* partition: `ADD` genuinely is a different computation from
   `SORT`. A POS-based partition of English tokens (§4) is a **linguistic
   prior**, not the objectively correct division of labour for next-token
   prediction. This weakens what routing accuracy can mean and forces the
   permutation-invariant metrics into the primary role (§4.4).
3. **The task is autoregressive, so context matters.** The arithmetic model had
   no self-attention and did not need any: each step-token's arguments were
   sufficient. Language is not learnable without context, so the recursive block
   gains a causal-attention sublayer (§6). This is the largest code change in the
   migration and the largest new leakage surface (Gate L1).

The primary predictive metric becomes **validation causal-LM cross-entropy**
(`val/task_loss`, nats/token), with **perplexity** = `exp(task_loss)` reported
alongside as the conventional unit. Everything `updated_rules.md` §8 says about
never using `total_loss`, reporting terms separately, `N/A` over sentinels, and
mean±std over the frozen seed set applies verbatim.

Outcomes A / B / C from `updated_objective.md` are unchanged and none may be
optimized toward. It must be recorded now, before any number exists, that
**Outcome C on language is a plausible and publishable result** and that the
arithmetic Outcome C creates no expectation either way.

---

## 3. Dataset

### 3.1 Source

**WikiText** (Merity et al. 2016), `wikitext-103-raw-v1` and `wikitext-2-raw-v1`,
via HuggingFace `datasets`. Chosen over the alternatives for reasons that matter
to *this* repo rather than to leaderboard comparability:

- it ships **document-disjoint** `train` / `validation` / `test` splits produced
  by the dataset authors, so split cleanliness is a property we can *audit*
  (Gate L1) rather than a property we manufacture with our own random split;
- `-raw-` means no `<unk>` substitution, so we control tokenization end to end
  and the vocabulary is not pre-truncated by someone else's decision;
- two sizes with identical structure: `wikitext-2` is the **development and gate**
  corpus (fast enough for a CPU smoke run), `wikitext-103` is the **canonical**
  corpus. Same loader, same manifest schema, same code path.

TinyStories was considered and rejected as canonical: it is synthetic
LLM-generated text, which would reintroduce exactly the "controlled synthetic
benchmark" caveat the migration is meant to escape. It stays available as a
labelled robustness ablation.

**FineWeb-Edu was raised and rejected as canonical (decision recorded 2026-09-03).**
The case for it is real and it is about downstream capability: HuggingFace's own
ablations show `FineWeb-Edu` giving a 12–24% relative improvement on MMLU, ARC and
OpenBookQA over general web snapshots. That result does not transfer to this study,
and the reason is scale, not doubt about the result. Those ablations were run at
roughly 1.8 B parameters; this spec's frozen shape is `d_model = 256`, 4 heads,
`num_blocks = 1`, `V = 8192`, tied LM head — single-digit millions of parameters.
A model that size sits at chance on MMLU (25%) whatever it trained on, so the one
axis on which FineWeb-Edu is measurably better is an axis this paper cannot
measure. Buying it would mean giving up the three properties that ARE load-bearing
here:

- **Author-provided document-disjoint splits.** Gate L2 audits split cleanliness as
  a property of the dataset (zero exact-content overlap across all 526,320 canonical
  train blocks). FineWeb-Edu ships no canonical val/test split, so we would construct
  one — and near-duplicate contamination across a self-made split of a CommonCrawl
  derivative becomes *our* claim to defend rather than the authors'. That is the most
  dangerous reviewer objection available against a small-scale LM result, and it is
  currently a one-line answer.
- **A floor a reader can situate.** WikiText-103 perplexity is among the most-reported
  numbers in LM research, so `primary_metric_floor = 4.9849` nats (ppl 146.2) sits in
  a literature. At single-digit millions of parameters on FineWeb-Edu there is no
  reference point at all, and "is 4.9 good" becomes unanswerable.
- **No sampling decision.** FineWeb-Edu is used via a subsample, which makes *which
  sample* and *which seed* provenance fields the proxy guard must enforce, and invites
  "did you pick a favourable sample". WikiText-103 is used whole.

The legitimate part of the objection — that a routing partition learned on
encyclopedic register might be an artifact of that register — is answered by a
**second corpus as a labelled robustness ablation**, exactly as TinyStories is above,
not by moving the canonical arm. That gives "the specialization finding replicates on
a differently-distributed corpus" while keeping the audit properties on the arm the
headline table is drawn from. `FineWeb-Edu` is the preferred candidate for that
ablation if Phase L-10 has budget.

### 3.2 Tokenizer — trained on the train split only, small vocab

A **byte-level BPE** trained with `tokenizers` on the **train split alone**,
target vocab **8192**, with `<|endoftext|>` as the single special token.

Two decisions here, both load-bearing:

**(a) Fitted on train only.** Fitting a tokenizer on the full corpus leaks
validation and test token statistics into the model's input representation. It is
a small leak and the LM literature routinely ignores it; `updated_rules.md` §10
does not permit us to. The tokenizer is therefore fitted on `train`, and its
merge table is part of the dataset manifest and its provenance.

**(b) 8192, not GPT-2's 50257.** With `d_model = 256`, a 50257-entry embedding is
12.9 M parameters. The entire MoRE expert stack is ~3.2 M. The architectural
difference between MoE, MoR and MoRE would be 6% of the parameter count and the
`|P_MoR - P_MoRE| / P_MoRE < 0.05` budget-matching bound of `updated_rules.md` §6
would be satisfiable *trivially* — matched on the embedding, unmatched on the part
under study. At 8192 the tied embedding is 2.1 M, comparable to the expert stack,
so the budget-matching constraint continues to constrain the thing it is about.
Parameter counts are reported **twice**: total, and excluding embedding/LM-head.

Consequence to state in the write-up: absolute perplexities are **not comparable**
to published WikiText numbers, because neither the vocabulary nor the model scale
matches. Only within-matrix comparisons are claimed. This is the same restriction
the arithmetic POC operated under and is not a new weakness.

### 3.3 Packing and the on-disk format

Per split: concatenate documents in **dataset order** (deterministic, recorded),
inserting one `<|endoftext|>` between documents, tokenize, then chunk into
contiguous blocks of exactly `seq_len` tokens. **The trailing partial block is
dropped.**

Dropping the remainder is not laziness — it makes every sequence full-length, so
`step_mask` is all-`True` for language. That in turn means:

- no `key_padding_mask` is needed in attention, so the all-padded-row NaN in
  `nn.MultiheadAttention` is unreachable by construction rather than guarded
  against;
- the ACT loop's `real_tokens`, `avg_depth`, `forced_exit_rate` and
  `mean_remainder` accounting is used **unmodified** and its denominators are
  exact;
- the number of dropped tokens is `< seq_len` per split and is recorded in the
  manifest as a measurement.

Stored as `uint16` NumPy arrays — `data/lang/<corpus>/{train,val,test}.npy` — not
JSONL. At 8192 vocab every id fits in `uint16`, a 103 M-token corpus is 206 MB on
disk, and the array is `mmap`-loaded so a `Dataset.__getitem__` is a slice with no
per-record Python object. The arithmetic path keeps JSONL: 70 000 records with
nested step structure is what JSONL is good at, and 10^8 integers is not.

**Sequence-level shuffling happens in the DataLoader, never on disk.** The on-disk
token order is the corpus order, so the file hash is reproducible from the
generator inputs alone, and shuffling remains a function of the run seed.

### 3.4 Manifest

`data/lang/<corpus>/dataset_meta.json`, mirroring the field structure of
[data/dataset_meta.json](data/dataset_meta.json) so `export_results.py` and the
provenance checks need no new schema:

```text
dataset_version         lang-<corpus>-bpe<V>-len<S>-<hash8>
generator_script        data/lang/build_language_dataset.py
generator_commit        git SHA at build time
hf_dataset / hf_config / hf_revision      exact HF coordinates
tokenizer_sha256        hash of tokenizer.json
vocab_size, seq_len
splits                  {train,val,test} -> number of SEQUENCES
token_counts            {train,val,test} -> number of TOKENS
dropped_tail_tokens     per split, a measurement
sha256                  per split, over the raw .npy bytes
family_manifest         the six language families, §4
family_counts           per split, TYPE-level and TOKEN-level, plus `unmapped`
depth_curriculum        "none" in canonical; the ablation table when enabled
split_overlap_documents {train|val, train|test, val|test} -> 0 required
split_overlap_ngrams    n=32 shingle collision counts, a measurement
input_feature_definition / target_definition
trivial_baselines       uniform / unigram / bigram cross-entropy (§5.3)
gate_l1_audit           checks_passed / checks_total and each check's number
```

`family_counts` is reported at both **type** and **token** level because the two
disagree strongly in language: `PUNCT_SYM` is a handful of types and a large
share of tokens. A load-balance figure read against the wrong one is misleading.

---

## 4. Oracle family partition — six POS families

### 4.1 The manifest

Six families, to keep `num_experts = 6` canonical and every `E>1` metric
comparable with the arithmetic result:

```text
L1 FUNCTION      determiners, prepositions, conjunctions, pronouns, particles,
                 auxiliary/copular verbs
L2 NOUN          common and proper nouns
L3 VERB          main verbs, participles, gerunds
L4 MODIFIER      adjectives, adverbs
L5 PUNCT_SYM     punctuation and symbols
L6 NUM_SUBWORD   numerals, and subword continuation pieces with no standalone POS
```

W&B-safe labels (no `/` inside a label, `updated_rules.md` §9):
`L1_FUNCTION, L2_NOUN, L3_VERB, L4_MODIFIER, L5_PUNCT_SYM, L6_NUM_SUBWORD`.

### 4.2 Assigned at the token-TYPE level

The label is a property of the **BPE id**, not of the occurrence: one
`token_family[V]` `int8` lookup array, built once at dataset-build time and stored
in the manifest directory. Per-token oracle labels in the training loop are then a
single gather, `token_family[input_ids]`, costing nothing.

Built by decoding each vocabulary entry to a surface string and tagging it with
nltk's averaged-perceptron tagger inside a minimal one-word sentence context,
then mapping the Penn tag to a family. Where a type is ambiguous across contexts
(`run` is L2 or L3), the **majority POS over the train split's occurrences** wins,
counted with the same tagger run over the raw train text. The counts behind every
majority decision are written to the manifest.

Why type-level rather than per-occurrence: a per-occurrence oracle would require
running a POS tagger over 10^8 tokens (slow, and its own error surface), and would
make the oracle label depend on context — which means routing accuracy would be
measuring the model's ability to reproduce a tagger's contextual decisions, a
different and much stronger claim than "the router groups tokens by lexical
class". Type-level is the weaker, honest, cheap choice, and it is stated as such.

### 4.3 No fallback: unmapped types are IGNORED, never invented

`updated_rules.md` §5 requires that an unmapped operation **raise** rather than
route to a catch-all. For language, raising is wrong — a byte-level BPE vocabulary
legitimately contains fragments that have no POS (`zx`, a lone accent byte). The
letter of the rule is preserved by the alternative that carries no silent
information: such types get **`-1`**, the documented ignore label already used by
`FAMILY_TO_IDX["MIXED"]` and already honoured by `ignore_index=-1` and by the
`(family >= 0)` masks in `engine.py` and `model.py`.

The guarantee that replaces `raise` is a **budget**: the unmapped **token**
fraction is measured at build time and Gate L2 fails the build if it exceeds
**2%** of train tokens. An ignore label whose share is unmeasured is exactly the
E7 catch-all in another costume; an ignore label whose share is published and
bounded is a measurement.

### 4.4 STRICTER THAN ARITHMETIC: routing "accuracy" is demoted

`updated_rules.md` §8 makes `mean(predicted_expert == oracle_expert)` the
*authoritative* first-step routing accuracy. On arithmetic that was defensible:
`OP_TO_EXPERT` is a functional ground truth, so disagreeing with it is genuinely
worse routing.

On language it is not. POS is a linguistic prior. A router that discovered
"tokens that begin a rare multi-piece name" versus "tokens that continue one"
could be a *better* partition for next-token prediction while scoring near chance
against POS. Reporting raw accuracy as the headline specialization number would
therefore assert something the experiment cannot support.

So for `task = language`:

- `routing_accuracy` is still computed, still equal to the confusion diagonal
  within tolerance, and still reported — as **`routing_agreement_with_pos`**, a
  named agreement measure, never as "accuracy" or "correctness";
- the **primary specialization metrics are the permutation-invariant ones**:
  Hungarian-matched agreement, AMI, cluster purity, plus normalized load entropy
  as the balance diagnostic and mean/max pairwise expert cosine similarity;
- a **routing-vs-nothing control** is mandatory: the same metrics computed against
  a *random* type-level 6-way partition with the observed family token
  proportions, over the frozen seed set. AMI against POS is only evidence of
  linguistic specialization if it exceeds AMI against the shuffled control. This
  control did not exist in the arithmetic POC and is required here.

Everything `updated_rules.md` §8 forbids still holds: high entropy is not proof of
specialization; low cosine similarity is not proof of orthogonality.

---

## 5. Depth: no invented curriculum

### 5.1 Canonical language runs use PURE UNSUPERVISED ACT

`families.OP_TARGET_DEPTH` was defensible because a naive sequential
decomposition of `SORT` really does take more steps than `ADD`, and the table's
status as *a human design choice* was documented at the point of definition.

There is **no equivalent for English**. Any per-token depth target we wrote would
be a hypothesis about what the model *should* do, dressed as data. Writing one and
then reporting "depth allocation error" against it would be the single most
misleading thing this migration could produce — it would make a designed
correlation look like a discovery, which `updated_rules.md` §2.3 and
`updated_objective.md`'s prohibited-behaviour list both forbid.

Therefore, in canonical language configuration:

```text
model.halting_supervision      = false
loss_weights.halting_supervision = 0.0
depth/allocation_error_abs     = N/A
depth/allocation_error_rel     = N/A
```

`N/A` here is the honest report of "this configuration has no depth target",
exactly as `entropy = N/A` at `E = 1` is the honest report of "there is no
distribution over one expert". It is not a missing measurement.

### 5.2 What replaces it: correlation against MEASURED difficulty

The Second Objective — *does MoRE actually learn adaptive computation?* — is
tested against properties of the corpus that we did not choose:

| Metric | Definition | Why it is not circular |
|---|---|---|
| `depth/spearman_vs_logfreq` | Spearman ρ between a token type's mean exit depth and its log train-frequency rank | frequency is a property of WikiText, fixed before any model exists |
| `depth/spearman_vs_unigram_surprisal` | ρ between mean exit depth and `-log p_unigram(token)` under the train unigram model | same; the unigram model is frozen at build time |
| `depth/spearman_vs_model_loss` | ρ between per-token exit depth and that token's own cross-entropy, same batch | measures whether the model spends compute where *it* finds the task hard — a behavioural claim, not a designed one |
| `depth/mean_by_family` | mean exit depth per POS family | descriptive; no target |
| `depth/hist` | exit-depth histogram over `1..max_depth` | descriptive |

`depth/spearman_vs_model_loss` is the one to watch. A model that has learned
adaptive computation should show ρ > 0: hard positions get more steps. A
collapsed-depth model shows ρ ≈ 0 with near-zero variance in exit depth, which is
what the arithmetic POC found (93–97% of tokens at exactly 2 steps) and what the
constant-depth null model then beat.

**The constant-depth null model is mandatory here too.** `code/depth_null_model.py`
already implements the right test — the best single constant depth policy, scored
against the learned allocation. On language it is re-instantiated as: does the
learned depth distribution predict per-token loss better than the best constant
depth? If not, adaptive depth is not doing work, and that is Outcome C for the
MoR axis regardless of what the loss table says.

### 5.3 Trivial baselines (the `primary_metric_floor` analogue)

`updated_rules.md` §10 requires trivial predictors before a learned loss is
interpretable. For causal LM, computed on the val split at build time and frozen
into the manifest:

```text
uniform_ce      = ln(V)                       ~ 9.011 at V = 8192
unigram_ce      = cross-entropy of val under the add-k smoothed train unigram
bigram_ce       = cross-entropy of val under a Katz-backoff train bigram
```

A learned model that does not beat `bigram_ce` has not learned anything a lookup
table cannot do, and no comparison between architectures above that floor is
meaningful. `bigram_ce` becomes `canonical_spec_language.json:primary_metric_floor`.

### 5.4 The frequency-decile curriculum exists, as a LABELLED ABLATION

So the halting-supervision machinery is still exercised and the question "would a
depth target have helped?" is answerable, a curriculum **is** defined — and it is
never canonical:

```text
token_target_depth[V] : train-frequency decile -> depth in 1..max_depth
```

Enabled only by `--halting_supervision`, which `resolve_variant` already tags
`supervised_curriculum`. Its provenance says so, the exporter cannot mix it with
canonical runs, and the write-up must state that the target is a design choice
derived from corpus frequency, not a property the model discovered.

---

## 6. Architecture change: causal self-attention inside the recursive block

### 6.1 Why it is required

The current model has **no attention anywhere**. `MoREModel.forward` projects each
step row independently, adds an operation embedding, runs the per-token MoE-FFN
recursion, then masked-mean-pools to a single vector and regresses a scalar. No
token ever sees another token.

For arithmetic that was sufficient and was the right minimal design. For causal
LM it is not merely weak, it is **degenerate**: with no cross-position path, the
best achievable model is the unigram distribution, every architecture in the
matrix would converge to `unigram_ce`, and the MoE/MoR/MoRE comparison would
measure nothing. Attention is not an enhancement here; without it the experiment
has no content.

### 6.2 Where it goes, and how weight sharing is preserved

One `nn.MultiheadAttention` per `MoREWrapper`, constructed once, **applied at
every recursion depth with the same parameters** — the same weight-sharing
contract the MoE-FFN already honours (`updated_rules.md` §2.1: *"The same
recursive block parameters MUST be reused at every recursion depth"*). Depth
remains computation through time.

Per depth step, per block:

```text
state [B,S,D]
  -> causal self-attention over the FULL sequence          (shared params)
  -> residual + LayerNorm                                   -> h_attn
  -> gather ACTIVE token rows from h_attn
  -> MoE block: router -> top-1 -> selected expert only     (shared params)
  -> residual + LayerNorm
  -> write back to ACTIVE positions only; halted positions untouched
  -> per-expert halt head -> ACT update (unchanged)
```

This is the Universal Transformer's layout (attention sublayer, then transition
sublayer, both shared across steps) with the transition sublayer replaced by the
Top-1 MoE block. Nothing about routing, halting or the balance objective changes.

### 6.3 Halted tokens remain attendable — and stay frozen

Attention is computed over **all** `S` positions, using each position's *current*
state. A halted position's state is its frozen exit state, so:

- halted tokens **are** valid attention keys and values. They must be: position 4
  halting early cannot be allowed to blind position 40 to the word at position 4;
- halted tokens are **not** written to. `updated_rules.md` §2.2's "halted states
  are frozen and are not recursively recomputed" is enforced exactly as it is
  today, by writing updates only into `active_positions`;
- attention *outputs* at halted positions are computed and discarded. That wastes
  compute proportional to the halted fraction and is accepted for clarity: a
  variable-length gather of queries would complicate the mask and the exactness of
  the depth accounting for no scientific gain. The waste is **reported**, not
  hidden — `route_stats.attn_query_waste_fraction` is a measurement.

Gate L3 asserts the freeze bit-identically, reusing the Phase 3 gate's method.

### 6.4 Strict causality, and the leakage surface it creates

The mask is an explicit upper-triangular boolean, `True` = disallowed, so position
`i` attends only to `j <= i`. `is_causal=True` fast paths are not used: they are
version-sensitive and silently no-op in some configurations, and this is the one
place in the model where a silent no-op means **target leakage**.

`x[:, :-1]` predicts `x[:, 1:]`. Combined with strict causality, position `i`'s
logits depend only on tokens `0..i` and predict token `i+1`.

**Gate L1 tests this by perturbation, not by inspection:** for random `t` and
`k >= 1`, mutate `input_ids[b, t+k]`, re-run, and assert the logits at position `t`
are **bit-identical**. This is the language analogue of the arithmetic
`result_mutation_feature_changes = 0` and `expert_permutation_feature_changes = 0`
checks in `data/dataset_meta.json:gate1_audit`, and it is the single most important
new test in the migration. A model that fails it will show an implausibly good
perplexity and nothing else will look wrong.

**AMENDMENT (T-L4.4, measured 2026-09-03): bit-identity holds EXACTLY only at
`num_experts = 1, max_depth = 1`, and the residual above that is not a causality
defect.** Measured worst deviation at positions `<= t` over eight `(t, k)` pairs:

| configuration | masked | mask removed |
|---|---|---|
| E=1, depth 1 | **0.0 exactly** | 0.37 |
| E=1, depth 4/7 | 1.2e-07 (1 ULP) | 0.93–0.95 |
| E=6, depth 1/4/7 | 2.4e-07 (2 ULP) | 0.37–0.51 |

The cause is **grouped Top-1 dispatch**, not attention. When a perturbation flips
the perturbed token's expert, the per-expert row counts change (measured:
`[9,6,4,3,2,0] -> [8,7,5,2,2,0]`), so two `nn.Linear` GEMMs get different shapes and
tile differently, which moves the shared rows in their last one or two bits. The same
happens for `E = 1` at depth > 1, where a changed halt decision changes `N_active`.
It is isolated by construction: `E=6, depth 1` **with attention entirely absent**
still shifts the past by 2.4e-07, while `E=1, depth 1` is exactly 0.0.

So the gate is written in three parts, which together assert more than a bare
`torch.equal` could:

1. **exact** bit-identity where the confound is absent (`E=1`, depth 1);
2. a **bound of 8 ULP** (9.5e-07) everywhere else, with the mechanism *proved* by the
   E=6-no-attention comparison rather than asserted;
3. the mask-removed variant must exceed that bound by **>= 1e5** (measured smallest
   ratio 2.1e+06), so the tolerance cannot be concealing a real leak.

`torch.equal` alone would have been either unachievable or, if the tolerance had been
loosened without explanation, unfalsifiable.

### 6.5 Arithmetic must be bit-identical

`MoREWrapper(attention=False)` constructs **no** attention module. Not a module
bypassed by a flag at forward time — no module at all, so:

- the arithmetic `state_dict` is unchanged, and Gate 4's "no unexpected parameter
  in the canonical checkpoint" check still means what it meant;
- arithmetic parameter counts are unchanged, so `README.md`'s published counts and
  the 0.120% MoR/MoRE budget residual stay correct;
- Gate L0 (the 356-assertion suite) passes without any test being edited.

This mirrors the `router_noise` precedent in `model.py`: the canonical path has no
parameter for the thing it does not do, because absence is verifiable and a
zeroed-out parameter is not.

### 6.6 Attention hyperparameters are part of the frozen protocol

`n_heads` (4 at `d_model = 256`), attention dropout (= `model.dropout`), and
learned absolute positional embeddings of size `[seq_len, d_model]` are frozen in
`canonical_spec_language.json` and identical across MoE, MoR and MoRE. They are
**not** tuned per architecture — `updated_objective.md`'s canonical-hyperparameter
principle forbids it. Learned absolute positions are chosen over RoPE/ALiBi as the
simplest thing that cannot interact with recursion depth; a positional scheme that
varied with step count would confound the depth analysis.

**MoR keeps attention.** `architecture = mor` means one expert, not no context.
Removing attention from MoR would make it a different model class rather than the
"recursion without multiple experts" arm `updated_objective.md` §4 defines.

### 6.7 Parameter budget under attention

Attention adds `4 * d_model^2 + 4 * d_model` ≈ 263 K per block to **all three**
architectures equally, so it does not disturb the MoR/MoRE matching logic — which
operates on the FFN width via `CANONICAL_FFN_MULT` (`{moe: 4, mor: 24, more: 4}`).
`ffn_mult` for MoR must be **re-derived** for language rather than assumed: the
tied embedding is now a large shared term, so the arithmetic 0.120% residual will
not reproduce. T-L6 recomputes it and freezes the value that lands inside the 5%
bound, reporting both total and non-embedding counts.

---

## 7. Heads and loss assembly

### 7.1 Input adapter and output head

| | arithmetic | language |
|---|---|---|
| input | `step_proj: Linear(step_feat_dim, d_model)` + `op_embed` | `tok_embed: Embedding(V, d_model)` + `pos_embed: Embedding(seq_len, d_model)` |
| pooling | masked mean over real steps | **none** — per-position outputs are the prediction |
| task head | `regression_head -> 1`, MSE | `lm_head: Linear(d_model, V)`, **weight-tied** to `tok_embed` |
| whole-seq head | `cls_head -> num_families`, CE | **not constructed** |
| per-token head | `step_cls_head -> num_families`, in the objective | `family_probe -> num_families`, **detached** (§7.3) |

Weight tying is standard, halves the largest parameter block, and keeps the
embedding under gradient from both directions. It is frozen in the spec, not
optional, so it cannot vary across arms.

### 7.2 The task loss

```python
logits = lm_head(h)                              # [B, S, V]
task_loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, V),
    input_ids[:, 1:].reshape(-1),
)                                                # nats/token
perplexity = exp(task_loss)                      # reported, never optimized
```

Reported per `updated_rules.md` §8, each term separately: `task_loss`,
`routing_balance_loss` with `entropy_term` and `switch_aux_term` split out,
`ponder_cost`, `halting_supervision_loss` (ablation only), `probe/family_ce`
(outside the trunk), and `weighted_total_loss`. `val/task_loss` is primary.

### 7.3 STRICTER THAN ARITHMETIC: the family head becomes a detached probe

In the arithmetic POC, `loss_weights.family_cls = 0.5` put a 6-way family
cross-entropy into the objective as its **second-largest term** (T8.3 found it as
a bare `0.5` literal inside `engine.py`). That is legitimate supervised
multi-tasking, but it permanently qualifies the specialization claim: "MoRE's
representation separates operation families" is much weaker when a family
cross-entropy was being minimised on that representation the whole time. The
arithmetic write-up handles this with the labelled `no_family_supervision`
ablation, which is the correct remedy after the fact.

The language migration does not inherit the confound. `family_probe` reads
**`h.detach()`**, so:

- its gradient reaches only its own `Linear`, never the trunk;
- the representation is shaped by the LM objective and the balance/ponder terms
  alone;
- `probe/family_ce` and every routing/AMI number derived from the trunk measure
  emergent structure, with **no** family signal anywhere in the trunk's objective;
- the probe's loss weight is therefore scientifically inert, and is documented as
  inert rather than tuned.

Enforced, not merely intended: `apply_task` **raises** if
`task == "language"` and `loss_weights.family_cls != 0.0`. There is no whole-
sequence family label in a packed LM sequence for that term to even use, so a
non-zero value could only be a copied config.

### 7.4 Oracle routing supervision: unchanged, ablation only

`loss_weights.step_routing = 0.0` in canonical. Non-zero trains the router against
the POS oracle and is tagged `oracle_routing` by `resolve_variant` already. Given
§4.4 this ablation is *more* interesting on language than on arithmetic — it
measures how much of the achievable loss is lost by forcing a POS partition — and
it still may never share a table with canonical runs.

---

## 8. Metrics

### 8.1 New

```text
val/perplexity                          exp(val/task_loss)
train/perplexity                        exp(train/task_loss)
depth/spearman_vs_logfreq               §5.2
depth/spearman_vs_unigram_surprisal     §5.2
depth/spearman_vs_model_loss            §5.2
depth/mean_by_family                    six keys, L1_FUNCTION .. L6_NUM_SUBWORD
routing/agreement_with_pos              the renamed accuracy (§4.4)
routing/ami_vs_shuffled_control         the §4.4 control
params/total, params/non_embedding      §3.2, §6.7
route_stats/attn_query_waste_fraction   §6.3
```

### 8.2 Forced to `N/A` under `task = language`

```text
depth/allocation_error_abs, depth/allocation_error_rel     no target (§5.1)
train/halting_loss                                          supervision off
```

plus everything `engine.py` already forces: the `halt/*` and `val/*` depth keys at
`max_depth <= 1` (i.e. all of MoE's), and the `routing_*` / `entropy_term` /
`switch_aux_term` keys at `num_experts <= 1` (i.e. all of MoR's). The existing
exhaustive `N/A` block in the `metrics.json` dump is extended, not replaced —
`updated_rules.md` §11: no sentinel may be reported as a measurement.

### 8.3 Adapted

`metrics.make_op_depth_bar_figure` carries a hard-coded 16-entry `preferred_order`
list of arithmetic op names. Under `task = language` the bars are the six POS
families in manifest order. The function takes the ordering from the active task's
manifest rather than a literal, so it cannot silently draw arithmetic labels on
language data — the same defect class as `expert_labels()` drawing six tick labels
on a 1×1 matrix at `E = 1`.

`results.tsv` gains `val_perplexity` and `depth_rho_model_loss`; the header is
extended, never reordered, so existing readers keep working.

---

## 9. Baselines and the canonical matrix

Per `updated_rules.md` §6, with attention present in all three:

```text
MoE   num_experts 6   max_depth 1   adaptive off   top-1 routing   attention on
MoR   num_experts 1   max_depth C   adaptive on    routing N/A     attention on
MoRE  num_experts 6   max_depth C   adaptive on    top-1 routing   attention on
```

All three receive **bit-identical `input_ids` for the same sequence index** — the
language instantiation of `updated_rules.md` §4's input-parity invariant, and
trivially testable since the input is the token array itself (Gate L2).

Seeds `42, 43, 44, 45, 46`; mean ± std; the exact randomization test in
`code/seed_stats.py` with its 0.0040 resolution floor at 5v5; Bonferroni across
however many ablations are actually run. Checkpoint selection is **final epoch**,
per the T11.0b amendment, for the same reason as before: best-val is an order
statistic whose downward bias scales with per-arm validation noise, and that noise
is not matched across arms.

### 9.1 Required ablations

Carried over from `updated_objective.md`, minus the ones language makes
meaningless, plus two that language makes necessary:

```text
A  MoRE adaptive depth      vs  fixed depth at the same max_depth
B  MoRE learned routing     vs  oracle (POS-supervised) routing
C  1 recursive block        vs  2
D  parameter-matched MoR    (ffn_mult re-derived, T-L6)
E  dense routing            vs  canonical top-1 sparse
F  supervised frequency-decile curriculum   vs  pure ACT     [NEW, §5.4]
G  POS oracle partition     vs  shuffled-control partition    [NEW, §4.4]
```

Dropped: "router noise none vs annealed vs trainable" is not run unless a measured
collapse motivates it (`updated_rules.md` §7: do not add multiple routing
regularizers at once).

`G` is not a training ablation — it is an evaluation-time recomputation of the
routing metrics against a control partition, so it costs no runs.

---

## 10. Config and spec layout

### 10.1 `code/config_language.json`

A new defaults file; `code/config.json` is **not edited**, so no arithmetic run's
`config_hash` changes and every archived resolved config stays reproducible.

```text
task            "language"
model           d_model 256, n_heads 4, num_experts 6, max_depth C, num_blocks 1,
                dropout 0.1, ffn_mult 4, routing_mode top1_sparse,
                router_noise none, tie_lm_head true, attention true
data            corpus, seq_len, vocab_size, token dir, dataset_version,
                train_split_version, subset_fraction 1.0
training        lr, weight_decay, batch_size, epochs, grad_clip 1.0
loss_weights    task 1.0, family_cls 0.0 (REQUIRED), routing_balance 0.001,
                step_routing 0.0, halting 0.001, halting_supervision 0.0
logging         wandb_project "MoRE-language-poc", log_interval
```

`model.max_steps` and `model.step_feat_dim` are absent — they are arithmetic
concepts. `data.max_val` / `data.pad_value` likewise.

### 10.2 `code/canonical_spec_language.json`

A sibling spec, selected by `run_context.load_canonical_spec` on `cfg["task"]`.
`code/canonical_spec.json` is untouched, so Gate 0 and the three test suites that
read it are unaffected.

The numeric protocol fields — `epochs`, `batch_size`, `lr`, `weight_decay`,
`seq_len`, `max_depth`, `primary_metric_floor`, `ffn_mult.mor` — start as
**`null`**. While any enforced field is `null` the existing guard refuses every
`experiment_group = canonical_phase_b` claim. That is the mechanism working as
designed, not a blocker: it is what stops a calibration run from entering the
headline table. They are frozen exactly once, by Gate L5, from measurement.

Run names follow the existing template with the task prefixed, so a language run
can never be confused with an arithmetic one in a directory listing or a W&B
project: `langB_MoE_seed42`, `langB_MoR_seed42`, `langB_MoRE_seed42`.

---

## 11. Gates

Each gate STOPS the pipeline on failure, per `plan.md` §20: **STOP → report the
actual output → identify the cause → fix → re-run.** Never revert a correctness
fix because the old number looked better.

| Gate | Asserts | Blocks |
|---|---|---|
| **L0** | `run_correctness_suite.py` still prints `TOTAL 356 356 0 0`; arithmetic param counts and a 1-epoch arithmetic `metrics.json` unchanged | every commit |
| **L1** | **causality + leakage.** Perturbing `input_ids[b, t+k]`, `k>=1`, leaves position `t` logits bit-identical; strict `x[:-1]→x[1:]` shift verified; zero document overlap across splits; 32-gram shingle collisions reported; trivial baselines computed | any training |
| **L2** | **dataset integrity.** Per-split SHA-256 recorded; tokenizer fitted on train only (hash pinned); unmapped-family token share ≤ 2%; family counts at type and token level; `input_ids` bit-identical across MoE/MoR/MoRE for the same index; dropped-tail token counts recorded | any training |
| **L3** | **recursion invariants under attention.** Same block params at every depth (identity of `id()` and of grads); halted states bit-identical after their exit step; halted positions still attendable; forced exit at max depth; balance loss depth- and block-invariant after normalisation (re-run of the Phase 3/4 gates with `attention=True`) | any training |
| **L4** | **halting gradient.** Every halt-head parameter has a non-`None`, non-zero grad after one backward pass on the LM loss alone (`halting` weight 0) — the language re-run of the T3.2/T3.4 check that caught 24 parameters with `grad is None` | any training |
| **L5** | **calibration + protocol freeze.** Measured throughput and memory on the actual GPU; a stated wall-clock budget per run; `epochs`/`batch_size`/`seq_len`/`lr`/corpus size chosen so the canonical matrix fits it; all three arms still improving or flat (not diverging) at the chosen epoch count; `canonical_spec_language.json` nulls filled and frozen in one commit | the canonical matrix |
| **L6** | **parameter budget.** `abs(P_MoR - P_MoRE)/P_MoRE < 0.05` on both total and non-embedding counts, with the re-derived `ffn_mult.mor` | the MoR/MoRE claim |
| **L7** | **floor.** Every canonical arm beats `uniform_ce`, `unigram_ce` and `bigram_ce` on val | any comparative claim |
| **L8** | **provenance.** Every canonical run records the full `updated_rules.md` §9 list plus `task`, `corpus`, `vocab_size`, `seq_len`, `tokenizer_sha256`, `n_heads`, `tie_lm_head`; exporter refuses mixed dataset versions or config hashes and emits `N/A` rather than fabricating | the results table |

---

## 12. Phase order

Phases are sized so each ends at a gate, and so a resumed session can tell from
[TASKS_LANGUAGE.md](TASKS_LANGUAGE.md) exactly where it is.

```text
L-0  Environment          correct the documented interpreter and GPU; get a CUDA
                          torch working in a dedicated env; keep CPU as fallback
L-1  Task axis            `task` in config/cli/run_context; arithmetic bit-identical
                          -> GATE L0
L-2  Data pipeline        HF download, BPE on train only, pack, .npy, manifest,
                          unigram/bigram baselines, POS family lookup, frequency
                          curriculum table (unused in canonical)
                          -> GATE L2
L-3  Leakage audit        code/audit_lang_leakage.py; causality perturbation test
                          -> GATE L1
L-4  Model               attention sublayer (flag-gated), token/pos embeddings,
                          tied LM head, detached family probe
                          -> GATES L0, L3, L4
L-5  Engine              LM loss path, metric namespace, N/A discipline,
                          results.tsv extension
                          -> GATE L0 + a real 1-epoch language run on wikitext-2
L-6  Metrics             perplexity, difficulty correlations, family-ordered
                          figures, shuffled-control routing metrics
L-7  Calibration         throughput/memory measurement, protocol choice, spec
                          freeze, parameter-budget re-derivation
                          -> GATES L5, L6, L7
L-8  Canonical matrix    3 architectures x 5 seeds on the frozen protocol
L-9  Ablations           A-G from §9.1
L-10 Export + write-up   results.csv / results.json / results_language.md against
                          `updated_objective.md`'s 17-item deliverable list
                          -> GATE L8
```

L-0 through L-6 are CPU-feasible: every gate above is a correctness gate and none
of them needs a trained model. Only L-7 onward requires the GPU. This ordering is
deliberate — it means a CUDA install failing does not block any of the
correctness work.

---

## 13. Prohibited, language-specific

In addition to `updated_rules.md` §11 and `updated_objective.md`'s list:

- **Do not invent a per-token depth target and report allocation error against
  it** (§5.1). If a curriculum is used, it is the labelled ablation and its
  designed nature is stated in the same sentence as its number.
- **Do not call POS agreement "routing accuracy" or "correct routing"** (§4.4).
- **Do not report perplexity against published WikiText numbers.** The vocabulary
  and scale differ; only within-matrix comparisons are claimed (§3.2).
- **Do not tune `n_heads`, `seq_len`, `lr` or `epochs` per architecture** (§6.6).
- **Do not fit the tokenizer, the unigram/bigram baselines, the POS majority
  counts, or the frequency curriculum on anything but the train split** (§3.2).
- **Do not put the family probe's gradient into the trunk** (§7.3).
- **Do not use `is_causal=True` or any attention fast path** in place of the
  explicit mask (§6.4).
- **Do not delete or re-run any arithmetic artifact** to make the two tasks look
  consistent. They are two experiments (§0).

---

## 14. Deliverables

Per `updated_objective.md`'s FINAL DELIVERABLE, into `results/language/`:

```text
results.csv
results.json
results_language.md
```

`results_language.md` carries the same 17 required sections, with three
language-specific additions folded in:

- §3 (dataset statistics and leakage audit) includes the tokenizer provenance, the
  type/token family split, and the Gate L1 perturbation results;
- §7 (expert routing analysis) leads with the permutation-invariant metrics and the
  shuffled control, and labels POS agreement as agreement (§4.4);
- §8 (adaptive-depth analysis) reports the measured difficulty correlations and the
  constant-depth null model in place of allocation error, and states that no depth
  target exists in the canonical configuration.

The two "stricter than arithmetic" decisions (§4.4, §7.3) belong in §16
(explicitly unsupported claims) as well as where they are used, so a reader who
reads only the claims section still learns that POS is a prior and that no family
signal entered the trunk.

---

## 15. When a gate fails

`plan.md` §20, unchanged:

1. **STOP.**
2. **Report the failure** — plainly, with the actual output.
3. **Identify the cause.**
4. **Fix it.**
5. **Re-run the gate.**

Never revert a correctness fix because the old number looked better. Never proceed
past a failing gate. Archive evidence, never delete it.

