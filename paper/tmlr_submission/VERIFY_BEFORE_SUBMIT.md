# VERIFY_BEFORE_SUBMIT.md — TMLR

**Status: no blockers remain.** The paper compiles clean and every item that was
a blocker in earlier revisions has been closed. What is left below is judgement
calls and things the build cannot check for you.

Measured on the last clean build (`python code/compile_papers.py`):

| | value |
|---|---|
| PDF total | **19 pages** |
| body (intro → conclusion) | **13 pages** |
| LaTeX errors | **0** |
| overfull boxes (content outside the text block) | **0** |
| undefined references / citations | **0** |
| bibliography entries rendered | **18 of 18** |

---

## 1. Closed since the last revision — do not re-raise these

- **The bibliography is fully verified.** `refs.bib` carries **zero** unverified
  entries. The EWoK entry was wrong in *every* guessed field and is now
  `ivanova2024ewok`, verified against the arXiv API; `vinh2010ami` was checked
  against JMLR's own BibTeX export; `raposo2024mod`, `bae2025mor` and
  `banino2021pondernet` were corrected or confirmed. Per-entry verification notes
  are comments in the file.
- **The page count is measured, not estimated.** An earlier revision estimated
  the body from source volume and was wrong by several pages; the figure above
  comes from `Output written on main.pdf` plus a page anchor at the Limitations
  heading read out of `main.aux`.
- **Nothing runs outside the text block.** Three display equations used to
  overflow the *ACL* column by up to 48%; TMLR's wider measure never showed it,
  and both are clean now. Verify with `python code/check_overflow.py`.

## 2. Judgement calls — yours, not blockers

- [ ] **Title.** Currently *"Adaptive Compute at Matched Parameters: Routing and
      Recursion Trade Off in Small Language Models."* The earlier candidate
      "Parameter- and FLOP-Matched" was rejected as inaccurate: the arms are
      parameter-matched but deliberately **not** FLOP-matched, and
      1.00×/8.87×/2.19× is the central measurement.
- [ ] **The Pareto figure costs one page.** `fig_pareto` took the body from 12 to
      13 pages. TMLR sets no limit and asks that length be justified by content;
      this figure is what revealed that MoR is Pareto-*dominated* out of domain,
      a result the table had understated. Keep it unless you disagree — but the
      ~12-page norm is a real convention.
- [ ] **`math_commands.tex` is shipped and never `\input`.** It comes with the
      official TMLR template and is kept so the folder matches it. Delete it if
      you prefer a minimal upload; nothing references it.

## 3. Anonymity — currently correct, do not "fix" it

`\usepackage{tmlr}` with **no option** is the anonymous submission mode, and
TMLR review is double-blind. The `\author{}` block is ignored and replaced by
"Anonymous authors / Paper under double-blind review". Leave it.

- Do **not** add `[accepted]` or `[preprint]` until acceptance.
- Checked and clean: no author names, no institutional strings, no repository
  links, no acknowledgements anywhere in the source.
- Supplementary material must be anonymised too. A preprint may be posted at any
  time but must not link back to a non-anonymous version.

## 4. Content checks the build cannot make

- [ ] **No arithmetic numbers anywhere.** The arithmetic study is withdrawn, and
      mixing its mean-squared-error table with this cross-entropy one is the
      worst error available in this repository. Grep for `0.06`, `more6`,
      `3,201,555`, `MSE`, `R^2`.
- [ ] **Every number traces to a generated file.** Headline from
      `results/language/results_tables.md`, battery from
      `results/language/eval_summary.md`, per-seed from
      `results/language/results.json` (`admitted_runs[]`). Nothing was
      hand-copied. **`results.csv` is unusable for per-seed claims** — its
      architecture and run-id columns are all `N/A`; use `results.json`.
- [ ] **Exploratory vs confirmatory tiering is intact.** The headline table and
      the pairwise tests are confirmatory, Holm-corrected over two
      pre-registered families. The evaluation battery, the forced-depth sweep,
      the stratified loss, the subspace and dynamic probes, and the seed-variance
      comparison are **exploratory and uncorrected** — no significance verdict
      from any of them may appear in the abstract or conclusions. Check the
      abstract still obeys this.
- [ ] **The two axes stay separate.** At matched parameters MoR is lowest-loss by
      0.1216 nats; per unit of inference compute MoRE is ahead. Neither statement
      licenses the other, and "comparable to MoR" appears only ever as a
      *per-FLOP* claim. A reviewer should not be able to say we switched axes to
      tell a better story.

## 5. What the paper discloses, and should keep disclosing

These are in the Limitations section and are why the rest is credible. Do not
trim them to save space:

- A declared Family-A control (the depth-matched fixed-depth arm) was **never
  run** and still consumes its Holm slot.
- Fixing the architecture to route once made the headline **worse** by 0.134
  nats; the spec-violating per-step arm is kept as a labelled ablation.
- The pre-registration is **partial** and the registration file says so.
- The matrix spans **two GPUs** and four dirty code revisions; ten of fifteen
  cells carry null GPU fields that were **not** backfilled.
- Three epochs is a **budget, not convergence**.
- A balance-loss diagnosis was published and then **withdrawn** when a sweep
  refuted its sign.

## 6. How to rebuild

Do not edit `sections/*.tex` or `fig/fig_architecture.tex` here — they are
generated copies and will be overwritten. Edit `paper/common/*.tex` (or
`paper/fig/`), then:

```bash
python paper/assemble_submissions.py
```

Then, in order:

```bash
python code/check_latex_lint.py
```

```bash
python code/compile_papers.py
```

```bash
python code/check_overflow.py
```

```bash
python code/make_overleaf_zips.py
```

The lint catches source defects that compile silently — most importantly
shell-expansion damage, which once turned `MoRE$-$MoE` into `MoREhBc` and set a
whole sentence in math italic with zero errors and zero warnings. **Never edit a
`.tex` file through a shell string**; write a `.py` file and run it.

## 7. What is in this folder

| file | note |
|---|---|
| `main.tex` | the document, including the Broader Impact and Reproducibility statements |
| `sections/01_intro.tex` | introduction + related work |
| `sections/02_method.tex` | architecture, ACT equations, one-engine-three-modes |
| `sections/03_setup.tex` | data, protocol, metrics, statistics, integrity |
| `sections/04_results.tex` | headline table, Pareto figure, depth curve, battery, strata |
| `sections/05_mechanism.tex` | dilution arithmetic, saturation, competition, refutation |
| `sections/06_limitations.tex` | limitations + disclosures |
| `sections/07_conclusion.tex` | conclusion |
| `refs.bib` | bibliography, 18 entries, all verified |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` | official TMLR style files, unmodified |
| `math_commands.tex` | template file, **not used** — see §2 |
| `fig/fig_architecture.tex` | the block diagram, pure TikZ (needs `\usepackage{tikz}`, already in the preamble) |
| `fig/fig_*.pdf` | four generated figures, from `code/make_paper_figures.py` |

`tmlr.sty` does **not** set the bibliography style, so `main.tex` must declare
`\bibliographystyle{tmlr}` — unlike the ACL version, where `acl.sty` sets it and
declaring it again silently broke bibtex.
