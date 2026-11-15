# VERIFY_BEFORE_SUBMIT.md — ACL Rolling Review (ARR)

**Read this before uploading. Four items are blockers, and the first is a
desk-rejection risk.** Everything here is something the LaTeX build will *not*
warn you about.

---

## 1. BLOCKER — the 8-page limit has never been measured, and overrunning is a desk rejection

ARR long papers are limited to **eight (8) pages of content**. The Limitations
section, the Ethical Considerations section, references and appendices do **not**
count.

**This body was written for TMLR, which has no page limit. It is very likely
over 8 pages here.** No LaTeX toolchain existed on the machine that assembled
this, so **the document has never been compiled** and the page count is unknown.

**Compile it before you edit anything else.** If it overruns, cut in this order —
the argument survives all four, and each is a block move to an appendix rather
than a rewrite. (The same list is in `main.tex`'s preamble.)

1. `sections/03_setup.tex` — the statistics and integrity-checks subsections →
   appendix (~0.75 page). The claims they support can be restated in one
   sentence each where used.
2. `sections/05_mechanism.tex` §"A plausible mechanism, refuted" → appendix
   (~0.5 page). Keep one sentence in the competition subsection saying trapping
   was tested and falsified, with a pointer.
3. `sections/04_results.tex` §"Stratified loss" and its figure → appendix
   (~0.5 page). Corroborating, not load-bearing.
4. `sections/01_intro.tex` Related Work → compress to one paragraph.

**Do not cut:** the headline table, the depth-curve figure, the depth-matched
comparison, or the dilution arithmetic. Those four are the paper.

## 2. BLOCKER — one bibliography entry is knowingly unverified

`refs.bib` contains `leong2023ewok` with the literal author field
`{UNVERIFIED, Author List}` and a placeholder arXiv ID. It will typeset and
BibTeX will not complain. ARR explicitly warns about "hallucitations".

**Fix:** verify it against arXiv or the ACL Anthology and replace the fields.

**Or delete it** — the EWoK result is "every arm at chance", reported as a scale
limitation, so nothing is lost. Then remove the `\citep{leong2023ewok}` in
`sections/04_results.tex` and `sections/01_intro.tex`.

## 3. BLOCKER — the Responsible NLP Checklist is mandatory and is not in this folder

It is submitted through the ARR form, not as part of this `.tex`. **Errors or
misleading entries can cause desk rejection**, so fill it from the generated
results rather than from memory. Facts you will need:

- 5 seeds per arm, 15 runs; one consumer laptop GPU; 3 epochs ≈ 404M tokens/run.
- Measured cost 5.7–7.9 h per recursive cell on an RTX 4060 (8 GB).
- Parameters: 5,584,908 (MoE, MoRE) and 5,581,063 (MoR).
- Data: WikiText-103, the corpus's own author-provided validation split.
- No human subjects, no annotation, no PII.

## 4. BLOCKER — two bibliography entries were never checked

`vinh2010ami` and `raposo2024mod` were not verified; present values match
expectation but were not confirmed against a publisher record. Several others
were not re-checked this round.

**Verified correct:** `jacobs1991moe`, `fedus2022switch`, `zoph2022stmoe`,
`shazeer2017moe`, `lepikhin2021gshard`, `warstadt2020blimp`.
**Verified and corrected:** `bae2025mor` (full author list; venue upgraded to
NeurIPS 2025) and `banino2021pondernet` (**8th ICML Workshop on AutoML**, not
ICML main track).

---

## 5. Anonymity — currently correct, do not "fix" it

`\usepackage[review]{acl}` is the anonymised review mode, and ARR uses two-way
anonymised review. It also enables line numbers and page numbers, which
reviewers expect.

- Do **not** switch to `[final]` or `[preprint]` for submission.
- Self-citations must be phrased in the third person — there are none in this
  draft; check again if you add any.
- **Links to trackable services (Dropbox, Google Drive) are banned.** If you
  link code or checkpoints, use an anonymised repository.
- Supplementary material must be anonymised too, and submitted as a single
  zip/tgz.

## 6. Structure requirements — currently satisfied, easy to break

- [ ] **A dedicated section titled "Limitations" is REQUIRED; its absence is a
      desk rejection.** It is `\section*{Limitations}` — starred and unnumbered —
      and sits **after** the conclusion and **before** the references. That is
      how `assemble_submissions.py` arranges it. Do not move it into the
      numbered body, and do not unstar it: starring is what keeps it out of the
      8-page count.
- [ ] Limitations "should not introduce new methods, analysis, or results" —
      padding it counts as abusing the page limit. The current section states
      scope, budget, resolution, what was declared-and-unrun, evaluation
      caveats, and disclosures. No new results.
- [ ] **Cross-references to the starred sections have been rewritten as prose.**
      A `\ref` to a `\section*` silently prints the previous numbered section's
      number — LaTeX gives no warning. If you add a reference to Limitations,
      write it out in words; do not use `\ref{sec:limitations}`.
- [ ] Appendices, if you add them, go **after** the references and keep the
      two-column format.

## 7. Content checks the build cannot make

- [ ] **Title is provisional.** Currently *"Routing and Recursion Compete: A
      Parameter-Matched Decomposition of Conditional Computation in Small
      Language Models."* Change it if you disagree.
- [ ] **No arithmetic numbers anywhere.** The arithmetic study is withdrawn and
      mixing its mean-squared-error table with this cross-entropy one is the
      worst available error here. Grep for `0.06`, `more6`, `3,201,555`, `MSE`,
      `R^2`.
- [ ] **Every number traces to a generated file** — headline from
      `results/language/results_tables.md`, battery from
      `results/language/eval_summary.md`, per-seed from
      `results/language/results.json` (`admitted_runs[]`). Regenerate, never
      hand-edit. `results.csv` is unusable for per-seed claims (its architecture
      and run-id columns are all `N/A`).
- [ ] **Exploratory vs confirmatory tiering is intact.** The headline table and
      pairwise tests are confirmatory, Holm-corrected over pre-registered
      families. The evaluation battery, forced-depth sweep, stratified loss, and
      subspace/dynamic probes are **exploratory and uncorrected**; no
      significance verdict from them may appear in the abstract or conclusion.
- [ ] **Figures regenerate:** `python code/make_paper_figures.py`, then
      `python paper/assemble_submissions.py`.

## 8. How to rebuild this folder

Do not edit `sections/*.tex` here — they are generated copies and will be
overwritten. Edit `paper/common/*.tex`, then:

```bash
python paper/assemble_submissions.py
```

That rewrites both venue trees and re-runs the self-containment check (every
`\input`, figure, citation key and `\ref` must resolve).

## 9. What is in this folder

| file | note |
|---|---|
| `main.tex` | the document; preamble carries the page-budget cut list |
| `sections/01_intro.tex` | introduction + related work |
| `sections/02_method.tex` | architecture, ACT equations, one-engine-three-modes |
| `sections/03_setup.tex` | data, protocol, metrics, statistics, integrity |
| `sections/04_results.tex` | headline table, depth curve, battery, strata |
| `sections/05_mechanism.tex` | dilution arithmetic, saturation, competition, refutation |
| `sections/06_limitations.tex` | **starred** Limitations + disclosures (page-exempt) |
| `sections/07_conclusion.tex` | conclusion |
| `refs.bib` | bibliography — see §2 and §4 above |
| `acl.sty`, `acl_natbib.bst` | official ACL style files, unmodified — do not edit |
| `fig/` | four figures; the three `fig_*.pdf` are generated |

An Ethical Considerations section is written inline in `main.tex` (optional for
ARR, page-exempt when placed at the end).
