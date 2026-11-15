# VERIFY_BEFORE_SUBMIT.md — TMLR

**Read this before uploading. Three items are blockers.** Everything here is
something the LaTeX build will *not* warn you about.

---

## 1. BLOCKER — one bibliography entry is knowingly unverified

`refs.bib` contains `leong2023ewok` with the literal author field
`{UNVERIFIED, Author List}` and a placeholder arXiv ID. It will typeset and
BibTeX will not complain.

**Fix:** open the arXiv abstract page or ACL Anthology entry for the EWoK
benchmark, and replace the author list, venue and ID outright.

**Or delete it.** The EWoK result is "every arm is at chance", reported as a
scale limitation rather than a finding, so the paper loses nothing. If you
delete the entry, also remove the `\citep{leong2023ewok}` in
`sections/04_results.tex` (§ "Where the composition does pay") and in
`sections/01_intro.tex` (Related Work, evaluation-at-small-scale paragraph).

## 2. BLOCKER — the page count has never been measured

No LaTeX toolchain existed on the machine that assembled this, so **this
document has never been compiled**. TMLR has no fixed page limit, but the
practical norm is ~12 pages of body and "unusually long papers are likely to
result in reviewing delays."

**Compile it first** and check: page count, float placement (three figures and
three tables), and that no table overruns the single-column width.

## 3. BLOCKER — two remaining bibliography entries were never checked

`vinh2010ami` and `raposo2024mod` were not verified. The values present match
expectation (`JMLR 11:2837–2854`; `arXiv:2404.02258`) but neither was confirmed
against a publisher record. Six further entries (`graves2016act`,
`dehghani2019ut`, `elbayad2020depthadaptive`, `schuster2022calm`,
`lan2020albert`, and `merity2017wikitext`/`gao2020pile` beyond their flagged
fields) were likewise not re-checked this round.

**Verified and correct:** `jacobs1991moe`, `fedus2022switch`, `zoph2022stmoe`,
`shazeer2017moe`, `lepikhin2021gshard`, `warstadt2020blimp`.

**Verified and corrected in place:** `bae2025mor` (author list was `and others`;
now the full eleven authors, venue upgraded from arXiv to NeurIPS 2025) and
`banino2021pondernet` (was flagged as possibly ICML main track; it is the
**8th ICML Workshop on AutoML**).

---

## 4. Anonymity — currently correct, do not "fix" it

`\usepackage{tmlr}` with **no option** is the anonymous submission mode, and
TMLR review is double-blind. The `\author{}` block in `main.tex` is ignored and
replaced by "Anonymous authors / Paper under double-blind review". Leave it.

- Do **not** add `[accepted]` or `[preprint]` until acceptance.
- Supplementary material must be anonymised too.
- A preprint may be posted at any time, but must not link back to a
  non-anonymous version.

## 5. Content checks the build cannot make

- [ ] **Title is provisional.** Currently *"Routing and Recursion Compete: A
      Parameter-Matched Decomposition of Conditional Computation in Small
      Language Models."* Chosen to lead with the finding and to pre-empt the
      scale objection in the title itself. Change it if you disagree.
- [ ] **No arithmetic numbers anywhere.** The arithmetic study is withdrawn, and
      mixing its mean-squared-error table with this cross-entropy one is the
      single most damaging error available in this repository. Grep the draft
      for `0.06`, `more6`, `3,201,555`, `MSE`, `R^2`.
- [ ] **Every number traces to a generated file.** Nothing was hand-copied: the
      headline comes from `results/language/results_tables.md`, the battery from
      `results/language/eval_summary.md`, the per-seed values from
      `results/language/results.json` (`admitted_runs[]`). If a number needs
      changing, regenerate rather than edit.
- [ ] **`results.csv` is unusable for per-seed claims** — its architecture and
      run-id columns are all `N/A`. Use `results.json`.
- [ ] **Exploratory vs confirmatory tiering is intact.** The headline table and
      pairwise tests are confirmatory with Holm correction over pre-registered
      families. Everything in the evaluation battery, the forced-depth sweep,
      the stratified loss, the subspace and dynamic probes is **exploratory and
      uncorrected**, and no significance verdict from them may appear in the
      abstract or conclusion. Check the abstract still obeys this.
- [ ] **Figures regenerate.** `python code/make_paper_figures.py`, then re-run
      `python paper/assemble_submissions.py`.

## 6. How to rebuild this folder

Do not edit `sections/*.tex` here — they are generated copies and will be
overwritten. Edit `paper/common/*.tex`, then:

```bash
python paper/assemble_submissions.py
```

That rewrites both venue trees and re-runs the self-containment check (every
`\input`, figure, citation key and `\ref` must resolve).

## 7. What is in this folder

| file | note |
|---|---|
| `main.tex` | the document; preamble comments explain each option |
| `sections/01_intro.tex` | introduction + related work |
| `sections/02_method.tex` | architecture, ACT equations, one-engine-three-modes |
| `sections/03_setup.tex` | data, protocol, metrics, statistics, integrity |
| `sections/04_results.tex` | headline table, depth curve, battery, strata |
| `sections/05_mechanism.tex` | dilution arithmetic, saturation, competition, refutation |
| `sections/06_limitations.tex` | limitations + disclosures |
| `sections/07_conclusion.tex` | conclusion |
| `refs.bib` | bibliography — see §1 and §3 above |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` | official TMLR style files, unmodified |
| `fig/` | four figures; the three `fig_*.pdf` are generated |

`math_commands.tex` ships with the TMLR template and is **not** `\input` by this
document. It is kept so the folder matches the official template; delete it if
you prefer a minimal upload.
