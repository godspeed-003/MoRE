"""shrink_acl.py -- one-shot (T-LA.5c): the last ACL page.

Measured state before this: counted content 9 pages against a limit of 8, read
from the .aux page record for the Limitations heading (page 10, so the body runs
through page 9). Three rounds of appendix moves got it from ~10 to 9 and then
stopped moving, for a reason worth recording: every moved block leaves a
self-contained summary in the body, because ARR reviewers are not obliged to read
appendices, and ten summaries at ~90 words are themselves most of a page. Moving
an eleventh block would have bought almost nothing.

So this trims the two things that are pure length rather than content:

  1. The appendix pointers. Each keeps its numbers and its claim and loses the
     restatement of context the surrounding prose already gives. No figure is
     dropped from any of them.
  2. The intro bullet list, 7 bullets -> 5, by merging the two pairs that were
     making one point each: the two Pareto bullets (efficiency and robustness)
     and the two mechanism bullets (competition and the excluded rivals). Every
     number survives the merge.

Nothing in the protected set is touched: the headline table, the depth-curve
figure, the depth-matched comparison, the dilution arithmetic, the ACT equations.
"""
import io
import re

# ---------------------------------------------------------------- pointers ----
# (file, old pointer prefix -> replacement). Matched on the marker line so the
# edit cannot touch body prose by accident.
POINTERS = {
    "engine": (
        r"% >>>ACLMOVE:engine|\subsection{One engine, three modes}\label{sec:engine} "
        r"MoE, MoR and MoRE are configurations of one implementation with an explicit "
        r"\texttt{architecture} mode, not three codebases, so a difference between arms "
        r"cannot be a difference in the harness. Equalising \emph{total} FFN hidden "
        r"width gives MoR's single FFN a multiplier of $24 = 4\times6$, so all three "
        r"arms hold one FFN parameter budget: \textbf{5{,}584{,}908 for MoE and MoRE, "
        r"identical, and 5{,}581{,}063 for MoR}---a $0.069\%$ deficit that is the router "
        r"MoR does not have, and which runs \emph{against} our headline since the "
        r"lighter arm wins in domain. Appendix~\ref{app:engine} derives it."
    ),
    "metrics": (
        r"% >>>ACLMOVE:metrics|\subsection{Metrics}\label{sec:metrics} The primary "
        r"metric is \textbf{validation task loss}, final-epoch per-token cross-entropy "
        r"in nats; \texttt{best} equals \texttt{last} on all 15 cells, so \emph{no model "
        r"selection occurred}. Routing is scored with permutation-invariant measures "
        r"(AMI, Hungarian, purity) against a per-run shuffled-label control and reported "
        r"as a z-score, since unsupervised expert indices carry no semantic identity. "
        r"Depth is reported with its exit rates alongside, because a depth--difficulty "
        r"correlation taken on an arm whose tokens almost all reach the ceiling is "
        r"computed against a near-constant and \textbf{says nothing about allocation}. "
        r"Appendix~\ref{app:metrics} defines each."
    ),
    "protocol": (
        r"% >>>ACLMOVE:protocol|\subsection{Training protocol}\label{sec:protocol} Five "
        r"seeds per arm, reported as mean$\,\pm\,$std, with one optimiser configuration "
        r"fixed before the comparison and \emph{not} swept per architecture. "
        r"\textbf{Three epochs is a compute budget, not convergence}: one epoch is 134.7M "
        r"tokens against $\approx$3.4M non-embedding parameters, so no arm is converged. "
        r"Every arm gets the identical budget, so the comparison holds; the consequences "
        r"are in the Limitations section. Appendix~\ref{app:protocol} gives the "
        r"hyperparameters."
    ),
    "blimp": (
        r"% >>>ACLMOVE:blimp|\textbf{Grammatical competence is real but modest, and "
        r"routing does not buy it.} On BLiMP \citep{warstadt2020blimp} both recursive "
        r"arms beat MoE (58.74\% and 58.36\% against 56.63\%) and MoR is not "
        r"distinguishable from MoRE ($p = 0.71$), so recursion buys measurable syntactic "
        r"sensitivity and routing adds none. On EWoK \citep{ivanova2024ewok} all three "
        r"arms sit at chance---a statement about scale, not architecture. "
        r"Appendix~\ref{app:blimp} reports both."
    ),
    "integrity": (
        r"% >>>ACLMOVE:integrity|\subsection{Integrity checks}\label{sec:integrity} "
        r"Three invariants hold across the matrix, each machine-checked: inputs are "
        r"bit-identical across arms per record; the target is absent from the input and "
        r"no oracle routing label is encoded in a feature slot; and the arms are "
        r"parameter-matched to 0.069\% with seed-blind configurations identical within "
        r"each arm. Three loss terms are structurally zero on language, one of which "
        r"\emph{strengthens} the specialisation claims. Appendix~\ref{app:integrity} "
        r"gives each in full."
    ),
}

FILES = ["paper/common/01_intro.tex", "paper/common/02_setup.tex",
         "paper/common/03_method.tex", "paper/common/04_results.tex",
         "paper/common/05_mechanism.tex", "paper/common/06_limitations.tex"]

n = 0
for path in FILES:
    s = io.open(path, encoding="utf-8").read()
    before = s
    for blk, new in POINTERS.items():
        s = re.sub(rf"^% >>>ACLMOVE:{blk}\|.*$", lambda _m, _n=new: _n, s,
                   count=1, flags=re.M)
    if s != before:
        io.open(path, "w", encoding="utf-8", newline="\n").write(s)
        n += 1
        print(f"  pointers shortened in {path}")

# ---------------------------------------------------------------- bullets -----
P = "paper/common/01_intro.tex"
s = io.open(P, encoding="utf-8").read()

MERGED_PARETO = r"""\item \textbf{The composition is the most compute-efficient arm, and the most
  robust out of domain.} MoRE improves on MoE by $7.9\%$ validation loss
  ($3.948 \rightarrow 3.637$ nats/token) at $2.19\times$ MoE's inference FLOPs,
  where MoR needs $9.07\times$: it captures $72\%$ of MoR's quality gain over MoE
  for $24\%$ of MoR's inference compute---$15\%$ of its \emph{extra} compute---which
  is roughly $3\times$ more quality per unit compute, and $6\times$ out of domain.
  On held-out text from a different distribution \citep{gao2020pile} it is the
  \emph{best} of the three (7.573 nats against MoR's 7.802 and MoE's 8.272),
  capturing $149\%$ of MoR's out-of-domain gain. The in-domain ordering inverts,
  and this is the one respect in which composing the mechanisms does something
  neither does alone."""

MERGED_MECH = r"""\item \textbf{The two mechanisms compete for one budget, and two rival
  explanations are excluded.} Depth and specialisation are not independent knobs
  in MoRE the way they are in MoR and MoE separately: a sweep that recovers
  adaptive depth by raising the ponder penalty destroys expert differentiation,
  while the arm that preserves differentiation saturates depth. Capacity dilution
  is ruled out by arithmetic---tokens per parameter are identical across arms by
  construction, and once recursion is counted MoRE receives slightly \emph{more}
  updates per parameter than MoR---and subspace trapping is falsified three ways:
  the recursive arms expand rather than collapse representational dimensionality,
  settle rather than oscillate, and update near-orthogonally rather than
  rescaling."""


def cut(start_marker, end_marker, replacement, text):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + replacement + "\n\n" + text[j:]


s = cut(r"\item \textbf{The composition is the most compute-efficient arm.}",
        r"\item \textbf{But recursion alone remains the lowest-loss arm in domain.}",
        MERGED_PARETO, s)
s = cut(r"\item \textbf{The two mechanisms compete for one budget.}",
        r"\end{itemize}", MERGED_MECH, s)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
bullets = s.count(r"\item \textbf{")
print(f"  intro bullets merged: now {bullets}")
