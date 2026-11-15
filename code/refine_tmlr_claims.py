"""refine_tmlr_claims.py -- T-LA.6: make the Pareto claim precise, and measure the variance.

TWO CORRECTIONS TO CLAIMS THE REFRAME INTRODUCED.

1. "MoRE is the compute-efficient point of the three" is imprecise and a
   reviewer checking it would be right to object. Computed on (FLOPs, loss),
   IN DOMAIN all three arms are Pareto NON-DOMINATED -- MoE is cheapest, MoR is
   best, MoRE sits between, and none dominates another. So there is no single
   "efficient point"; the three trace a frontier. MoRE's real in-domain claim is
   the best quality-per-unit-compute RATIO (0.142 vs MoR's 0.048 nats per
   FLOPs-multiplier), which is a statement about slope, not position.

2. OUT OF DOMAIN the claim is strictly STRONGER than the reframe asserted, and
   was being undersold: MoRE at (2.19x, 7.573) dominates MoR at (9.07x, 7.802)
   outright -- better loss at a quarter of the compute. MoR is Pareto-dominated
   there. "Best arm out of domain" understates it; dominance is the word.

AND ONE HAND-WAVE CONVERTED INTO A MEASUREMENT. The results section flagged
MoRE's tight seed spread and said we did not know what to make of it. The spread
difference is testable and significant: var(MoR)/var(MoRE) = 6.44, exact
one-sided permutation p = 1/252 = 0.0040, which is that test's floor. The
*interpretation* stays open -- a model whose halting has collapsed to a constant
has less to vary across seeds, and we cannot separate that from genuine
stabilisation -- but "there is a real difference in spread" is now measured
rather than asserted. EXPLORATORY and uncorrected: it is outside both declared
families.
"""
import io
import re

# ------------------------------------------------------------- abstract -------
OLD_ABS = (r"The composition (MoRE) is the compute-efficient point of the three. It improves on"
           "\nMoE by $7.9\\%$ validation loss at $2.19\\times$ MoE's inference FLOPs, where"
           "\nrecursion alone needs $9.07\\times$: it captures $72\\%$ of recursion's quality gain"
           "\nover MoE for $24\\%$ of recursion's compute---roughly $3\\times$ more quality per"
           "\nunit of compute, and $6\\times$ out of domain. On held-out text from a different"
           "\ndistribution it is the best of the three outright.")
NEW_ABS = (r"All three arms lie on the quality/compute frontier, and the composition (MoRE)"
           "\nbuys the most quality per unit of compute. It improves on MoE by $7.9\\%$"
           "\nvalidation loss at $2.19\\times$ MoE's inference FLOPs, where recursion alone needs"
           "\n$9.07\\times$: $72\\%$ of recursion's quality gain for $24\\%$ of its compute, or"
           "\n$3\\times$ the quality per FLOPs-multiplier. Out of domain it is stronger still---it"
           "\n\\emph{dominates} recursion alone, reaching lower loss at a quarter of the compute.")

for p in ("paper/tmlr_submission/main.tex", "paper/acl_submission/main.tex"):
    s = io.open(p, encoding="utf-8").read()
    assert OLD_ABS in s, f"abstract para not found in {p}"
    io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(OLD_ABS, NEW_ABS))
    print(f"  abstract refined: {p}")

# ------------------------------------------------------------- intro ----------
P = "paper/common/01_intro.tex"
s = io.open(P, encoding="utf-8").read()
OLD_B = r"""\item \textbf{The composition is the most compute-efficient arm, and the most
  robust out of domain.}"""
NEW_B = r"""\item \textbf{All three arms sit on the quality/compute frontier, and the
  composition buys the most quality per unit of compute.}"""
assert OLD_B in s, "pareto bullet head not found"
s = s.replace(OLD_B, NEW_B, 1)

OLD_T = (r"  On held-out text from a different distribution \citep{gao2020pile} it is the"
         "\n  \\emph{best} of the three (7.573 nats against MoR's 7.802 and MoE's 8.272),"
         "\n  capturing $149\\%$ of MoR's out-of-domain gain. The in-domain ordering inverts,"
         "\n  and this is the one respect in which composing the mechanisms does something"
         "\n  neither does alone.")
NEW_T = (r"  Out of domain \citep{gao2020pile} the claim strengthens from a ratio to outright"
         "\n  \\emph{dominance}: MoRE reaches 7.573 nats against MoR's 7.802 at a quarter of"
         "\n  the compute, so MoR is Pareto-dominated there, and MoRE captures $149\\%$ of"
         "\n  MoR's out-of-domain gain over MoE. The in-domain ordering inverts, and this is"
         "\n  the one respect in which composing the mechanisms does something neither does"
         "\n  alone.")
assert OLD_T in s, "pareto bullet tail not found"
s = s.replace(OLD_T, NEW_T, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  intro bullet refined")

# ------------------------------------------------------------- results --------
P = "paper/common/04_results.tex"
s = io.open(P, encoding="utf-8").read()

OLD_R = (r"\textbf{MoRE is the most robust arm out of domain.} On held-out Pile blocks"
         "\n\\citep{gao2020pile} it reaches 7.573 nats against MoR's 7.802 and MoE's 8.272---a"
         "\n$0.699$-nat advantage over MoE against MoR's $0.470$. The in-domain ordering"
         "\ninverts: the composition is the \\emph{worst} recursive arm in domain and the"
         "\n\\emph{best} out of it. Put in the same terms as the in-domain comparison, MoRE"
         "\ncaptures $149\\%$ of MoR's out-of-domain gain over MoE, against $72\\%$ in domain."
         "\nThat is a real and reportable property, and it is the one respect in which"
         "\ncomposing the two mechanisms does something neither does alone.")
NEW_R = (r"\textbf{In domain, all three arms are on the frontier; out of domain, MoRE"
         "\ndominates.} Fig.~\\ref{fig:pareto} plots loss against inference compute. In domain"
         "\nno arm Pareto-dominates another: MoE is cheapest, MoR is best, MoRE sits between"
         "\nthem, so MoRE's in-domain claim is a \\emph{ratio}---the most quality per unit of"
         "\ncompute---and not a position the others fail to occupy. We are explicit about this"
         "\nbecause the weaker reading is the one a reviewer should be able to check and find"
         "\nalready stated."
         "\n\nOut of domain the relationship changes in kind rather than degree. On held-out"
         "\nPile blocks \\citep{gao2020pile} MoRE reaches 7.573 nats against MoR's 7.802 and"
         "\nMoE's 8.272, at $2.19\\times$ MoE's FLOPs against MoR's $9.07\\times$---so MoRE is"
         "\nboth cheaper and better than MoR, and \\textbf{MoR is Pareto-dominated}. Its"
         "\nadvantage over MoE is $0.699$ nats against MoR's $0.470$, i.e. MoRE captures"
         "\n$149\\%$ of MoR's out-of-domain gain against $72\\%$ in domain. The in-domain"
         "\nordering inverts, and this is the one respect in which composing the two"
         "\nmechanisms does something neither does alone.")
assert OLD_R in s, "results OOD para not found"
s = s.replace(OLD_R, NEW_R, 1)

# the Pareto figure, placed with the efficiency discussion
FIGBLOCK = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\linewidth]{fig_pareto.pdf}
\caption{Loss against inference FLOPs per token, five seeds, error bars one
standard deviation. (a) In domain no arm dominates another---the three trace a
frontier, and MoRE's claim is the slope from MoE, not its position. (b) Out of
domain MoR is \emph{dominated}: MoRE reaches lower loss at a quarter of the
inference compute. Exploratory, uncorrected.}
\label{fig:pareto}
\end{figure*}
"""
anchor = r"\textbf{It gets there far more cheaply, on a different axis.}"
assert anchor in s, "efficiency para not found"
s = s.replace(anchor, FIGBLOCK.strip() + "\n\n" + anchor, 1)

# the variance observation, now measured
OLD_V = (r"Second, MoRE has by far the tightest seed spread (std 0.0134 against"
         "\nMoR's 0.0340), and we flag that we do not know what to make of it: it is the one"
         "\naxis on which the composition is unambiguous, and the runs that exist cannot"
         "\nseparate a genuinely stabilising effect from the simpler account that a model whose"
         "\nhalting has collapsed to a constant has less to vary across seeds.")
NEW_V = (r"Second, MoRE has by far the tightest seed spread, and the difference is real"
         "\nrather than impressionistic: its variance is $6.4\\times$ smaller than MoR's"
         "\n($0.000179$ against $0.001155$; std $0.0134$ against $0.0340$), and an exact"
         "\none-sided permutation test on the variance ratio returns $p = 1/252 = 0.0040$,"
         "\nwhich is that test's floor. \\textbf{What the number means is still open.} The runs"
         "\nthat exist cannot separate a genuinely stabilising effect from the simpler account"
         "\nthat a model whose halting has collapsed to a near-constant has less left to vary"
         "\nacross seeds---and the second reading is the more parsimonious one. We report the"
         "\nmeasurement and decline the interpretation. (Exploratory and uncorrected: the"
         "\nvariance comparison is outside both declared families.)")
assert OLD_V in s, "variance para not found"
s = s.replace(OLD_V, NEW_V, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  results: frontier/dominance split, Pareto figure, variance measured")

# ------------------------------------------------------------- conclusion -----
P = "paper/common/06_limitations.tex"
s = io.open(P, encoding="utf-8").read()
OLD_C = (r"5.6M parameters on WikiText-103 it buys efficiency and robustness rather than"
         "\npeak quality. MoRE is the compute-efficient point of the three: $7.9\\%$ better"
         "\nthan MoE on validation loss at $2.19\\times$ its inference FLOPs, capturing $72\\%$"
         "\nof recursion's quality gain for $24\\%$ of recursion's compute, which is roughly"
         "\nthree times more quality per unit of compute. Out of domain it is the best arm"
         "\noutright, ahead of both components on held-out text from a different"
         "\ndistribution---the one respect in which composing the mechanisms does something"
         "\nneither does alone.")
NEW_C = (r"5.6M parameters on WikiText-103 it buys efficiency and robustness rather than"
         "\npeak quality. In domain all three arms lie on the quality/compute frontier and"
         "\nMoRE buys the most quality per unit of compute: $7.9\\%$ better than MoE at"
         "\n$2.19\\times$ its inference FLOPs, $72\\%$ of recursion's gain for $24\\%$ of"
         "\nrecursion's compute, about three times the quality per FLOPs-multiplier. Out of"
         "\ndomain it is stronger than a ratio---MoRE \\emph{dominates} recursion alone,"
         "\nreaching lower loss at a quarter of the compute, which is the one respect in which"
         "\ncomposing the mechanisms does something neither does alone.")
assert OLD_C in s, "conclusion para not found"
io.open(P, "w", encoding="utf-8", newline="\n").write(s.replace(OLD_C, NEW_C, 1))
print("  conclusion refined")


# --------------------------------------------------------- abstract length ----
def wordcount(tex):
    t = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", tex)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$[^$]*\$", "X", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("---", " ").replace("{", "").replace("}", "")
    return len([w for w in t.split() if any(c.isalnum() for c in w)])


s = io.open("paper/tmlr_submission/main.tex", encoding="utf-8").read()
a = s.index("\\begin{abstract}")
b = s.index("\\end{abstract}") + len("\\end{abstract}")
print(f"\n  abstract now {wordcount(s[a:b])} words (target 180-220)")
