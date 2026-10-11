"""add_acl_cuts.py -- one-shot (T-LA.5): the final ~2 pages for the ARR limit.

Context. The compile put the ACL counted body at ~10 pages against a limit of 8,
so three more blocks move to the page-exempt appendix. These are chosen to leave
the four protected items untouched -- the headline table, the depth-curve figure,
the depth-matched comparison, and the dilution arithmetic -- and to leave the ACT
equations in the body, since writing the halting path down was the paper's
longest-standing blocking item and a reviewer cannot check the mechanism without it.

Each replacement is self-contained: a reader who never opens the appendix still
gets the parameter counts, the primary metric, and the saturation conclusion,
because ARR reviewers are under no obligation to read appendices.
"""
import io

EDITS = [
    # --- method: the engine/budget subsection. The parameter COUNTS stay in the
    # body (the headline claim is "matched parameters"); the derivation goes.
    ("paper/common/03_method.tex",
     "\\subsection{One engine, three modes}",
     r"% >>>ACLMOVE:engine|\subsection{One engine, three modes}\label{sec:engine} MoE, MoR and MoRE are configurations of one implementation with an explicit \texttt{architecture} mode, not three codebases: one data pipeline, one optimiser setup, one evaluation path, so a difference between arms cannot be a difference in the harness. Only the fields their definitions require differ. Equalising \emph{total} FFN hidden width rather than the per-FFN multiplier gives MoR's single FFN a multiplier of $24 = 4\times6$, so all three arms hold the same FFN parameter budget: \textbf{5{,}584{,}908 parameters for MoE and MoRE, identical, and 5{,}581{,}063 for MoR}---a $0.069\%$ deficit that is the router MoR does not have, and which runs \emph{against} our headline since the lighter arm wins in domain. Appendix~\ref{app:engine} gives the derivation.",
     ),
    # --- setup: the metrics subsection. The primary metric and the
    # depth-correlation caveat are the parts the body cannot lose.
    ("paper/common/02_setup.tex",
     "\\subsection{Metrics}",
     r"% >>>ACLMOVE:metrics|\subsection{Metrics}\label{sec:metrics} The primary metric is \textbf{validation task loss}, final-epoch per-token cross-entropy in nats. We do not select on best-epoch loss---an order statistic whose downward bias scales with each arm's validation noise, which is not matched across architectures---and in any case \texttt{best} equals \texttt{last} on all 15 cells, so \emph{no model selection occurred}. Routing is scored with permutation-invariant measures (AMI, Hungarian-matched accuracy, purity) against a per-run shuffled-label control, reported as a z-score rather than a raw value, because unsupervised expert indices carry no semantic identity. For depth we report mean steps, exit rates, halt mass, and the depth--difficulty correlation---the last \textbf{only interpretable alongside the exit rates}, since on an arm whose tokens almost all reach the ceiling the correlation is taken against a near-constant and a value near zero says nothing about allocation. Appendix~\ref{app:metrics} gives each definition.",
     ),
    # --- mechanism: the subset-sweep table. The CONCLUSION (saturation is a
    # data-scale effect, not an architectural limit) is the load-bearing part.
    ("paper/common/05_mechanism.tex",
     "\\subsection{Depth saturation is a data-scale effect, not an architectural limit}",
     r"% >>>ACLMOVE:saturation|\subsection{Depth saturation is a data-scale effect}\label{sec:saturation} \textbf{[MEASURED, four points at fixed hyperparameters.]} MoRE's halting sits at the ceiling on the full corpus (97.3\% forced-exit, mean depth 6.955 of 7), which invites the reading that the architecture cannot learn adaptive depth. A subset sweep falsifies that: at 2.5\%, 10\% and 40\% of the corpus the \emph{same} model at the \emph{same} ponder weight allocates depth adaptively---forced-exit rates near zero, mean depth between 1.9 and 2.4 inside a budget of 7---and the collapse appears only at full scale. \textbf{So the failure is a calibration finding, not an architectural one}: this ponder penalty cannot hold depth down at this corpus size. The subset axis is not cleanly monotone, so the honest phrasing is that saturation emerges somewhere between 40\% and 100\% of the data. The mechanism is consistent: the ponder term is linear in expected depth while the task gradient grows as the model fits, and once the halting sigmoid saturates its gradient vanishes \citep{graves2016act}. \citet{banino2021pondernet} is the principled fix; we did not run it. Appendix~\ref{app:saturation} gives the sweep.",
     ),
]

for path, anchor, marker in EDITS:
    s = io.open(path, encoding="utf-8").read()
    assert anchor in s, f"anchor not found in {path}: {anchor}"
    assert marker.split("|")[0] not in s, f"already marked: {path}"
    s = s.replace(anchor, marker + "\n" + anchor, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"  opened block in {path}")

# close each block at the next \subsection or EOF
CLOSERS = [
    ("paper/common/03_method.tex", "engine", None),            # runs to EOF
    ("paper/common/02_setup.tex", "metrics", "\\subsection{Statistics}"),
    ("paper/common/05_mechanism.tex", "saturation",
     "\\subsection{The two mechanisms compete for one budget}"),
]
for path, blk, before in CLOSERS:
    s = io.open(path, encoding="utf-8").read()
    close = f"% <<<ACLMOVE:{blk}\n"
    if before is None:
        s = s.rstrip("\n") + "\n" + close
    else:
        assert before in s, f"closer anchor missing in {path}"
        s = s.replace(before, close + before, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"  closed {blk} in {path}")
