"""add_acl_cuts2.py -- one-shot (T-LA.5b): the last page for the ARR limit.

After the first three moves the compiled ACL counted body was 9 pages against a
limit of 8. Two more blocks go, both corroborating rather than load-bearing:

  blimp     -- the BLiMP/EWoK paragraph. Keeps the result in one sentence; the
               per-phenomenon breakdown was never doing argumentative work.
  protocol  -- the training-protocol detail. The sentence that MUST stay in the
               body is "three epochs is a budget, not convergence", because the
               paper's scope depends on it; the hyperparameter list does not.

Still untouched, as required: the headline table, the depth-curve figure, the
depth-matched comparison, the dilution arithmetic, and the ACT equations.
"""
import io

BLIMP = (
    r"% >>>ACLMOVE:blimp|\textbf{Grammatical competence is real but modest, and "
    r"routing does not buy it.} On BLiMP \citep{warstadt2020blimp}, 67{,}000 minimal "
    r"pairs, both recursive arms beat MoE (58.74\% and 58.36\% against 56.63\%) and "
    r"MoR is not distinguishable from MoRE ($p = 0.71$); every arm clears the 50\% "
    r"floor, so recursion buys measurable syntactic sensitivity at this scale and "
    r"adding routing does not add more of it. On EWoK \citep{ivanova2024ewok} all "
    r"three arms sit within noise of chance, which is a statement about scale rather "
    r"than architecture. Appendix~\ref{app:blimp} reports both, including which "
    r"phenomena the gains concentrate on."
)

PROTOCOL = (
    r"% >>>ACLMOVE:protocol|\subsection{Training protocol}\label{sec:protocol} Every "
    r"arm runs at seeds $\{42,43,44,45,46\}$ and is reported as mean$\,\pm\,$std over "
    r"five runs, with one fixed optimiser configuration chosen before the comparison "
    r"and \emph{not} swept per architecture. \textbf{Three epochs is a fixed compute "
    r"budget, not convergence, and we state it as such}: one epoch is 134.7M tokens "
    r"against $\approx$3.4M non-embedding parameters, so the runs sit in an "
    r"underfitting regime. Every arm receives the identical budget, so the comparison "
    r"is valid even though no arm is converged; the consequences are in the "
    r"Limitations section. Appendix~\ref{app:protocol} gives the hyperparameters and "
    r"the reason for three epochs rather than two or eight."
)

# (path, anchor to insert the open marker before, closer anchor)
JOBS = [
    ("paper/common/04_results.tex",
     "\\textbf{Grammatical competence is real but modest, and routing does not buy it.}",
     "% >>>ACLMOVE:stratified|", BLIMP, "blimp"),
    ("paper/common/02_setup.tex",
     "\\subsection{Training protocol}",
     "% >>>ACLMOVE:metrics|", PROTOCOL, "protocol"),
]

for path, anchor, closer, marker, blk in JOBS:
    s = io.open(path, encoding="utf-8").read()
    assert anchor in s, f"anchor missing in {path}"
    assert f"ACLMOVE:{blk}" not in s, f"already marked: {blk}"
    s = s.replace(anchor, marker + "\n" + anchor, 1)
    assert closer in s, f"closer missing in {path}"
    s = s.replace(closer, f"% <<<ACLMOVE:{blk}\n" + closer, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print(f"  {blk} block wrapped in {path}")

# register in the appendix map
p = "paper/assemble_submissions.py"
s = io.open(p, encoding="utf-8").read()
old = '    ("saturation",  "The subset sweep behind the depth-saturation finding", "app:saturation"),\n]'
new = ('    ("saturation",  "The subset sweep behind the depth-saturation finding", "app:saturation"),\n'
       '    ("protocol",    "Training protocol and hyperparameters", "app:protocol"),\n'
       '    ("blimp",       "BLiMP and EWoK, in full", "app:blimp"),\n]')
assert old in s, "appendix map anchor missing"
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("  appendix map -> 10 blocks")
