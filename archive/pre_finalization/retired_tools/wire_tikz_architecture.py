"""wire_tikz_architecture.py -- T-LA.12: adopt the TikZ architecture figure.

The figure was replaced by hand in `paper/tmlr_submission/fig/`: the PNG deleted
and a 5,672-byte pure-TikZ `fig_architecture.tex` dropped in, with wiring
instructions in its own header. This follows them.

CORRECTNESS CHECKED FIRST, because the previous PNG's defect was exactly this.
`fig/More(3)(1).png` drew the RETIRED per-step architecture -- noisy top-K, one
expert per token per depth, the halt looping back into the router -- and
contradicted the method section on the paper's central claim. The new TikZ file
carries none of those markers and states the opposite explicitly: routing happens
ONCE from the post-embedding state, the token keeps e* for the whole trajectory
(the route-once specification), only the selected expert is evaluated, and the
same weights are reused at every depth. It is the correct diagram.

WHY THE DROP-IN COULD NOT STAY WHERE IT WAS PUT. `paper/tmlr_submission/fig/` is
GENERATED -- `assemble_submissions.py` rewrites it from source on every run and,
since T-LA.8, prunes files it did not write. The next assembly would have deleted
the new figure and restored the PNG. It moves to `paper/fig_lang/`, which is where
the assembler reads figures from, and the `FIGS` entry switches from the PNG to
the .tex.

BOTH VENUES GET IT. The figure block lives in `paper/common/03_method.tex`, which
feeds both trees, so ACL picks up the corrected diagram too -- which is what
content parity requires and what was asked for.

The superseded PNG stays in `paper/fig_lang/` rather than being deleted
(CLAUDE.md: archive, never delete), but is no longer copied into either tree.
"""
import io
import os
import shutil

DROP = "paper/tmlr_submission/fig/fig_architecture.tex"
SRC = "paper/fig_lang/fig_architecture.tex"

# ------------------------------------- 1. move the drop-in to the source ------
if os.path.exists(DROP):
    shutil.move(DROP, SRC)
    print(f"  moved the drop-in out of the generated tree:\n     {DROP}\n  -> {SRC}")
elif os.path.exists(SRC):
    print(f"  {SRC} already in place")
else:
    raise SystemExit("the TikZ figure is in neither location -- stopping")

# ------------------------------------------- 2. switch FIGS to the .tex -------
P = "paper/assemble_submissions.py"
s = io.open(P, encoding="utf-8").read()
old = ('    (os.path.join(HERE, "fig_lang", "fig_architecture.png"), '
       '"fig_architecture.png"),')
new = ('    # T-LA.12: the diagram is pure TikZ and ships as a COMPLETE figure*\n'
       '    # float -- wrapper, caption and \\label{fig:architecture} all inside the\n'
       '    # file -- so 03_method.tex only \\inputs it. Needs \\usepackage{tikz}.\n'
       '    # fig_architecture.png is SUPERSEDED: it drew the retired per-step\n'
       '    # architecture and contradicted the method section. Kept in fig_lang/\n'
       '    # as history, never copied into a venue tree again.\n'
       '    (os.path.join(HERE, "fig_lang", "fig_architecture.tex"), '
       '"fig_architecture.tex"),')
assert old in s, "FIGS png entry not found"
s = s.replace(old, new, 1)

# the renamer for the long-dead original is now dead code too
s = s.replace('        body = body.replace("More(3)(1).png", "fig_architecture.png")\n',
              '')
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  assembler: FIGS -> fig_architecture.tex; stale PNG renamer removed")

# ------------------------------- 3. \input it instead of \includegraphics -----
P = "paper/common/03_method.tex"
s = io.open(P, encoding="utf-8").read()
start = s.index("\\begin{figure}[t]")
end = s.index("\\end{figure}", start) + len("\\end{figure}")
# keep any comment block immediately above the float out of the way
lead = s.rfind("\n%", 0, start)
if lead != -1 and s.count("\n", lead, start) <= 8:
    start = s.index("\n", lead) + 1
replacement = (
    "% The architecture diagram is pure TikZ and ships as a COMPLETE float:\n"
    "% fig/fig_architecture.tex supplies the figure* wrapper, the caption and\n"
    "% \\label{fig:architecture}, so declaring any of them here would define\n"
    "% them twice. Requires \\usepackage{tikz} (present in both preambles); the\n"
    "% assembler demotes the spanning float for single-column TMLR.\n"
    "\\input{fig/fig_architecture}\n")
s = s[:start] + replacement + s[end + 1:]
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  03_method: figure block -> \\input{fig/fig_architecture}")
