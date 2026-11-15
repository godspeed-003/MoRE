"""tighten_abstract.py -- T-LA.4 follow-up: bring the abstract inside 180-220 words.

The first pass came out at 246. Trimmed by compressing the mechanism sentence
(the full argument is in the body) and the axis-separation clause, with no
numbers removed -- every figure in the abstract still appears, because dropping
one would be the wrong economy.
"""
import io
import re

NEW = r"""\begin{abstract}
Conditional computation chooses either \emph{which} parameters process a token
---sparse mixture-of-experts (MoE)---or \emph{how much} computation it receives
---adaptive-depth weight-tied recursion (MoR). Composing them should inherit
both. We test whether it pays: one engine, three
configurations, parameters matched to 0.069\% ($\approx$5.58M), bit-identical
inputs, five seeds, exact randomisation tests under a pre-registered correction,
on WikiText-103.

The composition (MoRE) is the compute-efficient point of the three. It improves on
MoE by $7.9\%$ validation loss at $2.19\times$ MoE's inference FLOPs, where
recursion alone needs $9.07\times$: it captures $72\%$ of recursion's quality gain
over MoE for $24\%$ of recursion's compute---roughly $3\times$ more quality per
unit of compute, and $6\times$ out of domain. On held-out text from a different
distribution it is the best of the three outright.

It is not the lowest-loss arm in domain. Recursion alone leads by $0.1216$ nats, a
gap significant at our resolution floor, and adding routing improves neither
in-domain quality nor routing quality over MoE; we keep the quality and compute
axes separate. We trace the deficit to a competition for one per-token budget,
excluding two rival explanations. Most transferably: a mean step count and an exit histogram cannot
establish input-dependent depth, since a uniform policy forced to the learned mean
reproduces both. Scoring against that control separates our recursive arms by a
factor of 24.
\end{abstract}"""


def wordcount(tex):
    t = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", tex)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$[^$]*\$", "X", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("---", " ").replace("{", "").replace("}", "")
    return len([w for w in t.split() if any(c.isalnum() for c in w)])


print(f"abstract word count: {wordcount(NEW)}  (target 180-220)")
for p in ("paper/tmlr_submission/main.tex", "paper/acl_submission/main.tex"):
    s = io.open(p, encoding="utf-8").read()
    a = s.index("\\begin{abstract}")
    b = s.index("\\end{abstract}") + len("\\end{abstract}")
    io.open(p, "w", encoding="utf-8", newline="\n").write(s[:a] + NEW + s[b:])
    print("  patched", p)
