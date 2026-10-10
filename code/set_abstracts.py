"""set_abstracts.py -- one-shot (T-LA.4): new title + reframed abstract, both venues.

The abstract is rewritten around the compute-adjusted result and split into three
paragraphs (the old one was a single ~330-word block). Order is deliberate:
Pareto result, then the honest caveat that MoRE is NOT lowest-loss in domain,
then the transferable diagnostic.

TITLE. Of the two candidates proposed, "A Parameter- and FLOP-Matched Study" is
not usable: the arms are parameter-matched but emphatically NOT FLOP-matched --
1.00x / 9.07x / 2.19x inference FLOPs is the paper's central measurement, and a
reviewer who read the title and then Table 2 would have a fair objection. The
credibility anchor is kept as "Matched Parameters".
"""
import io

TITLE_TMLR = r"""\title{Adaptive Compute at Matched Parameters: Routing and\\
Recursion Trade Off in Small Language Models}"""

TITLE_ACL = r"""\title{Adaptive Compute at Matched Parameters: Routing and Recursion\\
Trade Off in Small Language Models}"""

ABSTRACT = r"""\begin{abstract}
Conditional computation chooses either \emph{which} parameters process a token
---sparse mixture-of-experts (MoE)---or \emph{how much} computation it receives
---adaptive-depth weight-tied recursion (MoR). The two are orthogonal, so their
composition should inherit both. We test whether it pays, using one engine and
three configurations with parameters matched to 0.069\% ($\approx$5.58M),
bit-identical inputs, five seeds, and exact randomisation tests under a
pre-registered correction, on WikiText-103.

The composition (MoRE) is the compute-efficient point of the three. It improves
on MoE by $7.9\%$ validation loss at $2.19\times$ MoE's inference FLOPs, where
recursion alone needs $9.07\times$: it captures $72\%$ of recursion's quality gain
over MoE for $24\%$ of recursion's inference compute---roughly $3\times$ more
quality per unit of compute, and $6\times$ out of domain. On held-out text from a
different distribution it is the best of the three outright.

It is not the lowest-loss arm in domain. Recursion alone leads by $0.1216$ nats,
a gap significant at our resolution floor, and adding routing improves neither
in-domain quality nor routing quality over MoE alone; we keep the quality and
compute axes separate throughout rather than substituting one for the other. We
trace this to a competition for one per-token budget, excluding capacity dilution
by arithmetic and subspace trapping by three measurements. Finally, and most
transferably: a mean step count and an exit histogram cannot establish
input-dependent depth, since a uniform policy forced to the learned mean
reproduces both. Scoring against that control separates our two recursive arms by
a factor of 24.
\end{abstract}"""


def patch(path, title):
    s = io.open(path, encoding='utf-8').read()
    # --- title ---
    i = s.index('\\title{')
    j = s.index('}', s.index('\n', i)) + 1 if '\\\\' in s[i:i + 200] else s.index('}', i) + 1
    s = s[:i] + title + s[j:]
    # --- abstract ---
    a = s.index('\\begin{abstract}')
    b = s.index('\\end{abstract}') + len('\\end{abstract}')
    s = s[:a] + ABSTRACT + s[b:]
    io.open(path, 'w', encoding='utf-8', newline='\n').write(s)
    words = len(ABSTRACT.split())
    print(f'  {path}: title + abstract replaced (~{words} tokens of LaTeX source)')


patch('paper/tmlr_submission/main.tex', TITLE_TMLR)
patch('paper/acl_submission/main.tex', TITLE_ACL)

# plain-text word count of the abstract, for the 180-220 target
import re
txt = ABSTRACT
txt = re.sub(r'\\begin\{abstract\}|\\end\{abstract\}', '', txt)
txt = re.sub(r'\\emph\{([^}]*)\}', r'\1', txt)
txt = re.sub(r'\$[^$]*\$', 'X', txt)
txt = re.sub(r'\\[a-zA-Z]+', '', txt)
txt = txt.replace('---', ' ').replace('{', '').replace('}', '')
n = len([w for w in txt.split() if any(c.isalnum() for c in w)])
print(f'\nabstract plain-text word count: {n}  (target 180-220)')
