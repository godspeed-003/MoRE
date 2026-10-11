"""fix_acl_overflow.py -- T-LA.7: stop content running outside ACL's text block.

The overflow check found 7 boxes more than 15pt over in the ACL build and none at
all in TMLR, which is the whole diagnosis: every offender fits a 468pt
single-column measure and busts a 234pt two-column one. So these are venue
variants, not errors in the source.

FIVE OFFENDERS, two kinds.

Display math (`detected at line N` -- a single unbreakable box):
  eq:block     112.4pt over (48% of the column). The culprit is the trailing
               `\qquad t = 0,\dots,T-1` and the \left(...\right) pair; the range
               moves into the following prose.
  eq:loss       64.6pt over. A five-term sum with underbrace annotations cannot
               fit 234pt. Stacked into two lines via `aligned`, and the
               annotations move to the prose that already explains them, so no
               information is lost.
  eq:halt-*     55.4pt over. A THREE-column `align` with a `cases` block. Rebuilt
               as a two-column align with shorter case labels.

Unbreakable prose tokens (`in paragraph at lines N--M`):
  The 41-character dataset identifier and a file path, both in \texttt. TeX sets
  \hyphenchar to -1 for typewriter, so neither can break ANYWHERE and the whole
  token is shoved past the margin. In ACL both are replaced by a short phrase
  with a pointer to the appendix -- which also helps the page budget -- while
  TMLR keeps the literal strings, because provenance belongs in the paper that
  has room for it.

`regression/classification` is fixed in COMMON rather than per venue: a slash
offers TeX no break point either, and "regression or classification" is better
prose in both.
"""
import io

# --------------------------------------------------------------- eq:block -----
P = "paper/common/03_method.tex"
s = io.open(P, encoding="utf-8").read()

OLD_BLOCK = r"""\begin{equation}
\mathbf{h}^{(t+1)} \;=\; \mathrm{LayerNorm}\!\left(
  \mathbf{h}^{(t)} + g_{e^\star}\cdot \mathrm{FFN}_{e^\star}\!\left(\mathbf{h}^{(t)}\right)
\right), \qquad t = 0,\dots,T-1 .
\label{eq:block}
\end{equation}"""

NEW_BLOCK = r"""% >>>TMLRONLY
\begin{equation}
\mathbf{h}^{(t+1)} \;=\; \mathrm{LayerNorm}\!\left(
  \mathbf{h}^{(t)} + g_{e^\star}\cdot \mathrm{FFN}_{e^\star}\!\left(\mathbf{h}^{(t)}\right)
\right), \qquad t = 0,\dots,T-1 .
\label{eq:block}
\end{equation}
% <<<TMLRONLY
% >>>ACLONLY
% Narrow-column form: the \qquad range tail and the \left(...\right) pair put the
% single-column version 112pt (48%) outside a 234pt column. The range moves to
% the prose below; the mathematics is unchanged.
\begin{equation}
\mathbf{h}^{(t+1)} = \mathrm{LayerNorm}\big(
  \mathbf{h}^{(t)} + g_{e^\star}\,\mathrm{FFN}_{e^\star}(\mathbf{h}^{(t)})
\big)
\label{eq:block}
\end{equation}
for $t = 0,\dots,T-1$.
% <<<ACLONLY"""
assert OLD_BLOCK in s, "eq:block not found"
s = s.replace(OLD_BLOCK, NEW_BLOCK, 1)

# --------------------------------------------------------------- eq:halt-* ----
OLD_HALT = r"""\begin{align}
p_t &= \sigma\!\left(\mathbf{w}_{e^\star}^{\!\top}\mathbf{h}^{(t)}\right),
  & c_t &= \sum_{\tau \le t} p_\tau ,
  \label{eq:halt-p} \\[2pt]
\text{halt when } &\; c_t + p_t > 1 - \epsilon, & \epsilon &= 0.01,
  \label{eq:halt-cond} \\[2pt]
R &= 1 - c_t \ \text{(remainder)}, &
  w_t &= \begin{cases} p_t & \text{running} \\ R & \text{at the halt step}\end{cases}
  \label{eq:halt-w}
\end{align}"""

NEW_HALT = r"""% >>>TMLRONLY
\begin{align}
p_t &= \sigma\!\left(\mathbf{w}_{e^\star}^{\!\top}\mathbf{h}^{(t)}\right),
  & c_t &= \sum_{\tau \le t} p_\tau ,
  \label{eq:halt-p} \\[2pt]
\text{halt when } &\; c_t + p_t > 1 - \epsilon, & \epsilon &= 0.01,
  \label{eq:halt-cond} \\[2pt]
R &= 1 - c_t \ \text{(remainder)}, &
  w_t &= \begin{cases} p_t & \text{running} \\ R & \text{at the halt step}\end{cases}
  \label{eq:halt-w}
\end{align}
% <<<TMLRONLY
% >>>ACLONLY
% Narrow-column form: the three-column align with a cases block ran 55pt outside
% a 234pt column. Two columns, shorter case labels, epsilon stated in the prose.
\begin{align}
p_t &= \sigma\big(\mathbf{w}_{e^\star}^{\!\top}\mathbf{h}^{(t)}\big),
  \quad c_t = \textstyle\sum_{\tau \le t} p_\tau , \label{eq:halt-p} \\[2pt]
&\text{halt when } c_t + p_t > 1 - \epsilon , \label{eq:halt-cond} \\[2pt]
R &= 1 - c_t , \quad
  w_t = \begin{cases} p_t & \text{running} \\ R & \text{at halt} \end{cases}
  \label{eq:halt-w}
\end{align}
with $\epsilon = 0.01$ and $R$ the halting remainder.
% <<<ACLONLY"""
assert OLD_HALT in s, "eq:halt align not found"
s = s.replace(OLD_HALT, NEW_HALT, 1)

# --------------------------------------------------------------- eq:loss ------
OLD_LOSS = r"""\begin{equation}
\mathcal{L} \;=\;
\underbrace{\mathcal{L}_{\text{task}}}_{1.0} \;+\;
0.001\,\mathcal{L}_{\text{bal}} \;+\;
0.001\,\rho \;+\;
\underbrace{0.0\cdot\mathcal{L}_{\text{fam}}}_{h\,\text{detached}} \;+\;
\underbrace{0.0\cdot\mathcal{L}_{\text{route}}}_{\text{off}} .
\label{eq:loss}
\end{equation}"""

NEW_LOSS = r"""% >>>TMLRONLY
\begin{equation}
\mathcal{L} \;=\;
\underbrace{\mathcal{L}_{\text{task}}}_{1.0} \;+\;
0.001\,\mathcal{L}_{\text{bal}} \;+\;
0.001\,\rho \;+\;
\underbrace{0.0\cdot\mathcal{L}_{\text{fam}}}_{h\,\text{detached}} \;+\;
\underbrace{0.0\cdot\mathcal{L}_{\text{route}}}_{\text{off}} .
\label{eq:loss}
\end{equation}
% <<<TMLRONLY
% >>>ACLONLY
% Narrow-column form: five terms with underbrace annotations ran 65pt outside a
% 234pt column. Stacked, with the annotations moved into the prose below -- which
% already names the weight on each term, so nothing is lost.
\begin{equation}
\begin{aligned}
\mathcal{L} \;=\;& 1.0\,\mathcal{L}_{\text{task}}
   + 0.001\,\mathcal{L}_{\text{bal}} + 0.001\,\rho \\
  &+ 0.0\,\mathcal{L}_{\text{fam}} + 0.0\,\mathcal{L}_{\text{route}} .
\end{aligned}
\label{eq:loss}
\end{equation}
The last two weights are zero: $\mathcal{L}_{\text{fam}}$ reads a detached
representation, and $\mathcal{L}_{\text{route}}$ is off.
% <<<ACLONLY"""
assert OLD_LOSS in s, "eq:loss not found"
s = s.replace(OLD_LOSS, NEW_LOSS, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  03_method: eq:block, eq:halt-*, eq:loss given narrow-column ACL variants")

# ----------------------------------------------- unbreakable prose tokens -----
P = "paper/common/02_setup.tex"
s = io.open(P, encoding="utf-8").read()

OLD_ID = r"(\texttt{lang-wikitext-103-bpe8192-len256-800d6154}). We use the corpus's own"
NEW_ID = (r"% >>>TMLRONLY" "\n"
          r"(\texttt{lang-wikitext-103-bpe8192-len256-800d6154}). We use the corpus's own" "\n"
          r"% <<<TMLRONLY" "\n"
          r"% >>>ACLONLY" "\n"
          r"% The literal identifier is 41 characters of \texttt, and TeX sets" "\n"
          r"% \hyphenchar to -1 for typewriter, so it cannot break anywhere and ran 98pt" "\n"
          r"% past the margin. Short phrase here, literal string in the appendix." "\n"
          r"(identifier in Appendix~\ref{app:protocol}). We use the corpus's own" "\n"
          r"% <<<ACLONLY")
assert OLD_ID in s, "dataset identifier line not found"
s = s.replace(OLD_ID, NEW_ID, 1)

OLD_RC = "regression/classification target is absent from the input}, and no oracle routing"
NEW_RC = "regression or classification target is absent from the input}, and no oracle routing"
assert OLD_RC in s, "regression/classification not found"
s = s.replace(OLD_RC, NEW_RC, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  02_setup: dataset identifier -> ACL pointer; slash compound made breakable")

# the FLOPs file path sits in the results table caption
P = "paper/common/04_results.tex"
s = io.open(P, encoding="utf-8").read()
OLD_FP = r"measured mean depth, from \texttt{results/flops\_language.json}."
NEW_FP = r"measured mean depth, from the FLOP accounting described in \S\ref{sec:metrics}."
if OLD_FP in s:
    s = s.replace(OLD_FP, NEW_FP, 1)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("  04_results: long file path in a caption replaced by a cross-reference")
else:
    print("  04_results: FLOPs path not found in that form -- check manually")
