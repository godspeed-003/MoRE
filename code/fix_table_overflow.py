"""fix_table_overflow.py -- T-LA.7b: the last three ACL overflows are tables.

The box dumps and file attribution put all three remaining overfull boxes in
inline `tabular` blocks, not in prose: TeX reports them as "in paragraph at
lines N--M" because a centred tabular sits inside a paragraph, which is why the
first diagnosis chased unbreakable words instead.

  98.0pt  the subspace-probe table: a 31-character row label plus three
          `$x \\pm y$` cells is ~315pt against a 234pt column.
  72.8pt  the dilution table: the header row "per FFN parameter / MoE expert /
          MoR block / MoRE expert" is almost all of the width.
  17.4pt  the subset-sweep table: marginal, and `\\small` alone covers it.

Fixed in COMMON rather than as ACL variants, because every change is an
improvement in both venues: `\\small` on a four-row table costs nothing in a
single-column layout, the shortened headers lose no information once the column
group is named, and moving the standard deviations out of the widest table into
its caption is where they belong anyway -- the cells were carrying "41.5 +- 1.6"
where the comparison being made is between 41.5, 70.3 and 57.3.
"""
import io

P = "paper/common/05_mechanism.tex"
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------- dilution table ----
OLD1 = r"""\begin{center}
\begin{tabular}{lccc}
\toprule
per FFN parameter & MoE expert & MoR block & MoRE expert \\
\midrule
tokens seen        & 128.5 & 128.5 & 128.5 \\
FFN applications   & 128   & 865   & \textbf{894} \\
\bottomrule
\end{tabular}
\end{center}"""
NEW1 = r"""\begin{center}
\small
\begin{tabular}{lccc}
\toprule
per FFN parameter & MoE & MoR & MoRE \\
\midrule
tokens seen      & 128.5 & 128.5 & 128.5 \\
FFN applications & 128   & 865   & \textbf{894} \\
\bottomrule
\end{tabular}
\end{center}"""
assert OLD1 in s, "dilution table not found"
s = s.replace(OLD1, NEW1, 1)

# ---------------------------------------------------------- subspace table ----
OLD2 = r"""\begin{center}
\begin{tabular}{lccc}
\toprule
probe & MoE & MoR & MoRE \\
\midrule
final-state participation ratio & $41.5 \pm 1.6$ & $\mathbf{70.3 \pm 5.6}$ & $57.3 \pm 2.8$ \\
between-expert mean cosine & $0.630$ & \textsc{n/a} & $0.916$ \\
\bottomrule
\end{tabular}
\end{center}"""
NEW2 = r"""\begin{center}
\small
\begin{tabular}{lccc}
\toprule
probe & MoE & MoR & MoRE \\
\midrule
participation ratio    & 41.5 & \textbf{70.3} & 57.3 \\
inter-expert cosine    & 0.630 & \textsc{n/a} & 0.916 \\
\bottomrule
\end{tabular}
\end{center}
Participation ratio is the final-state value, $\pm$ 1.6, 5.6 and 2.8 across
seeds respectively; the cosine is the mean between-expert value."""
assert OLD2 in s, "subspace table not found"
s = s.replace(OLD2, NEW2, 1)

# ----------------------------------------------------------- subset table -----
OLD3 = r"""\begin{center}
\begin{tabular}{lcc}
\toprule
training data & mean depth & forced-exit rate \\
\midrule
2.5\% of the corpus & 2.43 & $\approx 0$ \\
10\% & 1.92 & $\approx 0$ \\
40\% & 1.97 & $\approx 0$ \\
100\% (canonical) & 6.96 & 0.973 \\
\bottomrule
\end{tabular}
\end{center}"""
NEW3 = r"""\begin{center}
\small
\begin{tabular}{lcc}
\toprule
training data & mean depth & forced exit \\
\midrule
2.5\%             & 2.43 & $\approx 0$ \\
10\%              & 1.92 & $\approx 0$ \\
40\%              & 1.97 & $\approx 0$ \\
100\% (canonical) & 6.96 & 0.973 \\
\bottomrule
\end{tabular}
\end{center}"""
assert OLD3 in s, "subset table not found"
s = s.replace(OLD3, NEW3, 1)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  05_mechanism: three inline tables narrowed (\\small + shorter headers;")
print("                standard deviations moved out of the widest one)")
