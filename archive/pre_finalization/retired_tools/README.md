RETIRED. Do not run anything in this directory.

## wire_tikz_architecture.py — retired at T-LA.18

Written at T-LA.12 as a one-shot migration to adopt a pure-TikZ architecture
figure, wiring `paper/common/03_method.tex` to `\input{fig/fig_architecture}`
and pointing the assembler's `FIGS` at a `.tex` instead of the PNG.

**That migration was reverted at T-LA.15.** The TikZ figure positioned nodes in
a 0--100 coordinate space while sizing their text in points, so text overflowed
its nodes at native scale and `\resizebox` preserved the overlap rather than
fixing it: unreadable in single-column TMLR, collapsed into fragments over the
Related Work text in two-column ACL. TikZ overlap raises no overfull-hbox
warning, so four consecutive compiles reported 0 errors, 0 overfull and 0
underfull while the paper's only architecture diagram was illegible. Both venues
now use the generated PNG from `code/make_architecture_figure.py`.

**Why it is in `archive/` and not in `code/`.** It is the only thing in the
repository that writes `\input{fig/fig_architecture}`. Running it re-breaks the
shared method source and re-points `FIGS` at a `.tex` that no longer ships, which
fails the build with

    ! LaTeX Error: File `fig/fig_architecture.tex' not found.

That exact failure appeared mid-session at T-LA.18 from a stale generated
`02_method.tex`, and this script is the only mechanism that produces that
content. Kept for history under the archive-never-delete rule; moved out of
`code/` so it cannot be invoked by habit or by a glob.

If the architecture figure ever becomes TikZ again, write a new script against
the geometry lesson above — do not resurrect this one.

## main_content_pages.py — retired at T-LA.18

Written at T-LA.14 to measure TMLR's main content when no such tool was known to
exist here. One already did: `code/measure_counted_pages.py`.

**Two tools measuring the same quantity under different rules is the defect, not
the duplication.** This one ended main content at the *Broader Impact Statement*;
`measure_counted_pages.py` ends it at *References*, which is TMLR's actual rule
and includes the Broader Impact and Reproducibility statements. The ~1-page gap
between them is exactly those two statements, and it is why T-LA.14 reported
"12.0 pages, WITHIN" for a paper that was really ~13.0 and over.

It also measured two-column ACL by counting interleaved `pdftotext -layout`
lines, which cannot separate the columns: it read 8.6 where the coordinate-based
tool reads 8.29.

Kept for history. Use `code/measure_counted_pages.py` — it reads real glyph
coordinates, knows both venues' rules, and raises rather than defaulting when it
cannot tell which venue a PDF is.
