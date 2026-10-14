"""adopt_tikz_architecture.py -- T-LA.8b: take up the TikZ architecture figure.

WHAT TURNED UP. `paper/acl_submission/fig/fig_architecture(1).tex` appeared on
disk, untracked, dated after the last assembly, and `fig_architecture.png` had
been removed from that same folder -- which is why the ACL build began failing
with "File `fig_architecture.png' not found" and then cascaded (a halted first
pass leaves an incomplete .aux, bibtex errors, and the final passes report 41
undefined citations). The 2 errors and 41 undefined refs were that cascade, not
two independent faults.

The new file is a complete, self-contained `figure*` float -- 86 lines of pure
TikZ, its own `\\caption` and `\\label{fig:architecture}` -- and its header states
the intended wiring: add `\\usepackage{tikz}`, then replace the old block with
`\\input{fig/fig_architecture}`. Adopting it rather than restoring the PNG is the
right call: a vector diagram beats a 2870x1139 raster at print resolution.

THREE THINGS THE DROP-IN DID NOT HANDLE.

1. It was sitting in a GENERATED folder. `assemble_submissions.py` rewrites the
   venue trees from source, so the next assembly would have deleted it. It moves
   to `paper/fig/`, the source of truth, and is registered in `FIGS`.

2. The filename carries `(1)` from a browser download, and parentheses break
   `\\input` on most TeX installations -- the same trap that made the PNG ship as
   `fig_architecture.png` rather than `More(3)(1).png`. Renamed.

3. It declares its own caption and label, so the block in `03_method.tex` must
   stop declaring them or both are defined twice.

ONE CONSEQUENCE TO FLAG, NOT HIDE. The TikZ float is `figure*`, which spans both
columns in ACL. T-LA.5 had deliberately demoted the architecture figure to a
single-column `figure` to claw back a page against the 8-page limit. Spanning is
the right choice for a diagram this wide -- squeezed into 3.25in it would be
illegible -- so the page is given back and ACL's budget decision stands open.
"""
import io
import os
import shutil

SRC = "paper/acl_submission/fig/fig_architecture(1).tex"
DST = "paper/fig/fig_architecture.tex"

# ----------------------------------------------- 1. move it to the source -----
if os.path.exists(SRC):
    shutil.move(SRC, DST)
    print(f"  moved {SRC}\n     -> {DST}  (out of a generated folder, parens dropped)")
elif os.path.exists(DST):
    print(f"  {DST} already in place")
else:
    raise SystemExit("neither the drop-in nor the destination exists -- stopping")

# ------------------------------------- 2. register it with the assembler ------
P = "paper/assemble_submissions.py"
s = io.open(P, encoding="utf-8").read()
old = '    (os.path.join(HERE, "fig", "More(3)(1).png"), "fig_architecture.png"),'
new = ('    # T-LA.8b: the architecture diagram is now pure TikZ, \\input as a\n'
       '    # complete figure* float. The superseded PNG stays in paper/fig/ as the\n'
       '    # original but is no longer copied -- nothing references it, and it was\n'
       '    # 256 KB of the upload bundle.\n'
       '    (os.path.join(HERE, "fig", "fig_architecture.tex"), "fig_architecture.tex"),')
assert old in s, "FIGS entry for the PNG not found"
s = s.replace(old, new, 1)

# TMLR is single-column; apply the same float demotion to copied .tex figures
# that already applies to the section files, so a spanning float cannot strand.
old_copy = """    # figures
    for src, dst in FIGS:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fig, dst))
        else:
            print(f"  WARNING missing figure: {src}")"""
new_copy = """    # figures. A .tex figure is COPIED THROUGH THE SAME float conversion as
    # the sections: it carries its own figure* float, and in single-column TMLR
    # a spanning float can strand. Binary figures are copied verbatim.
    for src, dst in FIGS:
        if not os.path.exists(src):
            print(f"  WARNING missing figure: {src}")
            continue
        if dst.endswith(".tex") and not two_col:
            t = open(src, encoding="utf-8").read()
            t = t.replace("{figure*}", "{figure}").replace("{table*}", "{table}")
            open(os.path.join(fig, dst), "w",
                 encoding="utf-8", newline="\\n").write(t)
        else:
            shutil.copy2(src, os.path.join(fig, dst))"""
assert old_copy in s, "figure-copy block not found"
s = s.replace(old_copy, new_copy, 1)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  assembler: fig_architecture.tex registered; .tex figures get the TMLR"
      " float conversion")

# --------------------------------- 3. \input it instead of \includegraphics ---
P = "paper/common/03_method.tex"
s = io.open(P, encoding="utf-8").read()
start = s.index("% Plain `figure`, NOT `figure*`, deliberately.")
end = s.index("\\end{figure}", start) + len("\\end{figure}")
replacement = (
    "% The architecture diagram is pure TikZ and ships as a COMPLETE float --\n"
    "% figure* wrapper, caption and \\label{fig:architecture} all live in\n"
    "% fig/fig_architecture.tex -- so nothing may be declared here or they are\n"
    "% defined twice. Needs \\usepackage{tikz} in the preamble; the assembler\n"
    "% demotes the spanning float for single-column TMLR.\n"
    "\\input{fig/fig_architecture}")
s = s[:start] + replacement + s[end:]
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("  03_method: \\includegraphics block replaced by \\input{fig/fig_architecture}")

# ------------------------------------------- 4. tikz in both preambles --------
for main, anchor in (("paper/tmlr_submission/main.tex", "\\usepackage{graphicx}"),
                     ("paper/acl_submission/main.tex", "\\usepackage{graphicx}")):
    s = io.open(main, encoding="utf-8").read()
    if "\\usepackage{tikz}" in s:
        print(f"  {main}: tikz already present")
        continue
    assert anchor in s, f"graphicx not found in {main}"
    s = s.replace(anchor,
                  anchor + "\n\\usepackage{tikz}        % architecture figure is"
                           " drawn, not included", 1)
    io.open(main, "w", encoding="utf-8", newline="\n").write(s)
    print(f"  {main}: \\usepackage{{tikz}} added")
