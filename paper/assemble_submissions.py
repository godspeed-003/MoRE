"""assemble_submissions.py -- build two self-contained submission trees.

WHY A SCRIPT. The two papers share one body on purpose: the failure mode when a
result is written twice is that the numbers drift apart and one venue gets a
stale table. `paper/common/` is the single source; this script materialises it
into two INDEPENDENT trees so each can be uploaded to its own Overleaf project
with no external paths, no symlinks and nothing to resolve outside the folder.
Re-run it after any edit to paper/common/ or paper/refs.bib.

TWO THINGS IT FIXES THAT ARE NOT COSMETIC.

1. SECTION ORDER. The files in paper/common/ are numbered by the order they were
   written, not the order they are read: 02_setup was drafted before 03_method.
   The reading order is intro -> architecture -> setup -> results -> mechanism,
   so the mapping below renames on copy. Do not "fix" the source numbering --
   the mapping is the authority and both venues consume it.

2. LIMITATIONS PLACEMENT. TMLR takes limitations as an ordinary numbered section
   before the conclusion. ACL requires "a dedicated section titled Limitations"
   placed AFTER the conclusion and before the references, and it does not count
   toward the 8-page limit -- so for ACL it must be \\section*{Limitations}
   (starred, after the conclusion) or the paper is desk-rejected. The source file
   06_limitations.tex therefore gets split at the \\section{Conclusion} line and
   reassembled per venue.

AND ONE THAT IS COSMETIC BUT BREAKS THE BUILD. TMLR is single-column, where
`table*`/`figure*` are legal but can strand floats; ACL is two-column, where the
wide tables NEED `table*` to span both columns. The TMLR copy gets them
downgraded to `table`/`figure`.

AND ONE CROSS-REFERENCE BUG THAT COMPILES SILENTLY AND PRINTS A WRONG NUMBER.
Because ACL's Limitations section is STARRED it has no number, so a `\\ref` to it
resolves to whatever counter was last incremented -- i.e. the Conclusion's
number -- and LaTeX reports no error. Four places in the body and one in ACL's
main.tex point at it. For ACL those references are rewritten to prose, and the
subsections inside Limitations are starred too, so they do not inherit a stale
parent number. For TMLR nothing changes: there the section is numbered and the
references are correct as written.

The architecture figure ships from the arithmetic-era draft as `More(3)(1).png`.
Parentheses in a filename break \\includegraphics on most TeX installations, so
it is copied under a safe name.
"""

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))          # paper/
COMMON = os.path.join(HERE, "common")
BIB = os.path.join(HERE, "refs.bib")

# source file -> destination name, in READING order (see docstring note 1)
ORDER = [
    ("01_intro.tex",       "01_intro.tex"),        # intro + related work
    ("03_method.tex",      "02_method.tex"),       # architecture + ACT equations
    ("02_setup.tex",       "03_setup.tex"),        # data, protocol, metrics, stats
    ("04_results.tex",     "04_results.tex"),
    ("05_mechanism.tex",   "05_mechanism.tex"),
]
LIMITS_SRC = "06_limitations.tex"                   # split into limitations + conclusion

FIGS = [
    (os.path.join(HERE, "fig", "More(3)(1).png"), "fig_architecture.png"),
    (os.path.join(HERE, "fig_lang", "fig_depth_curve.pdf"), "fig_depth_curve.pdf"),
    (os.path.join(HERE, "fig_lang", "fig_stratified.pdf"),  "fig_stratified.pdf"),
    (os.path.join(HERE, "fig_lang", "fig_ood.pdf"),         "fig_ood.pdf"),
]

STYLES = {
    "tmlr_submission": [("tmlr", f) for f in
                        ("tmlr.sty", "tmlr.bst", "fancyhdr.sty", "math_commands.tex")],
    "acl_submission":  [("acl", f) for f in
                        ("acl.sty", "acl_natbib.bst")],
}


def split_limitations():
    """Return (limitations_body, conclusion_body) from the combined source."""
    src = open(os.path.join(COMMON, LIMITS_SRC), encoding="utf-8").read()
    marker = "\\section{Conclusion}"
    assert marker in src, "conclusion marker missing from " + LIMITS_SRC
    i = src.index(marker)
    return src[:i].rstrip() + "\n", src[i:]


def unnumber_refs(text):
    """ACL only: point at the starred Limitations section in prose, not by number.

    A \\ref to a \\section* prints the previous numbered section's number and
    LaTeX issues no warning, so this cannot be caught by reading the log.
    """
    return (text
            .replace("\\S\\ref{sec:limitations}", "the Limitations section")
            .replace("Section~\\ref{sec:limitations}", "the Limitations section")
            .replace("\\S\\ref{sec:honesty}", "the disclosures in the Limitations section")
            .replace("Section~\\ref{sec:honesty}", "the disclosures in the Limitations section"))


def build(venue):
    out = os.path.join(HERE, venue)
    sec = os.path.join(out, "sections")
    fig = os.path.join(out, "fig")
    for d in (sec, fig):
        os.makedirs(d, exist_ok=True)

    two_col = (venue == "acl_submission")

    # body sections
    for src, dst in ORDER:
        body = open(os.path.join(COMMON, src), encoding="utf-8").read()
        if not two_col:
            # single column: avoid stranded spanning floats
            body = body.replace("{table*}", "{table}").replace("{figure*}", "{figure}")
        else:
            body = unnumber_refs(body)
        # the architecture figure is renamed on copy (parentheses break TeX)
        body = body.replace("More(3)(1).png", "fig_architecture.png")
        open(os.path.join(sec, dst), "w", encoding="utf-8", newline="\n").write(body)

    limits, concl = split_limitations()
    if not two_col:
        limits = limits.replace("{table*}", "{table}").replace("{figure*}", "{figure}")
        concl = concl.replace("{table*}", "{table}").replace("{figure*}", "{figure}")
    else:
        # ACL: Limitations is a STARRED section after the conclusion, and is
        # excluded from the page limit. Unstarred would both number it wrongly
        # and put it inside the counted body. Its subsections are starred to
        # match, or they inherit the Conclusion's number as their parent.
        limits = limits.replace("\\section{Limitations}", "\\section*{Limitations}", 1)
        limits = limits.replace("\\subsection{", "\\subsection*{")
        limits = unnumber_refs(limits)
        concl = unnumber_refs(concl)

    open(os.path.join(sec, "06_limitations.tex"), "w",
         encoding="utf-8", newline="\n").write(limits)
    open(os.path.join(sec, "07_conclusion.tex"), "w",
         encoding="utf-8", newline="\n").write(concl)

    # bibliography
    shutil.copy2(BIB, os.path.join(out, "refs.bib"))

    # style files
    for sub, name in STYLES[venue]:
        shutil.copy2(os.path.join(HERE, sub, name), os.path.join(out, name))

    # figures
    for src, dst in FIGS:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fig, dst))
        else:
            print(f"  WARNING missing figure: {src}")

    n = len(os.listdir(sec))
    print(f"  {venue}: {n} section files, "
          f"{len(os.listdir(fig))} figures, "
          f"{'two' if two_col else 'single'}-column floats")


def check(venue):
    """Every \\input target and \\includegraphics target must exist in-tree."""
    out = os.path.join(HERE, venue)
    main = open(os.path.join(out, "main.tex"), encoding="utf-8").read()
    bad = []
    for m in re.finditer(r"\\input\{([^}]+)\}", main):
        if not os.path.exists(os.path.join(out, m.group(1) + ".tex")):
            bad.append("missing \\input: " + m.group(1))
    body = main
    for f in sorted(os.listdir(os.path.join(out, "sections"))):
        body += open(os.path.join(out, "sections", f), encoding="utf-8").read()
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        if not os.path.exists(os.path.join(out, "fig", m.group(1))):
            bad.append("missing figure: " + m.group(1))
    # citations resolve?
    cited = {k.strip() for mm in re.finditer(r"cite[pt]?\{([^}]*)\}", body)
             for k in mm.group(1).split(",")
             if re.fullmatch(r"[A-Za-z0-9_]+", k.strip())}
    have = set(re.findall(r"@[a-z]+\{([A-Za-z0-9_]+),",
                          open(os.path.join(out, "refs.bib"), encoding="utf-8").read()))
    for k in sorted(cited - have):
        bad.append("undefined citation: " + k)
    # labels referenced but never defined
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    for r in sorted(refs - labels):
        bad.append("undefined \\ref: " + r)
    return bad


if __name__ == "__main__":
    print("assembling self-contained submission trees from paper/common/ ...")
    for v in ("tmlr_submission", "acl_submission"):
        build(v)
    print("\nself-containment check:")
    ok = True
    for v in ("tmlr_submission", "acl_submission"):
        bad = check(v)
        if bad:
            ok = False
            print(f"  {v}: {len(bad)} PROBLEM(S)")
            for b in bad:
                print("     -", b)
        else:
            print(f"  {v}: OK -- all inputs, figures, citations and refs resolve")
    print("\ndone." if ok else "\nFIX THE ABOVE BEFORE UPLOADING.")
