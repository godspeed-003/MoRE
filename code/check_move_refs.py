"""check_move_refs.py -- what breaks if a block moves to the appendix?

A block that moves takes its \\label definitions with it. Two hazards:

  1. The body keeps a \\ref to a label now defined in the appendix. That still
     compiles, but "\\S\\ref{sec:stratified}" then prints an appendix number
     behind a section sign, which reads wrong.
  2. The short replacement re-declares a label the moved copy also defines ->
     duplicate \\label, which LaTeX reports only as a soft warning while \\ref
     silently picks one. The assembler's strip_labels handles this; this
     verifies it applies to every block being moved.

Run before adding ids to a venue's move list.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(HERE, "paper", "common")


def proposed():
    """The blocks TMLR moves, read from the assembler rather than duplicated.

    This list started as a hand-written 8-block proposal here and the assembler
    then settled on 11. A second copy of a list like that drifts silently and
    the check quietly stops covering what actually ships, so it is read from the
    one place that defines it.
    """
    p = os.path.join(HERE, "paper", "assemble_submissions.py")
    src = io.open(p, encoding="utf-8").read()
    i = src.index("TMLR_APPENDIX = [")
    block = src[i:src.index("]", i)]
    return re.findall(r'"([A-Za-z0-9_]+)"', block)


def load():
    blocks, whole = {}, []
    for fn in sorted(os.listdir(COMMON)):
        if not fn.endswith(".tex"):
            continue
        src = io.open(os.path.join(COMMON, fn), encoding="utf-8").read()
        whole.append(src)
        for m in re.finditer(
                r"^% >>>ACLMOVE:([A-Za-z0-9_]+)\|(.*?)$(.*?)^% <<<ACLMOVE:\1\s*$",
                src, re.S | re.M):
            blocks[m.group(1)] = {"repl": m.group(2), "body": m.group(3),
                                  "file": fn}
    return blocks, "\n".join(whole)


def main():
    blocks, allsrc = load()
    PROPOSED = proposed()
    # body text with every movable block removed = what stays inline
    staying = allsrc
    for bid in PROPOSED:
        staying = staying.replace(blocks[bid]["body"], "")

    print("blocks TMLR moves to the appendix (read from the assembler):\n")
    bad = []
    for bid in PROPOSED:
        b = blocks[bid]
        labs = re.findall(r"\\label\{([^}]+)\}", b["body"])
        repl_labs = set(re.findall(r"\\label\{([^}]+)\}", b["repl"]))
        print(f"  {bid:14} ({b['file']})")
        if labs:
            print(f"     defines : {labs}")
        if repl_labs:
            # These are SAFE. strip_labels removes the label from the moved copy
            # and the replacement re-declares it in the body, so a body \ref
            # still resolves to a body number. This is the designed behaviour and
            # over-reporting it buried the one real hazard on the first run.
            print(f"     replacement keeps {sorted(repl_labs)} in the body "
                  f"-> body refs still resolve to a body number (safe)")
        for lab in labs:
            if lab in repl_labs:
                continue                       # stays defined in the body
            hits = re.findall(r"(\\S~?\\ref|Section~\\ref|\\ref|\\autoref)"
                              r"\{" + re.escape(lab) + r"\}", staying)
            if not hits:
                continue
            styles = sorted(set(hits))
            sectionish = [s for s in styles if "S\\ref" in s or "Section" in s]
            print(f"     referenced from the remaining body: {lab} via {styles}")
            if sectionish:
                print(f"       ^ HAZARD: prints a section sign before an "
                      f"appendix number")
                bad.append((bid, lab, sectionish))
    print()
    if not bad:
        print("  no cross-reference hazard in the source.")
        return 0

    # A hazard in paper/common/ is EXPECTED and not itself a defect: whether a
    # block sits inline or in the appendix is per-venue, so the source cannot
    # commit to either wording. What must hold is that no SHIPPED tree still
    # cites a moved label as a section. appendix_refs() in the assembler does
    # that rewrite, and this is the assertion that it actually ran.
    print(f"  {len(bad)} label(s) leave the body and are cited section-style")
    print("  in paper/common/ -- expected; appendix_refs() rewrites them per")
    print("  venue. Verifying the shipped trees:\n")
    shipped = 0
    for venue in ("tmlr_submission", "acl_submission"):
        d = os.path.join(HERE, "paper", venue, "sections")
        if not os.path.isdir(d):
            print(f"     {venue:18} not assembled -- cannot verify")
            continue
        text = "".join(io.open(os.path.join(d, f), encoding="utf-8").read()
                       for f in sorted(os.listdir(d)) if f.endswith(".tex"))
        leaks = []
        for _bid, lab, _st in bad:
            for pat in (r"\\S~?\\ref\{%s\}", r"Section~\\ref\{%s\}"):
                if re.search(pat % re.escape(lab), text):
                    leaks.append(lab)
        if leaks:
            shipped += len(leaks)
            print(f"     {venue:18} LEAKED: {sorted(set(leaks))}")
        else:
            print(f"     {venue:18} clean -- all cited as Appendix~\\ref")
    print()
    if shipped:
        print("  A shipped tree cites an appendix label as a section. LaTeX")
        print("  cannot warn about this: the \\ref resolves, only the prose")
        print("  around it is wrong. Fix appendix_refs() and reassemble.")
        return 1
    print("  rewrite verified in every assembled tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
