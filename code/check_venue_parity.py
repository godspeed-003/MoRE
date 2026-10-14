"""check_venue_parity.py -- does TMLR carry everything ACL carries?

The requirement is content parity with format divergence: the two venues should
differ only in how the same material is laid out, never in what it says. That is
easy to believe and easy to get wrong, because the assembler deliberately
produces different trees -- ACL moves blocks into a page-exempt appendix, swaps
in narrow-column equations, and uses a one-paragraph related-work variant, while
TMLR keeps everything inline at full width.

So "the files differ" is expected and uninformative. What matters is whether any
CLAIM, NUMBER, CITATION, FIGURE or LABEL exists in one venue and not the other.
This compares those, counting the appendix as part of ACL's content, and
classifies each difference as expected-by-design or a real divergence.

Known and intended asymmetries, which this does NOT report as faults:
  - ACL's Ethical Considerations vs TMLR's Broader Impact Statement.
  - TMLR's Reproducibility Statement, which has no ACL counterpart because ARR
    collects the same facts on its mandatory Responsible NLP Checklist.
  - ACL's one-paragraph related work against TMLR's four paragraphs; TMLR having
    MORE prose is not a parity failure.
"""
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# statements that exist in exactly one venue on purpose
EXPECTED_ONLY = {
    "acl_submission": {"Ethical Considerations"},
    "tmlr_submission": {"Broader Impact Statement", "Reproducibility Statement"},
}


def appendix_wrapper_titles():
    """The ACL appendix's own section headings, read from the assembler.

    ACL lifts blocks into a page-exempt appendix and wraps each in a
    `\\section{...}` with an `app:*` label. TMLR keeps the SAME CONTENT inline
    under its original subsection headings, so those wrappers exist in exactly
    one venue by construction. Counting them as "missing from TMLR" reported 23
    phantom gaps on the first run and buried the one real finding, so they are
    read from the single source that defines them rather than hard-coded here.
    """
    p = os.path.join(REPO, "paper", "assemble_submissions.py")
    src = io.open(p, encoding="utf-8").read()
    block = src[src.index("APPENDIX = ["):src.index("]", src.index("APPENDIX = ["))]
    titles = set(re.findall(r'"\s*([^"]+)"\s*,\s*"app:', block))
    labels = {"app:" + x for x in re.findall(r'"app:([A-Za-z0-9_]+)"', block)}
    return titles, labels


def live(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))


def corpus(venue):
    """All live LaTeX for a venue, appendix included."""
    parts = []
    d = os.path.join(REPO, "paper", venue, "sections")
    for f in sorted(os.listdir(d)):
        if f.endswith(".tex"):
            parts.append(io.open(os.path.join(d, f), encoding="utf-8").read())
    parts.append(io.open(os.path.join(REPO, "paper", venue, "main.tex"),
                         encoding="utf-8").read())
    return live("\n".join(parts))


def facts(text):
    cites = set()
    for m in re.finditer(r"cite[pt]?\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", k):
                cites.add(k)
    return {
        "citations": cites,
        "labels": set(re.findall(r"\\label\{([^}]+)\}", text)),
        "figures": set(re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)),
        # every number with a decimal point: the claims are carried by these
        "numbers": set(re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", text)),
        "sections": set(re.findall(r"\\(?:sub)*section\*?\{([^}]+)\}", text)),
    }


def main():
    t = facts(corpus("tmlr_submission"))
    a = facts(corpus("acl_submission"))
    app_titles, app_labels = appendix_wrapper_titles()

    # strip the appendix wrapper from ACL before comparing: it is scaffolding
    # around content TMLR holds inline, not content of its own.
    a["labels"] -= app_labels
    a["sections"] -= app_titles

    print("venue parity -- content should match, layout may differ")
    print(f"  (ignoring {len(app_labels)} ACL appendix anchors and "
          f"{len(app_titles)} appendix headings: scaffolding, not content)\n")
    bad = 0
    for key in ("citations", "figures", "labels", "numbers"):
        only_t = t[key] - a[key]
        only_a = a[key] - t[key]
        status = "OK" if not (only_t or only_a) else "DIFFERS"
        print(f"  {key:11} TMLR {len(t[key]):>3}   ACL {len(a[key]):>3}   {status}")
        if only_a:
            bad += len(only_a)
            print(f"     IN ACL ONLY  (TMLR is missing these): {sorted(only_a)}")
        if only_t:
            print(f"     in TMLR only (extra, usually fine)  : {sorted(only_t)}")

    # Sections, compared on the part of the title before the first comma. The
    # ACL variants shorten some headings by truncating a trailing clause --
    # "Depth saturation is a data-scale effect" against "...effect, not an
    # architectural limit" -- which is a wording difference, not a missing
    # section, and reporting it as a gap hid the one real finding.
    norm = lambda ss: {s.split(",")[0].strip() for s in ss}
    ts, as_ = norm(t["sections"]), norm(a["sections"])
    only_t = ts - as_ - norm(EXPECTED_ONLY["tmlr_submission"])
    only_a = as_ - ts - norm(EXPECTED_ONLY["acl_submission"])
    print(f"  {'sections':11} TMLR {len(ts):>3}   ACL {len(as_):>3}   "
          f"{'OK' if not (only_t or only_a) else 'DIFFERS'}")
    if only_a:
        bad += len(only_a)
        print(f"     IN ACL ONLY  (TMLR is missing these): {sorted(only_a)}")
    if only_t:
        print(f"     in TMLR only (extra, usually fine)  : {sorted(only_t)}")

    print()
    if bad:
        print(f"  {bad} item(s) present in ACL and absent from TMLR. ACL is the")
        print("  reference, so each of these is a real gap to close.")
        return 1
    print("  TMLR carries everything ACL carries. Remaining differences are")
    print("  layout and the intended per-venue statements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
