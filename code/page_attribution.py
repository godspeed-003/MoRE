"""page_attribution.py -- which pages each section occupies, from the pdflatex log.

Why this exists. Three rounds of appendix moves took the ACL counted body from
~10 pages to ~9 and then stopped moving it, because every moved block leaves a
self-contained summary behind (ARR reviewers are not obliged to read appendices)
and ten such summaries are themselves most of a page. Guessing which section to
cut next was not working, so this attributes pages to sections directly.

Method. pdflatex's log interleaves file opens `(./sections/NN_x.tex` with page
shipouts `[N]`. Walking it in order gives the page in flight when each section
started, hence the span each occupies. Boundaries are approximate by one page
either way -- a section that starts mid-page is charged to the page it starts on.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def attribute(venue):
    log = os.path.join(REPO, "paper", venue, "main.log")
    if not os.path.exists(log):
        print(f"  {venue}: no log")
        return
    text = open(log, encoding="utf-8", errors="replace").read()

    # events in document order: a section file opening, or a page being shipped
    ev = re.finditer(r"\(\.?[\\/]?sections[\\/](\d\d_[a-z]+)\.tex|\[(\d+)[\]\s{]", text)
    page, order, start = 0, [], {}
    for m in ev:
        if m.group(1):
            if m.group(1) not in start:
                start[m.group(1)] = max(page, 1)
                order.append(m.group(1))
        elif m.group(2):
            page = int(m.group(2))
    total = page

    print(f"  {venue}  (total {total} pages)")
    print(f"     {'section':22}{'starts p.':>10}{'spans':>8}")
    for i, s in enumerate(order):
        nxt = start[order[i + 1]] if i + 1 < len(order) else total
        print(f"     {s:22}{start[s]:>10}{max(nxt - start[s], 0):>8}")
    print()
    return start, total, order


if __name__ == "__main__":
    for v in ("acl_submission", "tmlr_submission"):
        attribute(v)
    print("  Boundaries are +/- one page: a section starting mid-page is charged")
    print("  to the page it starts on, so spans sum to the total only loosely.")
