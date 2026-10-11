"""page_report.py -- real page counts, and the ACL *counted* page total.

Reads the pdflatex logs rather than the PDFs: both documents compress their page
tree into object streams, so the plain-text /Count regex finds nothing, while
pdfTeX states the total outright ("Output written on main.pdf (N pages...)").

THE NUMBER THAT MATTERS FOR ARR IS NOT THE PDF TOTAL. Eight pages of *content*;
the Limitations section, Ethical Considerations, references and appendices are all
exempt. So this walks the log in order, tracking page shipouts (`[N]`) and noting
the page in flight when the exempt tail begins -- the point where
sections/06_limitations.tex is opened. Pages before that are the counted body.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the first file of the page-exempt tail, per venue layout
EXEMPT_FROM = {
    "acl_submission": "06_limitations",   # Limitations -> Ethics -> refs -> appendix
    "tmlr_submission": None,              # TMLR has no page limit; nothing exempt
}
LIMIT = 8


def report(venue):
    log = os.path.join(REPO, "paper", venue, "main.log")
    if not os.path.exists(log):
        print(f"  {venue}: no log")
        return
    text = open(log, encoding="utf-8", errors="replace").read()

    m = re.search(r"Output written on \S+ \((\d+) pages?, (\d+) bytes\)", text)
    total = int(m.group(1)) if m else None
    size = int(m.group(2)) / 1024 if m else None

    counted = None
    marker = EXEMPT_FROM.get(venue)
    if marker:
        page = 0
        for tok in re.finditer(r"\[(\d+)[\]\s{]|" + re.escape(marker), text):
            if tok.group(1):
                page = int(tok.group(1))
            else:
                counted = page
                break

    over = len(re.findall(r"Overfull \\hbox", text))
    under = len(re.findall(r"Underfull \\hbox", text))
    errs = re.findall(r"^! (.+)$", text, re.M)
    bad_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    bad_cit = sorted(set(re.findall(r"Citation `([^']+)' on page", text)))

    print(f"  {venue}")
    print(f"     PDF total           : {total} pages ({size:.0f} KB)")
    if marker:
        if counted:
            verdict = "WITHIN" if counted <= LIMIT else f"OVER by {counted - LIMIT}"
            print(f"     counted content     : ~{counted} pages   "
                  f"(limit {LIMIT} -> {verdict})")
            print(f"     page-exempt tail    : ~{total - counted} pages "
                  f"(Limitations, Ethics, references, appendix)")
        else:
            print("     counted content     : could not locate the exempt boundary")
    print(f"     errors              : {len(errs)}")
    print(f"     overfull / underfull: {over} / {under}")
    print(f"     undefined refs      : {bad_ref or 'none'}")
    print(f"     undefined citations : {bad_cit or 'none'}")
    print()


if __name__ == "__main__":
    print("page report, from the pdflatex logs\n")
    for v in ("tmlr_submission", "acl_submission"):
        report(v)
