"""main_content_pages.py -- how many pages of MAIN CONTENT does each venue have?

The number that matters is not the PDF total. For TMLR the author guide
recommends 12 pages of main content, and main content ends at the end of the
Conclusion -- everything after it (Broader Impact, Reproducibility, references,
appendix) is exempt. For ACL the limit is 8 and the exempt tail starts at
Limitations.

Measured to a fraction of a page by finding, on the page the exempt tail starts,
how far down that page the heading sits. That is how the 14.4 figure arises:
the Conclusion ended ~40% into page 15.

Needs pdftotext (ships with TinyTeX / poppler).
"""
import io
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "build_tmp")

# (venue, recommended main-content pages, heading that ENDS main content)
VENUES = [
    ("tmlr_submission", 12, ["Broader Impact Statement"]),
    ("acl_submission", 8, ["Limitations"]),
]


def page_text(pdf):
    os.makedirs(TMP, exist_ok=True)
    txt = os.path.join(TMP, os.path.basename(os.path.dirname(pdf)) + ".txt")
    subprocess.run(["pdftotext", "-layout", pdf, txt],
                   check=True, capture_output=True)
    pages = io.open(txt, encoding="utf-8", errors="replace").read().split("\f")
    # pdftotext emits a trailing form feed, so the split yields one empty chunk
    # past the last page. Leaving it in reported 20 pages for a 19-page PDF.
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


def main():
    if not shutil.which("pdftotext"):
        print("pdftotext not on PATH -- cannot measure fractional pages")
        return 2
    rc = 0
    for venue, limit, enders in VENUES:
        pdf = os.path.join(HERE, "paper", venue, "main.pdf")
        if not os.path.exists(pdf):
            print(f"{venue}: no main.pdf -- compile first")
            rc = 2
            continue
        pages = page_text(pdf)
        total = len(pages)
        found = None
        for i, p in enumerate(pages, 1):
            for e in enders:
                # a heading, i.e. at the start of a line
                if re.search(r"^\s*(?:\d+\s+)?" + re.escape(e), p, re.M):
                    found = (i, p, e)
                    break
            if found:
                break
        print(f"\n{'=' * 64}\n  {venue}\n{'=' * 64}")
        print(f"  PDF total                 : {total} pages")
        if not found:
            print(f"  could not locate {enders} -- main content NOT measured")
            rc = 2
            continue
        pg, ptext, ender = found
        lines = ptext.split("\n")
        at = next(i for i, ln in enumerate(lines)
                  if re.match(r"^\s*(?:\d+\s+)?" + re.escape(ender), ln))
        # Lines before the heading that are NOT content: the running header
        # ("Under review as submission to TMLR") and a bare page number. Counting
        # the header made a heading at the very top of a page read as 2% down,
        # which reported exactly-12 pages as "OVER by 0.0".
        def is_furniture(ln):
            s = ln.strip()
            return (not s or s.isdigit()
                    or s.startswith("Under review")
                    or s.startswith("Published in"))

        body_lines = len([ln for ln in lines[:at] if not is_furniture(ln)])
        page_lines = max(len([ln for ln in lines if not is_furniture(ln)]), 1)
        frac = body_lines / page_lines
        content = (pg - 1) + frac
        verdict = "WITHIN" if content <= limit else f"OVER by {content - limit:.1f}"
        print(f"  main content ends         : p{pg}, ~{frac * 100:.0f}% down "
              f"(at '{ender}')")
        print(f"  MAIN CONTENT              : ~{content:.1f} pages  "
              f"(recommended {limit}) -> {verdict}")
        print(f"  exempt tail               : ~{total - content:.1f} pages")
        if content > limit:
            rc = 1
    print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
