r"""measure_counted_pages.py -- the counted body length of a paper, measured.

    python code/measure_counted_pages.py [path/to/main.pdf] [--venue acl|tmlr]

Each venue counts a different thing, so the tool reads the venue off the path
(see detect_venue for why not off the PDF) and applies that venue's rule.
Pointing the ACL rule at the TMLR PDF printed "10.25 pages (limit 8)" -- a real
number against the wrong landmark and the wrong limit, which is exactly the kind
of figure that ends up quoted in a decision.

  ACL/ARR   two-column, 8 pages of CONTENT. Limitations, Ethics, references and
            appendices are EXEMPT, so the count ends where Limitations starts.
  TMLR      single-column, no hard limit; the author guide's norm is ~12 pages of
            main body and the editors' reviewing-time split sits there (<=12 ->
            ~2 weeks, >12 -> ~4). Limitations is an ORDINARY NUMBERED SECTION and
            counts; references and appendices do not. So the count ends at
            References -- a later landmark than ACL's, on purpose.

FOUR TRAPS, each of which silently gives a plausible wrong answer.

1. MATCH THE HEADING, NOT THE WORD. "Limitations" and "References" both occur in
   running prose (the ACL build points at the starred section by name because
   \ref to a \section* prints the previous numbered section's number). A plain
   text search finds the prose hit first and reports a body of ~1.1 pages. Filter
   on font size.

2. A TWO-COLUMN PAGE IS NOT MEASURED BY HEIGHT. A heading 9% down the page is 9%
   into whichever COLUMN it sits in. If that is column 2, the page is already
   (1 + 0.09)/2 = 54% used, not 9%. Getting this wrong understated every
   measurement in one session by about half a page -- 8.09 reported against 8.54
   actual -- and made text cuts look more effective than they were. Single-column
   TMLR needs no such correction, which is why the venue has to be known.

3. HEADINGS ARE SPLIT ONE SPAN PER WORD. pdflatex emits "Broader", "Impact",
   "Statement" as three spans, so comparing a span's text against the full
   phrase never matches and the tool reports "not found". Join the spans in a
   line before comparing.

4. A NUMBERED HEADING CARRIES ITS NUMBER. The joined line is "7 Limitations",
   not "Limitations". Strip a leading number.
"""
import io
import os
import re
import sys

import fitz

HEADING_PT = 11.5          # body text is 10.1 pt in both venues; headings are 12

# venue -> (landmark headings, columns, soft limit, note)
#
# The measurement stops at whichever landmark appears FIRST in the document.
#
# T-LA.18: TMLR gained a second landmark when the bibliography moved after the
# appendix. With only 'references' to go on, the landmark then sat past the
# appendix and the whole appendix counted as body. Stopping at the earlier of
# References / the appendix makes the measurement independent of back-matter
# order.
#
# THE APPENDIX IS MATCHED BY ITS SECTION TITLES, not by a pattern. A first
# attempt matched "a single letter, whitespace, then anything" -- meant to catch
# the "A <Title>" heading the \appendix switch produces. It matched the display
# equation "h = LayerNorm(...)" on page 3 and reported 2.50 pages: a confidently
# wrong answer, the exact failure mode detect_venue() already warns about.
# Display math in TMLR is set \large since T-LA.16, so it clears HEADING_PT and
# looks like a heading. Titles come from the assembler, the one place that
# defines them.
RULES = {
    "acl":  (("limitations",), 2, 8,
             "Limitations/Ethics/refs/appendix exempt"),
    "tmlr": (("references",), 1, 12,
             "Limitations counts; refs/appendix do not"),
}


def appendix_titles() -> set:
    """Lowercased appendix section titles, read from assemble_submissions.py."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "paper", "assemble_submissions.py")
    try:
        src = io.open(p, encoding="utf-8").read()
        blk = src[src.index("APPENDIX = ["):src.index("]", src.index("APPENDIX = ["))]
        return {t.strip().lower() for t in re.findall(r'"\s*([^"]+)"\s*,\s*"app:', blk)}
    except (OSError, ValueError):
        return set()


def detect_venue(path: str) -> str:
    """Venue from the path, not from the pixels.

    Two attempts at inferring the layout from the PDF both failed on real input,
    and they failed by returning a confident wrong answer rather than by
    erroring:

      * line-start distribution -- a single-column page whose content is mostly
        TABLE CELLS has a quarter of its line starts right of the midpoint, so
        TMLR was detected as two-column;
      * median line width -- pdflatex fragments a text line into per-word
        spans at different granularity in the two builds (ACL's median
        "line" is 2.6% of the page width, about one word), so the two
        layouts are not separable on this axis at all.

    The venue is already encoded unambiguously in the directory name, so use
    that and let --venue override it. An unrecognised path raises rather than
    defaulting, because defaulting is what produced the wrong number: the ACL
    rule pointed at the TMLR PDF printed "10.25 pages (limit 8)", a real
    measurement against the wrong landmark and the wrong limit.
    """
    low = path.replace("\\", "/").lower()
    for v in RULES:
        if f"{v}_submission" in low:
            return v
    raise SystemExit(
        f"cannot tell which venue '{path}' is -- no '<venue>_submission' in the "
        f"path. Pass it explicitly: --venue {'|'.join(RULES)}")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    venue = None
    if "--venue" in args:
        i = args.index("--venue")
        venue = args[i + 1]
        del args[i:i + 2]
        if venue not in RULES:
            raise SystemExit(f"--venue must be one of {list(RULES)}")
    path = args[0] if args else "paper/acl_submission/main.pdf"
    venue = venue or detect_venue(path)
    landmarks, cols, limit, note = RULES[venue]
    stops = set(landmarks)
    if venue == "tmlr":
        stops |= appendix_titles()

    doc = fitz.open(path)
    for i, page in enumerate(doc):
        W, H = page.rect.width, page.rect.height
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                big = [s for s in ln["spans"] if s["size"] >= HEADING_PT]
                if not big:
                    continue
                txt = " ".join(s["text"] for s in big).strip().lower()
                # Spans are one per word AND carry their own trailing spaces, so
                # joining yields runs of whitespace: "the   resolution   floor".
                # A single-word landmark ("references") survives that; a
                # multi-word appendix title does not, and the tool silently ran
                # past the appendix to the bibliography on page 19 and reported
                # 18.28 pages.
                txt = re.sub(r"\s+", " ", txt)
                # strip a leading section number, and the letter the \appendix
                # switch puts in front of an appendix section title
                txt = re.sub(r"^[0-9.\s]+", "", txt).strip()
                if txt not in stops:
                    txt = re.sub(r"^[a-z]\.?\s+", "", txt).strip()
                if txt not in stops:
                    continue
                frac = ln["bbox"][1] / H
                if cols == 2:
                    col = 0 if ln["bbox"][0] < W / 2 else 1
                    pages = i + (col + frac) / 2
                else:
                    pages = i + frac
                verdict = "OK" if pages <= limit else f"OVER by {pages - limit:.2f}"
                print(f"  venue          : {venue.upper()} ({cols}-column)")
                print(f"  rule           : count to "
                      f"{' or '.join(l.title() for l in landmarks)}"
                      f" or the appendix, whichever is first; {note}"
                      if venue == "tmlr" else
                      f"  rule           : count to "
                      f"'{landmarks[0].title()}'; {note}")
                print(f"  stopped at     : '{txt}' on page {i + 1}")
                print(f"  COUNTED BODY   : {pages:.2f} pages  (limit {limit}) "
                      f"-> {verdict}")
                print(f"  total PDF      : {doc.page_count} pages")
                return 0

    print(f"  venue          : {venue.upper()} ({cols}-column)")
    print(f"  none of {sorted(stops)} found -- cannot measure. "
          f"Check the heading exists and is set at >= {HEADING_PT} pt.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
