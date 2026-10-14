"""tmlr_length_map.py -- where does TMLR's main content actually end?

The user measures 14.4 pages against a 12-page recommendation; our tooling
reports 18 PDF pages. Both can be true: TMLR's recommended length counts MAIN
CONTENT, excluding references and appendix, and 18 is the PDF total. This prints
the page each section starts on so the boundary is a measurement, not a guess.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def label_pages(venue):
    aux = io.open(os.path.join(HERE, "paper", venue, "main.aux"),
                  encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux):
        lab, num, pg = m.group(1), m.group(2), int(m.group(3))
        if lab.endswith(("@cref", "@gtempa")):
            continue
        out.setdefault(lab, (num, pg))
    return out


def refs_page(venue):
    """Page the bibliography starts on, from the .log's shipout trace."""
    log = io.open(os.path.join(HERE, "paper", venue, "main.log"),
                  encoding="utf-8", errors="replace").read()
    m = re.findall(r"Output written on .*?\((\d+) pages", log)
    total = int(m[-1]) if m else None
    return total


def main():
    for venue in ("tmlr_submission", "acl_submission"):
        labs = label_pages(venue)
        total = refs_page(venue)
        print(f"\n{'=' * 66}\n  {venue}   PDF total {total} pages\n{'=' * 66}")
        # section-level labels only, in page order
        secs = [(pg, num, lab) for lab, (num, pg) in labs.items()
                if re.fullmatch(r"\d+(\.\d+)?", num or "")]
        for pg, num, lab in sorted(secs):
            kind = "section   " if "." not in (num or "") else "  subsec  "
            print(f"  p{pg:>3}  {kind} {num:<6} {lab}")
    print()


if __name__ == "__main__":
    main()
