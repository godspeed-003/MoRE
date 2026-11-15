"""section_mass.py -- where the ACL counted body's length actually sits (T-LA.4).

Run after estimate_pages.py. That script says whether the document fits; this one
says which sections to cut to make it fit, so the editorial decision is made on
measured mass rather than on impression.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "paper", "acl_submission", "sections")

COUNTED = ["01_intro.tex", "02_method.tex", "03_setup.tex",
           "04_results.tex", "05_mechanism.tex", "07_conclusion.tex"]

CHARS_PER_PAGE = 3294      # calibrated in estimate_pages.py from the ~15pp anchor
SPAN_FLOAT = 0.33
FLOAT_PAGES_TOTAL = 1.32   # floats that survive in the counted body
LIMIT = 8.0


def live(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.split("\n"))


rows, total = [], 0
for f in COUNTED:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        continue
    t = live(open(p, encoding="utf-8").read())
    spans = len(re.findall(r"\\begin\{(?:table|figure)\*\}", t))
    t = re.sub(r"\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\}",
               "", t, flags=re.S)
    c = len(re.sub(r"\s+", " ", t))
    total += c
    rows.append((f, c, spans))

print(f"{'section':24}{'chars':>9}{'share':>8}{'floats':>8}{'est pp':>8}")
for f, c, sp in rows:
    print(f"{f:24}{c:>9,}{c / total * 100:>7.1f}%{sp:>8}"
          f"{c / CHARS_PER_PAGE + sp * SPAN_FLOAT:>8.1f}")
print(f"{'TOTAL prose':24}{total:>9,}{100.0:>7.1f}%")
print(f"{'+ floats':24}{'':>9}{'':>8}{'':>8}{FLOAT_PAGES_TOTAL:>8.2f}")
print(f"{'= ESTIMATED':24}{'':>9}{'':>8}{'':>8}"
      f"{total / CHARS_PER_PAGE + FLOAT_PAGES_TOTAL:>8.1f}")
print()
budget = int((LIMIT - FLOAT_PAGES_TOTAL) * CHARS_PER_PAGE)
print(f"prose budget at {LIMIT:.0f} pages : {budget:,} chars")
print(f"must remove            : {total - budget:,} chars "
      f"({(total - budget) / total * 100:.0f}% of the counted body)")
