"""estimate_pages.py -- estimate ACL counted-body length before/after the T-LA.4 cuts.

NOT A SUBSTITUTE FOR COMPILING. There is no TeX toolchain on this machine, so
this measures the only thing measurable here: the volume of live (non-comment)
LaTeX that lands in the page-counted part of the document, before and after the
reframe and the appendix moves. Anchored on the operator's measured ~15 pages
for the pre-cut document, the ratio gives a scaled estimate.

WHAT COUNTS FOR ARR. Eight pages of content. Excluded: the Limitations section,
Ethical Considerations, references, and appendices. So 06_limitations.tex,
07_conclusion.tex's successors in the exempt region, 08_appendix.tex and the
inline ethics block are all removed from the measurement -- counting them would
flatter the result.

FLOATS. Tables and figures occupy page area out of proportion to their source
length, so they are counted as area rather than characters: a two-column-spanning
float is charged a nominal fraction of a page. The constants are rough and are
labelled as such; they matter for the absolute estimate, not for the ratio.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACL = os.path.join(REPO, "paper", "acl_submission")

# sections inside ARR's 8-page count
COUNTED = ["01_intro.tex", "02_method.tex", "03_setup.tex",
           "04_results.tex", "05_mechanism.tex", "07_conclusion.tex"]

# nominal page area per float, two-column ACL layout. Rough by construction.
SPAN_FLOAT = 0.33     # table*/figure* spanning both columns
COL_FLOAT = 0.17      # single-column float


def live(tex):
    """Strip comments; LaTeX ignores them and so must any length estimate."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.split("\n"))


def measure(texts):
    chars, spans, cols = 0, 0, 0
    for t in texts:
        t = live(t)
        spans += len(re.findall(r"\\begin\{(?:table|figure)\*\}", t))
        cols += len(re.findall(r"\\begin\{(?:table|figure)\}", t))
        # remove float bodies before counting prose
        t = re.sub(r"\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\}",
                   "", t, flags=re.S)
        chars += len(re.sub(r"\s+", " ", t))
    return chars, spans, cols


def at_head(path):
    """The committed version of a file, or None if it was not in HEAD."""
    rel = os.path.relpath(path, REPO).replace("\\", "/")
    r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"],
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def report(label, texts):
    chars, spans, cols = measure(texts)
    float_pages = spans * SPAN_FLOAT + cols * COL_FLOAT
    return chars, spans, cols, float_pages


now = [open(os.path.join(ACL, "sections", f), encoding="utf-8").read()
       for f in COUNTED if os.path.exists(os.path.join(ACL, "sections", f))]
old_raw = [at_head(os.path.join(ACL, "sections", f)) for f in
           ["01_intro.tex", "02_method.tex", "03_setup.tex",
            "04_results.tex", "05_mechanism.tex", "07_conclusion.tex"]]
old = [t for t in old_raw if t]

if not old:
    print("no committed version to compare against -- cannot scale")
    sys.exit(0)

oc, osp, ocl, ofl = report("before", old)
nc, nsp, ncl, nfl = report("after", now)

ANCHOR = 15.0            # operator's measured page count for the pre-cut document
# split the anchor into prose pages and float pages, then rescale prose by chars
old_prose_pages = ANCHOR - ofl
chars_per_page = oc / old_prose_pages if old_prose_pages > 0 else float("nan")
new_est = nc / chars_per_page + nfl

print(f"{'':22} {'chars':>9} {'span flt':>9} {'col flt':>8} {'float pp':>9}")
print(f"{'before (HEAD)':22} {oc:>9,} {osp:>9} {ocl:>8} {ofl:>9.2f}")
print(f"{'after  (working)':22} {nc:>9,} {nsp:>9} {ncl:>8} {nfl:>9.2f}")
print()
print(f"prose chars removed from the counted body : {oc - nc:,}  ({(oc-nc)/oc*100:.1f}%)")
print(f"calibration from the ~{ANCHOR:.0f}-page anchor      : {chars_per_page:,.0f} chars/page")
print()
print(f"ESTIMATED counted-body pages, after cuts  : {new_est:.1f}")
print(f"ARR limit                                 : 8.0")
print()
if new_est > 8.0:
    print(f"STILL OVER by ~{new_est - 8.0:.1f} pages. The cut list in main.tex has")
    print("more steps; next cheapest is compressing Section 2 further.")
else:
    print("Within the limit on this estimate -- but COMPILE to confirm.")
print()
print("This is an ESTIMATE from source volume, not a compile. Treat the ratio as")
print("more trustworthy than the absolute number: the float constants are rough,")
print("and the anchor itself was measured elsewhere.")
