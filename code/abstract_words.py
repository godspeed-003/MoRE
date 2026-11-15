"""abstract_words.py -- plain-text word count of each venue's abstract.

Exists because the abstract has a target length (180-220 words), the count has
had to be re-checked after every edit, and doing it inline kept tripping over
shell quoting of `$...$` math. Strips the math, the macros and the em-dashes so
the number is what a human would count, not what `wc -w` sees in LaTeX source.
"""
import io
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = (180, 220)


def words(tex):
    t = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", tex)
    t = re.sub(r"\\emph\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$[^$]*\$", "X", t)          # math counts as one word
    t = re.sub(r"\\[a-zA-Z]+", "", t)         # strip remaining macros
    t = t.replace("---", " ").replace("--", " ")
    t = t.replace("{", "").replace("}", "")
    return [w for w in t.split() if any(c.isalnum() for c in w)]


for v in ("tmlr_submission", "acl_submission"):
    p = os.path.join(REPO, "paper", v, "main.tex")
    if not os.path.exists(p):
        print(f"  {v:20} no main.tex")
        continue
    s = io.open(p, encoding="utf-8").read()
    try:
        a = s.index("\\begin{abstract}")
        b = s.index("\\end{abstract}") + len("\\end{abstract}")
    except ValueError:
        print(f"  {v:20} no abstract found")
        continue
    n = len(words(s[a:b]))
    lo, hi = TARGET
    flag = "ok" if lo <= n <= hi else (f"{n - hi} over" if n > hi else f"{lo - n} under")
    paras = s[a:b].count("\n\n")
    print(f"  {v:20} {n:>4} words  ({flag}; target {lo}-{hi})  paragraphs: {paras}")
