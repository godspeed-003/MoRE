"""check_overflow.py -- find content that runs outside the text block.

Answers "is any equation or anything flowing out of the body" from the pdflatex
logs, which is where TeX records it and where it is otherwise easy to miss: an
overfull box is a WARNING, not an error, so both papers can report "0 errors"
while text sits in the margin.

WHAT COUNTS AS A PROBLEM, and what does not:

  Overfull \\hbox   -- REAL. Content is wider than the column and is printed
                      outside it. In two-column ACL a column is 3.25in = 234pt,
                      so anything past ~20pt is visible in the gutter and
                      anything past 50pt is unmistakable.
  Overfull \\vbox   -- REAL. Content is taller than the page body.
  Underfull \\vbox  -- COSMETIC. Whitespace left by float placement. A paper
                      with full-width floats always has these; they are not
                      overflow and are reported separately so they cannot be
                      mistaken for one.
  Underfull \\hbox  -- COSMETIC. Loose inter-word spacing in justified text.

Each overfull box is attributed to the file that was open when TeX hit it, by
tracking the `(filename` / `)` nesting the log uses. "detected at line N" rather
than "in paragraph at lines N--M" means the offender is a single box -- a display
equation, a tabular, or an \\includegraphics -- not flowing prose, which matters
because the two need different fixes: prose rewraps, a wide equation does not.
"""
import io
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COL_PT = {"acl_submission": 234.5,      # 3.25in, two-column
          "tmlr_submission": 468.0}     # ~6.5in, single-column

FILE_RE = re.compile(r"\((\.?[/\\]?[\w./\\-]+\.(?:tex|sty|cls|bbl))")
OVER_H = re.compile(
    r"Overfull \\hbox \(([\d.]+)pt too wide\)"
    r"(?: (?:detected at line (\d+)|in paragraph at lines ([\d\-]+)))?")
OVER_V = re.compile(r"Overfull \\vbox \(([\d.]+)pt too high\)")


def box_text(block):
    """Readable text of an overfull box, when the log carries the dump.

    `pdflatex "\\showboxbreadth=12 \\showboxdepth=4 \\input{main}"` makes TeX
    print the contents of each overfull box. The dump is one line per item --
    glyphs appear as `\\T1/ptm/m/n/10 w`, kerns and glue as their own lines -- so
    the words are recovered by keeping the font lines and stitching their tails.
    Without the dump this returns "" and the caller falls back to line numbers,
    which is why the plain compile is still useful on its own.

    THE WINDOW MATTERS. A fixed 6000-character window bled across several dumps
    and reported fragments of unrelated paragraphs together, which sent the first
    diagnosis after the wrong paragraphs entirely. The dump ends at the next
    warning or at a blank line, so cut there.
    """
    end = len(block)
    for pat in (r"\n\s*\n", r"\nOverfull ", r"\nUnderfull ", r"\n\) *\n"):
        m = re.search(pat, block)
        if m:
            end = min(end, m.start())
    glyphs = re.findall(r"\\(?:T1|OT1|TU|TS1)/[^ ]+ (.*)", block[:end])
    if not glyphs:
        return ""
    s = re.sub(r"\s+", " ", "".join(glyphs)).strip()
    return s[:200]


def scan(venue):
    log = os.path.join(REPO, "paper", venue, "main.log")
    if not os.path.exists(log):
        print(f"  {venue}: no log -- compile first")
        return []
    text = io.open(log, encoding="utf-8", errors="replace").read()

    # Attribute each warning to the LAST .tex opened before it. The earlier
    # version kept a paren stack, but the log's parentheses are not balanced --
    # font names and warning prose contain them -- so the stack drifted and
    # every finding came back as "?". "Most recently opened file" is approximate
    # (it does not pop on close) but TeX reads files in order, so it is right in
    # practice and it actually names something.
    opens = [(m.start(), os.path.basename(m.group(1)))
             for m in re.finditer(
                 r"\((?:\./)?((?:sections[/\\])?[\w.-]+\.(?:tex|bbl))", text)]

    def owner(pos):
        name = "?"
        for at, n in opens:
            if at > pos:
                break
            name = n
        return name

    found = []
    for m in re.finditer(
            r"Overfull \\hbox \(([\d.]+)pt too wide\)"
            r"(?: (?:detected at line (\d+)|in paragraph at lines ([\d\-]+)))?",
            text):
        found.append(("hbox", float(m.group(1)), owner(m.start()),
                      m.group(2) or m.group(3) or "?",
                      "single box" if m.group(2) else "prose",
                      box_text(text[m.end():m.end() + 4000])))
    for m in re.finditer(r"Overfull \\vbox \(([\d.]+)pt too high\)", text):
        found.append(("vbox", float(m.group(1)), owner(m.start()),
                      "?", "page body", ""))

    under_v = len(re.findall(r"Underfull \\vbox", text))
    under_h = len(re.findall(r"Underfull \\hbox", text))
    return found, under_h, under_v


def report(venue):
    res = scan(venue)
    if not res:
        return 0
    found, under_h, under_v = res
    col = COL_PT.get(venue, 234.5)
    print(f"  {venue}   (text width {col:.0f}pt)")
    if not found:
        print("     NOTHING runs outside the text block: 0 overfull hbox, 0 overfull vbox")
    else:
        found.sort(key=lambda r: -r[1])
        for kind, pt, f, where, what, txt in found:
            sev = "SEVERE" if pt > 50 else ("moderate" if pt > 15 else "minor")
            print(f"     {kind} {pt:>7.1f}pt ({pt / col * 100:.0f}% of width)  "
                  f"{f}  at {where}  {what}, {sev}")
            if txt:
                print(f"        text: {txt}")
    print(f"     cosmetic only: {under_h} underfull hbox, {under_v} underfull vbox "
          f"(loose spacing / float whitespace, NOT overflow)")
    print()
    return len([1 for k, pt, *_ in found if pt > 15])


if __name__ == "__main__":
    print("overflow check -- content running outside the text block\n")
    bad = 0
    for v in ("tmlr_submission", "acl_submission"):
        bad += report(v)
    if bad:
        print(f"  {bad} box(es) more than 15pt over. Fixes, by `what`:")
        print("     single box  -> a display equation, tabular or graphic is too wide.")
        print("                    Equations: break with \\\\ / use split or multline, or")
        print("                    move to a figure*/table* that spans both columns.")
        print("                    Tabulars: \\small, \\resizebox, or fewer columns.")
        print("     prose       -> an unbreakable token (a long \\texttt, URL or hyphenless")
        print("                    compound). Add \\- hyphenation points or \\sloppy locally.")
    else:
        print("  nothing meaningful outside the text block in either paper.")
    sys.exit(0)
