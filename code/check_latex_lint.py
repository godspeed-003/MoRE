"""check_latex_lint.py -- source-level defects that compile silently.

Motivated by one that reached both venue trees: `MoRE$-$MoE` became `MoREhBc`
and `$-0.147$` became `hBc0.147$`, because an edit was applied through a
DOUBLE-QUOTED shell string and bash expanded `$-` to the current shell option
flags (`hBc`). LaTeX then set the surrounding prose in math italic with no
inter-word spaces -- visibly broken output, zero errors, zero warnings.

`$` followed by `-` is the dangerous case because `$-` is a real bash variable,
but `$0`..`$9` (positional parameters) expand to nothing, which is worse: they
vanish without a trace. They are also NOT detectable here -- `$7.9\\%$` and
`$0.1216$` are ordinary math spans beginning with a digit, and flagging them gave
dozens of false positives per real defect. That check was removed rather than
tuned, because the honest statement is that silent positional expansion cannot be
found after the fact. That is the argument for never editing .tex through a shell
string: write a .py file and run it.

Checks, in rough order of how badly each would embarrass us:
  shell-expansion fingerprints  -- what `$-` leaves behind (bash option flags)
  unbalanced $                  -- per file, ignoring escaped \\$
  markdown leakage              -- `**bold**` from drafting in prose
  tabs                          -- a literal tab inside a .tex line
  non-ASCII                     -- pdflatex chokes or silently mis-sets it
  unresolved assembler markers  -- an ACLMOVE/ACLONLY/TMLRONLY that survived
"""
import io
import os
import re
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRS = [
    os.path.join(REPO, "paper", "common"),
    os.path.join(REPO, "paper", "tmlr_submission", "sections"),
    os.path.join(REPO, "paper", "acl_submission", "sections"),
]
MAINS = [
    os.path.join(REPO, "paper", "tmlr_submission", "main.tex"),
    os.path.join(REPO, "paper", "acl_submission", "main.tex"),
]

# bash's `$-` yields the current option flags; these are what we have actually
# seen, plus the common interactive forms.
SHELL_FLAGS = ("hBc", "himBHc", "himBH", "hB")


def dollars(line):
    """Count real math delimiters, ignoring \\$ (a literal dollar sign)."""
    return len(re.findall(r"(?<!\\)\$", line))


def check(path):
    bad = []
    raw = io.open(path, encoding="utf-8", errors="replace").read()
    lines = raw.split("\n")

    total = 0
    for i, ln in enumerate(lines, 1):
        # longest first, and stop at the first hit: 'hB' is a substring of
        # 'hBc', so reporting every match double-counts one defect.
        for flag in sorted(SHELL_FLAGS, key=len, reverse=True):
            if flag in ln:
                bad.append((i, "SHELL EXPANSION", f"contains '{flag}' "
                            f"(bash $- flags); a $-prefixed token was eaten"))
                break
        for m in re.finditer(r"(?<!\\)\$-", ln):
            # `$-` is the one shell variable that leaves a visible trace. A
            # legitimate `$-0.147$` is indistinguishable from damage at the
            # character level, so this only reports when the flag string is
            # ALSO present on the line (handled above) -- kept as a no-op hook
            # documenting why the general case cannot be checked.
            pass
        if "**" in ln:
            bad.append((i, "MARKDOWN", "'**' -- markdown bold, use \\textbf{}"))
        if "\t" in ln:
            bad.append((i, "TAB", "literal tab character"))
        for ch in ln:
            if ord(ch) > 127:
                bad.append((i, "NON-ASCII",
                            f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"))
                break
        for mk in re.findall(r">>>(?:ACLMOVE|ACLONLY|TMLRONLY)", ln):
            if "sections" in path:      # markers are expected in common/
                bad.append((i, "MARKER", f"unresolved {mk} in a generated tree"))
        total += dollars(ln)

    if total % 2:
        bad.append((0, "MATH BALANCE",
                    f"{total} unescaped '$' in the file -- odd, so one math "
                    f"span is unterminated"))
    return bad


if __name__ == "__main__":
    targets = []
    for d in DIRS:
        if os.path.isdir(d):
            targets += [os.path.join(d, f) for f in sorted(os.listdir(d))
                        if f.endswith(".tex")]
    targets += [m for m in MAINS if os.path.exists(m)]

    n = 0
    for p in targets:
        bad = check(p)
        if bad:
            n += len(bad)
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            print(f"  {rel}")
            for line, kind, msg in bad:
                where = f"line {line}" if line else "file"
                print(f"     {kind:16} {where:10} {msg}")
    print()
    if n:
        print(f"  {n} issue(s). Shell-expansion and math-balance findings are")
        print("  real defects that compile without any warning -- fix before building.")
        sys.exit(1)
    print(f"  clean: {len(targets)} .tex files, no shell-expansion damage,")
    print("  balanced math, no markdown leakage, no tabs, no non-ASCII.")
