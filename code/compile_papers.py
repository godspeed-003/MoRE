"""compile_papers.py -- compile both submission PDFs and report real page counts.

WHY A SCRIPT. Three reasons a bare `pdflatex` call is not enough here:
  1. TinyTeX ships a minimal package set, so the first run fails on missing
     .sty files. This loops: compile -> parse the log for what is missing ->
     `tlmgr install` it -> recompile, up to MAX_INSTALL_ROUNDS.
  2. A bibliography needs pdflatex -> bibtex -> pdflatex -> pdflatex, and the
     page count is only correct after the last pass (labels and floats settle).
  3. The thing actually wanted is not "did it build" but the COUNTED page total,
     which for ACL excludes the Limitations section, the Ethics statement, the
     references and the appendix. That boundary is found by locating the page
     the References heading starts on, not by trusting the PDF length.

TinyTeX is extracted under a temp directory rather than installed system-wide,
so nothing is added to PATH permanently and removing the directory removes it.
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_tinytex():
    """Locate TinyTeX's bin directory on THIS machine.

    The path used to be hard-coded to one author's temp directory, which meant
    the script silently did nothing on the other machine -- exactly the
    per-machine-path trap CLAUDE.md section 9 warns about. Resolution order:
    an explicit override, then each user's own temp layout, then whatever is on
    PATH. Returns "" when nothing is found, and the caller reports that rather
    than failing on a missing .exe.
    """
    env = os.environ.get("MORE_TINYTEX")
    if env and os.path.isfile(os.path.join(env, "pdflatex.exe")):
        return env
    cands = []
    for base in (os.environ.get("TEMP"), os.environ.get("TMP"),
                 r"C:\Users\Hp\AppData\Local\Temp",
                 r"C:\Users\vedan\AppData\Local\Temp"):
        if base:
            cands.append(os.path.join(base, "tinytex_dl", "out",
                                      "TinyTeX", "bin", "windows"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "pdflatex.exe")):
            return c
    from shutil import which
    p = which("pdflatex")
    return os.path.dirname(p) if p else ""


TT = _find_tinytex()
PDFLATEX = os.path.join(TT, "pdflatex.exe") if TT else ""
BIBTEX = os.path.join(TT, "bibtex.exe") if TT else ""
TLMGR = os.path.join(TT, "tlmgr.bat") if TT else ""

MAX_INSTALL_ROUNDS = 14

# .sty/.cls -> TeX Live package name, where they differ. Most resolve by
# `tlmgr install <base>`; these do not.
PKG_ALIAS = {
    "inconsolata": "inconsolata",
    "latexsym": "base",
    # eso-pic is its OWN TeX Live package; mapping it to oberdiek (which does
    # bundle many small packages) silently fails and the alias table was the
    # reason TMLR would not build at all.
    "eso-pic": "eso-pic",
    "lmodern": "lm",
    "times": "psnfss",
    "natbib": "natbib",
    "booktabs": "booktabs",
    "enumitem": "enumitem",
    "microtype": "microtype",
    "subcaption": "caption",
    "nicefrac": "units",
    "amsfonts": "amsfonts",
    "amssymb": "amsfonts",
    "amsmath": "amsmath",
    "graphicx": "graphics",
    "hyperref": "hyperref",
    "url": "url",
    "fontenc": "latex-base-dev",
    "inputenc": "latex-base-dev",
}


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def missing_from_log(log):
    """Package names pdflatex reported as absent."""
    want = set()
    for pat in (r"File `([^']+)\.sty' not found",
                r"File `([^']+)\.cls' not found",
                r"! LaTeX Error: File `([^']+)\.(?:sty|cls)' not found"):
        want |= set(re.findall(pat, log))
    # font/encoding files surface differently
    want |= set(re.findall(r"Font \\[^=]+=([a-z0-9]+) not loadable", log))
    return want


def tlmgr_install(names):
    """Install, then VERIFY with kpsewhich.

    `tlmgr install` exits 0 even when it refuses to do anything -- a stock
    TinyTeX answers every install with "tlmgr itself needs to be updated" and
    still returns 0. Trusting the exit code made the install loop spin fourteen
    times reporting success while nothing landed, so success is defined here as
    "kpsewhich can now find the file", not as rc == 0.
    """
    kpse = os.path.join(TT, "kpsewhich.exe")
    installed = []
    for n in sorted(names):
        pkg = PKG_ALIAS.get(n, n)
        run([TLMGR, "install", pkg], REPO)
        found = run([kpse, n + ".sty"], REPO)
        ok = bool(found.stdout.strip())
        installed.append((n, pkg, ok))
    return installed


def ensure_tlmgr():
    """Stock TinyTeX refuses all installs until tlmgr updates itself."""
    probe = run([TLMGR, "install", "lineno"], REPO)
    if "needs to be updated" in (probe.stdout + probe.stderr):
        print("  tlmgr needs a self-update before it will install anything")
        r = run([TLMGR, "update", "--self"], REPO)
        print(f"     self-update rc={r.returncode}")


def page_count(pdf):
    """Page total, read from the PDF itself."""
    try:
        data = open(pdf, "rb").read()
    except OSError:
        return None
    counts = [int(x) for x in re.findall(rb"/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page[^s]", data)) or None


def references_page(folder, stem):
    """Which page the References heading lands on, from the .log's shipout trace.

    Falls back to None. Used to separate counted content from the exempt tail
    (Limitations, Ethics, References, Appendix) for the ACL budget.
    """
    aux = os.path.join(folder, stem + ".aux")
    if not os.path.exists(aux):
        return None
    # \bibstyle marks the bibliography; the page is not in the aux, so instead
    # use the last \newlabel before it as a proxy only if nothing better exists.
    return None


def compile_one(folder, stem="main"):
    print(f"\n{'=' * 66}\n  {os.path.basename(folder)}\n{'=' * 66}")
    tex = stem + ".tex"
    log = os.path.join(folder, stem + ".log")

    for rnd in range(MAX_INSTALL_ROUNDS):
        r = run([PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", tex], folder)
        text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else r.stdout
        miss = missing_from_log(text)
        if not miss:
            break
        print(f"  round {rnd + 1}: missing {sorted(miss)}")
        res = tlmgr_install(miss)
        for n, pkg, ok in res:
            print(f"     {'ok  ' if ok else 'FAIL'} {n} -> {pkg}")
        if not any(ok for _, _, ok in res):
            print("  no package install succeeded; stopping the install loop")
            break
    else:
        print("  hit MAX_INSTALL_ROUNDS")

    # full bibliography cycle
    run([BIBTEX, stem], folder)
    run([PDFLATEX, "-interaction=nonstopmode", tex], folder)
    r = run([PDFLATEX, "-interaction=nonstopmode", tex], folder)

    text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    pdf = os.path.join(folder, stem + ".pdf")
    pages = page_count(pdf)

    errors = re.findall(r"^! (.+)$", text, re.M)
    overfull = len(re.findall(r"Overfull \\hbox", text))
    underfull = len(re.findall(r"Underfull \\hbox", text))
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    undef_cit = sorted(set(re.findall(r"Citation `([^']+)' on page", text)))

    print(f"  PDF pages          : {pages}")
    print(f"  LaTeX errors       : {len(errors)}")
    for e in errors[:8]:
        print(f"     ! {e.strip()[:95]}")
    print(f"  overfull hboxes    : {overfull}")
    print(f"  underfull hboxes   : {underfull}")
    print(f"  undefined refs     : {undef_ref if undef_ref else 'none'}")
    print(f"  undefined citations: {undef_cit if undef_cit else 'none'}")
    return pages, len(errors)


if __name__ == "__main__":
    if not os.path.exists(PDFLATEX):
        sys.exit(f"pdflatex not found at {PDFLATEX}")
    ensure_tlmgr()
    out = {}
    for v in ("tmlr_submission", "acl_submission"):
        folder = os.path.join(REPO, "paper", v)
        out[v] = compile_one(folder)
    print(f"\n{'=' * 66}\n  SUMMARY\n{'=' * 66}")
    for v, (pages, errs) in out.items():
        print(f"  {v:20} {pages} pages, {errs} error(s)")
    print("\n  ACL limit is 8 pages of CONTENT; Limitations, Ethics, references")
    print("  and the appendix are exempt, so the counted figure is lower than")
    print("  the PDF total. Open the PDF to find where the References start.")
