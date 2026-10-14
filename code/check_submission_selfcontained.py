# -*- coding: utf-8 -*-
r"""Zip-readiness check for a submission folder.

    python code/check_submission_selfcontained.py paper/tmlr_submission

Confirms the folder can be zipped and dropped into Overleaf with nothing else:
every \input and \includegraphics resolves inside it, no path escapes it, and
the bibliography it names is present. The assembler's own check runs against
paper/common/ before the copy; this one runs against what you would actually
upload.
"""
import glob
import os
import re
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "paper/tmlr_submission"
os.chdir(root)

tex = {}
for f in ["main.tex"] + sorted(glob.glob("sections/*.tex")):
    tex[f] = open(f, encoding="utf-8").read()

missing, outside = [], []
for f, s in tex.items():
    body = "\n".join(l.split("%")[0] for l in s.splitlines())
    for m in re.finditer(r"\\input\{([^}]+)\}", body):
        t = m.group(1)
        if not any(os.path.exists(c) for c in (t, t + ".tex")):
            missing.append(f"{f}: input {t}")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        t = m.group(1)
        if not (os.path.exists(os.path.join("fig", t)) or os.path.exists(t)):
            missing.append(f"{f}: graphic {t}")
    for m in re.finditer(r"\.\./|[A-Za-z]:[\\/]", body):
        outside.append(f"{f}: {body[max(0, m.start()-40):m.start()+25]!r}")

bib = re.findall(r"\\bibliography\{([^}]+)\}", tex["main.tex"])
sty = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex["main.tex"])
local_sty = [s for s in sum((b.split(",") for b in sty), [])
             if os.path.exists(s.strip() + ".sty")]

print(f"  files: main.tex + {len(tex)-1} sections, "
      f"{len(glob.glob('fig/*'))} figures")
print(f"  unresolved inputs/graphics : {missing if missing else 'NONE'}")
print(f"  paths escaping the folder  : {outside if outside else 'NONE'}")
print(f"  bibliography               : {bib} present={all(os.path.exists(b+'.bib') for b in bib)}")
print(f"  local style files shipped  : {local_sty}")
print(f"  build artefacts present    : "
      f"{[f for f in ('main.aux','main.bbl','main.log','main.blg') if os.path.exists(f)]}")
sys.exit(1 if (missing or outside) else 0)
