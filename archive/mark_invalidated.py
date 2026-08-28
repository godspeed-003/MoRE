"""
mark_invalidated.py — stamp the archived pre-finalization record as invalidated.

plan.md Phase 0 / 2.1 requires historical result artifacts to carry an explicit
header:

    INVALIDATED FOR SCIENTIFIC COMPARISON
    Reason: confirmed target leakage and/or broken halting/routing implementation.

and "Keep the history. Do not delete evidence."

Strategy (evidence-preserving, grep-verifiable, idempotent):

  * text documents (.md, .py, .txt, extensionless notes)
        -> the header is PREPENDED in-place, commented where the file is code.
           Lossless: the original content follows verbatim.

  * data artifacts (.csv, .tsv, .json, .pt, .wandb, logs)
        -> left BYTE-IDENTICAL. A sibling `<name>.INVALIDATED.txt` carries the
           header instead, so no measurement file is ever mutated.

  * every archive directory
        -> gets an INVALIDATED.md notice.

Re-running is safe: files already carrying the header are skipped.

Usage:
    python archive/mark_invalidated.py            # apply
    python archive/mark_invalidated.py --check    # verify only, exit 1 if gaps
"""

from __future__ import annotations

import argparse
import os
import sys

MARKER = "INVALIDATED FOR SCIENTIFIC COMPARISON"
REASON = ("Reason: confirmed target leakage and/or broken halting/routing "
          "implementation.")

_ARCHIVE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_ARCHIVE_DIR)

# Text documents get the header prepended in-place.
PREPEND_EXT = {".md", ".txt", ".rst"}
COMMENT_EXT = {".py"}
# Extensionless files in these directories are treated as text notes.
TEXT_NO_EXT = {"final_plan"}

# Data artifacts are never modified; they get a sibling marker file.
SIBLING_EXT = {".csv", ".tsv", ".json", ".pt", ".yaml", ".yml", ".log", ".wandb"}

# Directories whose individual files are not marked one-by-one (raw W&B run
# payloads: hundreds of machine-written files). The directory notice covers them.
DIR_NOTICE_ONLY = {"logs", "files", "media", "tmp"}

SKIP_NAMES = {"mark_invalidated.py", "INVALIDATED.md"}


def _plain_header() -> str:
    return f"{MARKER}\n{REASON}\n"


def _md_header() -> str:
    return (
        f"> **{MARKER}**\n"
        f">\n"
        f"> {REASON}\n"
        f">\n"
        f"> Retained as research history only. Superseded by `plan.md`,\n"
        f"> `updated_rules.md`, `updated_objective.md`. Nothing in this file may\n"
        f"> be cited, exported, or used as a baseline for scientific comparison.\n"
        f"\n---\n\n"
    )


def _comment_header() -> str:
    return (
        f"# {MARKER}\n"
        f"# {REASON}\n"
        f"#\n"
        f"# Retained as research history only. Do not run against the canonical\n"
        f"# pipeline; paths and assumptions here predate plan.md Phase 0.\n"
        f"\n"
    )


def _has_marker(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return MARKER in f.read(4096)
    except Exception:
        return False


def _prepend(path: str, header: str, dry: bool) -> bool:
    if _has_marker(path):
        return False
    if dry:
        return True
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        body = f.read()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + body)
    return True


def _sibling(path: str, dry: bool) -> bool:
    target = path + ".INVALIDATED.txt"
    if os.path.exists(target):
        return False
    if dry:
        return True
    with open(target, "w", encoding="utf-8") as f:
        f.write(
            _plain_header()
            + "\n"
            f"Applies to: {os.path.basename(path)}\n\n"
            "This measurement file is preserved byte-for-byte as evidence and is\n"
            "NOT modified. It is invalid for scientific comparison: it was\n"
            "produced by a pipeline with confirmed target leakage, an untrained\n"
            "halting mechanism, dense (not Top-1) routing, and a balance loss\n"
            "that dominated and scaled with depth.\n\n"
            "Do not load this file into any results table, figure, or baseline.\n"
        )
    return True


def _dir_notice(directory: str, dry: bool) -> bool:
    target = os.path.join(directory, "INVALIDATED.md")
    if os.path.exists(target):
        return False
    if dry:
        return True
    rel = os.path.relpath(directory, _REPO_DIR).replace("\\", "/")
    with open(target, "w", encoding="utf-8") as f:
        f.write(
            f"# {MARKER}\n\n"
            f"{REASON}\n\n"
            f"**Scope: every file in `{rel}/` and all subdirectories.**\n\n"
            "## Why\n\n"
            "The pipeline that produced these artifacts had six confirmed defects:\n\n"
            "1. **Target leakage** — each computation step's `result` was written\n"
            "   into that step's input feature row, so the regression target was\n"
            "   present in the input for 100% of validation records. Reported\n"
            "   losses are worse than a closed-form copy baseline.\n"
            "2. **Untrained halting** — the halting loss had no `grad_fn`; every\n"
            "   halt-head parameter received `grad = None`. Depth allocation was\n"
            "   never learned.\n"
            "3. **Dense routing** — every expert was evaluated and blended, so\n"
            "   these are not Top-1 sparse MoE/MoRE results.\n"
            "4. **Objective scaling** — the balance term reached ~146x the task\n"
            "   loss and accumulated per recursion depth, so total loss was\n"
            "   dominated by, and confounded with, depth.\n"
            "5. **Dimension hard-coding** — 7-class heads in 6-expert runs.\n"
            "6. **Input mismatch** — MoE / MoR / MoRE received different feature\n"
            "   scalings, and the oracle expert label occupied input slot 0.\n\n"
            "## What this means\n\n"
            "- These numbers are **research history**, not results.\n"
            "- They must not appear in any table, figure, or baseline.\n"
            "- The exporter must refuse them (`experiment_group` filter).\n"
            "- They are kept because deleting evidence is prohibited\n"
            "  (`updated_rules.md` 12).\n\n"
            "## Authoritative replacements\n\n"
            "`plan.md`, `updated_rules.md`, `updated_objective.md`, and — once the\n"
            "corrected canonical matrix has run — `runs/<experiment_id>/` plus\n"
            "`results.md`.\n"
        )
    return True


def walk(root: str, dry: bool) -> dict:
    stats = {"prepended": [], "siblings": [], "notices": [], "untouched": []}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()

        # Directory notices only at the archive root and its immediate children
        # (one per artifact collection). Writing one into every nested W&B
        # media/ subdirectory would bury the signal in hundreds of copies.
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth <= 1 and _dir_notice(dirpath, dry):
            stats["notices"].append(dirpath)

        if os.path.basename(dirpath) in DIR_NOTICE_ONLY:
            continue

        for name in sorted(filenames):
            if name in SKIP_NAMES or name.endswith(".INVALIDATED.txt"):
                continue
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()

            if ext in COMMENT_EXT:
                if _prepend(path, _comment_header(), dry):
                    stats["prepended"].append(path)
                else:
                    stats["untouched"].append(path)
            elif ext in PREPEND_EXT or (ext == "" and name in TEXT_NO_EXT):
                if _prepend(path, _md_header(), dry):
                    stats["prepended"].append(path)
                else:
                    stats["untouched"].append(path)
            elif ext in SIBLING_EXT or ext == "":
                if _sibling(path, dry):
                    stats["siblings"].append(path)
                else:
                    stats["untouched"].append(path)
            else:
                if _sibling(path, dry):
                    stats["siblings"].append(path)
                else:
                    stats["untouched"].append(path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report what is missing; change nothing; exit 1 if gaps")
    ap.add_argument("--root", default=os.path.join(_ARCHIVE_DIR, "pre_finalization"))
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"[ERROR] archive root not found: {a.root}", file=sys.stderr)
        return 1

    stats = walk(a.root, dry=a.check)

    verb = "WOULD mark" if a.check else "marked"
    print(f"[archive] root: {a.root}")
    print(f"[archive] {verb} {len(stats['prepended'])} text docs (header prepended)")
    print(f"[archive] {verb} {len(stats['siblings'])} data files (sibling marker)")
    print(f"[archive] {verb} {len(stats['notices'])} directories (INVALIDATED.md)")
    print(f"[archive] already marked: {len(stats['untouched'])}")

    for key in ("prepended", "siblings", "notices"):
        for p in stats[key][:200]:
            print(f"   [{key}] {os.path.relpath(p, _REPO_DIR)}")

    if a.check:
        gaps = len(stats["prepended"]) + len(stats["siblings"]) + len(stats["notices"])
        if gaps:
            print(f"\n[FAIL] {gaps} archive artifact(s) are not marked invalidated.")
            return 1
        print("\n[OK] every archive artifact is marked invalidated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
