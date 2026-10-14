"""measure_move_blocks.py -- how much body does each ACLMOVE block cost?

TMLR's main content measures ~14.4 pages against a 12-page recommendation, so
~2.4 pages must leave the body. The 11 ACLMOVE blocks are the material already
judged appendix-able, each with a self-contained short replacement, so the
question is which subset to move for TMLR -- and that needs the sizes measured
rather than guessed.

Net saving per block = full block text - its short replacement, since the
replacement stays in the body. Reported in characters and in estimated TMLR
pages (single column, 468pt: ~3,600 characters of body text per page, calibrated
against the 14.4-page measurement).
"""
import io
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(HERE, "paper", "common")

# calibration: TMLR body chars / measured main-content pages
CHARS_PER_PAGE = 3600


def blocks():
    out = []
    for fn in sorted(os.listdir(COMMON)):
        if not fn.endswith(".tex"):
            continue
        src = io.open(os.path.join(COMMON, fn), encoding="utf-8").read()
        for m in re.finditer(
                r"^% >>>ACLMOVE:([A-Za-z0-9_]+)\|(.*?)$(.*?)^% <<<ACLMOVE:\1\s*$",
                src, re.S | re.M):
            name, repl, full = m.group(1), m.group(2), m.group(3)
            # strip comment-only lines from the measured body text
            live = "\n".join(ln for ln in full.split("\n")
                             if not ln.lstrip().startswith("%"))
            out.append({
                "name": name,
                "file": fn,
                "full": len(live.strip()),
                "repl": len(repl.strip()),
                "net": len(live.strip()) - len(repl.strip()),
            })
    return out


def main():
    bs = sorted(blocks(), key=lambda b: -b["net"])
    total_body = 0
    for fn in sorted(os.listdir(COMMON)):
        if fn.endswith(".tex"):
            s = io.open(os.path.join(COMMON, fn), encoding="utf-8").read()
            total_body += len("\n".join(
                ln for ln in s.split("\n") if not ln.lstrip().startswith("%")))

    print(f"shared body: {total_body:,} live characters across "
          f"{len([f for f in os.listdir(COMMON) if f.endswith('.tex')])} files")
    print(f"calibration: ~{CHARS_PER_PAGE:,} chars/page -> "
          f"~{total_body / CHARS_PER_PAGE:.1f} pages of prose\n")
    print(f"  {'block':14} {'file':18} {'full':>7} {'repl':>6} {'net':>7} "
          f"{'~pages':>7}")
    print("  " + "-" * 62)
    run = 0.0
    for b in bs:
        pages = b["net"] / CHARS_PER_PAGE
        run += pages
        print(f"  {b['name']:14} {b['file']:18} {b['full']:>7,} "
              f"{b['repl']:>6,} {b['net']:>7,} {pages:>6.2f}")
    print("  " + "-" * 62)
    print(f"  {'ALL 11':14} {'':18} {'':>7} {'':>6} "
          f"{sum(b['net'] for b in bs):>7,} {run:>6.2f}")
    print("\n  cumulative, largest first (target ~2.4 pages):")
    c = 0.0
    for b in bs:
        c += b["net"] / CHARS_PER_PAGE
        flag = "  <-- reaches target" if c >= 2.4 else ""
        print(f"     + {b['name']:14} {c:>5.2f} pages{flag}")
        if c >= 2.4:
            break


if __name__ == "__main__":
    main()
