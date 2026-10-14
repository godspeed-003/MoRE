# -*- coding: utf-8 -*-
"""make_architecture_figure.py -- the MoRE block diagram, as a PNG.

    python code/make_architecture_figure.py

Writes paper/fig_lang/fig_architecture.png, which is the directory
assemble_submissions.py copies figures FROM. Writing straight into the
submission trees would be undone by the next assemble run.

WHY THIS SCRIPT REPLACES THE PREVIOUS FIGURE. The hand-drawn PNG it supersedes
(paper/fig/More(3)(1).png) depicted the architecture this study RETIRED:

  * it labelled the router "noisy top-K", but the canonical router is top-1 with
    router_noise = none;
  * it said "ONE expert per token PER DEPTH" and looped the halt decision back
    into the router -- that is per-step re-routing, the T-LX.12 defect. The
    canonical path routes ONCE and persists the assignment for the whole
    trajectory, which is the specification the method section states and the
    single most load-bearing architectural claim in the paper;
  * it omitted the gate multiply g_{e*}, the only path carrying task gradient
    into the router (the argmax is not differentiated);
  * it omitted the ACT output aggregation o = sum_t w_t h^(t), the only path
    carrying task gradient into the halt head.

A reviewer reading the figure against the method section would have found it
contradicting the text on the paper's central claim.

COMPACT ON PURPOSE. This is a one-column float in a two-column venue, so its
HEIGHT is what it costs. The first version of this script ran 300 pt tall in a
655 pt column -- about 23% of a page -- while the ACL body was already over its
limit. Five rows instead of seven, side by side wherever the flow allows.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # noqa: E402

REPO = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUTS = [REPO / "paper" / "fig_lang" / "fig_architecture.png"]

INK = "#1a1a1a"
GREY = "#8a8a8a"
FILL = "#f2f2f2"
SEL = "#cfcfcf"
DIM = "#fbfbfb"


def box(ax, x, y, w, h, text, *, fc=FILL, ec=INK, lw=1.0, fs=6.4, weight=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, linespacing=1.4, fontweight=weight)


def arrow(ax, p, q, *, color=INK, lw=1.0, style="-|>", z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=color, zorder=z,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(3.35, 2.42), dpi=400)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    cx, LOOP = 50.0, 95.5

    # ---- Row A: tokens -> embed, side by side ------------------------
    box(ax, 20, 93, 30, 10, "tokens\n$B\\times S$")
    box(ax, 67, 93, 42, 10, "embed + position\n$h^{(0)}$")
    arrow(ax, (35.3, 93), (46.2, 93))

    # ---- Row B: the router, OUTSIDE the recursion --------------------
    arrow(ax, (67, 87.8), (67, 82.3))
    box(ax, cx, 76.5, 96, 11,
        "router, once per token\n"
        "$g=\\mathrm{softmax}(W_r h^{(0)})$,  $e^{\\star}=\\arg\\max_e g_e$",
        lw=1.3)

    # ---- Row C: expert bank, only e* evaluated -----------------------
    names = ["$\\mathrm{FFN}_1$", "$\\mathrm{FFN}_2$", "$e^{\\star}$",
             "$\\mathrm{FFN}_4$", "$\\cdots$", "$\\mathrm{FFN}_E$"]
    bw, gap = 14.2, 1.6
    x0 = cx - (len(names) * bw + (len(names) - 1) * gap) / 2 + bw / 2
    ex = x0 + 2 * (bw + gap)
    arrow(ax, (cx, 71.0), (cx, 65.6))
    for i, nm in enumerate(names):
        chosen = (i == 2)
        box(ax, x0 + i * (bw + gap), 60.0, bw, 9.0, nm,
            fc=SEL if chosen else DIM, ec=INK if chosen else GREY,
            lw=1.5 if chosen else 0.7, fs=6.2,
            weight="bold" if chosen else None)
    ax.text(99, 51.0, "top-1, no noise; $e^{\\star}$ persisted",
            ha="right", va="center", fontsize=5.6, color=GREY)
    arrow(ax, (ex, 55.5), (ex, 47.0), lw=1.4)

    # ---- Row D: the recursion block ----------------------------------
    box(ax, cx, 36.0, 96, 20,
        "$h^{(t+1)}=\\mathrm{LN}\\!\\left(h^{(t)}"
        "+g_{e^{\\star}}\\cdot\\mathrm{FFN}_{e^{\\star}}(h^{(t)})\\right)$\n\n"
        "halt head:  $p_t=\\sigma(w_{e^{\\star}}^{\\top}h^{(t)})$", lw=1.3)
    ax.text(2.5, 23.5, "same $e^{\\star}$ weights every step",
            ha="left", va="center", fontsize=5.6, color=GREY)

    # the gate multiply: the only differentiable path into the router
    gx = 2.2
    arrow(ax, (2.0, 71.0), (gx, 71.0), style="-", color=GREY, lw=0.8)
    arrow(ax, (gx, 71.0), (gx, 36.0), style="-", color=GREY, lw=0.8)
    arrow(ax, (gx, 36.0), (cx - 48.0, 36.0), color=GREY, lw=0.8)
    ax.text(3.6, 54.0, "$g_{e^{\\star}}$", ha="left", va="center",
            fontsize=6.4, color=INK)

    # ---- Row E: exit test -> output, side by side --------------------
    arrow(ax, (cx, 26.0), (cx, 19.6))
    box(ax, 27, 12.0, 46, 13, "exit if $c_t+p_t>1-\\epsilon$\nor $t=T$", fs=6.2)
    box(ax, 77, 12.0, 40, 13,
        "$o=\\sum_t w_t h^{(t)}$\n$\\rightarrow$ logits", lw=1.2)
    arrow(ax, (50.2, 12.0), (56.8, 12.0))

    # the single recursion return path
    arrow(ax, (27, 5.5), (27, 2.0), style="-", color=GREY, lw=0.9)
    arrow(ax, (27, 2.0), (LOOP, 2.0), style="-", color=GREY, lw=0.9)
    arrow(ax, (LOOP, 2.0), (LOOP, 36.0), style="-", color=GREY, lw=0.9)
    arrow(ax, (LOOP, 36.0), (cx + 48.0, 36.0), color=GREY, lw=0.9)
    ax.text(LOOP - 1.5, 23.5, "no: $t\\!\\leftarrow\\!t\\!+\\!1$", ha="right",
            va="center", fontsize=5.6, color=GREY)

    # The T=1 / E=1 reductions are stated in the caption rather than drawn: in
    # the compact layout the only free space left collided with the exit box.

    fig.subplots_adjust(left=0.004, right=0.996, top=0.998, bottom=0.002)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, facecolor="white")
        print(f"wrote {out.relative_to(REPO)}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
