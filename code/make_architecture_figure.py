# -*- coding: utf-8 -*-
"""make_architecture_figure.py -- the MoRE block diagram, as a PNG.

    python code/make_architecture_figure.py

Writes fig_architecture.png into both submission trees.

WHY THIS SCRIPT REPLACES THE PREVIOUS FIGURE. The hand-drawn PNG it supersedes
depicted the architecture this study RETIRED, not the one it reports:

  * it labelled the router "noisy top-K", but the canonical router is top-1 with
    `router_noise = none` (CLAUDE.md 2);
  * it said "ONE expert per token PER DEPTH" and looped the halt decision back
    into the router -- that is per-step re-routing, the T-LX.12 defect. The
    canonical path routes ONCE and persists the assignment for the whole
    trajectory, which is the specification the paper's method section states and
    the single most load-bearing architectural claim in the paper;
  * it omitted the gate multiply g_{e*}, which is the only path carrying task
    gradient into the router (the argmax is not differentiated);
  * it omitted the ACT output aggregation o = sum_t w_t h^(t), which is the only
    path carrying task gradient into the halt head.

A reviewer who read the figure against the method section would have found it
contradicting the text on the paper's central claim.

LAYOUT. Vertical and narrow on purpose: T-LA.5 moved this from `figure*` to a
plain one-column `figure` to save page budget, so at ACL it renders about 3.2 in
wide. A horizontal flow would be illegible at that width.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon  # noqa: E402

REPO = Path(os.path.dirname(os.path.abspath(__file__))).parent
# Written into paper/fig_lang/, which is the directory assemble_submissions.py
# copies figures FROM. Writing straight into the submission trees would be undone
# by the next assemble run.
OUTS = [REPO / "paper" / "fig_lang" / "fig_architecture.png"]

INK = "#1a1a1a"
GREY = "#8a8a8a"
FILL = "#f2f2f2"
SEL = "#cfcfcf"
DIM = "#fbfbfb"


def box(ax, x, y, w, h, text, *, fc=FILL, ec=INK, lw=1.0, fs=7.2, weight=None,
        rounding=0.022):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=3, linespacing=1.45, fontweight=weight)


def arrow(ax, p, q, *, color=INK, lw=1.1, style="-|>", ls="-", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 lw=lw, color=color, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(3.35, 4.6), dpi=400)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cx = 50.0                      # main spine
    LOOP = 95.0                    # x of the single recursion return path

    # ---- 1. input ----------------------------------------------------
    box(ax, cx, 96.0, 40, 6.0, "tokens  $B\\times S$")
    arrow(ax, (cx, 93.0), (cx, 89.8))

    # ---- 2. embedding ------------------------------------------------
    box(ax, cx, 86.6, 52, 6.0, "embed + position  $\\rightarrow$  $h^{(0)}$")
    arrow(ax, (cx, 83.6), (cx, 80.2))

    # ---- 3. the router, OUTSIDE the loop -----------------------------
    box(ax, cx, 74.8, 80, 9.2,
        "router  (once per token)\n"
        "$g=\\mathrm{softmax}(W_r h^{(0)})$,   "
        "$e^{\\star}=\\arg\\max_e g_e$", lw=1.3)
    ax.text(9.0, 68.4, "top-1, no noise; $e^{\\star}$ persisted",
            ha="left", va="center", fontsize=5.9, color=GREY)

    # ---- 4. expert bank: only e* is evaluated ------------------------
    names = ["$\\mathrm{FFN}_1$", "$\\mathrm{FFN}_2$", "$e^{\\star}$",
             "$\\mathrm{FFN}_4$", "$\\cdots$", "$\\mathrm{FFN}_E$"]
    bw, gap = 11.6, 1.5
    total = len(names) * bw + (len(names) - 1) * gap
    x0 = cx - total / 2 + bw / 2
    ex_star = x0 + 2 * (bw + gap)
    arrow(ax, (cx, 70.2), (cx, 66.6))
    for i, nm in enumerate(names):
        xi = x0 + i * (bw + gap)
        chosen = (i == 2)
        box(ax, xi, 62.6, bw, 6.6, nm,
            fc=SEL if chosen else DIM, ec=INK if chosen else GREY,
            lw=1.5 if chosen else 0.7, fs=6.4,
            weight="bold" if chosen else None)
    # caption sits RIGHT of the descending arrow (which leaves from e*), so the
    # two cannot collide.
    ax.text(99.0, 56.2, "$E$ experts; only $e^{\\star}$ runs",
            ha="right", va="center", fontsize=5.9, color=GREY)
    arrow(ax, (ex_star, 59.3), (ex_star, 53.0), lw=1.4)

    # ---- 5. the recursion block --------------------------------------
    box(ax, cx, 43.6, 86, 17.2,
        "$h^{(t+1)}=\\mathrm{LN}\\!\\left(h^{(t)}"
        "+g_{e^{\\star}}\\cdot\\mathrm{FFN}_{e^{\\star}}(h^{(t)})\\right)$\n\n\n"
        "halt head:   $p_t=\\sigma(w_{e^{\\star}}^{\\top}h^{(t)})$", lw=1.3)
    # Weight tying stated OUTSIDE the block and left-aligned: inside it collided
    # with the halt-head line, and centred it collided with the arrow to the
    # halting test.
    ax.text(4.0, 32.2, "same $e^{\\star}$ weights every step",
            ha="left", va="center", fontsize=5.8, color=GREY, zorder=3)

    # the gate multiply: the only differentiable path into the router
    gx = 3.0
    arrow(ax, (cx - 40.0, 71.6), (gx, 71.6), style="-", color=GREY, lw=0.9)
    arrow(ax, (gx, 71.6), (gx, 47.0), style="-", color=GREY, lw=0.9)
    arrow(ax, (gx, 47.0), (cx - 43.0, 47.0), color=GREY, lw=0.9)
    ax.text(gx + 1.2, 58.5, "$g_{e^{\\star}}$", ha="left", va="center",
            fontsize=6.8, color=INK)

    arrow(ax, (cx, 35.0), (cx, 29.8))

    # ---- 6. the halting decision -------------------------------------
    dw, dh, dy = 29.0, 7.2, 25.6
    ax.add_patch(Polygon([[cx, dy + dh], [cx + dw, dy], [cx, dy - dh],
                          [cx - dw, dy]], closed=True, fc=FILL, ec=INK,
                         lw=1.0, zorder=2))
    ax.text(cx, dy, "$c_t+p_t>1-\\epsilon$\nor  $t=T$", ha="center",
            va="center", fontsize=6.4, color=INK, zorder=3, linespacing=1.4)
    ax.text(cx - dw - 1.5, dy - 4.6, "yes", ha="right", va="center",
            fontsize=6.0, color=GREY)

    # ONE return path: "no" -> right -> up -> back into the block.
    arrow(ax, (cx + dw, dy), (LOOP, dy), style="-", lw=1.0)
    arrow(ax, (LOOP, dy), (LOOP, 43.6), style="-", lw=1.0)
    arrow(ax, (LOOP, 43.6), (cx + 43.0, 43.6), lw=1.0)
    ax.text(LOOP - 1.0, 33.0, "no:  $t\\!\\leftarrow\\!t\\!+\\!1$", ha="right",
            va="center", fontsize=5.9, color=GREY)

    arrow(ax, (cx, dy - dh), (cx, 14.4))

    # ---- 7. output ----------------------------------------------------
    box(ax, cx, 10.6, 72, 7.2,
        "$o=\\sum_t w_t\\,h^{(t)}$   $\\rightarrow$   logits", lw=1.2)
    ax.text(cx, 4.6,
            "$T=1\\Rightarrow$ MoE      $E=1\\Rightarrow$ MoR",
            ha="center", va="center", fontsize=6.4, color=INK)
    ax.text(cx, 0.8, "halted states are frozen, not recomputed",
            ha="center", va="center", fontsize=5.7, color=GREY)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.998, bottom=0.002)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, facecolor="white")
        print(f"wrote {out.relative_to(REPO)}  ")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
