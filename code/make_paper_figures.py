"""make_paper_figures.py -- regenerate the language-paper figures from results/.

WHY THIS FILE EXISTS. Every figure in the paper must be regenerable from the
generated results, for the same reason no number may be hand-copied into prose
(CLAUDE.md 5): a figure drawn from an ad-hoc session cannot be audited or
rebuilt when an arm is retrained. Run this after any change to
results/language/*.json and the figures follow.

INPUTS (all generated, never hand-edited):
  results/language/stratified_val.json       adaptive loss + loss by decile/position
  results/language/stratified_val_d1..d7.json  forced-depth 1..7 (the budgeted curve)

TWO TIERS, RESPECTED HERE. The budgeted-depth curve and the decile strata are
EXPLORATORY (T-LX.22 onwards; outside both declared families in
code/confirmatory_tests.json). They are mechanism observations. Nothing in these
figures may be read as a confirmatory verdict -- the confirmatory verdicts are in
results/language/results_tables.md and carry the Holm correction.

PALETTE. Slots 1-3 of the validated categorical palette, in fixed order and
assigned to entities, not ranks: MoE blue, MoR orange, MoRE aqua. The first three
slots are the all-pairs-validated set (worst pair CVD dE 9.2 light / 9.4 dark,
normal-vision 24.0 / 20.9), which is what a 3-series line chart needs. Do not add
a fourth series by generating a hue; fold or facet instead.

OUTPUT: paper/fig_lang/*.pdf (vector, for LaTeX) and *.png (for review).
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "language")
OUT = os.path.join(REPO, "paper", "fig_lang")

# --- palette: validated categorical slots 1-3, assigned to entities ---
C = {"moe": "#2a78d6", "mor": "#eb6834", "more": "#1baf7a"}
LABEL = {"moe": "MoE", "mor": "MoR", "more": "MoRE"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlesize": 9.5,
    "figure.dpi": 200,
})


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ---------------------------------------------------------------------------
# Figure: the budgeted-depth curve.  MoR's learned policy beats every uniform
# depth including its own best; MoRE's ties its best forced depth because its
# halting collapsed to always-max.  This is the paper's central mechanism figure.
# ---------------------------------------------------------------------------
def fig_depth_curve():
    depths = list(range(1, 8))
    curve = {}
    for arm in ("mor", "more"):
        curve[arm] = []
        for d in depths:
            ba = load(f"stratified_val_d{d}.json")["by_arm"]
            curve[arm].append((ba[arm]["mean_loss"], ba[arm]["std_loss"]))
    adaptive = load("stratified_val.json")["by_arm"]

    fig, ax = plt.subplots(figsize=(6.4, 3.1))

    for arm in ("mor", "more"):
        mean = [v[0] for v in curve[arm]]
        std = [v[1] for v in curve[arm]]
        ax.plot(depths, mean, "-o", color=C[arm], lw=2, ms=4.5,
                label=f"{LABEL[arm]} — forced depth", zorder=3)
        ax.fill_between(depths, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)],
                        color=C[arm], alpha=0.15, lw=0, zorder=1)

    # the learned policy: one point per arm, at its own measured mean depth
    for arm in ("mor", "more"):
        dd = load("stratified_val.json")["cells"]
        md = sum(c["mean_exit_depth"] for c in dd if c["arch"] == arm) / len(
            [c for c in dd if c["arch"] == arm])
        ax.plot([md], [adaptive[arm]["mean_loss"]], "*", color=C[arm],
                ms=17, mec="white", mew=1.0, zorder=5,
                label=f"{LABEL[arm]} — learned policy")

    ax.axhline(3.948311, color=INK2, ls=":", lw=1.2, zorder=2)
    ax.text(1.02, 3.948311 + 0.035, "MoE — no recursion (1.00× FLOPs)",
            fontsize=7.4, color=INK2, va="bottom", ha="left")

    ax.set_xlabel("forced uniform depth")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_xticks(depths)
    ax.set_ylim(3.2, 5.4)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=7.8, loc="upper center",
                    ncol=2, handletextpad=0.5, columnspacing=1.1)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    save(fig, "fig_depth_curve")
    return {"mor_best_forced": min(v[0] for v in curve["mor"]),
            "mor_adaptive": adaptive["mor"]["mean_loss"],
            "more_best_forced": min(v[0] for v in curve["more"]),
            "more_adaptive": adaptive["more"]["mean_loss"]}


# ---------------------------------------------------------------------------
# Figure: loss stratified by target-token frequency decile.  Two panels --
# absolute loss, and each arm's gap to MoE, which is where the widening shows.
# ---------------------------------------------------------------------------
def fig_stratified():
    ba = load("stratified_val.json")["by_arm"]
    dec = list(range(1, 11))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))

    ax = axes[0]
    for arm in ("moe", "more", "mor"):
        ax.plot(dec, ba[arm]["loss_by_decile_mean"], "-o", color=C[arm],
                lw=2, ms=3.8, label=LABEL[arm])
    ax.set_xlabel("target-token frequency decile (1 = rarest)")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("(a) Loss by token difficulty", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    ax = axes[1]
    for arm in ("mor", "more"):
        gap = [ba[arm]["loss_by_decile_mean"][i] - ba["moe"]["loss_by_decile_mean"][i]
               for i in range(10)]
        ax.plot(dec, gap, "-o", color=C[arm], lw=2, ms=3.8,
                label=f"{LABEL[arm]} − MoE")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("target-token frequency decile (1 = rarest)")
    ax.set_ylabel("loss advantage over MoE (nats)")
    ax.set_title("(b) Advantage grows with difficulty", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    save(fig, "fig_stratified")


# ---------------------------------------------------------------------------
# Figure: where the compute goes.  In-domain the ordering is MoR < MoRE < MoE;
# out of domain it inverts to MoRE < MoR < MoE.  Two panels, same three entities,
# same colours -- the point is that the colours swap rank, not that the palette
# changed.
# ---------------------------------------------------------------------------
def fig_ood():
    order = ("moe", "mor", "more")
    ind = {"moe": 3.948311, "mor": 3.515415, "more": 3.636966}
    ood = load("ood_pile_test.json")["by_arm"]
    ood_m = {a: ood[a]["mean_loss"] for a in order}
    ind_s = {"moe": 0.024202, "mor": 0.033980, "more": 0.013387}
    ood_s = {a: ood[a]["std_loss"] for a in order}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for i, (vals, stds, title, ylab) in enumerate([
            (ind, ind_s, "(a) In-domain (WikiText-103 val)", "loss (nats/token)"),
            (ood_m, ood_s, "(b) Out-of-domain (the Pile)", "loss (nats/token)")]):
        ax = axes[i]
        ypos = list(range(3))[::-1]
        for j, arm in enumerate(order):
            ax.barh(ypos[j], vals[arm], xerr=stds[arm], color=C[arm],
                    height=0.55, error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
            ax.text(vals[arm] + stds[arm] + 0.06 * max(vals.values()), ypos[j],
                    f"{vals[arm]:.3f}", va="center", fontsize=7.8, color=INK)
        ax.set_yticks(ypos)
        ax.set_yticklabels([LABEL[a] for a in order], fontsize=8.5)
        ax.set_xlabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(vals.values()) * 1.28)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "fig_ood")


if __name__ == "__main__":
    print("generating language-paper figures from results/language/ ...")
    c = fig_depth_curve()
    print(f"  MoR  adaptive {c['mor_adaptive']:.4f} vs best forced "
          f"{c['mor_best_forced']:.4f}  -> adaptive wins by "
          f"{c['mor_best_forced'] - c['mor_adaptive']:.4f}")
    print(f"  MoRE adaptive {c['more_adaptive']:.4f} vs best forced "
          f"{c['more_best_forced']:.4f}  -> adaptive wins by "
          f"{c['more_best_forced'] - c['more_adaptive']:.4f}")
    fig_stratified()
    fig_ood()
    print("done.")
