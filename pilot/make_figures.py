#!/usr/bin/env python3
"""Turns the JSON + CSV that pilot_measurement.py spits out into the
two figures I want in the paper. Run this *after* a successful pilot
run, otherwise there's nothing to draw.

What comes out:
    pilot/fig_trainable_sweep.pdf   The sweep figure: per-update wall
                                    time on top, marginal energy on
                                    the bottom, both as bars across
                                    the freeze configs. mu_2pct is
                                    coloured so it pops.
    pilot/fig_power_timeline.pdf    Power vs time, smoothed, with
                                    phase-coloured background bands,
                                    per-phase mean overlays, and the
                                    idle baseline as a dashed line.

PDF for the paper, PNG alongside for quick eyeballing. Sized for
IEEE column widths (single-column for the sweep, double-column for
the timeline) so I'm not fighting LaTeX later.

- Marty
"""

import csv
import json
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
RESULTS = json.loads((HERE / "pilot_results.json").read_text())
CSV_PATH = HERE / "pilot_data.csv"

# IEEE column widths in inches. Don't change these unless the
# template changes - matplotlib figsize is what makes the fonts
# come out the right size when the PDF is dropped in.
SINGLE_COL_W = 3.5
DOUBLE_COL_W = 7.16


def rolling_mean(values, window):
    # Plain trailing-mean smoother. The smoothed line ends up lagging
    # by ~window/2 samples - I don't care for the timeline figure.
    if window <= 1:
        return list(values)
    out = []
    buf = deque(maxlen=window)
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf))
    return out


# --------------------------------------------------------------
# Figure 1: the trainable-fraction sweep.
# Top panel: ms/update. Bottom panel: mJ/update (marginal over idle).
# Same x-axis order, mu_2pct highlighted.
# --------------------------------------------------------------

def fig_trainable_sweep():
    sweep = RESULTS["trainable_fraction_sweep"]
    labels = [r["config"] for r in sweep]
    # Hand-rolled labels with the actual trainable-fraction percentages.
    # I'm hardcoding these because the LaTeX mu-sign is fiddly to do
    # programmatically and the numbers are stable per model arch.
    pretty = {
        "last_layer": "last-layer\n(0.33%)",
        "mu_2pct":    r"$\mu$-train" + "\n(2.06%)",
        "mu_10pct":   r"$\mu$-train" + "\n(2.97%)",
        "half":       "half-net\n(85.3%)",
        "full":       "full-net\n(100%)",
    }
    pretty_labels = [pretty[l] for l in labels]
    ms = [r["per_update_ms"] for r in sweep]
    mJ_marg = [r["energy_mJ_per_update_marginal"] for r in sweep]

    # Grey for the context bars, blue for the one config the paper is
    # actually arguing for. Cheap way to make the figure self-narrate.
    bar_colors = ["#bbbbbb"] * len(labels)
    for i, l in enumerate(labels):
        if l == "mu_2pct":
            bar_colors[i] = "#1a6dbb"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(SINGLE_COL_W, 2.8),
        sharex=True, gridspec_kw=dict(hspace=0.15))

    x = list(range(len(labels)))
    ax1.bar(x, ms, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("ms per update", fontsize=8)
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    ax2.bar(x, mJ_marg, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("mJ per update\n(marginal)", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pretty_labels, fontsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate the full-vs-mu ratio directly on the figure. This is
    # the headline number ("mu is Nx cheaper"), so I want it readable
    # without consulting the table.
    full_idx = labels.index("full")
    mu_idx = labels.index("mu_2pct")
    ratio_e = mJ_marg[full_idx] / mJ_marg[mu_idx]
    ratio_t = ms[full_idx] / ms[mu_idx]
    ax1.annotate(
        f"{ratio_t:.1f}x", xy=(mu_idx, ms[mu_idx]),
        xytext=(mu_idx + 0.4, ms[full_idx] * 0.55),
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
    ax2.annotate(
        f"{ratio_e:.1f}x", xy=(mu_idx, mJ_marg[mu_idx]),
        xytext=(mu_idx + 0.4, mJ_marg[full_idx] * 0.55),
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6))

    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.18)
    out = HERE / "fig_trainable_sweep.pdf"
    fig.savefig(out)
    # PNG companion for sticking in slides / quickly checking the
    # figure without opening a PDF viewer.
    fig.savefig(HERE / "fig_trainable_sweep.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------
# Figure 2: power vs time. Background bands show what phase we were
# in. Smoothed black line is the trace. Blue ticks are per-phase
# means. Dashed line is idle, just for reference.
# --------------------------------------------------------------

def fig_power_timeline():
    # Pull just the columns we actually plot. Skipping malformed rows
    # (occasional PMIC parse hiccup) instead of bailing on the run.
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((
                    float(r["timestamp_unix"]),
                    r["phase"],
                    r.get("step_label", ""),
                    float(r["total_W"]),
                ))
            except (KeyError, ValueError):
                continue
    if not rows:
        print("no rows; skipping power timeline")
        return

    # Re-zero time on the first sample so the x-axis is "seconds
    # since run start" rather than an unreadable unix epoch.
    t0 = rows[0][0]
    times = [r[0] - t0 for r in rows]
    powers = [r[3] for r in rows]
    phases = [r[1] for r in rows]
    # window=10 samples @ 5 Hz = 2 s of smoothing. Enough to kill the
    # PMIC jitter without flattening the actual phase transitions.
    powers_smooth = rolling_mean(powers, window=10)

    # Pastel-ish palette, one per phase. Tuned so the black power
    # trace still reads clearly on top - earlier version was too dark.
    phase_color = {
        "setup":            "#cccccc",
        "idle":             "#a8c8e8",
        "timing":           "#ffd599",
        "inference":        "#ffae5d",
        "training":         "#7fbf6e",
        "drift_detection":  "#c98fc8",
        "bwt":              "#e88a8a",
        "teardown":         "#cccccc",
    }

    # Compress the per-sample phase tags into contiguous blocks so I
    # can draw one axvspan per visible band instead of thousands.
    blocks = []
    cur_phase = phases[0]
    cur_start = times[0]
    for i in range(1, len(phases)):
        if phases[i] != cur_phase:
            blocks.append((cur_start, times[i], cur_phase))
            cur_phase = phases[i]
            cur_start = times[i]
    blocks.append((cur_start, times[-1], cur_phase))

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 1.85),
                            constrained_layout=True)

    # Draw background bands first so the trace sits on top.
    for s, e, p in blocks:
        ax.axvspan(s, e, color=phase_color.get(p, "#ffffff"),
                   alpha=0.5, linewidth=0)

    # Smoothed line only - the raw 5 Hz trace was dominating the
    # figure with measurement noise without adding signal beyond what
    # the rolling mean already shows.
    ax.plot(times, powers_smooth, color="black", linewidth=1.0,
            label="2 s rolling mean")

    # Overlay the post-hoc per-phase mean as a short horizontal blue
    # segment on top of each block. Lets the eye check "did this
    # phase actually steady-state where the JSON says it did".
    phase_stats = RESULTS["phase_power_stats"]
    for s, e, p in blocks:
        # phase_stats is keyed by "phase" or "phase/step_label", so a
        # phase with multiple sweep points has multiple entries here.
        candidates = [k for k in phase_stats
                       if phase_stats[k]["phase"] == p]
        if not candidates:
            continue
        # Just average the per-step means together for the overlay.
        # Not statistically pure but it's a visual reference, not a
        # number I'm quoting.
        mean = sum(phase_stats[k]["power_W_mean"]
                   for k in candidates) / len(candidates)
        ax.hlines(mean, s, e, colors="#1a6dbb", linewidth=1.6,
                  zorder=3)

    # Dashed idle line - everything else should be read relative to this.
    idle_w = RESULTS["power"]["idle_W"]
    ax.axhline(idle_w, color="#1a6dbb", linewidth=0.8, linestyle="--",
               label=f"idle baseline {idle_w:.2f} W")

    ax.set_xlabel("time since run start (s)", fontsize=9)
    ax.set_ylabel("total power (W)", fontsize=9)
    ax.set_xlim(0, times[-1])
    ax.set_ylim(bottom=max(0, min(powers) - 0.3))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.legend(loc="upper left", fontsize=7, frameon=True,
              framealpha=0.85, ncol=3)

    # Second legend underneath: phase colour key. Built from the
    # phases that actually appear in the run (in order of first
    # occurrence) so we don't have orphan entries if a phase got
    # skipped.
    used_phases = []
    for _, _, p in blocks:
        if p not in used_phases:
            used_phases.append(p)
    handles = [Patch(facecolor=phase_color.get(p, "#fff"),
                      edgecolor="black", linewidth=0.3,
                      label=p.replace("_", " "))
               for p in used_phases]
    # Place the phase legend underneath the axes; constrained_layout
    # then routes it cleanly so the x-axis label is not clipped.
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(used_phases), fontsize=7, frameon=False)
    out = HERE / "fig_power_timeline.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "fig_power_timeline.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_trainable_sweep()
    fig_power_timeline()
