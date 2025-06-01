#!/usr/bin/env python
"""
Make publication-ready plots from KCNav trajectory folders.

Usage
-----
python plot/plot_complexity_metrics.py \
       --exp_dirs EXP_DIR [EXP_DIR ...] \
       --out_dir  FIG_OUT_DIR \
       [--heatmap]
"""
from __future__ import annotations
import argparse, json, pathlib, zlib
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
#                               Data loading                                  #
# --------------------------------------------------------------------------- #
def load_folder(folder: pathlib.Path) -> pd.DataFrame:
    """Return long DF with per-step rows + one summary row (step == −1)."""
    rows: List[dict] = []

    for fp in folder.glob("trajectory_*.json"):
        with fp.open() as f:
            data = json.load(f)

        traj_id   = data["problem_id"]
        model_tag = folder.name            # use folder name as model label
        is_corr   = data["is_correct_assessment"]
        status    = data["status"]

        cumulative_bytes = b""
        max_glob = 0.0

        for step in data["steps"]:
            step_idx  = step["step_index"]
            stype     = step["step_type"]
            ts        = step["timestamp"]
            ncd_local = step["metadata"]["ncd_local"]

            # stored value or recompute on the fly
            if "ncd_global" in step["metadata"]:
                ncd_global = step["metadata"]["ncd_global"]
            else:
                step_text = (step.get("thought", "") +
                             step.get("action",  "")).encode()
                cumulative_bytes += step_text
                ncd_global = len(zlib.compress(cumulative_bytes))

            rows.append(dict(model=model_tag, traj=traj_id, step=step_idx,
                             stype=stype, ncd_local=ncd_local,
                             ncd_global=ncd_global, timestamp=ts,
                             correct=is_corr, status=status))
            max_glob = max(max_glob, ncd_global)

        rows.append(dict(model=model_tag, traj=traj_id, step=-1,
                         stype="TRAJ", ncd_local=np.nan,
                         ncd_global=max_glob, timestamp=np.nan,
                         correct=is_corr, status=status))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def safe_save(fig_path: pathlib.Path) -> None:
    """Skip empty figs cleanly; report what was written."""
    if not plt.gca().has_data():
        print(f" – skipped empty plot: {fig_path.name}")
    else:
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        print(f" ✓ wrote {fig_path.name}")
    plt.clf()


def make_out_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    make_out_dir(out_dir)

    dfs = [load_folder(pathlib.Path(p)) for p in args.exp_dirs]
    if not dfs:
        print("No trajectory files found — nothing to plot.")
        return

    df       = pd.concat(dfs, ignore_index=True)
    step_df  = df[df.step >= 0].copy()
    traj_df  = df[df.step == -1].copy()

    # ---------------------------------------------------------------- A  mean trajectory
    plt.figure(figsize=(6, 4))
    for model, grp in step_df.groupby("model"):
        stats = (grp.groupby(["correct", "step"])
                    .ncd_global.agg(["mean", "std"]).reset_index())
        for is_ok, col in zip([True, False], ["C0", "C3"]):
            sub = stats[stats.correct == is_ok]
            if sub.empty:
                continue
            lbl = f"{model}-{'succ' if is_ok else 'fail'}"
            plt.plot(sub.step, sub["mean"], label=lbl, color=col)
            plt.fill_between(sub.step,
                             sub["mean"]-sub["std"],
                             sub["mean"]+sub["std"],
                             alpha=0.15, color=col)
    plt.title("Complexity trajectory (mean ±1σ)")
    plt.xlabel("Step index"); plt.ylabel("Mean NCD global"); plt.legend()
    safe_save(out_dir / "traj_mean.png")

    # ---------------------------------------------------------------- B  max-NCD histogram
    g = sns.displot(traj_df, x="ncd_global", hue="correct",
                    col="model", kde=True,
                    facet_kws=dict(sharex=False, sharey=False))
    g.fig.suptitle("Distribution of max NCD global", y=1.02)
    g.savefig(out_dir / "hist_max_complex.png", dpi=180)
    print(" ✓ wrote hist_max_complex.png")
    plt.close(g.fig)

    # ---------------------------------------------------------------- C  step count violin
    counts = (step_df.groupby(["model", "traj"])
                        .agg(max_step=("step", "max"),
                             correct=("correct", "max"))
                        .reset_index())
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=counts, x="correct", y="max_step",
                   hue="model", split=True, inner="quartile")
    plt.title("Step count distribution")
    plt.xlabel("Correct?"); plt.ylabel("#steps")
    safe_save(out_dir / "violin_steps.png")

    # ---------------------------------------------------------------- D  steps vs max-NCD scatter
    counts = counts.merge(traj_df[["traj", "model", "ncd_global"]],
                          on=["traj", "model"])
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=counts, x="max_step", y="ncd_global",
                    hue="correct", style="model")
    plt.title("Complexity vs. length")
    plt.xlabel("#steps"); plt.ylabel("max NCD global")
    safe_save(out_dir / "scatter_steps_max.png")

    # ---------------------------------------------------------------- F  scoreboard
    scoreboard = (traj_df.groupby("model")
                          .agg(accuracy=("correct", "mean"),
                               mean_max_ncd=("ncd_global", "mean"))
                          .join(counts.groupby("model")["max_step"]
                                .mean().rename("mean_steps"))
                          .reset_index())
    melt = scoreboard.melt("model", var_name="metric", value_name="value")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=melt, x="metric", y="value", hue="model")
    plt.title("Model scoreboard")
    safe_save(out_dir / "scoreboard.png")

    # ---------------------------------------------------------------- G  reliability curve
    if not traj_df.empty:
        traj_df["bucket"] = pd.qcut(traj_df.ncd_global, q=6,
                                    duplicates="drop")
        rel = (traj_df.groupby(["bucket", "model"])
                        .correct.mean().reset_index())
        rel["centroid"] = rel.bucket.apply(lambda b: b.mid)
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=rel, x="centroid", y="correct",
                     hue="model", marker="o")
        plt.title("Reliability: success rate vs max-NCD")
        plt.xlabel("Max NCD global bucket centre"); plt.ylabel("Success rate")
        plt.ylim(0, 1)
        safe_save(out_dir / "reliability.png")

    # ---------------------------------------------------------------- H  percentile heatmap
    for model, grp in step_df.groupby("model"):
        pct = (grp.groupby("step")
                  .ncd_global.quantile([.1, .25, .5, .75, .9])
                  .unstack(level=1))
        if pct.empty:
            continue
        plt.figure(figsize=(5, 6))
        sns.heatmap(pct, cmap="crest",
                    cbar_kws=dict(label="NCD global"))
        plt.title(f"Percentile bands – {model}")
        plt.xlabel("Percentile"); plt.ylabel("Step")
        safe_save(out_dir / f"whisker_{model}.png")

    # ---------------------------------------------------------------- I  latency vs complexity
    if "timestamp" in step_df.columns:
        step_df["latency"] = (step_df.groupby(["model", "traj"])
                                      .timestamp.transform(lambda s: s.max()-s.min()))
        lat = (step_df[["traj", "model", "latency"]]
                      .drop_duplicates()
                      .merge(traj_df[["traj", "model", "ncd_global"]],
                             on=["traj", "model"]))
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=lat, x="ncd_global", y="latency",
                        hue="model")
        plt.title("Latency vs complexity")
        plt.xlabel("Max NCD global"); plt.ylabel("Latency (s)")
        safe_save(out_dir / "latency.png")

    # ---------------------------------------------------------------- J  CDF of step length
    plt.figure(figsize=(6, 4))
    sns.ecdfplot(data=counts, x="max_step", hue="correct", linestyle="--")
    plt.title("CDF of step length"); plt.xlabel("#steps")
    safe_save(out_dir / "cdf_steps.png")

    # ---------------------------------------------------------------- E  optional per-trajectory heatmap
    if args.heatmap:
        for model in step_df.model.unique():
            succ = step_df[(step_df.model == model) & (step_df.correct)]
            if succ.empty:
                continue
            pivot = succ.pivot(index="traj", columns="step",
                               values="ncd_local")
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, cmap="YlOrBr",
                        cbar_kws=dict(label="NCD local"),
                        rasterized=True)
            plt.title(f"NCD local heatmap – {model} (success)")
            plt.xlabel("Step"); plt.ylabel("Trajectory")
            safe_save(out_dir / f"heatmap_local_{model}.png")

    print(f"\nAll done. Plots are in {out_dir.resolve()}\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dirs", nargs="+", required=True,
                    help="Trajectory folders (each with trajectory_*.json).")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write PNGs.")
    ap.add_argument("--heatmap", action="store_true",
                    help="Also make heavy per-trajectory heatmaps.")
    main(ap.parse_args())
