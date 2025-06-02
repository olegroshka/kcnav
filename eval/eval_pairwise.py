#!/usr/bin/env python3
# tools/eval_pairwise.py
"""
Compare two sets of JSON trajectory summaries (baseline vs. guided).

Each trajectory_*.json must contain at least:
    ─ problem_id                 (str / int)
    ─ is_correct_assessment      (bool)
    ─ metadata.risk_score        (float, optional)

The script prints:
    • baseline accuracy
    • guided   accuracy
    • confusion matrix vs. baseline (TP / FP / FN / TN as defined below)
    • McNemar χ² on the 2x2 table
    • AUROC of the guided model’s risk score (if provided)
"""
import argparse
import json
import pathlib
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def load_run(folder: pathlib.Path) -> Dict[str, dict]:
    """Load all *trajectory_*.json files inside a folder into a dict keyed by problem_id."""
    run = {}
    for fp in folder.glob("trajectory_*.json"):
        with fp.open() as f:
            rec = json.load(f)
        run[str(rec["problem_id"])] = rec
    if not run:
        raise FileNotFoundError(f"No trajectory_*.json files found in {folder}")
    return run


def eval_pairwise(baseline_dir: pathlib.Path, guided_dir: pathlib.Path) -> None:
    base = load_run(baseline_dir)
    guided = load_run(guided_dir)

    if base.keys() != guided.keys():
        raise ValueError("The two folders do not contain the same set of problem_ids")

    total = len(base)
    base_correct   = sum(rec["is_correct_assessment"] for rec in base.values())
    guided_correct = sum(rec["is_correct_assessment"] for rec in guided.values())

    # Confusion matrix where:
    #   – “positive”  = baseline wrong *and* guided correct  (guided fixes a miss)
    #   – “negative”  = baseline correct
    tp = fp = tn = fn = 0
    auc_y: List[int]   = []
    auc_p: List[float] = []

    for pid in base:
        b_ok = base[pid]["is_correct_assessment"]
        g_ok = guided[pid]["is_correct_assessment"]

        if not b_ok and g_ok:
            tp += 1
        elif not b_ok and not g_ok:
            fp += 1
        elif b_ok and g_ok:
            tn += 1
        else:  # b_ok and not g_ok
            fn += 1

        # Gather risk score for AUROC
        score = guided[pid].get("metadata", {}).get("risk_score")
        if score is not None:
            auc_y.append(int(g_ok))  # 1 = correct, 0 = wrong
            auc_p.append(-score)     # lower risk ⇒ better ⇒ invert for AUROC

    # --- Statistics ---------------------------------------------------------
    # McNemar χ² with continuity correction
    mcnemar_chi2 = ((abs(tp - fn) - 1) ** 2) / (tp + fn) if (tp + fn) else 0

    print(f"Processed {total} problems\n")

    # Accuracies -------------------------------------------------------------
    base_acc   = base_correct   / total * 100
    guided_acc = guided_correct / total * 100
    print(f"Baseline accuracy : {base_correct}/{total} = {base_acc:.2f}%")
    print(f"Guided   accuracy : {guided_correct}/{total} = {guided_acc:.2f}%\n")

    # Confusion matrix & McNemar --------------------------------------------
    print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"McNemar χ² (1 dof, continuity-corrected) = {mcnemar_chi2:.3f}")

    # AUROC ------------------------------------------------------------------
    if auc_y:
        auc = roc_auc_score(auc_y, auc_p)
        print(f"AUROC (risk_score, guided)            = {auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pair-wise evaluation of baseline vs guided runs")
    parser.add_argument("baseline", type=pathlib.Path, help="Folder with baseline trajectory_*.json files")
    parser.add_argument("guided",   type=pathlib.Path, help="Folder with guided   trajectory_*.json files")
    args = parser.parse_args()

    eval_pairwise(args.baseline, args.guided)
