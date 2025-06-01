# tools/eval_pairwise.py
import argparse, json, pathlib, collections, math
from sklearn.metrics import roc_auc_score

def load(folder):
    d={}
    for fp in pathlib.Path(folder).glob("trajectory_*.json"):
        j=json.load(open(fp))
        d[j["problem_id"]] = j
    return d

def main(baseline, guided):
    b = load(baseline); g = load(guided)
    assert b.keys()==g.keys()
    tp=fp=tn=fn=0
    auc_y, auc_p = [], []
    for pid in b:
        if b[pid]["is_correct_assessment"]:
            if g[pid]["is_correct_assessment"]: tn+=1
            else: fn+=1
        else:
            if g[pid]["is_correct_assessment"]: tp+=1
            else: fp+=1
        # collect risk score if stored
        s = g[pid]["metadata"].get("risk_score", None)
        if s is not None:
            auc_y.append(g[pid]["is_correct_assessment"])
            auc_p.append(-s)           # lower risk = better
    mcnemar = (abs(tp-fn)-1)**2/(tp+fn) if tp+fn else 0
    auc = roc_auc_score(auc_y, auc_p) if auc_y else None
    print("TP FN FP TN:", tp, fn, fp, tn)
    print("McNemar χ²:", round(mcnemar,3))
    if auc: print("AUROC:", round(auc,3))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("baseline"); p.add_argument("guided")
    main(**vars(p.parse_args()))
