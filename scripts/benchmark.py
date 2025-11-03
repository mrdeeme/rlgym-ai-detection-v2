import argparse, json, csv, os
from collections import defaultdict
def load_jsonl(fp):
    with open(fp,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    rows = list(load_jsonl(args.preds))
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("tier","UNK"), r.get("lang","UNK"))].append(r)
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None
    import numpy as np
    with open(args.out_csv,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f); w.writerow(["tier","lang","n","auc","tnr@fpr<=5%","abstention_rate"])
        for (tier,lang), items in groups.items():
            y = [int(it.get("label",0)) for it in items]
            p = [float(it.get("prob_llm",0.5)) for it in items]
            d = [it.get("decision") for it in items]
            n = len(items)
            auc = roc_auc_score(y,p) if (roc_auc_score and len(set(y))>1) else ""
            # TNR at FPR<=5%
            y_arr = np.array(y); p_arr = np.array(p)
            best=None
            for t in np.linspace(0,1,1001):
                y_pred = (p_arr>=t).astype(int)
                tn = ((y_pred==0)&(y_arr==0)).sum(); fp=((y_pred==1)&(y_arr==0)).sum()
                fpr = fp/max(1,(fp+tn))
                if fpr<=0.05:
                    tnr = tn/max(1,(tn+fp))
                    best = tnr if best is None or tnr>best else best
            abst = sum(1 for x in d if x=="abstain")/n if n else ""
            w.writerow([tier,lang,n,auc,best if best is not None else "",abst])
if __name__=="__main__":
    main()
