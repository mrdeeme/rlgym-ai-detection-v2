import argparse, json, os, matplotlib.pyplot as plt, csv
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
    ap.add_argument("--report")
    ap.add_argument("--outdir", default="dash_artifacts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = list(load_jsonl(args.preds))
    probs = [float(r.get("prob_llm",0.5)) for r in rows]
    plt.figure(); plt.hist(probs, bins=20); plt.title("prob_llm"); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"prob_hist.png")); plt.close()
    by_tier = defaultdict(list)
    for r in rows: by_tier[r.get("tier","UNK")].append(r)
    tiers = sorted(by_tier.keys())
    abst = [ sum(1 for x in by_tier[t] if x.get("decision")=="abstain")/len(by_tier[t]) for t in tiers ]
    plt.figure(); plt.bar(range(len(tiers)), abst); plt.xticks(range(len(tiers)), tiers, rotation=30, ha="right"); plt.title("abstention by tier"); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"abstention_by_tier.png")); plt.close()
    if args.report and os.path.exists(args.report):
        recs=[]; 
        with open(args.report,"r",encoding="utf-8") as f:
            rdr=csv.DictReader(f)
            for r in rdr: recs.append(r)
        tnr = {}
        for r in recs:
            v=r.get("tnr@fpr<=5%","")
            if v!="":
                tnr.setdefault(r["tier"], []).append(float(v))
        if tnr:
            lab = sorted(tnr.keys())
            val = [sum(tnr[k])/len(tnr[k]) for k in lab]
            plt.figure(); plt.bar(range(len(lab)), val); plt.xticks(range(len(lab)), lab, rotation=30, ha="right"); plt.title("avg TNR@FPR<=5% by tier"); plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"tnr_by_tier.png")); plt.close()
if __name__=="__main__":
    main()
