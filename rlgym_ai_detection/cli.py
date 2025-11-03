import argparse, json, sys
from rlgym_ai_detection.pipeline import DetectionPipeline, default_embedder
from rlgym_ai_detection.policies import load_policy

def read_jsonl(fp):
    with open(fp,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def cmd_fit(args):
    try:
        import joblib
    except Exception:
        raise SystemExit("joblib is required. pip install joblib")
    texts, y, langs = [], [], []
    for ex in read_jsonl(args.train):
        texts.append(ex.get("text",""))
        y.append(int(ex.get("label",0)))
        langs.append(ex.get("lang","EN"))
    pipe = DetectionPipeline(policy=load_policy(), embed_fn=default_embedder())
    pipe.fit(texts, y, langs=langs)
    joblib.dump(pipe, args.out)
    print(f"saved model -> {args.out}")

def cmd_eval(args):
    try:
        import joblib
    except Exception:
        raise SystemExit("joblib is required. pip install joblib")
    from sklearn.metrics import roc_auc_score, f1_score
    pipe = joblib.load(args.model)
    y_true, probs = [], []
    for ex in read_jsonl(args.test):
        rep = pipe.predict_one(ex.get("text",""), risk_tier=ex.get("risk_tier","medium"), lang=ex.get("lang"))
        y_true.append(int(ex.get("label",0)))
        probs.append(float(rep.get("prob_llm",0.5)))
    auc = roc_auc_score(y_true, probs) if len(set(y_true))>1 else None
    print(json.dumps({"n":len(y_true),"auc":auc}, ensure_ascii=False))

def cmd_run(args):
    try:
        import joblib
    except Exception:
        raise SystemExit("joblib is required. pip install joblib")
    pipe = joblib.load(args.model)
    txt = args.text if args.text else sys.stdin.read()
    rep = pipe.predict_one(txt, risk_tier=args.risk, lang=args.lang)
    print(json.dumps(rep, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser("detect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit"); a.add_argument("--train", required=True); a.add_argument("--out", required=True); a.set_defaults(func=cmd_fit)
    a = sub.add_parser("eval"); a.add_argument("--test", required=True); a.add_argument("--model", required=True); a.set_defaults(func=cmd_eval)
    a = sub.add_parser("run"); a.add_argument("--model", required=True); a.add_argument("--text"); a.add_argument("--risk", default="medium"); a.add_argument("--lang"); a.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
