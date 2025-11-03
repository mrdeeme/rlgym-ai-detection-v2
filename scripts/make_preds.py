import argparse, json
def load_jsonl(fp):
    with open(fp,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import joblib
    pipe = joblib.load(args.model)
    with open(args.out,"w",encoding="utf-8") as w:
        for ex in load_jsonl(args.data):
            rep = pipe.predict_one(ex.get("text",""), risk_tier=ex.get("risk_tier","medium"), lang=ex.get("lang"))
            out = {"doc_id":ex.get("doc_id"),"tier":ex.get("tier"),"lang":ex.get("lang","EN"),"label":ex.get("label"),
                   "prob_llm":rep.get("prob_llm"),"decision":rep.get("decision")}
            w.write(json.dumps(out, ensure_ascii=False)+"\n")
if __name__=="__main__":
    main()
