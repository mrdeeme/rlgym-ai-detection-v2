import os, yaml

def load_policy(path: str | None = None) -> dict:
    p = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "policy.yaml")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"policy_version":"fallback","tiers":{"medium":{"abstain_band":[0.4,0.6],"ood_threshold":1.2}}}
