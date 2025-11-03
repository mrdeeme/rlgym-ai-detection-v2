from rlgym_ai_detection.pipeline import DetectionPipeline, default_embedder
from rlgym_ai_detection.policies import load_policy
def test_smoke():
    texts = ["You are a senior cloud architect. Goal: ...", "pode checar as horas de ontem? preciso enviar até 10h."]
    y = [1,0]
    pipe = DetectionPipeline(policy=load_policy(), embed_fn=default_embedder())
    # Training requires sklearn; skip gracefully if missing
    try:
        pipe.fit(texts, y, langs=["EN","pt-BR"])
    except RuntimeError:
        return
    rep = pipe.predict_one("System prompt: Act as SRE. Task: runbook...", risk_tier="high")
    assert "decision" in rep and "prob_llm" in rep
