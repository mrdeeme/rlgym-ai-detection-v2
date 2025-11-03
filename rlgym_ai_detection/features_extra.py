import re, numpy as np
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

FUNCTION_WORDS_EN = ["the","and","of","to","in","that","it","is","for","on","as","with","at","by","from","this","be","or","an","are","was","were","which"]

def tokenize_words(text: str):
    return re.findall(r"\w+", text.lower())

def tokenize_sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+[\s\n]+", text) if s.strip()]

def function_word_profile(tokens, vocab=FUNCTION_WORDS_EN):
    if not tokens:
        return np.zeros(len(vocab), dtype=float)
    counts = np.array([tokens.count(w) for w in vocab], dtype=float)
    return counts / max(1.0, float(len(tokens)))

def coherence_graph_metrics(sent_embeddings: np.ndarray, tau: float = 0.65):
    n = len(sent_embeddings)
    if n < 3:
        return 1.0, 0.0
    if cosine_similarity is None:
        X = sent_embeddings
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
    else:
        S = cosine_similarity(sent_embeddings)
    np.fill_diagonal(S, 0.0)
    density = float((S > tau).sum() / (S.shape[0] * (S.shape[1] - 1) + 1e-9))
    row_sums = S.sum(axis=1, keepdims=True) + 1e-9
    N = S / row_sums
    vals = np.linalg.eigvals(N)
    vals = np.sort(np.real(vals))[::-1]
    gap = float(vals[0] - vals[1]) if len(vals) > 1 else 0.0
    return density, gap
