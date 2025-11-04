# Extra low-cost stylometry and discourse features.
# English comments per user preference.
from __future__ import annotations
import re, math
from collections import Counter

def safe_div(a, b, default=0.0):
    return (a / b) if b else default

def word_tokens(text: str):
    return re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)

def sentence_tokens(text: str):
    s = text.strip()
    return re.split(r"(?<=[.!?])\s+", s) if s else []

def ttr(tokens):
    return safe_div(len(set(tokens)), len(tokens))

def hapax_ratio(tokens):
    c = Counter(tokens)
    hapax = sum(1 for _,v in c.items() if v == 1)
    return safe_div(hapax, len(tokens))

def yules_k(tokens):
    N = len(tokens)
    if N == 0: return 0.0
    c = Counter(tokens)
    m2 = sum(v*v for v in c.values())
    return 1e4 * (m2 - N) / (N*N)

def mean_word_len(tokens): return safe_div(sum(len(w) for w in tokens), len(tokens))
def mean_sent_len(sents): return safe_div(sum(len(word_tokens(s)) for s in sents), len(sents))

def ngram_repetition(tokens, n=3):
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    c = Counter(grams)
    repeated = sum(1 for _,v in c.items() if v >= 2)
    return safe_div(repeated, len(c))

def punctuation_entropy(text: str):
    p = re.findall(r"[.,;:!?()\[\]{}\"'–—-]", text)
    if not p: return 0.0
    c = Counter(p); total = len(p)
    return -sum((v/total)*math.log2(v/total) for v in c.values())

DISCOURSE_PT = {"portanto","contudo","assim","além disso","no entanto","porém","dessa forma"}
DISCOURSE_EN = {"however","therefore","moreover","furthermore","thus","nevertheless"}

def discourse_markers_ratio(text: str, lang: str = "pt"):
    markers = DISCOURSE_PT if lang.startswith("pt") else DISCOURSE_EN
    t = text.lower()
    hits = sum(t.count(m) for m in markers)
    return hits

def build_extra2(text: str, lang: str = "pt"):
    toks = [w.lower() for w in word_tokens(text)]
    sents = sentence_tokens(text)
    return {
        "ttr": ttr(toks),
        "hapax_ratio": hapax_ratio(toks),
        "yules_k": yules_k(toks),
        "mean_word_len": mean_word_len(toks),
        "mean_sent_len": mean_sent_len(sents),
        "ngram3_rep": ngram_repetition(toks, 3),
        "ngram5_rep": ngram_repetition(toks, 5),
        "punct_entropy": punctuation_entropy(text),
        "discourse_markers": discourse_markers_ratio(text, lang),
    }
