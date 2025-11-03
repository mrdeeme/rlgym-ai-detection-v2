import re, numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

class TextFeaturizer:
    def __init__(self):
        self.vec = None

    def fit(self, texts):
        if TfidfVectorizer is None:
            self.vec = None
            return self
        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.98)
        self.vec.fit(texts)
        return self

    def transform(self, texts):
        # sparse and basic dense features
        if self.vec is not None:
            Xs = self.vec.transform(texts)
        else:
            Xs = None
        dense = []
        for t in texts:
            n_chars = len(t)
            n_words = len(re.findall(r"\w+", t))
            n_code = len(re.findall(r"```", t))
            ttr = len(set(re.findall(r"\w+", t))) / max(1, n_words)
            dense.append([n_chars, n_words, n_code, ttr])
        Xd = np.array(dense, dtype=float)
        return Xs, Xd
