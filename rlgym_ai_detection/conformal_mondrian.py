import numpy as np
from collections import defaultdict
from typing import Iterable, Tuple

class MondrianConformal:
    def __init__(self, q: float = 0.1):
        self.q = float(q)
        self.scores = defaultdict(list)

    @staticmethod
    def _group(meta: Iterable[str]) -> Tuple[str, ...]:
        return tuple(meta)

    def fit(self, probs, y_true, metas):
        probs = np.asarray(probs).ravel()
        y_true = np.asarray(y_true).ravel()
        for p, y, m in zip(probs, y_true, metas):
            self.scores[self._group(m)].append(abs(float(y) - float(p)))
        return self

    def threshold_for(self, meta):
        g = self._group(meta)
        s = self.scores.get(g, [])
        if not s: return 0.10
        return float(np.quantile(s, 0.9))

    def abstain(self, p: float, meta) -> bool:
        tau = self.threshold_for(meta)
        return abs(p - round(p)) <= tau
