import numpy as np
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

def fit_iforest(X: np.ndarray, contamination: float = 0.08):
    if IsolationForest is None:
        return None
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    clf.fit(X)
    return clf

def iforest_score(clf, X: np.ndarray):
    if clf is None:
        return np.zeros(X.shape[0], dtype=float)
    df = clf.decision_function(X)
    return -df
