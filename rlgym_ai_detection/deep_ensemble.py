import numpy as np

class HeteroEnsemble:
    def __init__(self, learners):
        self.learners = learners
        self.fitted = False

    def fit(self, X_sparse_texts, y):
        for l in self.learners:
            space = l.get("space")
            if space == "sparse":
                XA = l["vec"].fit_transform(X_sparse_texts)
                l["XA"] = XA
                l["clf"].fit(XA, y)
            elif space == "dense":
                l["clf"].fit(l["Xd"], y)
            elif space == "sem":
                l["clf"].fit(l["Xsem"], y)
        self.fitted = True
        return self

    def predict_proba(self, X_texts):
        Ps = []
        for l in self.learners:
            space = l.get("space")
            if space == "sparse":
                XA = l["vec"].transform(X_texts)
                Ps.append(l["clf"].predict_proba(XA)[:,1])
            elif space == "dense":
                Ps.append(l["clf"].predict_proba(l["Xd_test"])[:,1])
            elif space == "sem":
                Ps.append(l["clf"].predict_proba(l["Xsem_test"])[:,1])
        if not Ps:
            return np.zeros(len(X_texts), dtype=float)
        return np.vstack(Ps).mean(0)
