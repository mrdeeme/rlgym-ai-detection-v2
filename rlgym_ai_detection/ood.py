import numpy as np

class MahalanobisOOD:
    def __init__(self, eps=1e-6):
        self.mu = None
        self.Si = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mu = X.mean(axis=0, keepdims=True)
        S = np.cov(X.T) + self.eps*np.eye(X.shape[1])
        self.Si = np.linalg.pinv(S)
        return self

    def score(self, X: np.ndarray):
        if self.mu is None:
            return np.zeros(X.shape[0], dtype=float)
        D = X - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.Si, D))
