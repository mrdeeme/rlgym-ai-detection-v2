"""Calibration module for supervised learning"""
from __future__ import annotations
import json
from typing import List
import numpy as np


class Calibrator:
    """
    Simple logistic regression calibrator (no sklearn) for detector scores.
    Fit on (X, y), where X is an array of feature columns; y in {0,1} (1 = LLM).
    Persist as JSON (weights + bias).
    """
    
    def __init__(self):
        self.w = None
        self.b = 0.0
        self.feature_names: List[str] = []
    
    @staticmethod
    def _sigmoid(z):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 800, l2: float = 0.0):
        """
        Train logistic regression on features X and labels y.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) with values in {0, 1}
            lr: Learning rate
            epochs: Number of training iterations
            l2: L2 regularization strength
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        
        for _ in range(epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            
            # Gradients
            grad_w = (X.T @ (p - y)) / n + l2 * self.w
            grad_b = float(np.sum(p - y) / n)
            
            # Update
            self.w -= lr * grad_w
            self.b -= lr * grad_b
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples in X.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples,)
        """
        z = X @ self.w + self.b
        return self._sigmoid(z)
    
    def save_json(self, path: str, feature_names: List[str]):
        """
        Save calibrator to JSON file.
        
        Args:
            path: Output file path
            feature_names: Names of features (for documentation)
        """
        payload = {
            "w": self.w.tolist(),
            "b": float(self.b),
            "features": feature_names
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    def load_json(self, path: str):
        """
        Load calibrator from JSON file.
        
        Args:
            path: Input file path
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.w = np.array(obj["w"], dtype=float)
        self.b = float(obj["b"])
        self.feature_names = obj.get("features", [])

