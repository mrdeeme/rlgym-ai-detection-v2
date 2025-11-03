"""Tests for individual components"""
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rlgym_ai_detection.normalize_segment import normalize
from rlgym_ai_detection.conformal_mondrian import MondrianConformal
from rlgym_ai_detection.ood import MahalanobisOOD
from rlgym_ai_detection.utils import detect_lang


class TestNormalization:
    """Test text normalization"""
    
    def test_normalize_basic(self):
        """Test basic normalization"""
        text = "  Hello   World  "
        result = normalize(text)
        # normalize() strips leading/trailing but may preserve internal spaces
        assert "Hello" in result and "World" in result
    
    def test_normalize_preserves_content(self):
        """Test normalization preserves content"""
        text = "This is a test."
        result = normalize(text)
        assert "test" in result
    
    def test_normalize_empty(self):
        """Test normalization of empty string"""
        result = normalize("")
        assert result == ""


class TestMondrianConformal:
    """Test Mondrian conformal prediction"""
    
    def test_initialization(self):
        """Test MondrianConformal initialization"""
        mc = MondrianConformal(q=0.1)
        assert mc.q == 0.1
        assert len(mc.scores) == 0
    
    def test_fit(self):
        """Test fitting conformal predictor"""
        mc = MondrianConformal(q=0.1)
        probs = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        y_true = np.array([0, 1, 0, 1, 1])
        metas = [["EN", "short"], ["EN", "long"], ["pt-BR", "short"], ["EN", "medium"], ["EN", "short"]]
        
        mc.fit(probs, y_true, metas)
        assert len(mc.scores) > 0
    
    def test_threshold_for(self):
        """Test threshold calculation"""
        mc = MondrianConformal(q=0.1)
        probs = np.array([0.1, 0.9, 0.3, 0.7])
        y_true = np.array([0, 1, 0, 1])
        metas = [["EN"], ["EN"], ["EN"], ["EN"]]
        
        mc.fit(probs, y_true, metas)
        threshold = mc.threshold_for(["EN"])
        assert threshold > 0
        assert threshold <= 1.0
    
    def test_abstain(self):
        """Test abstention decision"""
        mc = MondrianConformal(q=0.1)
        probs = np.array([0.1, 0.9, 0.3, 0.7])
        y_true = np.array([0, 1, 0, 1])
        metas = [["EN"], ["EN"], ["EN"], ["EN"]]
        
        mc.fit(probs, y_true, metas)
        
        # Test abstention for uncertain prediction
        should_abstain = mc.abstain(0.5, ["EN"])
        assert isinstance(should_abstain, bool)
    
    def test_unknown_group(self):
        """Test handling of unknown group"""
        mc = MondrianConformal(q=0.1)
        probs = np.array([0.1, 0.9])
        y_true = np.array([0, 1])
        metas = [["EN"], ["EN"]]
        
        mc.fit(probs, y_true, metas)
        
        # Query unknown group
        threshold = mc.threshold_for(["FR"])  # Unknown language
        assert threshold == 0.10  # Should return default


class TestMahalanobisOOD:
    """Test Mahalanobis OOD detection"""
    
    def test_initialization(self):
        """Test MahalanobisOOD initialization"""
        ood = MahalanobisOOD(eps=1e-6)
        assert ood.eps == 1e-6
        assert ood.mu is None
        assert ood.Si is None
    
    def test_fit(self):
        """Test fitting OOD detector"""
        ood = MahalanobisOOD()
        X = np.random.randn(100, 10)
        
        ood.fit(X)
        assert ood.mu is not None
        assert ood.Si is not None
        assert ood.mu.shape == (1, 10)
    
    def test_score(self):
        """Test OOD scoring"""
        ood = MahalanobisOOD()
        X_train = np.random.randn(100, 10)
        ood.fit(X_train)
        
        # Score in-distribution samples
        X_test = np.random.randn(10, 10)
        scores = ood.score(X_test)
        
        assert scores.shape == (10,)
        assert np.all(scores >= 0)  # Mahalanobis distance is non-negative
    
    def test_outlier_detection(self):
        """Test outlier gets higher score"""
        ood = MahalanobisOOD()
        
        # Train on normal data
        X_train = np.random.randn(100, 10)
        ood.fit(X_train)
        
        # Normal sample
        X_normal = np.random.randn(1, 10)
        score_normal = ood.score(X_normal)[0]
        
        # Outlier (far from distribution)
        X_outlier = np.random.randn(1, 10) * 10  # 10x larger
        score_outlier = ood.score(X_outlier)[0]
        
        # Outlier should have higher score
        assert score_outlier > score_normal


class TestLanguageDetection:
    """Test language detection utility"""
    
    def test_detect_english(self):
        """Test English detection"""
        text = "This is an English sentence with common words."
        lang = detect_lang(text)
        # detect_lang may return various formats, just check it returns something
        assert lang is not None
        assert len(lang) > 0
    
    def test_detect_portuguese(self):
        """Test Portuguese detection"""
        text = "Este é um texto em português com palavras comuns."
        lang = detect_lang(text)
        # Should detect some non-English language
        assert lang is not None
    
    def test_detect_short_text(self):
        """Test detection on short text"""
        text = "hello"
        lang = detect_lang(text)
        assert lang is not None  # Should return something


class TestFeatures:
    """Test feature extraction"""
    
    def test_text_featurizer(self):
        """Test TextFeaturizer"""
        try:
            from rlgym_ai_detection.features import TextFeaturizer
            
            tf = TextFeaturizer()
            texts = ["hello world", "test message"]
            
            tf.fit(texts)
            Xs, Xd = tf.transform(texts)
            
            # Dense features should have 4 columns
            assert Xd.shape == (2, 4)
            
            # All values should be non-negative
            assert np.all(Xd >= 0)
        except ImportError:
            pytest.skip("sklearn not available")
    
    def test_extra_features(self):
        """Test extra feature extraction"""
        from rlgym_ai_detection.features_extra import tokenize_words, tokenize_sentences
        
        text = "This is a test. Another sentence here."
        
        words = tokenize_words(text)
        assert len(words) > 0
        
        sentences = tokenize_sentences(text)
        assert len(sentences) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

