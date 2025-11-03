"""Comprehensive tests for DetectionPipeline"""
import pytest
import sys
import os
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rlgym_ai_detection.pipeline import DetectionPipeline, default_embedder
from rlgym_ai_detection.policies import load_policy


class TestDetectionPipeline:
    """Test suite for DetectionPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample training data"""
        texts = [
            "You are a senior cloud architect. Goal: design a Terraform module for AWS deployment.",
            "pode ver se o relatório saiu? preciso enviar até 10h.",
            "Role: Technical Writer. Objective: write comprehensive OpenAPI specification.",
            "moved the call to 14:30, link in calendar.",
            "As an experienced DevOps engineer, create a CI/CD pipeline using GitHub Actions.",
            "alguém sabe onde ficou o documento da reunião?",
        ]
        y = [1, 0, 1, 0, 1, 0]
        langs = ["EN", "pt-BR", "EN", "EN", "EN", "pt-BR"]
        return texts, y, langs
    
    @pytest.fixture
    def trained_pipeline(self, sample_data):
        """Provide a trained pipeline"""
        texts, y, langs = sample_data
        try:
            pipe = DetectionPipeline(policy=load_policy(), embed_fn=default_embedder())
            pipe.fit(texts, y, langs=langs)
            return pipe
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        pipe = DetectionPipeline()
        assert pipe is not None
        assert pipe.model_ready is False
        assert pipe.policy is not None
        assert pipe.embed is not None
    
    def test_pipeline_fit(self, sample_data):
        """Test pipeline can be fitted with data"""
        texts, y, langs = sample_data
        try:
            pipe = DetectionPipeline()
            result = pipe.fit(texts, y, langs=langs)
            assert result is pipe  # Should return self
            assert pipe.model_ready is True
            assert pipe.learners is not None
            assert pipe.calib is not None
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise
    
    def test_pipeline_predict_one(self, trained_pipeline):
        """Test pipeline can make predictions"""
        report = trained_pipeline.predict_one(
            "System prompt: Act as SRE. Task: write runbook for incident response.",
            risk_tier="high"
        )
        
        # Validate report structure
        assert "decision" in report
        assert "prob_llm" in report
        assert "ood_scores" in report
        assert "confidence_band" in report
        assert "signals" in report
        
        # Validate decision values
        assert report["decision"] in ["likely_human", "likely_llm", "abstain"]
        
        # Validate probability range
        assert 0.0 <= report["prob_llm"] <= 1.0
        
        # Validate OOD scores
        assert "mahalanobis" in report["ood_scores"]
        assert "iforest" in report["ood_scores"]
        
        # Validate signals
        assert "struct" in report["signals"]
        assert "style" in report["signals"]
        assert "lex" in report["signals"]
    
    def test_pipeline_predict_human_text(self, trained_pipeline):
        """Test prediction on clearly human text"""
        report = trained_pipeline.predict_one(
            "hey, can you send me the link? thx",
            risk_tier="low"
        )
        # Should have low probability of being LLM
        assert report["prob_llm"] < 0.7
    
    def test_pipeline_predict_llm_text(self, trained_pipeline):
        """Test prediction on clearly LLM text"""
        report = trained_pipeline.predict_one(
            "As a senior software engineer, your task is to implement a comprehensive "
            "authentication system with JWT tokens, refresh mechanisms, and OAuth2 support. "
            "Ensure proper error handling and security best practices.",
            risk_tier="high"
        )
        # Should have high probability of being LLM
        assert report["prob_llm"] > 0.3
    
    def test_pipeline_multilingual(self, trained_pipeline):
        """Test pipeline handles multiple languages"""
        # Portuguese
        report_pt = trained_pipeline.predict_one(
            "preciso do relatório até amanhã",
            risk_tier="low"
        )
        assert report_pt is not None
        
        # English
        report_en = trained_pipeline.predict_one(
            "need the report by tomorrow",
            risk_tier="low"
        )
        assert report_en is not None
    
    def test_pipeline_different_risk_tiers(self, trained_pipeline):
        """Test pipeline respects different risk tiers"""
        text = "Implement a microservices architecture with proper service discovery."
        
        # Low risk tier
        report_low = trained_pipeline.predict_one(text, risk_tier="low")
        
        # High risk tier
        report_high = trained_pipeline.predict_one(text, risk_tier="high")
        
        # Both should return valid reports
        assert report_low["decision"] in ["likely_human", "likely_llm", "abstain"]
        assert report_high["decision"] in ["likely_human", "likely_llm", "abstain"]
    
    def test_pipeline_signals_structure(self, trained_pipeline):
        """Test signals contain expected metrics"""
        report = trained_pipeline.predict_one(
            "This is a test message with some code: ```python\nprint('hello')\n```",
            risk_tier="medium"
        )
        
        signals = report["signals"]
        
        # Structural signals
        assert "length_chars" in signals["struct"]
        assert "code_fences" in signals["struct"]
        assert signals["struct"]["code_fences"] > 0  # Should detect code fence
        
        # Style signals
        assert "coherence_density" in signals["style"]
        assert "coherence_gap" in signals["style"]
        
        # Lexical signals
        assert "ttr" in signals["lex"]
        assert 0.0 <= signals["lex"]["ttr"] <= 1.0
    
    def test_pipeline_version_info(self, trained_pipeline):
        """Test report includes version information"""
        report = trained_pipeline.predict_one("test", risk_tier="medium")
        
        assert "version" in report
        assert report["version"] == "1.4.0"
        assert "model_version" in report
        assert "policy_version" in report
    
    def test_pipeline_not_fitted_error(self):
        """Test error when predicting without fitting"""
        pipe = DetectionPipeline()
        
        with pytest.raises(RuntimeError, match="Pipeline not fitted"):
            pipe.predict_one("test", risk_tier="medium")
    
    def test_default_embedder(self):
        """Test default embedder works"""
        embedder = default_embedder()
        texts = ["hello world", "test message"]
        embeddings = embedder(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Should have some dimensions
    
    def test_pipeline_with_custom_policy(self, sample_data):
        """Test pipeline with custom policy"""
        custom_policy = {
            "policy_version": "test-1.0",
            "tiers": {
                "medium": {
                    "abstain_band": [0.3, 0.7],
                    "ood_threshold": 1.5
                }
            }
        }
        
        texts, y, langs = sample_data
        try:
            pipe = DetectionPipeline(policy=custom_policy)
            pipe.fit(texts, y, langs=langs)
            
            report = pipe.predict_one("test", risk_tier="medium")
            assert report["policy_version"] == "test-1.0"
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_text(self):
        """Test handling of empty text"""
        pipe = DetectionPipeline()
        # Should not crash
        try:
            texts = ["hello", "world"]
            y = [1, 0]
            pipe.fit(texts, y)
            report = pipe.predict_one("", risk_tier="low")
            assert report is not None
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        pipe = DetectionPipeline()
        try:
            texts = ["hello " * 100, "world " * 100]
            y = [1, 0]
            pipe.fit(texts, y)
            
            long_text = "test " * 1000
            report = pipe.predict_one(long_text, risk_tier="medium")
            assert report is not None
            assert report["signals"]["struct"]["length_chars"] > 4000
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise
    
    def test_special_characters(self):
        """Test handling of special characters"""
        pipe = DetectionPipeline()
        try:
            texts = ["hello @#$%", "world 123"]
            y = [1, 0]
            pipe.fit(texts, y)
            
            report = pipe.predict_one("test @#$% 123 !@#", risk_tier="low")
            assert report is not None
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

