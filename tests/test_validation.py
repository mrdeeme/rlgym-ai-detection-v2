"""Tests for input validation"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rlgym_ai_detection.pipeline import DetectionPipeline


class TestFitValidation:
    """Test validation in fit() method"""
    
    def test_fit_empty_texts(self):
        """Test error on empty texts"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="texts cannot be empty"):
            pipe.fit([], [])
    
    def test_fit_empty_labels(self):
        """Test error on empty labels"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="y cannot be empty"):
            pipe.fit(["test"], [])
    
    def test_fit_length_mismatch(self):
        """Test error on length mismatch"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="must have same length"):
            pipe.fit(["test1", "test2"], [1])
    
    def test_fit_too_few_samples(self):
        """Test error on too few samples"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="Need at least 4 samples"):
            pipe.fit(["test1", "test2"], [1, 0])
    
    def test_fit_invalid_labels(self):
        """Test error on invalid labels"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            pipe.fit(["a", "b", "c", "d"], [1, 2, 3, 4])
    
    def test_fit_single_class(self):
        """Test error on single class"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="must contain both classes"):
            pipe.fit(["a", "b", "c", "d"], [1, 1, 1, 1])
    
    def test_fit_langs_mismatch(self):
        """Test error on langs length mismatch"""
        pipe = DetectionPipeline()
        
        with pytest.raises(ValueError, match="must match texts"):
            pipe.fit(["a", "b", "c", "d"], [1, 0, 1, 0], langs=["EN", "pt-BR"])


class TestPredictValidation:
    """Test validation in predict_one() method"""
    
    @pytest.fixture
    def trained_pipe(self):
        """Provide a trained pipeline"""
        pipe = DetectionPipeline()
        try:
            pipe.fit(
                ["test1", "test2", "test3", "test4"],
                [1, 0, 1, 0]
            )
            return pipe
        except RuntimeError as e:
            if "scikit-learn" in str(e):
                pytest.skip("scikit-learn not installed")
            raise
    
    def test_predict_not_fitted(self):
        """Test error when predicting without fitting"""
        pipe = DetectionPipeline()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.predict_one("test")
    
    def test_predict_non_string(self, trained_pipe):
        """Test error on non-string input"""
        with pytest.raises(TypeError, match="text must be str"):
            trained_pipe.predict_one(123)
    
    def test_predict_empty_string(self, trained_pipe):
        """Test error on empty string"""
        with pytest.raises(ValueError, match="cannot be empty"):
            trained_pipe.predict_one("")
    
    def test_predict_whitespace_only(self, trained_pipe):
        """Test error on whitespace-only string"""
        with pytest.raises(ValueError, match="cannot be empty"):
            trained_pipe.predict_one("   ")
    
    def test_predict_invalid_risk_tier(self, trained_pipe):
        """Test error on invalid risk_tier"""
        with pytest.raises(ValueError, match="must be 'low', 'medium', or 'high'"):
            trained_pipe.predict_one("test", risk_tier="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

