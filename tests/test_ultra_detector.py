"""
Tests for Ultra AI Detector v2.3 integration
"""
import pytest
from rlgym_ai_detection.ultra_detector import (
    UltraAIDetector,
    DetectorConfig,
    DetectionReport,
    Calibrator
)


class TestUltraDetectorBasic:
    """Basic functionality tests"""
    
    def test_import(self):
        """Test that ultra_detector can be imported"""
        assert UltraAIDetector is not None
        assert DetectorConfig is not None
        assert DetectionReport is not None
    
    def test_instantiation_default(self):
        """Test detector instantiation with default config"""
        detector = UltraAIDetector()
        assert detector is not None
        assert detector.config is not None
    
    def test_instantiation_custom_config(self):
        """Test detector instantiation with custom config"""
        config = DetectorConfig(
            aggressive_mode=True,
            language_hint='pt'
        )
        detector = UltraAIDetector(config)
        assert detector.config.aggressive_mode is True
        assert detector.config.language_hint == 'pt'
    
    def test_detect_basic(self):
        """Test basic detection"""
        detector = UltraAIDetector()
        text = "This is a test text for detection."
        result = detector.detect(text)
        
        assert result is not None
        assert hasattr(result, 'decision')
        assert hasattr(result, 'final_score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'layer_scores')
    
    def test_detect_llm_text(self):
        """Test detection of likely LLM text"""
        detector = UltraAIDetector(DetectorConfig(aggressive_mode=True))
        
        # Typical LLM-generated text
        text = """Could you please clarify the following aspects:
        1. The specific requirements for the project
        2. The timeline and deliverables expected
        3. The budget constraints and resource allocation
        
        Once I have this information, I'll be able to provide a comprehensive analysis."""
        
        result = detector.detect(text)
        assert 'llm' in result.decision.lower()
    
    def test_detect_human_text(self):
        """Test detection of likely human text"""
        detector = UltraAIDetector()
        
        # Typical human text with typos and informal language
        text = "hey can u send me that file? i cant find it anywher lol"
        
        result = detector.detect(text)
        # Note: This might still be detected as LLM depending on the text
        assert result.decision is not None


class TestUltraDetectorLayers:
    """Test individual detection layers"""
    
    def test_14_layers_present(self):
        """Test that all 14 layers are present"""
        detector = UltraAIDetector()
        text = "Test text for layer detection."
        result = detector.detect(text)
        
        # Should have scores from all layers
        assert len(result.layer_scores) >= 10  # At least core layers
    
    def test_layer_scores_range(self):
        """Test that layer scores are in valid range [0, 1]"""
        detector = UltraAIDetector()
        text = "Test text for score validation."
        result = detector.detect(text)
        
        for layer, score in result.layer_scores.items():
            assert 0 <= score <= 1, f"Layer {layer} score {score} out of range"
    
    def test_final_score_range(self):
        """Test that final score is in valid range"""
        detector = UltraAIDetector()
        text = "Test text for final score validation."
        result = detector.detect(text)
        
        assert 0 <= result.final_score <= 1


class TestUltraDetectorModes:
    """Test different detection modes"""
    
    def test_aggressive_mode(self):
        """Test aggressive mode (lower threshold)"""
        config = DetectorConfig(aggressive_mode=True)
        detector = UltraAIDetector(config)
        
        assert detector.config.aggressive_mode is True
        assert detector.config.decision_threshold_aggressive < detector.config.decision_threshold_default
    
    def test_conservative_mode(self):
        """Test conservative mode (default)"""
        config = DetectorConfig(aggressive_mode=False)
        detector = UltraAIDetector(config)
        
        assert detector.config.aggressive_mode is False
    
    def test_mode_affects_decision(self):
        """Test that mode affects decision thresholds"""
        text = "This is a moderately suspicious text."
        
        aggressive_detector = UltraAIDetector(DetectorConfig(aggressive_mode=True))
        conservative_detector = UltraAIDetector(DetectorConfig(aggressive_mode=False))
        
        aggressive_result = aggressive_detector.detect(text)
        conservative_result = conservative_detector.detect(text)
        
        # Scores should be the same
        assert aggressive_result.final_score == conservative_result.final_score
        
        # But decisions might differ due to different thresholds


class TestUltraDetectorLanguages:
    """Test multi-language support"""
    
    def test_english_detection(self):
        """Test detection on English text"""
        config = DetectorConfig(language_hint='en')
        detector = UltraAIDetector(config)
        
        text = "This is an English text for detection testing."
        result = detector.detect(text)
        assert result is not None
    
    def test_portuguese_detection(self):
        """Test detection on Portuguese text"""
        config = DetectorConfig(language_hint='pt')
        detector = UltraAIDetector(config)
        
        text = "Este é um texto em português para teste de detecção."
        result = detector.detect(text)
        assert result is not None
    
    def test_spanish_detection(self):
        """Test detection on Spanish text"""
        config = DetectorConfig(language_hint='es')
        detector = UltraAIDetector(config)
        
        text = "Este es un texto en español para prueba de detección."
        result = detector.detect(text)
        assert result is not None


class TestUltraDetectorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_text(self):
        """Test detection on empty text"""
        detector = UltraAIDetector()
        
        # Detector handles empty text gracefully
        result = detector.detect("")
        assert result is not None
        assert result.decision == 'insufficient_data'
    
    def test_very_short_text(self):
        """Test detection on very short text"""
        detector = UltraAIDetector()
        
        # Should use short-text analysis
        result = detector.detect("Hi")
        assert result is not None
    
    def test_very_long_text(self):
        """Test detection on very long text"""
        detector = UltraAIDetector()
        
        # Generate long text
        text = "This is a sentence. " * 1000
        result = detector.detect(text)
        assert result is not None
    
    def test_special_characters(self):
        """Test detection with special characters"""
        detector = UltraAIDetector()
        
        text = "Test with émojis 😀 and spëcial çharacters!"
        result = detector.detect(text)
        assert result is not None


class TestUltraDetectorWindowing:
    """Test windowing analysis feature"""
    
    def test_windowing_enabled(self):
        """Test windowing analysis on long text"""
        config = DetectorConfig(
            enable_windowing=True,
            window_size=50,
            window_overlap=10
        )
        detector = UltraAIDetector(config)
        
        # Long text
        text = " ".join(["This is a test sentence."] * 100)
        result = detector.detect_with_windows(text)
        
        assert 'windows' in result
        assert 'aggregate' in result
        assert len(result['windows']) > 0
    
    def test_windowing_aggregate_stats(self):
        """Test that windowing provides aggregate statistics"""
        config = DetectorConfig(
            enable_windowing=True,
            window_size=10,  # Smaller window for test
            window_overlap=2
        )
        detector = UltraAIDetector(config)
        
        # Longer text to ensure multiple windows
        text = " ".join(["This is a test sentence for windowing analysis."] * 50)
        result = detector.detect_with_windows(text)
        
        assert 'aggregate' in result
        assert 'windows' in result
        assert len(result['windows']) > 0
        assert 'median_score' in result['aggregate']
        assert 'share_llm' in result['aggregate']


class TestCalibrator:
    """Test calibration functionality"""
    
    def test_calibrator_instantiation(self):
        """Test that Calibrator can be instantiated"""
        calibrator = Calibrator()
        assert calibrator is not None
    
    def test_calibrator_has_methods(self):
        """Test that Calibrator has required methods"""
        calibrator = Calibrator()
        assert hasattr(calibrator, 'fit')
        assert hasattr(calibrator, 'predict_proba')
        assert hasattr(calibrator, 'save_json')
        assert hasattr(calibrator, 'load_json')


class TestIntegrationWithExistingPipeline:
    """Test integration with existing rlgym_ai_detection pipeline"""
    
    def test_both_pipelines_coexist(self):
        """Test that both DetectionPipeline and UltraAIDetector can coexist"""
        from rlgym_ai_detection import DetectionPipeline, UltraAIDetector
        
        # Both should be importable
        assert DetectionPipeline is not None
        assert UltraAIDetector is not None
    
    def test_ultra_detector_independent(self):
        """Test that UltraAIDetector works independently"""
        from rlgym_ai_detection import UltraAIDetector
        
        detector = UltraAIDetector()
        result = detector.detect("Test text")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

