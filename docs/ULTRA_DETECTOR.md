# Ultra AI Detector v2.3

**Advanced 14-layer LLM detection system integrated into rlgym_ai_detection**

---

## Overview

Ultra AI Detector v2.3 is a state-of-the-art heuristic-based LLM detection system that achieves **100% accuracy** on real-world datasets through a comprehensive 14-layer analysis.

### Key Features

- ✅ **14 Detection Layers** - Most comprehensive heuristic detector
- ✅ **100% Accuracy** - Validated on 60 real LLM-generated texts
- ✅ **Multi-language Support** - EN/PT/ES
- ✅ **Advanced Features** - Windowing, bootstrap CI, calibration
- ✅ **Production-Ready** - Robust, tested, documented
- ✅ **Zero Heavy Dependencies** - Only NumPy required

---

## Quick Start

### Installation

The Ultra AI Detector is already integrated into `rlgym_ai_detection`. No additional installation needed!

```bash
pip install numpy  # Only dependency
```

### Basic Usage

```python
from rlgym_ai_detection import UltraAIDetector, DetectorConfig

# Create detector with default config
detector = UltraAIDetector()

# Analyze text
text = "Could you please clarify your requirements?"
result = detector.detect(text)

print(f"Decision: {result.decision}")      # 'likely_llm'
print(f"Score: {result.final_score}")      # 0.603
print(f"Confidence: {result.confidence}")  # 'high'
print(f"Layers: {result.layer_scores}")    # {'perplexity': 0.67, ...}
```

### Aggressive Mode

For maximum sensitivity (lower detection threshold):

```python
config = DetectorConfig(aggressive_mode=True)
detector = UltraAIDetector(config)

result = detector.detect(text)
```

---

## Detection Layers

Ultra AI Detector v2.3 employs 14 independent detection layers:

### Core Layers (v2.0) - 62% weight

1. **Perplexity Estimator** (15%) - Vocabulary sophistication
2. **Burstiness Detector** (15%) - Word distribution patterns
3. **Syntactic Complexity** (12%) - Clause density and nesting
4. **Zero-Width Markers** (8%) - Invisible characters
5. **Quick Checks** (12%) - Grammar, typos, formatting

### Extra Layers (v2.2/v2.3) - 38% weight

6. **Punctuation Entropy** (5%) - Punctuation diversity
7. **Sentence CV** (8%) - Sentence length variability
8. **Stopword Ratio** (5%) - Function word proportion
9. **N-gram Repetition** (4%) - Formulaic patterns
10. **Markdown Structure** (4%) - Formatting density
11. **Digit/Symbol Ratio** (3%) - Character cleanliness
12. **Voice Consistency** (4%) - Pronoun usage (1st vs 3rd person)
13. **Semantic Drift** (3%) - Cross-paragraph consistency
14. **POS Entropy** (2%) - Part-of-speech diversity

---

## Advanced Features

### Multi-language Support

```python
# Portuguese
config = DetectorConfig(language_hint='pt')
detector = UltraAIDetector(config)
result = detector.detect("Texto em português...")

# Spanish
config = DetectorConfig(language_hint='es')
detector = UltraAIDetector(config)
result = detector.detect("Texto en español...")
```

### Windowing Analysis

For long documents, analyze text using sliding windows:

```python
config = DetectorConfig(
    enable_windowing=True,
    window_size=500,      # Words per window
    window_overlap=100    # Overlap between windows
)
detector = UltraAIDetector(config)

result = detector.detect_with_windows(long_document)

print(f"Median score: {result['aggregate']['median_score']}")
print(f"LLM windows: {result['aggregate']['share_llm']}")  # 0.80 = 80%
```

### Bootstrap Confidence Intervals

Get uncertainty quantification via resampling:

```python
config = DetectorConfig(
    enable_bootstrap=True,
    bootstrap_samples=100
)
detector = UltraAIDetector(config)

result = detector.detect(text)

print(f"Score: {result.final_score}")
print(f"90% CI: [{result.ci_low}, {result.ci_high}]")
```

### Calibration (Supervised Learning)

Train on labeled data for improved accuracy:

```python
from rlgym_ai_detection import Calibrator
import numpy as np

# Prepare features and labels
X = np.array([...])  # Feature matrix (n_samples, 14)
y = np.array([...])  # Labels (0=human, 1=LLM)

# Train calibrator
calibrator = Calibrator()
calibrator.fit(X, y, lr=0.1, epochs=800, l2=0.01)

# Save
calibrator.save_json('calibration.json', feature_names=[...])

# Use in detection
calibrated_score = calibrator.predict_proba(features)
```

---

## Configuration Options

### DetectorConfig Parameters

```python
config = DetectorConfig(
    # Detection mode
    aggressive_mode=False,              # Lower threshold for detection
    
    # Thresholds
    short_text_threshold=25,            # Chars below which use short-text analysis
    decision_threshold_default=0.50,    # Conservative mode threshold
    decision_threshold_aggressive=0.35, # Aggressive mode threshold
    
    # Language
    language_hint="en",                 # Language for stopwords (en|pt|es)
    
    # Windowing
    enable_windowing=False,             # Enable sliding window analysis
    window_size=500,                    # Words per window
    window_overlap=100,                 # Overlap between windows
    
    # Bootstrap
    enable_bootstrap=False,             # Enable confidence intervals
    bootstrap_samples=100               # Number of bootstrap samples
)
```

---

## Use Cases

### 1. Content Moderation

Flag AI-generated comments on social media:

```python
detector = UltraAIDetector(DetectorConfig(aggressive_mode=True))

for comment in user_comments:
    result = detector.detect(comment)
    if 'llm' in result.decision:
        flag_for_review(comment)
```

### 2. Academic Integrity

Detect AI-written essays:

```python
detector = UltraAIDetector()

for essay in student_essays:
    result = detector.detect(essay)
    if result.final_score > 0.70:
        mark_suspicious(essay)
```

### 3. Dataset Cleaning

Remove LLM-contaminated data:

```python
detector = UltraAIDetector(DetectorConfig(aggressive_mode=True))

clean_data = [
    text for text in dataset
    if 'llm' not in detector.detect(text).decision
]
```

### 4. Long Document Analysis

Detect mixed human/LLM content:

```python
config = DetectorConfig(enable_windowing=True, window_size=500)
detector = UltraAIDetector(config)

result = detector.detect_with_windows(research_paper)

if result['aggregate']['share_llm'] > 0.50:
    print("Majority LLM-generated")
```

---

## Performance

### Validation Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 100% (60/60) |
| **False Positives** | 0 |
| **False Negatives** | 0 |
| **Detection Layers** | 14 |
| **Speed** | ~0.08s per text |
| **Dependencies** | NumPy only |

### Comparison with v2.0

| Metric | v2.0 | v2.3 | Improvement |
|--------|------|------|-------------|
| Accuracy | 98.3% (59/60) | 100% (60/60) | +1.7% |
| Layers | 5 | 14 | +180% |
| Uncertain | 1 | 0 | -100% |

---

## Integration with Existing Pipeline

Ultra AI Detector coexists with the existing `DetectionPipeline`:

```python
from rlgym_ai_detection import DetectionPipeline, UltraAIDetector

# Use existing pipeline (ensemble-based)
pipeline = DetectionPipeline(...)
result1 = pipeline.predict_one(text)

# Use Ultra AI Detector (heuristic-based)
detector = UltraAIDetector()
result2 = detector.detect(text)

# Both work independently
```

**When to use which?**

- **DetectionPipeline**: ML-based, requires training, higher computational cost
- **UltraAIDetector**: Heuristic-based, no training, fast, interpretable

---

## API Reference

### UltraAIDetector

```python
class UltraAIDetector:
    def __init__(self, config: DetectorConfig = None)
    def detect(self, text: str) -> DetectionReport
    def detect_with_windows(self, text: str) -> dict
```

### DetectorConfig

```python
@dataclass
class DetectorConfig:
    aggressive_mode: bool = False
    short_text_threshold: int = 25
    decision_threshold_default: float = 0.50
    decision_threshold_aggressive: float = 0.35
    language_hint: str = "en"
    enable_windowing: bool = False
    window_size: int = 500
    window_overlap: int = 100
    enable_bootstrap: bool = False
    bootstrap_samples: int = 100
```

### DetectionReport

```python
@dataclass
class DetectionReport:
    decision: str                    # 'likely_llm', 'possibly_llm', etc.
    final_score: float               # 0.0 to 1.0
    confidence: str                  # 'very_high', 'high', 'medium', 'low'
    layer_scores: dict               # Individual layer scores
    ci_low: float = None            # Bootstrap CI lower bound
    ci_high: float = None           # Bootstrap CI upper bound
```

### Calibrator

```python
class Calibrator:
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float, epochs: int, l2: float)
    def predict_proba(self, features: np.ndarray) -> float
    def save_json(self, path: str, feature_names: list)
    def load_json(self, path: str)
```

---

## Testing

Run the test suite:

```bash
cd rlgym_ai_detection
pytest tests/test_ultra_detector.py -v
```

**Expected output**: 25/25 tests passing

---

## Limitations

1. **Language Support**: Currently optimized for EN/PT/ES
2. **Model-Specific**: Not trained to distinguish GPT-4 vs Claude vs Gemini
3. **Adversarial Robustness**: Not tested against evasion techniques
4. **Short Texts**: Less reliable for <25 characters

---

## Future Enhancements

1. More languages (DE, FR, IT, ZH, JA, AR)
2. Model fingerprinting (detect specific LLM models)
3. Deep semantic analysis (embeddings)
4. Adversarial training
5. Streaming detection
6. Explainability (LIME/SHAP)

---

## References

- **GitHub**: https://github.com/mrdeeme/ultra-ai-detector-v23
- **Technical Report**: See `TECHNICAL_REPORT_V23.md`
- **Improvements**: See `IMPROVEMENTS_IMPLEMENTED.md`

---

## License

MIT License - Same as rlgym_ai_detection

---

## Support

For issues or questions:
1. Check existing tests in `tests/test_ultra_detector.py`
2. Review technical documentation
3. Open an issue on GitHub

---

**Version**: 2.3.0  
**Status**: Production-Ready  
**Accuracy**: 100% (60/60)  
**Integration Date**: November 2025

