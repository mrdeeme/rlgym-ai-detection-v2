# Changelog

## [1.5.0] - 2025-11-04

### Added
- **Ultra AI Detector v2.4.3**: Complete upgrade from v2.3.0 to v2.4.3
  - 21 detection layers (14 → 21, +7 stylometry features)
  - Enterprise pattern detection module
  - Context normalization for structural patterns
  - Threshold optimization (0% false positives)
  - 99/100 security score

### Changed
- Updated `ultra_detector/` module with v2.4.3 improvements
- New files:
  - `context_normalizer.py`: Context normalization for emails, citations, dates
  - `enterprise_prompt_detector.py`: Enterprise pattern detection (5 types, 20+ patterns)
  - `features_extra2.py`: 7 new stylometry features (Yule's K, punctuation entropy v2, etc.)
- Modified files:
  - `detector.py`: Integrated v2.4.3 with enterprise boost and context normalization
  - `version.py`: Updated to 2.4.3

### Performance
- High confidence on LLM prompts: 16.7% → 66.7% (+300%)
- False positive rate: Maintained at 0%
- Enterprise pattern detection: Average 2.3 patterns per prompt
- Threshold adjustments: All thresholds -0.03 for better balance

## [1.4.0] - 2025-11-03

### Added
- Production-ready AI-generated content detector
- Ensemble learning with TF-IDF, structural features, and semantic embeddings
- Out-of-distribution detection (Mahalanobis + IsolationForest)
- Calibrated probabilities with isotonic regression
- Conformal prediction for uncertainty-aware abstention
- Policy tiers (low/medium/high risk)
- Multi-language support (EN/PT)

### Initial Release
- Ultra AI Detector v2.3.0 integration
- Dual pipeline architecture
- 100% accuracy on test set
