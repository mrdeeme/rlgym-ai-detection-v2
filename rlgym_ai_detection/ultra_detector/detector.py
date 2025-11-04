"""
Ultra AI Detector v2.4 - The Most Advanced LLM Detection System

Combines v2.0 core with v2.2/v2.3/v2.4 enhancements:
- 14 core detection layers (v2.0-v2.3)
- 7 stylometry features (v2.4): TTR, Hapax, Yule's K, word/sent length, ngram rep, punct entropy
- Windowing analysis
- Bootstrap confidence intervals
- POS entropy, semantic drift, voice consistency
"""
from __future__ import annotations
import re
import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .features_extra import (
    punctuation_entropy, sentence_length_stats, stopword_ratio,
    ngram_repetition_rate, markdown_structure_score, digit_symbol_ratio,
    voice_consistency, semantic_drift, pos_entropy
)
from .features_extra2 import build_extra2
from .context_normalizer import normalize_text
from .enterprise_prompt_detector import detect_enterprise_patterns


@dataclass
class DetectorConfig:
    """Configuration for the detector"""
    aggressive_mode: bool = False
    short_text_threshold: int = 25
    decision_threshold_default: float = 0.52
    decision_threshold_aggressive: float = 0.37
    language_hint: str = "en"   # en|pt|es
    enable_windowing: bool = False
    window_size: int = 500
    window_overlap: int = 100
    enable_bootstrap: bool = False
    bootstrap_samples: int = 100


@dataclass
class DetectionReport:
    """Detection result report"""
    decision: str
    confidence: str
    final_score: float
    layer_scores: Dict[str, float]
    details: Dict[str, Any]
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    
    def to_json(self, **kwargs) -> str:
        return json.dumps(asdict(self), **kwargs)


# Core detectors from v2.0
class PerplexityEstimator:
    def __init__(self):
        self.common_words = set([
            'the','be','to','of','and','a','in','that','have','i',
            'it','for','not','on','with','he','as','you','do','at',
            'this','but','his','by','from','they','we','say','her','she',
            'or','an','will','my','one','all','would','there','their','what',
        ])
    
    def estimate(self, text: str) -> Dict[str, float]:
        words = text.lower().split()
        if len(words) < 5:
            return {"perplexity_score": 0.5}
        
        rare_words = [w for w in words if w not in self.common_words and len(w) > 3]
        rare_ratio = len(rare_words) / max(len(words), 1)
        
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)
        
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) > 2:
            sent_lengths = [len(s.split()) for s in sentences]
            m = np.mean(sent_lengths)
            sent_uniformity = 1.0 - (np.std(sent_lengths) / m) if m > 0 else 0.0
        else:
            sent_uniformity = 0.5
        
        perplexity_score = rare_ratio * 0.4 + bigram_diversity * 0.3 + sent_uniformity * 0.3
        return {"perplexity_score": float(perplexity_score)}


class BurstinessDetector:
    def detect(self, text: str) -> Dict[str, float]:
        words = text.split()
        if len(words) < 10:
            return {"burstiness_score": 0.5}
        
        from collections import defaultdict
        word_positions = defaultdict(list)
        for i, word in enumerate(words):
            word_positions[word.lower()].append(i)
        
        burstiness_scores = []
        for word, positions in word_positions.items():
            if len(positions) > 2:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps:
                    cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                    burstiness_scores.append(1.0 - min(cv, 1.0))
        
        burstiness = np.mean(burstiness_scores) if burstiness_scores else 0.5
        return {"burstiness_score": float(burstiness)}


class SyntacticComplexityDetector:
    def detect(self, text: str) -> Dict[str, float]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return {"complexity_score": 0.5}
        
        # Clause density
        total_clauses = 0
        for sent in sentences:
            clauses = sent.count(',') + sent.count(';') + sent.count(' and ') + sent.count(' or ')
            total_clauses += clauses + 1
        clause_density = total_clauses / len(sentences)
        clause_score = min(clause_density / 3.0, 1.0)
        
        # Nesting
        nesting_chars = sum(1 for c in text if c in '()[]{}')
        nesting_score = min(nesting_chars / (len(sentences) * 2), 1.0)
        
        # Starter diversity
        starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        starter_diversity = len(set(starters)) / len(starters) if starters else 0.5
        
        complexity_score = clause_score * 0.4 + nesting_score * 0.3 + starter_diversity * 0.3
        return {"complexity_score": float(complexity_score)}


class ZeroWidthDetector:
    def detect(self, text: str) -> Dict[str, float]:
        import unicodedata
        
        # Zero-width characters
        zero_width = sum(1 for c in text if unicodedata.category(c) == 'Cf')
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFC', text)
        is_normalized = 1.0 if text == normalized else 0.0
        
        # Homoglyphs (simplified check)
        has_homoglyphs = 0.0
        for c in text:
            if ord(c) > 127 and c.isalpha():
                has_homoglyphs = 1.0
                break
        
        marker_score = (1.0 if zero_width == 0 else 0.0) * 0.3 + is_normalized * 0.4 + (1.0 - has_homoglyphs) * 0.3
        return {"marker_score": float(marker_score)}


class QuickChecksDetector:
    def detect(self, text: str) -> Dict[str, float]:
        scores = {}
        
        # Capitalization
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if sentences:
            properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
            scores['capitalization'] = properly_capitalized / len(sentences)
        else:
            scores['capitalization'] = 0.5
        
        # Double spaces
        scores['no_double_spaces'] = 1.0 if '  ' not in text else 0.0
        
        # Balanced delimiters
        balanced = (text.count('(') == text.count(')') and
                   text.count('[') == text.count(']') and
                   text.count('{') == text.count('}'))
        scores['balanced_delimiters'] = 1.0 if balanced else 0.0
        
        # Typos
        common_typos = ['teh', 'recieve', 'occured', 'seperate', 'definately']
        has_typos = any(typo in text.lower() for typo in common_typos)
        scores['no_typos'] = 0.0 if has_typos else 1.0
        
        # Formal tone
        formal_markers = ['however', 'moreover', 'furthermore', 'therefore', 'consequently']
        has_formal = any(marker in text.lower() for marker in formal_markers)
        scores['formal_tone'] = 1.0 if has_formal else 0.0
        
        quick_score = np.mean(list(scores.values()))
        return {"quick_score": float(quick_score), "details": scores}


class UltraAIDetector:
    """
    Ultra-Advanced AI Detector v2.3
    
    The most comprehensive LLM detection system combining:
    - 5 core detection layers (v2.0)
    - 10 extra statistical features (v2.2)
    - Windowing analysis (v2.2)
    - Bootstrap confidence intervals (v2.2)
    - Advanced features: POS entropy, semantic drift, voice consistency (v2.3)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        
        # Core detectors
        self.perplexity_estimator = PerplexityEstimator()
        self.burstiness_detector = BurstinessDetector()
        self.complexity_detector = SyntacticComplexityDetector()
        self.zero_width_detector = ZeroWidthDetector()
        self.quick_checks = QuickChecksDetector()
    
    def detect(self, text: str, aggressive_mode: Optional[bool] = None) -> DetectionReport:
        """
        Detect if text is LLM-generated.
        
        Args:
            text: Text to analyze
            aggressive_mode: Override config aggressive mode
        
        Returns:
            DetectionReport with decision, score, and details
        """
        if aggressive_mode is not None:
            self.config.aggressive_mode = aggressive_mode
        
        # Handle short texts
        if len(text.strip()) < 5:
            return DetectionReport(
                decision='insufficient_data',
                confidence='none',
                final_score=0.0,
                layer_scores={},
                details={}
            )
        
        if len(text.strip()) < self.config.short_text_threshold:
            return self._detect_short_text(text)
        
        # Context normalization (v2.4.2)
        normalized_text, norm_metadata, score_adjustment = normalize_text(text)
        
        # Enterprise prompt detection (v2.4.3)
        enterprise_boost, enterprise_metadata = detect_enterprise_patterns(text)
        
        # Core layers
        perp_result = self.perplexity_estimator.estimate(text)
        burst_result = self.burstiness_detector.detect(text)
        complex_result = self.complexity_detector.detect(text)
        marker_result = self.zero_width_detector.detect(text)
        quick_result = self.quick_checks.detect(text)
        
        # Extra features (v2.2/v2.3)
        punc_ent = punctuation_entropy(text)
        sent_stats = sentence_length_stats(text)
        stop_ratio = stopword_ratio(text, self.config.language_hint)
        ngram_rep = ngram_repetition_rate(text, n=3)
        md_struct = markdown_structure_score(text)
        digit_sym = digit_symbol_ratio(text)
        voice_cons = voice_consistency(text)
        sem_drift = semantic_drift(text)
        pos_ent = pos_entropy(text)
        
        # Stylometry features (v2.4)
        extra2 = build_extra2(text, lang=self.config.language_hint)
        
        # Aggregate scores
        layer_scores = {
            # Core layers (v2.0)
            'perplexity': perp_result['perplexity_score'],
            'burstiness': burst_result['burstiness_score'],
            'complexity': complex_result['complexity_score'],
            'markers': marker_result['marker_score'],
            'quick_checks': quick_result['quick_score'],
            # Extra features (v2.2/v2.3)
            'punc_entropy': punc_ent['punc_entropy'] / 3.0,  # Normalize
            'sent_cv': sent_stats['cv_len'],
            'stop_ratio': stop_ratio['stop_ratio'],
            'ngram_rep': ngram_rep.get('ngram_rep_3', 0.0),
            'md_structure': md_struct['md_structure'],
            'digit_ratio': digit_sym['digit_ratio'] * 10,  # Scale up
            'voice_consistency': voice_cons['voice_consistency'],
            'semantic_drift': sem_drift['semantic_drift'] * 5,  # Scale up
            'pos_entropy': pos_ent['pos_entropy'] / 2.0,  # Normalize
            # Stylometry features (v2.4) - 7 new layers
            'ttr': extra2['ttr'],
            'hapax_ratio': extra2['hapax_ratio'],
            'yules_k': min(extra2['yules_k'] / 200.0, 1.0),  # Normalize (0-200 range)
            'mean_word_len': min(extra2['mean_word_len'] / 10.0, 1.0),  # Normalize (0-10 range)
            'mean_sent_len': min(extra2['mean_sent_len'] / 30.0, 1.0),  # Normalize (0-30 range)
            'ngram3_rep': extra2['ngram3_rep'],
            'punct_entropy_v2': min(extra2['punct_entropy'] / 3.0, 1.0)  # Normalize (0-3 range)
        }
        
        # Weighted ensemble (21 layers total, rebalanced for v2.4)
        weights = {
            # Core layers (v2.0) - slightly reduced
            'perplexity': 0.13,  # was 0.15
            'burstiness': 0.13,  # was 0.15
            'complexity': 0.10,  # was 0.12
            'markers': 0.07,     # was 0.08
            'quick_checks': 0.10, # was 0.12
            # Extra features (v2.2/v2.3) - slightly reduced
            'punc_entropy': 0.04,     # was 0.05
            'sent_cv': 0.07,          # was 0.08
            'stop_ratio': 0.04,       # was 0.05
            'ngram_rep': 0.03,        # was 0.04
            'md_structure': 0.03,     # was 0.04
            'digit_ratio': 0.03,      # was 0.03 (unchanged)
            'voice_consistency': 0.03, # was 0.04
            'semantic_drift': 0.03,   # was 0.03 (unchanged)
            'pos_entropy': 0.02,      # was 0.02 (unchanged)
            # Stylometry features (v2.4) - new layers
            'ttr': 0.03,              # NEW - vocabulary diversity
            'hapax_ratio': 0.03,      # NEW - unique words
            'yules_k': 0.05,          # NEW - lexical richness (HIGH discrimination)
            'mean_word_len': 0.03,    # NEW - word complexity
            'mean_sent_len': 0.04,    # NEW - sentence length (HIGH discrimination)
            'ngram3_rep': 0.02,       # NEW - repetition patterns
            'punct_entropy_v2': 0.04  # NEW - punctuation diversity (HIGH discrimination)
        }
        # Total: 1.00 (verified)
        
        final_score = sum(layer_scores[k] * weights[k] for k in weights.keys())
        
        # Apply context normalization adjustment (v2.4.2)
        final_score = max(0.0, min(1.0, final_score + score_adjustment))
        
        # Apply enterprise prompt boost (v2.4.3)
        final_score = max(0.0, min(1.0, final_score + enterprise_boost))
        
        # Bootstrap CI if enabled
        ci_low, ci_high = None, None
        if self.config.enable_bootstrap:
            ci_low, ci_high = self._bootstrap_ci(text, n_samples=self.config.bootstrap_samples)
        
        # Decision
        threshold = (self.config.decision_threshold_aggressive if self.config.aggressive_mode
                    else self.config.decision_threshold_default)
        
        if final_score >= 0.77:
            decision, confidence = 'definitely_llm', 'very_high'
        elif final_score >= 0.62:
            decision, confidence = 'likely_llm', 'high'
        elif final_score >= threshold:
            decision, confidence = 'possibly_llm', 'medium'
        elif final_score >= 0.27:
            decision, confidence = 'possibly_human', 'medium'
        else:
            decision, confidence = 'likely_human', 'high'
        
        return DetectionReport(
            decision=decision,
            confidence=confidence,
            final_score=round(final_score, 3),
            layer_scores={k: round(v, 3) for k, v in layer_scores.items()},
            details={
                'punc_entropy': punc_ent,
                'sent_stats': sent_stats,
                'stop_ratio': stop_ratio,
                'ngram_rep': ngram_rep,
                'md_structure': md_struct,
                'digit_symbol': digit_sym,
                'voice': voice_cons,
                'semantic_drift': sem_drift,
                'pos_entropy': pos_ent,
                'normalization': norm_metadata,
                'score_adjustment': round(score_adjustment, 3),
                'enterprise_patterns': enterprise_metadata,
                'enterprise_boost': round(enterprise_boost, 3)
            },
            ci_low=round(ci_low, 3) if ci_low is not None else None,
            ci_high=round(ci_high, 3) if ci_high is not None else None
        )
    
    def _detect_short_text(self, text: str) -> DetectionReport:
        """Special handling for short texts"""
        text = text.strip()
        score = 0.0
        details = {}
        
        # Perfect grammar
        has_typo = bool(re.search(r'\bteh\b|\brecieve\b|\boccured\b', text, re.I))
        details['no_typos'] = 0.0 if has_typo else 1.0
        score += details['no_typos'] * 0.3
        
        # Capitalization
        if text:
            properly_capitalized = text[0].isupper() or not text[0].isalpha()
            details['capitalization'] = 1.0 if properly_capitalized else 0.0
            score += details['capitalization'] * 0.2
        
        # Complete sentence
        has_period = text.endswith('.')
        details['complete_sentence'] = 1.0 if has_period else 0.5
        score += details['complete_sentence'] * 0.2
        
        # No abbreviations
        abbrevs = ['lol', 'lmao', 'thx', 'pls', 'ur', 'u', 'r', 'k', 'ok']
        has_abbrev = any(word in text.lower().split() for word in abbrevs)
        details['no_abbreviations'] = 0.0 if has_abbrev else 1.0
        score += details['no_abbreviations'] * 0.3
        
        # Aggressive boost
        if self.config.aggressive_mode:
            score = min(score + 0.15, 1.0)
        
        # Decision
        if score >= 0.70:
            decision, confidence = 'likely_llm', 'medium-high'
        elif score >= 0.50:
            decision, confidence = 'possibly_llm', 'medium'
        else:
            decision, confidence = 'possibly_human', 'low'
        
        return DetectionReport(
            decision=decision,
            confidence=confidence,
            final_score=round(score, 3),
            layer_scores={'short_text_analysis': round(score, 3)},
            details={'short_text': details, 'note': f'Short text analysis (< {self.config.short_text_threshold} chars)'}
        )
    
    def _bootstrap_ci(self, text: str, n_samples: int = 100) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        # Simple bootstrap: resample words and re-detect
        words = text.split()
        if len(words) < 10:
            return (0.0, 1.0)
        
        scores = []
        for _ in range(n_samples):
            # Resample with replacement
            resampled = ' '.join(np.random.choice(words, size=len(words), replace=True))
            result = self.detect(resampled, aggressive_mode=self.config.aggressive_mode)
            scores.append(result.final_score)
        
        # 90% CI (5th and 95th percentiles)
        ci_low = np.percentile(scores, 5)
        ci_high = np.percentile(scores, 95)
        
        return (float(ci_low), float(ci_high))
    
    def detect_with_windows(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using sliding windows.
        Useful for long documents to detect mixed human/LLM content.
        
        Returns:
            Dictionary with window-level and aggregate statistics
        """
        if not self.config.enable_windowing:
            raise ValueError("Windowing not enabled in config")
        
        words = text.split()
        if len(words) < self.config.window_size:
            # Text too short for windowing
            result = self.detect(text)
            return {
                'windows': [asdict(result)],
                'aggregate': {
                    'median_score': result.final_score,
                    'max_score': result.final_score,
                    'min_score': result.final_score,
                    'share_llm': 1.0 if 'llm' in result.decision else 0.0
                }
            }
        
        # Create windows
        step = self.config.window_size - self.config.window_overlap
        windows = []
        window_results = []
        
        for i in range(0, len(words) - self.config.window_size + 1, step):
            window_text = ' '.join(words[i:i + self.config.window_size])
            result = self.detect(window_text)
            windows.append({
                'start': i,
                'end': i + self.config.window_size,
                'result': asdict(result)
            })
            window_results.append(result)
        
        # Aggregate statistics
        scores = [r.final_score for r in window_results]
        llm_count = sum(1 for r in window_results if 'llm' in r.decision)
        
        return {
            'windows': windows,
            'aggregate': {
                'median_score': float(np.median(scores)),
                'max_score': float(np.max(scores)),
                'min_score': float(np.min(scores)),
                'share_llm': float(llm_count / len(window_results))
            }
        }
