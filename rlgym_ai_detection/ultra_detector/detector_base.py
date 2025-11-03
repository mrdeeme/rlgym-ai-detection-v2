#!/usr/bin/env python3
"""
Ultra-Advanced AI Detection System - Revolutionary Approach
Uses cutting-edge techniques to achieve near-perfect detection
"""

import re
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any
import math


class PerplexityEstimator:
    """Estimates perplexity-like scores without a language model"""
    
    def __init__(self):
        # Common word frequencies (approximation)
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        ])
    
    def estimate(self, text: str) -> Dict[str, float]:
        """Estimate perplexity-like metrics"""
        words = text.lower().split()
        
        if len(words) < 5:
            return {'perplexity_score': 0.5, 'details': {}}
        
        # 1. Rare word ratio (LLMs use more diverse vocabulary)
        rare_words = [w for w in words if w not in self.common_words and len(w) > 3]
        rare_ratio = len(rare_words) / len(words)
        
        # 2. Bigram surprisal (LLMs create smoother transitions)
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # 3. Sentence-level uniformity (LLMs are more uniform)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 2:
            sent_lengths = [len(s.split()) for s in sentences]
            sent_uniformity = 1.0 - (np.std(sent_lengths) / np.mean(sent_lengths)) if np.mean(sent_lengths) > 0 else 0
        else:
            sent_uniformity = 0.5
        
        # Combined perplexity score
        # LLMs: higher rare_ratio, higher bigram_diversity, higher uniformity
        perplexity_score = (rare_ratio * 0.4 + bigram_diversity * 0.3 + sent_uniformity * 0.3)
        
        return {
            'perplexity_score': perplexity_score,
            'details': {
                'rare_ratio': rare_ratio,
                'bigram_diversity': bigram_diversity,
                'sentence_uniformity': sent_uniformity
            }
        }


class BurstinessDetector:
    """Detects burstiness patterns (humans are bursty, LLMs are uniform)"""
    
    def detect(self, text: str) -> Dict[str, float]:
        """Detect burstiness in word usage"""
        words = text.lower().split()
        
        if len(words) < 20:
            return {'burstiness_score': 0.5, 'details': {}}
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # 1. Burstiness coefficient (humans repeat words in bursts)
        # Calculate variance in word positions
        burstiness_scores = []
        for word, count in word_counts.items():
            if count > 1:
                positions = [i for i, w in enumerate(words) if w == word]
                if len(positions) > 1:
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    if gaps:
                        # Low variance in gaps = bursty (human)
                        # High variance = uniform (LLM)
                        gap_cv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
                        burstiness_scores.append(1.0 - min(gap_cv, 1.0))
        
        burstiness = np.mean(burstiness_scores) if burstiness_scores else 0.5
        
        # 2. Lexical diversity over time (LLMs maintain consistent diversity)
        chunk_size = max(len(words) // 4, 10)
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        
        if len(chunks) > 2:
            chunk_ttrs = [len(set(chunk)) / len(chunk) for chunk in chunks if chunk]
            ttr_variance = np.std(chunk_ttrs) if chunk_ttrs else 0
            # Low variance = LLM (consistent), high variance = human (variable)
            diversity_uniformity = 1.0 - min(ttr_variance / 0.2, 1.0)
        else:
            diversity_uniformity = 0.5
        
        # LLMs score lower on burstiness, higher on uniformity
        llm_score = (1.0 - burstiness) * 0.6 + diversity_uniformity * 0.4
        
        return {
            'burstiness_score': llm_score,
            'details': {
                'burstiness': burstiness,
                'diversity_uniformity': diversity_uniformity
            }
        }


class SyntacticComplexityDetector:
    """Detects syntactic complexity patterns"""
    
    def detect(self, text: str) -> Dict[str, float]:
        """Detect syntactic complexity"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return {'complexity_score': 0.5, 'details': {}}
        
        scores = {}
        
        # 1. Clause density (LLMs use more clauses per sentence)
        total_clauses = 0
        for sent in sentences:
            # Count commas, semicolons, and conjunctions as clause indicators
            clauses = sent.count(',') + sent.count(';') + sent.count(' and ') + sent.count(' but ') + sent.count(' or ')
            total_clauses += clauses + 1  # +1 for main clause
        
        clause_density = total_clauses / len(sentences)
        # LLMs: 2-4 clauses per sentence, humans: more variable
        scores['clause_density'] = min(clause_density / 3.0, 1.0)
        
        # 2. Dependency depth (approximate via nested structures)
        nested_count = 0
        for sent in sentences:
            # Count parentheses, dashes, colons as nesting indicators
            nested_count += sent.count('(') + sent.count('--') + sent.count(':')
        
        nesting_ratio = nested_count / len(sentences)
        scores['nesting'] = min(nesting_ratio / 0.5, 1.0)
        
        # 3. Sentence starter diversity (LLMs vary sentence starters)
        starters = []
        for sent in sentences:
            words = sent.split()
            if words:
                starter = words[0].lower()
                starters.append(starter)
        
        if starters:
            starter_diversity = len(set(starters)) / len(starters)
            scores['starter_diversity'] = starter_diversity
        else:
            scores['starter_diversity'] = 0.5
        
        # Combined complexity score
        complexity_score = np.mean(list(scores.values()))
        
        return {
            'complexity_score': complexity_score,
            'details': scores
        }


class ZeroWidthDetector:
    """Detects zero-width characters and hidden markers"""
    
    def detect(self, text: str) -> Dict[str, float]:
        """Detect hidden characters and markers"""
        scores = {}
        
        # 1. Zero-width characters (sometimes used in AI watermarking)
        zero_width_chars = [
            '\u200B',  # Zero width space
            '\u200C',  # Zero width non-joiner
            '\u200D',  # Zero width joiner
            '\uFEFF',  # Zero width no-break space
        ]
        
        zw_count = sum(text.count(char) for char in zero_width_chars)
        scores['zero_width'] = min(zw_count / 5.0, 1.0)
        
        # 2. Unicode normalization (LLMs use consistent normalization)
        try:
            import unicodedata
            nfc = unicodedata.normalize('NFC', text)
            nfd = unicodedata.normalize('NFD', text)
            # If text is already normalized, it's more likely LLM
            scores['normalized'] = 1.0 if text == nfc else 0.3
        except:
            scores['normalized'] = 0.5
        
        # 3. Homoglyph detection (LLMs rarely use lookalike characters)
        homoglyphs = ['а', 'е', 'о', 'р', 'с', 'у', 'х']  # Cyrillic that look like Latin
        homoglyph_count = sum(text.count(char) for char in homoglyphs)
        # Absence of homoglyphs suggests LLM
        scores['no_homoglyphs'] = 1.0 - min(homoglyph_count / 3.0, 1.0)
        
        # Combined score
        marker_score = np.mean(list(scores.values()))
        
        return {
            'marker_score': marker_score,
            'details': scores
        }


class UltraAdvancedDetector:
    """Ultra-advanced AI detector with revolutionary techniques"""
    
    def __init__(self):
        self.perplexity_estimator = PerplexityEstimator()
        self.burstiness_detector = BurstinessDetector()
        self.complexity_detector = SyntacticComplexityDetector()
        self.zero_width_detector = ZeroWidthDetector()
    
    def detect(self, text: str, aggressive_mode: bool = True) -> Dict[str, Any]:
        """
        Detect LLM-generated text with ultra-advanced techniques
        
        Args:
            text: Text to analyze
            aggressive_mode: If True, lower threshold for LLM detection
        
        Returns:
            Detection report
        """
        if not text or len(text.strip()) < 5:
            return {
                'decision': 'insufficient_data',
                'confidence': 0.0,
                'final_score': 0.0,
                'details': {}
            }
        
        # Special handling for very short texts (5-25 chars)
        if len(text.strip()) < 25:
            return self._detect_short_text(text, aggressive_mode)
        
        # Run all detectors
        perplexity_result = self.perplexity_estimator.estimate(text)
        burstiness_result = self.burstiness_detector.detect(text)
        complexity_result = self.complexity_detector.detect(text)
        marker_result = self.zero_width_detector.detect(text)
        
        # Additional quick checks
        quick_scores = self._quick_llm_checks(text)
        
        # Weighted ensemble (aggressive weights)
        if aggressive_mode:
            weights = {
                'perplexity': 0.25,
                'burstiness': 0.25,
                'complexity': 0.20,
                'markers': 0.10,
                'quick': 0.20,
            }
            threshold_llm = 0.35  # Lower threshold
        else:
            weights = {
                'perplexity': 0.20,
                'burstiness': 0.20,
                'complexity': 0.20,
                'markers': 0.15,
                'quick': 0.25,
            }
            threshold_llm = 0.50
        
        final_score = (
            perplexity_result['perplexity_score'] * weights['perplexity'] +
            burstiness_result['burstiness_score'] * weights['burstiness'] +
            complexity_result['complexity_score'] * weights['complexity'] +
            marker_result['marker_score'] * weights['markers'] +
            quick_scores['quick_score'] * weights['quick']
        )
        
        # Decision logic (aggressive)
        if final_score >= 0.65:
            decision = 'definitely_llm'
            confidence = 'very_high'
        elif final_score >= 0.50:
            decision = 'likely_llm'
            confidence = 'high'
        elif final_score >= threshold_llm:
            decision = 'possibly_llm'
            confidence = 'medium'
        elif final_score >= 0.25:
            decision = 'possibly_human'
            confidence = 'medium'
        else:
            decision = 'likely_human'
            confidence = 'low'
        
        # In aggressive mode, bias toward LLM
        if aggressive_mode and final_score >= threshold_llm:
            if decision == 'possibly_llm':
                decision = 'likely_llm'
                confidence = 'medium-high'
        
        return {
            'decision': decision,
            'confidence': confidence,
            'final_score': round(final_score, 3),
            'layer_scores': {
                'perplexity': round(perplexity_result['perplexity_score'], 3),
                'burstiness': round(burstiness_result['burstiness_score'], 3),
                'complexity': round(complexity_result['complexity_score'], 3),
                'markers': round(marker_result['marker_score'], 3),
                'quick_checks': round(quick_scores['quick_score'], 3),
            },
            'details': {
                'perplexity': perplexity_result['details'],
                'burstiness': burstiness_result['details'],
                'complexity': complexity_result['details'],
                'markers': marker_result['details'],
                'quick_checks': quick_scores['details'],
            }
        }
    
    def _quick_llm_checks(self, text: str) -> Dict[str, Any]:
        """Quick heuristic checks for LLM patterns"""
        scores = {}
        
        # 1. Perfect capitalization (LLMs are consistent)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            properly_capitalized = sum(1 for s in sentences if s[0].isupper())
            scores['capitalization'] = properly_capitalized / len(sentences)
        else:
            scores['capitalization'] = 0.5
        
        # 2. Consistent spacing (LLMs don't have typos)
        double_spaces = text.count('  ')
        scores['no_double_spaces'] = 1.0 - min(double_spaces / 5.0, 1.0)
        
        # 3. Balanced quotes and parentheses (LLMs always balance)
        open_parens = text.count('(')
        close_parens = text.count(')')
        open_quotes = text.count('"') + text.count('"')
        close_quotes = text.count('"') + text.count('"')
        
        paren_balanced = 1.0 if open_parens == close_parens else 0.0
        quote_balanced = 1.0 if open_quotes == close_quotes or (open_quotes + close_quotes) % 2 == 0 else 0.0
        
        scores['balanced_delimiters'] = (paren_balanced + quote_balanced) / 2.0
        
        # 4. No typos (approximate via spell-check patterns)
        # Common typos humans make
        typo_patterns = [
            r'\bteh\b', r'\brecieve\b', r'\boccured\b', r'\bseperate\b',
            r'\bdefinately\b', r'\bwierd\b', r'\buntill\b'
        ]
        typo_count = sum(len(re.findall(p, text, re.I)) for p in typo_patterns)
        scores['no_typos'] = 1.0 - min(typo_count / 2.0, 1.0)
        
        # 5. Consistent tense (LLMs maintain tense better)
        past_tense = len(re.findall(r'\b\w+ed\b', text))
        present_tense = len(re.findall(r'\b\w+s\b', text))
        total_verbs = past_tense + present_tense
        
        if total_verbs > 5:
            tense_ratio = max(past_tense, present_tense) / total_verbs
            scores['consistent_tense'] = tense_ratio
        else:
            scores['consistent_tense'] = 0.5
        
        # 6. Formal tone markers
        formal_markers = [
            'however', 'moreover', 'furthermore', 'therefore', 'thus',
            'additionally', 'consequently', 'nevertheless', 'nonetheless'
        ]
        formal_count = sum(1 for word in text.lower().split() if word in formal_markers)
        scores['formal_tone'] = min(formal_count / 3.0, 1.0)
        
        # Combined quick score
        quick_score = np.mean(list(scores.values()))
        
        return {
            'quick_score': quick_score,
            'details': scores
        }
    
    def _detect_short_text(self, text: str, aggressive_mode: bool) -> Dict[str, Any]:
        """Special detection for very short texts (< 25 chars)"""
        text = text.strip()
        score = 0.0
        details = {}
        
        # 1. Perfect grammar (no typos)
        has_typo = bool(re.search(r'\bteh\b|\brecieve\b|\boccured\b', text, re.I))
        details['no_typos'] = 0.0 if has_typo else 1.0
        score += details['no_typos'] * 0.3
        
        # 2. Proper capitalization
        if text:
            properly_capitalized = text[0].isupper() or not text[0].isalpha()
            details['capitalization'] = 1.0 if properly_capitalized else 0.0
            score += details['capitalization'] * 0.2
        
        # 3. Complete sentence structure
        has_period = text.endswith('.')
        details['complete_sentence'] = 1.0 if has_period else 0.5
        score += details['complete_sentence'] * 0.2
        
        # 4. Formal/complete words (not abbreviations)
        abbrevs = ['lol', 'lmao', 'thx', 'pls', 'ur', 'u', 'r', 'k', 'ok']
        has_abbrev = any(word in text.lower().split() for word in abbrevs)
        details['no_abbreviations'] = 0.0 if has_abbrev else 1.0
        score += details['no_abbreviations'] * 0.3
        
        # In aggressive mode, assume short well-formed text is LLM
        if aggressive_mode:
            score = min(score + 0.15, 1.0)
        
        # Decision
        if score >= 0.65:
            decision = 'likely_llm'
            confidence = 'medium-high'
        elif score >= 0.45:
            decision = 'possibly_llm'
            confidence = 'medium'
        else:
            decision = 'possibly_human'
            confidence = 'low'
        
        return {
            'decision': decision,
            'confidence': confidence,
            'final_score': round(score, 3),
            'layer_scores': {
                'short_text_analysis': round(score, 3),
            },
            'details': {
                'short_text': details,
                'note': 'Short text analysis (< 25 chars)'
            }
        }


if __name__ == '__main__':
    detector = UltraAdvancedDetector()
    
    # Test on LLM-like text
    test_text = """
    Could you please clarify a few things to help me provide the best answer?
    
    1. What specific industry are you targeting?
    2. What is your budget range for this project?
    
    Once I have this information, I can tailor my recommendations to your needs.
    """
    
    result = detector.detect(test_text, aggressive_mode=True)
    print(json.dumps(result, indent=2))

