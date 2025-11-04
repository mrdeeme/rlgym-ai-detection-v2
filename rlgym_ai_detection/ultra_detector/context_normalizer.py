"""
Context Normalization Module for Ultra AI Detector v2.4.2

Handles special contexts that artificially inflate detection scores:
- Email threads and formatting
- Date inconsistencies in disclaimers
- Citation markers
- Unicode attacks (homoglyphs, zero-width)
"""
import re
from typing import Tuple, Dict, Any


class ContextNormalizer:
    """Normalizes text by removing or adjusting context-specific patterns"""
    
    def __init__(self):
        # Email patterns
        self.email_header_pattern = re.compile(
            r'^(From|To|Sent|Subject|Date|Cc|Bcc):.*?$',
            re.MULTILINE | re.IGNORECASE
        )
        self.email_quote_pattern = re.compile(r'^>\s*', re.MULTILINE)
        
        # Citation patterns
        self.citation_pattern = re.compile(r'\[\d+\]')
        
        # Date inconsistency patterns (in notes/disclaimers)
        self.disclaimer_pattern = re.compile(
            r'(nota|note|disclaimer|observa[çc][ãa]o):.*?(20\d{2}).*?(20\d{2})',
            re.IGNORECASE | re.DOTALL
        )
        
        # Unicode attack patterns
        self.cyrillic_lookalikes = {
            'А': 'A', 'а': 'a', 'В': 'B', 'Е': 'E', 'е': 'e',
            'К': 'K', 'к': 'k', 'М': 'M', 'Н': 'H', 'О': 'O',
            'о': 'o', 'Р': 'P', 'р': 'p', 'С': 'C', 'с': 'c',
            'Т': 'T', 'у': 'y', 'Х': 'X', 'х': 'x'
        }
        
        # Zero-width characters
        self.zero_width_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
        ]
    
    def normalize(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Normalize text by handling special contexts
        
        Returns:
            Tuple of (normalized_text, metadata)
        """
        metadata = {
            'email_detected': False,
            'citations_removed': 0,
            'disclaimer_detected': False,
            'unicode_normalized': False,
            'zero_width_removed': 0,
        }
        
        normalized = text
        
        # 1. Handle email threads
        if self._is_email_thread(normalized):
            normalized = self._extract_email_body(normalized)
            metadata['email_detected'] = True
        
        # 2. Remove citation markers
        citations_count = len(self.citation_pattern.findall(normalized))
        if citations_count > 0:
            normalized = self.citation_pattern.sub('', normalized)
            metadata['citations_removed'] = citations_count
        
        # 3. Handle date inconsistencies in disclaimers
        if self.disclaimer_pattern.search(normalized):
            normalized = self._remove_disclaimer(normalized)
            metadata['disclaimer_detected'] = True
        
        # 4. Normalize unicode attacks
        if self._has_cyrillic_lookalikes(normalized):
            normalized = self._normalize_cyrillic(normalized)
            metadata['unicode_normalized'] = True
        
        # 5. Remove zero-width characters
        zero_width_count = sum(normalized.count(char) for char in self.zero_width_chars)
        if zero_width_count > 0:
            for char in self.zero_width_chars:
                normalized = normalized.replace(char, '')
            metadata['zero_width_removed'] = zero_width_count
        
        return normalized, metadata
    
    def _is_email_thread(self, text: str) -> bool:
        """Check if text appears to be an email thread"""
        # Look for email headers
        has_headers = bool(self.email_header_pattern.search(text))
        # Look for quoted text
        has_quotes = bool(self.email_quote_pattern.search(text))
        return has_headers or has_quotes
    
    def _extract_email_body(self, text: str) -> str:
        """Extract the main body from an email thread"""
        # Remove email headers
        text = self.email_header_pattern.sub('', text)
        
        # Remove quoted lines (starting with >)
        lines = text.split('\n')
        body_lines = [line for line in lines if not line.strip().startswith('>')]
        
        # Rejoin and clean up
        body = '\n'.join(body_lines)
        body = re.sub(r'\n{3,}', '\n\n', body)  # Remove excessive newlines
        
        return body.strip()
    
    def _remove_disclaimer(self, text: str) -> str:
        """Remove disclaimer sections with date inconsistencies"""
        # Find and remove disclaimer sections
        parts = self.disclaimer_pattern.split(text)
        if len(parts) > 1:
            # Keep only the main text before the disclaimer
            return parts[0].strip()
        return text
    
    def _has_cyrillic_lookalikes(self, text: str) -> bool:
        """Check if text contains Cyrillic lookalike characters"""
        return any(char in text for char in self.cyrillic_lookalikes.keys())
    
    def _normalize_cyrillic(self, text: str) -> str:
        """Replace Cyrillic lookalikes with Latin equivalents"""
        for cyrillic, latin in self.cyrillic_lookalikes.items():
            text = text.replace(cyrillic, latin)
        return text
    
    def should_adjust_score(self, metadata: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if score should be adjusted based on normalization metadata
        
        Returns:
            Tuple of (should_adjust, adjustment_amount)
        """
        adjustment = 0.0
        
        # Email formatting adds artificial structure
        if metadata['email_detected']:
            adjustment -= 0.08  # Reduce score by 0.08
        
        # Citations add artificial formality
        if metadata['citations_removed'] > 0:
            adjustment -= min(0.05, metadata['citations_removed'] * 0.02)
        
        # Disclaimers add artificial inconsistency
        if metadata['disclaimer_detected']:
            adjustment -= 0.10  # Reduce score by 0.10
        
        # Unicode attacks should increase suspicion
        if metadata['unicode_normalized']:
            adjustment += 0.03  # Increase score by 0.03
        
        # Zero-width characters are suspicious
        if metadata['zero_width_removed'] > 0:
            adjustment += min(0.05, metadata['zero_width_removed'] * 0.01)
        
        should_adjust = abs(adjustment) > 0.001
        return should_adjust, adjustment


def normalize_text(text: str) -> Tuple[str, Dict[str, Any], float]:
    """
    Convenience function to normalize text and get score adjustment
    
    Returns:
        Tuple of (normalized_text, metadata, score_adjustment)
    """
    normalizer = ContextNormalizer()
    normalized, metadata = normalizer.normalize(text)
    should_adjust, adjustment = normalizer.should_adjust_score(metadata)
    
    return normalized, metadata, adjustment if should_adjust else 0.0

