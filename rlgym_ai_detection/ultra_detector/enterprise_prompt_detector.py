"""
Enterprise Prompt Detection Module for Ultra AI Detector v2.4.3

Detects patterns common in LLM-generated enterprise/technical prompts:
- Explicit enumeration requests ("3 things", "five ways")
- Superlative phrasing ("best", "fastest", "most modern")
- Formal exclusion clauses ("not X", "without having to Y")
- Technical precision markers (exact service names, formal capitalization)
"""
import re
from typing import Tuple, Dict, Any


class EnterprisePromptDetector:
    """Detects LLM-like patterns in enterprise/technical prompts"""
    
    def __init__(self):
        # Explicit enumeration patterns
        self.enumeration_patterns = [
            r'\b(the\s+)?(fastest|quickest|best|top|main)\s+\d+\s+(things|ways|steps|items|points)',
            r'\b\d+\s+(scariest|biggest|most\s+important|key|critical)\s+(things|issues|risks)',
            r'\bgive\s+me\s+(the\s+)?\d+',
            r'\blist\s+(the\s+)?\d+',
            r'\b(three|four|five|six|seven|eight|nine|ten)\s+(things|ways|steps|points)',
        ]
        
        # Superlative patterns
        self.superlative_patterns = [
            r'\babsolute\s+best\b',
            r'\bmost\s+(modern|efficient|effective|reliable|secure)\b',
            r'\bsimplest\s+(way|implementation|approach|solution)\b',
            r'\bfastest\s+(way|method|approach)\b',
            r'\bbest\s+practice',
        ]
        
        # Formal exclusion patterns
        self.exclusion_patterns = [
            r'\bnot\s+[\w\-]+\s+(jargon|buzzwords|fluff)\b',
            r'\bwithout\s+having\s+to\s+[\w\s]+\b',
            r'\bwithout\s+[\w\s]+ing\s+everything\b',
            r'\bnot\s+(super|too|overly)\s+\w+',
            r'\bor\s+just\s+(the\s+)?[\w\s]+\s+stuff\b',
        ]
        
        # Technical precision markers
        self.technical_precision_patterns = [
            r'\b(AWS|GCP|Azure)\s+[A-Z][A-Za-z0-9]+\b',  # Cloud services
            r'\bVMware\s+\w+\b',  # VMware products
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b.*\b(on|for|with)\s+(AWS|GCP|Azure)\b',  # "SAP on Azure"
            r'\b\d{3,4}\s+remote\s+users\b',  # Precise user counts
            r'\b(Postgres|PostgreSQL|MySQL)\b',  # Database names
        ]
        
        # Output format requests
        self.format_request_patterns = [
            r'\b(simple|quick)\s+(pro[/-]con|pros?\s+and\s+cons?)\s+list\b',
            r'\bjust\s+link\s+the\s+(patch|config|documentation)',
            r'\bneed\s+(actual|specific|concrete)\s+(steps|examples)',
            r'\bgive\s+me\s+a\s+list\b',
        ]
    
    def detect(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Detect enterprise prompt patterns
        
        Returns:
            Tuple of (score_boost, metadata)
        """
        text_lower = text.lower()
        
        metadata = {
            'enumeration_found': False,
            'superlatives_found': False,
            'exclusions_found': False,
            'technical_precision_found': False,
            'format_requests_found': False,
            'pattern_count': 0,
        }
        
        score_boost = 0.0
        pattern_count = 0
        
        # Check enumeration patterns
        for pattern in self.enumeration_patterns:
            if re.search(pattern, text_lower):
                metadata['enumeration_found'] = True
                pattern_count += 1
                score_boost += 0.02
                break
        
        # Check superlative patterns
        for pattern in self.superlative_patterns:
            if re.search(pattern, text_lower):
                metadata['superlatives_found'] = True
                pattern_count += 1
                score_boost += 0.015
                break
        
        # Check exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text_lower):
                metadata['exclusions_found'] = True
                pattern_count += 1
                score_boost += 0.01
                break
        
        # Check technical precision (case-sensitive)
        for pattern in self.technical_precision_patterns:
            if re.search(pattern, text):
                metadata['technical_precision_found'] = True
                pattern_count += 1
                score_boost += 0.01
                break
        
        # Check format requests
        for pattern in self.format_request_patterns:
            if re.search(pattern, text_lower):
                metadata['format_requests_found'] = True
                pattern_count += 1
                score_boost += 0.01
                break
        
        metadata['pattern_count'] = pattern_count
        
        # Cap the boost at +0.05
        score_boost = min(score_boost, 0.05)
        
        return score_boost, metadata


def detect_enterprise_patterns(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function to detect enterprise prompt patterns
    
    Returns:
        Tuple of (score_boost, metadata)
    """
    detector = EnterprisePromptDetector()
    return detector.detect(text)

