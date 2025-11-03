"""Extra feature extractors for v2.3"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List

# Minimal stopword lists for EN/PT/ES
STOP_EN = set("""a an the and or but if then else of to in on for from with by as is are was were be been being do does did done at this that these those it its it's you your we our they their he she his her them one all any not no yes which who whom whose what when where why how into out about over under more most less many few much such own same other some each either neither both between again further here there""".split())

STOP_PT = set("""o a os as e ou mas se entao senao de do da dos das em no na nos nas para por com como é são foi eram ser estar estarão fazer faz fez feito ao aos à às este esta isto aquele aquela aquilo você seu sua nós nosso nossa eles delas ele ela um uma uns umas não sim que quem cujo qual quando onde porquê porque quanto""".split())

STOP_ES = set("""el la los las y o pero si entonces sino de del en para por con como es son fue eran ser estar estarán hacer hace hizo hecho al a los las este esta esto aquel aquella aquello usted su sus nosotros nuestro nuestra ellos ellas él ella un una unos unas no sí que quien cuyo cual cuando donde porqué porque cuanto""".split())


def _sentences(text: str) -> List[str]:
    """Split text into sentences"""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _words(text: str) -> List[str]:
    """Extract words from text"""
    return [t for t in re.findall(r"[\w']+", text.lower()) if t]


def punctuation_entropy(text: str) -> Dict[str, float]:
    """
    Calculate entropy of punctuation marks.
    Higher entropy = more diverse punctuation (more human-like).
    LLMs tend to use punctuation more uniformly.
    """
    punc = [c for c in text if c in ",;:.!?"]
    if not punc:
        return {"punc_entropy": 0.0, "punc_count": 0}
    
    c = Counter(punc)
    total = sum(c.values())
    H = -sum((v/total) * math.log2(v/total) for v in c.values())
    
    return {"punc_entropy": float(H), "punc_count": int(total)}


def sentence_length_stats(text: str) -> Dict[str, float]:
    """
    Calculate statistics of sentence lengths.
    High CV (coefficient of variation) = more human-like variability.
    LLMs maintain more consistent sentence lengths.
    """
    sents = _sentences(text)
    if len(sents) < 2:
        return {"mean_len": 0.0, "std_len": 0.0, "cv_len": 0.0}
    
    lens = [len(s.split()) for s in sents]
    m = sum(lens) / len(lens)
    var = sum((x - m)**2 for x in lens) / (len(lens) - 1)
    std = var ** 0.5
    cv = std / m if m > 0 else 0.0
    
    return {"mean_len": float(m), "std_len": float(std), "cv_len": float(cv)}


def stopword_ratio(text: str, lang: str = "en") -> Dict[str, float]:
    """
    Calculate ratio of stopwords.
    Very low stopword ratio can indicate overly "clean" LLM text.
    """
    w = _words(text)
    if not w:
        return {"stop_ratio": 0.0}
    
    if lang == "pt":
        stop = STOP_PT
    elif lang == "es":
        stop = STOP_ES
    else:
        stop = STOP_EN
    
    sw = sum(1 for t in w if t in stop)
    return {"stop_ratio": float(sw / len(w))}


def ngram_repetition_rate(text: str, n: int = 3) -> Dict[str, float]:
    """
    Calculate n-gram repetition rate.
    High repetition can indicate formulaic LLM patterns.
    """
    w = _words(text)
    if len(w) < n + 1:
        return {f"ngram_rep_{n}": 0.0}
    
    grams = [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    c = Counter(grams)
    rep = sum(v for v in c.values() if v > 1) / len(grams)
    
    return {f"ngram_rep_{n}": float(rep)}


def markdown_structure_score(text: str) -> Dict[str, float]:
    """
    Evaluate markdown structure.
    Mid-range density looks human; extremely low or high looks suspect.
    LLMs love numbered lists and perfect formatting.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {"md_structure": 0.0}
    
    bullets = sum(1 for ln in lines if re.match(r"^[-*+]\s", ln))
    numbered = sum(1 for ln in lines if re.match(r"^\d+\.\s", ln))
    headers = sum(1 for ln in lines if re.match(r"^#{1,6}\s", ln))
    total = len(lines)
    density = (bullets + numbered + headers) / total
    
    # Mid-range density looks human; extremely low or high looks suspect
    if density < 0.02:
        score = 0.2
    elif density > 0.5:
        score = 0.3
    else:
        score = 0.8
    
    return {"md_structure": float(score)}


def digit_symbol_ratio(text: str) -> Dict[str, float]:
    """
    Calculate ratio of digits and special symbols.
    Very clean text (low ratio) can indicate LLM.
    """
    if not text:
        return {"digit_ratio": 0.0, "symbol_ratio": 0.0}
    
    digits = sum(1 for c in text if c.isdigit())
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    total_chars = len(text)
    
    return {
        "digit_ratio": float(digits / total_chars) if total_chars > 0 else 0.0,
        "symbol_ratio": float(symbols / total_chars) if total_chars > 0 else 0.0
    }


def voice_consistency(text: str) -> Dict[str, float]:
    """
    Detect voice consistency (1st person vs 3rd person).
    Frequent switches can indicate human writing.
    LLMs tend to maintain consistent voice.
    """
    first_person = len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', text, re.IGNORECASE))
    third_person = len(re.findall(r'\b(he|him|his|she|her|hers|they|them|their|theirs)\b', text, re.IGNORECASE))
    
    total_pronouns = first_person + third_person
    if total_pronouns == 0:
        return {"voice_consistency": 0.5, "first_person_ratio": 0.0}
    
    # High consistency (all one voice) = more LLM-like
    first_ratio = first_person / total_pronouns
    consistency = max(first_ratio, 1 - first_ratio)
    
    return {
        "voice_consistency": float(consistency),
        "first_person_ratio": float(first_ratio)
    }


def semantic_drift(text: str) -> Dict[str, float]:
    """
    Measure semantic drift between paragraphs.
    High drift = inconsistent style (more human).
    Low drift = consistent style (more LLM).
    
    Uses simple lexical overlap as proxy for semantic similarity.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        return {"semantic_drift": 0.0}
    
    # Calculate lexical overlap between consecutive paragraphs
    overlaps = []
    for i in range(len(paragraphs) - 1):
        words1 = set(_words(paragraphs[i]))
        words2 = set(_words(paragraphs[i+1]))
        
        if not words1 or not words2:
            continue
        
        # Jaccard similarity
        overlap = len(words1 & words2) / len(words1 | words2)
        overlaps.append(overlap)
    
    if not overlaps:
        return {"semantic_drift": 0.0}
    
    # High variance in overlap = high drift (more human)
    mean_overlap = sum(overlaps) / len(overlaps)
    var_overlap = sum((x - mean_overlap)**2 for x in overlaps) / len(overlaps)
    drift = var_overlap ** 0.5
    
    return {"semantic_drift": float(drift)}


def pos_entropy(text: str) -> Dict[str, float]:
    """
    Estimate POS (part-of-speech) entropy using simple heuristics.
    Higher entropy = more diverse syntax (more human).
    
    Note: This is a simplified version without spaCy/NLTK.
    For production, use actual POS tagging.
    """
    words = _words(text)
    if len(words) < 10:
        return {"pos_entropy": 0.0}
    
    # Simple heuristics for POS estimation
    pos_counts = {
        'verb': 0,      # words ending in -ing, -ed, -s
        'noun': 0,      # capitalized words, words ending in -tion, -ment
        'adj': 0,       # words ending in -ly, -ful, -ous
        'other': 0
    }
    
    for word in words:
        if re.search(r'(ing|ed|s)$', word):
            pos_counts['verb'] += 1
        elif re.search(r'(tion|ment|ness)$', word):
            pos_counts['noun'] += 1
        elif re.search(r'(ly|ful|ous|ive)$', word):
            pos_counts['adj'] += 1
        else:
            pos_counts['other'] += 1
    
    # Calculate entropy
    total = sum(pos_counts.values())
    if total == 0:
        return {"pos_entropy": 0.0}
    
    H = -sum((v/total) * math.log2(v/total) for v in pos_counts.values() if v > 0)
    
    return {"pos_entropy": float(H)}

