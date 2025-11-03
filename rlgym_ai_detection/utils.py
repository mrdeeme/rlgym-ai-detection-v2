import re, logging

PT_HINTS = {"não","pra","você","que","com","para","de","uma","numa","amanhã","hoje","relatório","rubrica"}
EN_HINTS = {"the","and","for","with","report","tomorrow","today","rubric","please","draft"}

def detect_lang(text: str) -> str:
    try:
        from langdetect import detect
        code = detect(text or "a")
        if code.startswith("pt"):
            return "pt-BR"
        return "EN"
    except Exception:
        t = (text or "").lower()
        pt = sum(1 for w in PT_HINTS if w in t)
        en = sum(1 for w in EN_HINTS if w in t)
        return "pt-BR" if pt >= en else "EN"

def get_logger(name: str = "rlgym_ai_detection"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
