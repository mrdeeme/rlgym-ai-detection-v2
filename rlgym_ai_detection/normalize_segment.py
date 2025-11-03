import re, unicodedata

INVIS = [
    "\u200b","\u200c","\u200d","\ufeff","\u2060"
]

def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    for ch in INVIS:
        t = t.replace(ch, "")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
