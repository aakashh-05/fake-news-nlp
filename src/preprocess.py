
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

def clean_for_bow(text: str) -> str:
    """
    Safer cleaning for classical models:
    - lowercase
    - remove URLs and HTML
    - keep alphabetic words
    - remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    # Replace punctuation with space but keep words
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
