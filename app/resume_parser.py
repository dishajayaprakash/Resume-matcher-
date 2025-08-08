from typing import Dict
import pdfplumber

# --- File reading ---

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_txt(file) -> str:
    raw = file.read()
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def build_resume_dict(files) -> Dict[str, str]:
    """Return {filename: text} for PDFs/TXTs; skip empties."""
    out: Dict[str, str] = {}
    for f in files:
        f.seek(0)
        txt = (
            extract_text_from_pdf(f)
            if f.type == "application/pdf"
            else extract_text_from_txt(f)
        )
        if txt:
            out[f.name] = txt
    return out