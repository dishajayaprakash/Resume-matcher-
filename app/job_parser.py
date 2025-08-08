from typing import Dict, List, Tuple
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Text utils ---

def preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip().lower()

# --- Embeddings/model ---

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Scoring ---

def calculate_similarity_scores(
    job_desc: str, resumes: Dict[str, str]
) -> List[Tuple[str, float, str]]:
    """Return list of (filename, similarity, raw_text), sorted desc."""
    if not job_desc.strip():
        return []

    model = load_model()
    job_emb = model.encode([preprocess_text(job_desc)])

    results: List[Tuple[str, float, str]] = []
    for filename, txt in resumes.items():
        resume_emb = model.encode([preprocess_text(txt)])
        sim = float(cosine_similarity(job_emb, resume_emb)[0][0])
        results.append((filename, sim, txt))

    return sorted(results, key=lambda x: x[1], reverse=True)