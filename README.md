# AI Resume Matcher — Brief README

## Approach

* **Parsing:** Extract text from uploaded PDF resumes (pdfplumber) and the Job Description (JD). Clean (lowercase, de-noise), split into sections (skills/experience/education) with regex heuristics.
* **Semantic Match:** Compute a quick **embedding similarity** (SentenceTransformers) for fast ranking, then call **Gemini (`google-genai`)** for a deeper, instruction-guided evaluation (fit score + rationale + skill gaps).
* **Output:** For each candidate: overall match %, evidence snippets, missing skills, and a short recommendation.
* **Resilience:** Centralized `genai.Client` (cached via `st.cache_resource`) + retry/backoff on 429/500/503; fallback between `gemini-1.5-flash` ↔ `gemini-1.5-pro`.

## Assumptions

* Resumes are text-selectable PDFs (not scans).
* English content; JD provided in plain text.
* Python **3.11** runtime.
* Keys provided via Streamlit **Secrets** in Cloud; `.env` is **local only**.
* Free-tier usage is sufficient; occasional rate/overload errors may occur.

## Setup (super short)

Local:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GENAI_API_KEY
streamlit run app.py
```

Streamlit Cloud:

* Advanced settings → Python **3.11**
* Secrets (TOML):

  ```toml
  GENAI_API_KEY = "your-gemini-key"
  ```
* Save → **Restart** app.

## Things to Know

* **.env isn’t uploaded** to Cloud; use **Secrets**.
* **503/429** = model busy → automatic retries + try the other model.
* We **don’t store** resumes/JDs; all processing is in-memory during the session.
* Scores are **heuristics**; use alongside human review.
* If PDFs are images, add OCR (e.g., Tesseract) in a future iteration.

## Known Limitations / Next Steps

* Add OCR for scanned resumes; handle tables in CVs better.
* Introduce reproducibility controls (seed) and budget caps (max tokens).
* Export results (CSV/JSON) and batch processing queue for large candidate sets.
