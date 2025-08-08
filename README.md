# AI Resume Scanner

A modern, AI-powered resume screening tool built with Python and BERT. This project compares resumes to job descriptions using semantic similarity (Sentence-BERT) and identifies missing keywords — enabling smarter resume tailoring and job targeting.

## 🔍 Key Features

- ✅ Extracts text from resumes (PDF, DOCX, TXT)
- ✅ Parses job descriptions (TXT)
- ✅ Calculates a semantic match score using **Sentence-BERT**
- ✅ Identifies missing keywords from the resume
- 🧠 Built for extensibility (Streamlit UI, AWS hosting, advanced NLP)

## 📁 Project Structure

```
AI-Resume-Scanner/
├── app/
│   ├── resume_parser.py
│   ├── job_parser.py
│   └── __init__.py
├── data/
│   ├── resumes/
│   └── job_descriptions/
├── main.py                  # BERT-based semantic comparison
├── requirements.txt
└── README.md
```

## ⚙️ How to Use

1. Clone the repository:
```bash
git clone https://github.com/Becky0x01/AI-Resume-Scanner.git
cd AI-Resume-Scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your resume and job description (in `data/`)

4. Run the script:
```bash
python3 main.py
```

## 🧠 Semantic AI Matching

This project uses [`sentence-transformers`](https://www.sbert.net/) to generate vector embeddings for resume and job description content, then compares them with cosine similarity.

### Example Output:
```
🤖 Semantic Match Score: 85.17%
🧩 Missing Keywords from Resume:
aws, scalable, cloud, communication, nlp, deployment
```

## 🚀 Future Extensions

- Add Streamlit UI for interactive uploading
- Deploy to AWS (EC2 or Streamlit Cloud)
- Semantic keyword feedback
- Resume version recommendations

## 👩‍💻 Built By

Becky Zhu  
Aspiring AI/Cloud Engineer | Lifelong Learner  
[LinkedIn](https://www.linkedin.com/in/rebeccaiit) | [GitHub](https://github.com/Becky0x01)

---
