import os
import time
import random
import re
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from google import genai

load_dotenv()  # enables .env for local dev


API_KEY = st.secrets.get("GENAI_API_KEY") or os.getenv("GENAI_API_KEY")



client = genai.Client(api_key=API_KEY)

def get_gemini_response(prompt: str) -> str:
    resp = client.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=prompt
    )
    return resp.text.strip()

# ----------------------------
# Page config + custom CSS
# ----------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .candidate-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .rank-badge {
        display: inline-block;
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-right: 1rem;
    }
    
    .ai-summary {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 5px;
        font-style: italic;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Helper functions
# ----------------------------
@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file."""
    raw = file.read()
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw

def preprocess_text(text: str) -> str:
    """Clean and normalize text for embedding."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip().lower()

def extract_candidate_name(resume_text: str) -> str:
    """Use Gemini to extract the candidate's full name."""
    prompt = (
        "You are a precise assistant. Given the raw text of a resume, "
        "extract only the candidate’s full name (first and last). "
        "If no clear name is found, reply exactly with 'Unknown'.\n\n"
        + resume_text
    )
    return get_gemini_response(prompt)

def calculate_similarity_scores(
    job_desc: str,
    resumes: Dict[str, str],
    model
) -> List[Tuple[str, float, str]]:
    """Compute cosine similarity between job description and each resume."""
    if not job_desc.strip():
        return []
    job_emb = model.encode([preprocess_text(job_desc)])
    results = []
    for filename, txt in resumes.items():
        resume_emb = model.encode([preprocess_text(txt)])
        sim = float(cosine_similarity(job_emb, resume_emb)[0][0])
        results.append((filename, sim, txt))
    return sorted(results, key=lambda x: x[1], reverse=True)


def generate_ai_summary(
    job_desc: str,
    resume_text: str,
    similarity_score: float,
    rank: int
) -> str:
    """
    Use Gemini to generate:
      1) Paragraph stating match %
      2) Paragraph of strengths
      3) Paragraph of gaps/advice
      4) Bullet list of missing keywords
    """
    prompt = f"""
You are an expert ATS evaluator.

Given the following:

• Candidate match score: {similarity_score*100:.1f}%  
• Candidate rank: {rank}  
• Resume text:
{resume_text}

• Job description:
{job_desc}

Please provide:
1. A short line stating the percentage match.
2. A paragraph summarizing the candidate’s strengths.
3. A bullet-point list under “Missing Keywords:” of key skills they lack.
"""
    return get_gemini_response(prompt)


# ----------------------------
# Main application
# ----------------------------
def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
        st.session_state.all_results = None
        st.session_state.names = {}
        st.session_state.job_description = ""
        st.session_state.analysis_complete = False

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    top_n = st.sidebar.slider(
        "Number of top candidates to return",
        min_value=1, max_value=20, value=10,
        help="Select how many top-ranked candidates you want to see"
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 About the Analysis")
    st.sidebar.info("""
        • Sentence Transformers for semantic embeddings  
        • Cosine Similarity for relevance scoring  
        • AI-powered summaries for candidate insights  
    """)
    if st.session_state.analysis_complete:
        if st.sidebar.button("🔄 Clear Results", type="secondary"):
            st.session_state.analysis_results = None
            st.session_state.all_results = None
            st.session_state.names = {}
            st.session_state.job_description = ""
            st.session_state.analysis_complete = False
            st.rerun()

    # Header
    st.markdown('<h1 class="main-header">AI Resume Matcher</h1>', unsafe_allow_html=True)
    st.markdown("### Find the best candidates for your job using advanced AI embeddings")

    # Job Description Input
    job_desc = st.text_area(
        "Paste the complete job description here:",
        value=st.session_state.job_description,
        height=200,
        placeholder="Include job title, requirements, skills, responsibilities...",
        help="Provide all relevant details for better matching."
    )
    if job_desc != st.session_state.job_description:
        st.session_state.job_description = job_desc

    # Resume Upload
    st.subheader("Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple PDF or TXT resumes."
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

    # Analyze Candidates button
    if st.button(" Analyze Candidates", type="primary", use_container_width=True):
        if not st.session_state.job_description.strip():
            st.error("❌ Please enter a job description")
            return
        if not uploaded_files:
            st.error("❌ Please upload at least one resume")
            return

        # Progress bar & status
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Extract text
        status_text.text("Extracting text from resumes...")
        resumes_dict = {}
        for i, f in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files) * 0.5)
            f.seek(0)
            if f.type == "application/pdf":
                txt = extract_text_from_pdf(f)
            else:
                txt = extract_text_from_txt(f)
            if txt:
                resumes_dict[f.name] = txt
            else:
                st.warning(f"⚠️ Could not extract text from {f.name}")

        if not resumes_dict:
            st.error("No valid resumes could be processed")
            return

        # Extract candidate names via Gemini
        names_dict = {}
        for fname, txt in resumes_dict.items():
            names_dict[fname] = extract_candidate_name(txt)

        # Compute similarity
        status_text.text("Analyzing candidates with AI...")
        progress_bar.progress(0.7)
        model = load_model()
        results = calculate_similarity_scores(
            st.session_state.job_description,
            resumes_dict,
            model
        )
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # Store top-N and all results + names
        st.session_state.analysis_results = results[:top_n]
        st.session_state.all_results = results
        st.session_state.names = names_dict
        st.session_state.analysis_complete = True

    # Display results
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        results = st.session_state.analysis_results

        # Stats row
        avg_score = np.mean([r[1] for r in results])
        max_score = max([r[1] for r in results])
        st.markdown(f"""
        <div class="stats-container">
          <div><strong>Total Candidates</strong><br>{len(results)}</div>
          <div><strong>Average Match</strong><br>{avg_score*100:.1f}%</div>
          <div><strong>Best Match</strong><br>{max_score*100:.1f}%</div>
          <div><strong>Showing Top</strong><br>{len(results)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"🏆 Top {len(results)} Candidates")

        # Candidate cards
        for rank, (filename, sim, txt) in enumerate(results, start=1):
            display_name = st.session_state.names.get(
                filename, filename.replace(".pdf","").replace(".txt","")
            )
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<span class="rank-badge">#{rank}</span> '
                        f'<strong>{display_name}</strong>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f'<div class="score-badge">{sim*100:.1f}% Match</div>',
                        unsafe_allow_html=True
                    )
                # Gemini-powered AI summary
                ai_summary = generate_ai_summary(
                    st.session_state.job_description,
                    txt,
                    sim,
                    rank
                )

                clean = ai_summary.lstrip()
                # st.markdown(
                #     f'<div class="ai-summary"><strong>AI Analysis:</strong><br>{ai_summary}</div>',
                #     unsafe_allow_html=True
                # )
                # 1) open the styled container
                st.markdown('<div class="ai-summary">', unsafe_allow_html=True)

                # 2) render the Gemini text _as Markdown_ so **bold** works
                st.markdown(clean, unsafe_allow_html=False)

                # 3) close the container
                st.markdown('</div>', unsafe_allow_html=True)
                with st.expander(f"📄 View resume preview – {filename}"):
                    preview = txt[:500] + "..." if len(txt) > 500 else txt
                    st.text(preview)
                    st.caption(f"File: {filename} | Length: {len(txt)} chars")
                st.markdown("---")

        # Detailed results table
        show_table = st.checkbox("Show detailed results table (All Candidates)")
        if show_table:
            df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Candidate_Name': st.session_state.names.get(
                        fn, fn.replace('.pdf','').replace('.txt','')
                    ),
                    'Filename': fn,
                    'Similarity_Score': f"{sim:.4f}",
                    'Match_Percentage': f"{sim*100:.1f}%",
                    'Text_Length': len(txt),
                    'File_Size_KB': f"{len(txt.encode('utf-8'))/1024:.1f} KB"
                }
                for i,(fn,sim,txt) in enumerate(st.session_state.all_results)
            ])
            st.dataframe(df, use_container_width=True)
            # Summary metrics
            all_scores = [sc for _,sc,_ in st.session_state.all_results]
            c1,c2,c3 = st.columns(3)
            with c1:
                st.metric("Total Analyzed", len(all_scores))
            with c2:
                st.metric("Average Score", f"{np.mean(all_scores)*100:.1f}%")
            with c3:
                st.metric("Score Range", f"{min(all_scores)*100:.1f}%–{max(all_scores)*100:.1f}%")
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Complete Results as CSV",
                data=csv,
                file_name=f"resume_analysis_{len(all_scores)}.csv",
                mime="text/csv"
            )

    elif st.session_state.analysis_complete and not st.session_state.analysis_results:
        st.warning("⚠️ No valid results found. Please check your files and try again.")

if __name__ == "__main__":
    main()
