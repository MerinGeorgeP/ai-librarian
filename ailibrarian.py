import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import shutil
import hashlib
from transformers import pipeline


# ============================================================
# 🔐 AUTHENTICATION & USER MANAGEMENT
# ============================================================

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

USERS_DIR = "users"
USERS_DB = os.path.join(USERS_DIR, "users_db.pkl")
os.makedirs(USERS_DIR, exist_ok=True)

if os.path.exists(USERS_DB):
    with open(USERS_DB, "rb") as f:
        users_db = pickle.load(f)
else:
    users_db = {}

def save_users_db():
    with open(USERS_DB, "wb") as f:
        pickle.dump(users_db, f)

def signup(username, password):
    if username in users_db:
        st.error("Username already exists.")
        return False
    users_db[username] = hash_password(password)
    save_users_db()
    os.makedirs(os.path.join(USERS_DIR, username, "uploaded_pdfs"), exist_ok=True)
    st.success("Signup successful! You can now log in.")
    return True

def login(username, password):
    if username not in users_db:
        st.error("Username does not exist.")
        return False
    if not verify_password(password, users_db[username]):
        st.error("Incorrect password.")
        return False
    st.session_state.username = username
    st.session_state.logged_in = True
    st.success(f"Logged in as {username}")
    return True

def get_user_paths(username):
    user_dir = os.path.join(USERS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return {
        "UPLOAD_DIR": os.path.join(user_dir, "uploaded_pdfs"),
        "INDEX_PATH": os.path.join(user_dir, "faiss_index.index"),
        "DOC_CHUNKS_PATH": os.path.join(user_dir, "doc_chunks.pkl")
    }

# ============================================================
# 🤖 MODEL LOADERS (Cached)
# ============================================================

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("text-generation", model="distilgpt2", device=-1)

# ============================================================
# 📘 PDF PROCESSING HELPERS
# ============================================================

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ============================================================
# 🧠 SUMMARIZATION HELPER (Fixed)
# ============================================================

def summarize_text(summarizer, text, max_chunk_length=800):
    """
    Simplified summarization function using extractive summarization
    """
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize effectively."
    
    # Split text into sentences
    sentences = text.split('. ')
    
    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 3:
        return ". ".join(sentences)
    
    # Score sentences based on length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Score based on position (earlier sentences get higher scores)
        position_score = 1.0 - (i / len(sentences))
        # Score based on length (prefer medium-length sentences)
        length_score = min(len(sentence.split()) / 20, 1.0)
        # Combined score
        combined_score = position_score * 0.6 + length_score * 0.4
        scored_sentences.append((combined_score, sentence, i))
    
    # Sort by score and select top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    # Select approximately 30% of sentences, but at least 3 and at most 10
    num_sentences = max(3, min(10, len(sentences) // 3))
    
    # Get the top sentences and sort them by original position
    top_sentences = scored_sentences[:num_sentences]
    top_sentences.sort(key=lambda x: x[2])  # Sort by original index
    
    # Extract the sentences in original order
    summary_sentences = [s[1] for s in top_sentences]
    
    # Combine into summary
    summary = ". ".join(summary_sentences)
    
    # Ensure it ends with a period
    if not summary.endswith('.'):
        summary += '.'
    
    return summary

# ============================================================
# 🗂️ DATA STORAGE / FAISS INDEX
# ============================================================

def build_faiss_index(doc_chunks):
    model = load_sentence_model()
    texts = [chunk for _, chunk in doc_chunks]
    embeddings = model.encode(texts, convert_to_tensor=False)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def save_data(index, doc_chunks, paths):
    if index is not None:
        faiss.write_index(index, paths["INDEX_PATH"])
    with open(paths["DOC_CHUNKS_PATH"], "wb") as f:
        pickle.dump(doc_chunks, f)

def load_data(paths):
    index = None
    doc_chunks = []
    if os.path.exists(paths["INDEX_PATH"]) and os.path.exists(paths["DOC_CHUNKS_PATH"]):
        index = faiss.read_index(paths["INDEX_PATH"])
        with open(paths["DOC_CHUNKS_PATH"], "rb") as f:
            doc_chunks = pickle.load(f)
    return index, doc_chunks

def clear_all_data(paths):
    st.session_state.index = None
    st.session_state.doc_chunks = []
    if os.path.exists(paths["UPLOAD_DIR"]):
        shutil.rmtree(paths["UPLOAD_DIR"])
    os.makedirs(paths["UPLOAD_DIR"], exist_ok=True)
    st.success("Cleared all stored PDFs.")

# ============================================================
# 🎨 STREAMLIT APP
# ============================================================

st.set_page_config(page_title="AI Librarian", layout="wide")
st.title("📚 AI Librarian + 🧠 Summarizer")

# --- Session State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Login / Signup ---
if not st.session_state.logged_in:
    st.header("👤 Login / Signup")
    choice = st.radio("Select Action", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if choice == "Signup":
            signup(username, password)
        else:
            login(username, password)

else:
    st.sidebar.write(f"👋 Logged in as **{st.session_state.username}**")
    user_paths = get_user_paths(st.session_state.username)
    st.session_state.index, st.session_state.doc_chunks = load_data(user_paths)

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    # Navigation
    menu = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload PDFs", "🔎 Search PDFs", "🧠 Summarize PDF"]
    )

    # ---------------- HOME ----------------
    if menu == "🏠 Home":
        st.header("Welcome to AI Librarian")
        st.write("Upload, search, and summarize your PDFs!")

    # ---------------- UPLOAD ----------------
    elif menu == "📤 Upload PDFs":
        st.header("Upload and Manage PDFs")
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files and st.button("Save Files"):
            for file in uploaded_files:
                with open(os.path.join(user_paths["UPLOAD_DIR"], file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully!")

            # ✅ Build FAISS index automatically
            st.info("Building FAISS index for uploaded PDFs...")
            doc_chunks = []
            for file in os.listdir(user_paths["UPLOAD_DIR"]):
                pdf_path = os.path.join(user_paths["UPLOAD_DIR"], file)
                text = extract_text_from_pdf(pdf_path)
                if text.strip():
                    chunks = chunk_text(text)
                    doc_chunks.extend([(file, c) for c in chunks])

            if doc_chunks:
                index = build_faiss_index(doc_chunks)
                save_data(index, doc_chunks, user_paths)
                st.session_state.index = index
                st.session_state.doc_chunks = doc_chunks
                st.success("✅ Index built successfully! You can now search or summarize.")
            else:
                st.warning("No valid text extracted from PDFs — index not built.")

        existing = os.listdir(user_paths["UPLOAD_DIR"])
        if existing:
            st.subheader("Stored PDFs:")
            for f in existing:
                st.write("•", f)

            if st.button("🔄 Rebuild FAISS Index"):
                st.info("Rebuilding FAISS index...")
                doc_chunks = []
                for file in os.listdir(user_paths["UPLOAD_DIR"]):
                    pdf_path = os.path.join(user_paths["UPLOAD_DIR"], file)
                    text = extract_text_from_pdf(pdf_path)
                    if text.strip():
                        chunks = chunk_text(text)
                        doc_chunks.extend([(file, c) for c in chunks])
                if doc_chunks:
                    index = build_faiss_index(doc_chunks)
                    save_data(index, doc_chunks, user_paths)
                    st.session_state.index = index
                    st.session_state.doc_chunks = doc_chunks
                    st.success("✅ FAISS index rebuilt successfully!")
                else:
                    st.warning("No valid text found to rebuild index.")

            if st.button("🗑️ Clear Library"):
                clear_all_data(user_paths)

    # ---------------- SEARCH ----------------
    elif menu == "🔎 Search PDFs":
        st.header("Search Stored PDFs")

        # ✅ Always reload saved FAISS index
        st.session_state.index, st.session_state.doc_chunks = load_data(user_paths)

        if st.session_state.index is None or not st.session_state.doc_chunks:
            st.warning("No indexed documents found. Please upload PDFs first.")
        else:
            query = st.text_input("Enter search query")
            if st.button("Search"):
                model = load_sentence_model()
                q_emb = model.encode([query])
                faiss.normalize_L2(q_emb)
                distances, indices = st.session_state.index.search(np.array(q_emb, dtype=np.float32), 10)

                seen_files = set()
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    filename, chunk = st.session_state.doc_chunks[idx]
                    if filename not in seen_files:
                        seen_files.add(filename)
                        results.append((filename, chunk, dist))

                if results:
                    st.subheader("🔍 Search Results:")
                    for filename, chunk, dist in results:
                        st.write(f"📄 **{filename}** — (distance: {dist:.2f})")
                        st.write(chunk[:400] + "...")
    
                        # 📎 Add clickable link to open/download PDF
                        pdf_path = os.path.join(user_paths["UPLOAD_DIR"], filename)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label=f"📥 Download {filename}",
                                data=pdf_file,
                                file_name=filename,
                                mime="application/pdf"
                            )

                        st.markdown("---")

                else:
                    st.info("No relevant results found.")

    # ---------------- SUMMARIZE ----------------
    elif menu == "🧠 Summarize PDF":
        st.header("Summarize Your PDFs")
        files = os.listdir(user_paths["UPLOAD_DIR"])
        if not files:
            st.warning("No PDFs available. Please upload first.")
        else:
            selected_file = st.selectbox("Select a PDF to summarize", files)
            
            # Add options for summary length
            summary_length = st.selectbox(
                "Select summary length",
                ["Short (100-150 words)", "Medium (150-250 words)", "Long (250-400 words)"]
            )
            
            if st.button("Generate Summary"):
                pdf_path = os.path.join(user_paths["UPLOAD_DIR"], selected_file)
                
                with st.spinner("Extracting text from PDF..."):
                    text = extract_text_from_pdf(pdf_path)
                
                if len(text.strip()) < 100:
                    st.error("PDF too short to summarize. Please select a PDF with more content.")
                else:
                    summarizer = load_summarizer()
                    
                    with st.spinner("Generating summary... This may take a few minutes..."):
                        try:
                            summary = summarize_text(summarizer, text)
                            
                            st.success("✅ Summary generated!")
                            st.subheader("📄 Summary")
                            st.write(summary)
                            
                            # Add download button for summary
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=f"{selected_file}_summary.txt",
                                mime="text/plain"
                            )
                            
                            # Show statistics
                            with st.expander("📊 Summary Statistics"):
                                original_words = len(text.split())
                                summary_words = len(summary.split())
                                compression_ratio = (summary_words / original_words) * 100
                                
                                st.write(f"**Original text:** {original_words:,} words")
                                st.write(f"**Summary:** {summary_words:,} words")
                                st.write(f"**Compression ratio:** {compression_ratio:.1f}%")
                                
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            st.info("Please try again with a different PDF or check if the PDF contains extractable text.")
