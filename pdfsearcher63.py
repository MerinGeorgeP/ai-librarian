import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import os
import pickle
import shutil

# --- Constants for Persistence ---
UPLOAD_DIR = "uploaded_pdfs"
INDEX_PATH = "faiss_index.index"
DOC_CHUNKS_PATH = "doc_chunks.pkl"

# --- Text Chunking Configuration ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200 # Characters to overlap between chunks

# --- Ensure upload directory exists ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="AI Librarian", layout="wide")
st.title("📚 AI Librarian")
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f1; }
    .stButton>button { width: 100%; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Core Functions ---

@st.cache_resource
def load_model():
    """Loads the SentenceTransformer model once."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file path."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                st.warning(f"Warning: Could not find any pages in '{os.path.basename(pdf_path)}'.")
                return ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading {os.path.basename(pdf_path)}: {e}")
        return ""
    return text.strip()

def build_faiss_index(doc_chunks):
    """Builds a FAISS index from document chunks."""
    if not doc_chunks:
        return None
    model = load_model()
    # We only need the text part of the chunks for embedding
    chunk_texts = [chunk for _, chunk in doc_chunks]
    embeddings = model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=True)

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# --- Persistence Functions ---

def save_data(index, doc_chunks):
    """Saves the index and document chunks to disk."""
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    with open(DOC_CHUNKS_PATH, "wb") as f:
        pickle.dump(doc_chunks, f)

def load_data():
    """Loads the index and document chunks from disk."""
    index = None
    doc_chunks = []
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_CHUNKS_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(DOC_CHUNKS_PATH, "rb") as f:
                doc_chunks = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading saved data: {e}. Clearing old data.")
            clear_all_data() # Use the new clear function
    return index, doc_chunks

def clear_all_data():
    """Clears all persisted data including PDFs, index, and chunks."""
    st.session_state.index = None
    st.session_state.doc_chunks = []
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(DOC_CHUNKS_PATH):
        os.remove(DOC_CHUNKS_PATH)
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True) # Recreate the folder
    st.success("Cleared all indexed data and stored PDFs.")


# --- Initialize Session State ---

if 'initialized' not in st.session_state:
    st.session_state.index, st.session_state.doc_chunks = load_data()
    st.session_state.page = 'home'
    st.session_state.initialized = True

def get_indexed_filenames():
    """Gets a unique, sorted list of filenames from the stored PDF files."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    return sorted(os.listdir(UPLOAD_DIR))

# --- Page Navigation Functions ---
def go_to_home(): st.session_state.page = 'home'
def go_to_upload(): st.session_state.page = 'upload'
def go_to_search(): st.session_state.page = 'search'


# --- UI Rendering ---

# Home Page
if st.session_state.page == 'home':
    st.header("What would you like to do?")
    col1, col2 = st.columns(2)
    with col1:
        st.button("📤 Upload & Manage PDFs", on_click=go_to_upload, type="primary")
    with col2:
        st.button("🔎 Search PDFs", on_click=go_to_search, type="primary", disabled=st.session_state.index is None)

    if st.session_state.index is None:
        st.info("The search function is disabled. Please upload and index at least one PDF.")

# Upload Page
elif st.session_state.page == 'upload':
    st.button("← Back to Home", on_click=go_to_home)
    st.header("1. Manage Your PDFs")
    uploaded_files = st.file_uploader(
        "Upload new PDF files to add them to your permanent library.",
        type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Store and Index PDFs", type="primary"):
            with st.spinner("Saving and Processing PDFs... This may take a moment."):
                indexed_files = get_indexed_filenames()
                new_files_processed = False

                # First, save all new files
                for pdf_file in uploaded_files:
                    if pdf_file.name not in indexed_files:
                        with open(os.path.join(UPLOAD_DIR, pdf_file.name), "wb") as f:
                            f.write(pdf_file.getbuffer())
                        st.write(f"✅ Saved '{pdf_file.name}' to permanent storage.")

                # Now, rebuild the index from scratch using all stored PDFs
                st.write("Rebuilding search index from all stored PDFs...")
                all_stored_pdfs = get_indexed_filenames()
                st.session_state.doc_chunks = [] # Reset chunks

                for filename in all_stored_pdfs:
                    st.write(f"Processing {filename}...")
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    full_text = extract_text_from_pdf(file_path)
                    if full_text:
                        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                        st.session_state.doc_chunks.extend([(filename, chunk) for chunk in chunks])
                        new_files_processed = True

                if new_files_processed:
                    st.write("Creating and saving new search index...")
                    st.session_state.index = build_faiss_index(st.session_state.doc_chunks)
                    save_data(st.session_state.index, st.session_state.doc_chunks)
                    st.success(f"Index updated successfully! Total documents in library: {len(all_stored_pdfs)}")
                else:
                    st.info("No new PDFs were added. All uploaded files are already in the library.")

    indexed_filenames = get_indexed_filenames()
    if indexed_filenames:
        st.subheader("Currently Stored PDFs:")
        with st.container(height=300):
            for name in indexed_filenames:
                st.write(f"• {name}")

        if st.button("🗑️ Clear Entire Library (including PDFs)", type="secondary"):
            clear_all_data()
            st.rerun()

# Search Page
elif st.session_state.page == 'search':
    st.button("← Back to Home", on_click=go_to_home)
    st.header("2. Search for Information")
    st.write("Enter a keyword or sentence to find the most relevant snippet from your PDFs.")

    with st.form(key="search_form"):
        search_query = st.text_input("Search Query")
        k_results = st.slider("Number of unique PDFs to show:", 1, 5, 3)
        search_button = st.form_submit_button("Search")

    if search_button and search_query:
        if st.session_state.index is not None:
            with st.spinner("Searching..."):
                model = load_model()
                query_embedding = model.encode([search_query])
                faiss.normalize_L2(query_embedding)

                DISTANCE_THRESHOLD = 1.3

                # Search for more results than needed to account for duplicates
                num_candidates = k_results * 5
                distances, indices = st.session_state.index.search(
                    np.array(query_embedding, dtype=np.float32), num_candidates
                )

                # --- Filter for unique filenames ---
                unique_results = []
                displayed_filenames = set()

                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    dist = distances[0][i]

                    if dist < DISTANCE_THRESHOLD:
                        filename, chunk_text = st.session_state.doc_chunks[idx]

                        # If we haven't shown this PDF yet, add it to our results
                        if filename not in displayed_filenames:
                            unique_results.append({
                                "filename": filename,
                                "chunk": chunk_text,
                                "distance": dist
                            })
                            displayed_filenames.add(filename)

                        # Stop once we have enough unique results
                        if len(unique_results) >= k_results:
                            break

                st.subheader(f"Top {len(unique_results)} Relevant Documents:")
                if not unique_results:
                    st.warning("Could not find any relevant snippets for your query.")
                else:
                    for result in unique_results:
                        st.markdown(f"---")
                        st.info(f"**📄 Source PDF:** `{result['filename']}`\n\n**Relevance Score (Distance):** {result['distance']:.2f}")
                        with st.expander("Show best matching snippet", expanded=True):
                            # Highlight the search query in the text for better context
                            highlighted_chunk = result['chunk'].replace(search_query, f"**{search_query}**")
                            st.write(f"...{highlighted_chunk}...")
