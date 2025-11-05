import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import shutil
import hashlib

# -----------------------------
# --- Helper Functions --------
# -----------------------------

# --- Password Hashing ---
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# --- Users DB ---
USERS_DIR = "users"
USERS_DB = os.path.join(USERS_DIR, "users_db.pkl")
os.makedirs(USERS_DIR, exist_ok=True)

# Load users database
if os.path.exists(USERS_DB):
    with open(USERS_DB, "rb") as f:
        users_db = pickle.load(f)
else:
    users_db = {}

def save_users_db():
    """Save the users database permanently."""
    with open(USERS_DB, "wb") as f:
        pickle.dump(users_db, f)

def signup(username, password):
    if username in users_db:
        st.error("Username already exists.")
        return False
    users_db[username] = hash_password(password)
    save_users_db()
    # Create user directory
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

# --- Model & FAISS ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size, chunk_overlap):
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
    if not doc_chunks:
        return None
    model = load_model()
    chunk_texts = [chunk for _, chunk in doc_chunks]
    embeddings = model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# --- Persistence ---
def save_data(index, doc_chunks, paths):
    if index is not None:
        faiss.write_index(index, paths["INDEX_PATH"])
    with open(paths["DOC_CHUNKS_PATH"], "wb") as f:
        pickle.dump(doc_chunks, f)

def load_data(paths):
    index = None
    doc_chunks = []
    if os.path.exists(paths["INDEX_PATH"]) and os.path.exists(paths["DOC_CHUNKS_PATH"]):
        try:
            index = faiss.read_index(paths["INDEX_PATH"])
            with open(paths["DOC_CHUNKS_PATH"], "rb") as f:
                doc_chunks = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading saved data: {e}. Clearing old data.")
            clear_all_data(paths)
    return index, doc_chunks

def clear_all_data(paths):
    st.session_state.index = None
    st.session_state.doc_chunks = []
    if os.path.exists(paths["INDEX_PATH"]):
        os.remove(paths["INDEX_PATH"])
    if os.path.exists(paths["DOC_CHUNKS_PATH"]):
        os.remove(paths["DOC_CHUNKS_PATH"])
    if os.path.exists(paths["UPLOAD_DIR"]):
        shutil.rmtree(paths["UPLOAD_DIR"])
    os.makedirs(paths["UPLOAD_DIR"], exist_ok=True)
    st.success("Cleared all indexed data and stored PDFs.")

def get_indexed_filenames(paths):
    if not os.path.exists(paths["UPLOAD_DIR"]):
        return []
    return sorted(os.listdir(paths["UPLOAD_DIR"]))

# -----------------------------
# --- Streamlit App Setup -----
# -----------------------------
st.set_page_config(page_title="AI Librarian", layout="wide")
st.title("📚 AI Librarian")
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f1; }
    .stButton>button { width: 100%; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# --- Session State Init ------
# -----------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'auth_choice' not in st.session_state:
    st.session_state.auth_choice = 'Login'

# -----------------------------
# --- Login / Signup Flow -----
# -----------------------------
if not st.session_state.logged_in:
    st.header("👤 Login or Signup")
    st.session_state.auth_choice = st.radio(
        "Choose action", ["Login", "Signup"],
        index=0 if st.session_state.auth_choice=="Login" else 1
    )

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        if st.session_state.auth_choice == "Signup":
            if signup(username, password):
                st.session_state.auth_choice = "Login"  # switch automatically to login
        elif st.session_state.auth_choice == "Login":
            if login(username, password):
                st.session_state.logged_in = True

# -----------------------------
# --- Main App After Login ----
# -----------------------------
else:
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = 'home'
        

    user_paths = get_user_paths(st.session_state.username)

    # Initialize user data
    if not st.session_state.initialized:
        st.session_state.index, st.session_state.doc_chunks = load_data(user_paths)
        st.session_state.initialized = True

    # --- Page Navigation ---
    def go_to_home(): st.session_state.page = 'home'
    def go_to_upload(): st.session_state.page = 'upload'
    def go_to_search(): st.session_state.page = 'search'

    # --- Home Page ---
    if st.session_state.page == 'home':
        st.header("What would you like to do?")
        col1, col2 = st.columns(2)
        with col1:
            st.button("📤 Upload & Manage PDFs", on_click=go_to_upload, type="primary")
        with col2:
            st.button("🔎 Search PDFs", on_click=go_to_search, type="primary", disabled=st.session_state.index is None)
        if st.session_state.index is None:
            st.info("The search function is disabled. Please upload and index at least one PDF.")

    # --- Upload Page ---
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
                    indexed_files = get_indexed_filenames(user_paths)
                    new_files_processed = False

                    # Save new files
                    for pdf_file in uploaded_files:
                        if pdf_file.name not in indexed_files:
                            with open(os.path.join(user_paths["UPLOAD_DIR"], pdf_file.name), "wb") as f:
                                f.write(pdf_file.getbuffer())
                            st.write(f"✅ Saved '{pdf_file.name}' to permanent storage.")

                    # Rebuild index
                    st.write("Rebuilding search index from all stored PDFs...")
                    all_stored_pdfs = get_indexed_filenames(user_paths)
                    st.session_state.doc_chunks = []

                    for filename in all_stored_pdfs:
                        st.write(f"Processing {filename}...")
                        file_path = os.path.join(user_paths["UPLOAD_DIR"], filename)
                        full_text = extract_text_from_pdf(file_path)
                        if full_text:
                            chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                            st.session_state.doc_chunks.extend([(filename, chunk) for chunk in chunks])
                            new_files_processed = True

                    if new_files_processed:
                        st.write("Creating and saving new search index...")
                        st.session_state.index = build_faiss_index(st.session_state.doc_chunks)
                        save_data(st.session_state.index, st.session_state.doc_chunks, user_paths)
                        st.success(f"Index updated successfully! Total documents: {len(all_stored_pdfs)}")
                    else:
                        st.info("No new PDFs were added.")

        # Show stored PDFs
        indexed_filenames = get_indexed_filenames(user_paths)
        if indexed_filenames:
            st.subheader("Currently Stored PDFs:")
            for name in indexed_filenames:
                st.write(f"• {name}")
            if st.button("🗑️ Clear Entire Library (including PDFs)", type="secondary"):
                clear_all_data(user_paths)
                

    # --- Search Page ---
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
                    num_candidates = k_results * 5
                    distances, indices = st.session_state.index.search(
                        np.array(query_embedding, dtype=np.float32), num_candidates
                    )

                    unique_results = []
                    displayed_filenames = set()
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        dist = distances[0][i]
                        if dist < DISTANCE_THRESHOLD:
                            filename, chunk_text = st.session_state.doc_chunks[idx]
                            if filename not in displayed_filenames:
                                unique_results.append({
                                    "filename": filename,
                                    "chunk": chunk_text,
                                    "distance": dist
                                })
                                displayed_filenames.add(filename)
                            if len(unique_results) >= k_results:
                                break

                    st.subheader(f"Top {len(unique_results)} Relevant Documents:")
                    if not unique_results:
                        st.warning("Could not find any relevant snippets for your query.")
                    else:
                        for result in unique_results:
                            st.markdown("---")
                            pdf_name = result['filename']
                            pdf_path = os.path.join(user_paths["UPLOAD_DIR"], pdf_name)

                        if os.path.exists(pdf_path):
                            pdf_link = f"[📄 {pdf_name}](file://{os.path.abspath(pdf_path)})"
                            st.markdown(pdf_link, unsafe_allow_html=True)
                        else:
                            st.warning(f"PDF file not found: {pdf_name}")


