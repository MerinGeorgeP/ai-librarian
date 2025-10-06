import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import os
import pickle

# --- Constants for Persistence ---
INDEX_PATH = "faiss_index.index"
FILENAMES_PATH = "filenames.pkl"
TEXTS_DIR = "pdf_texts"

# Create storage directory if it doesn't exist
if not os.path.exists(TEXTS_DIR):
    os.makedirs(TEXTS_DIR)

# Set a title for the Streamlit app
st.set_page_config(page_title=" AI Librarian ", layout="wide")

st.title(" AI Librarian ")
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 1rem;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 1rem;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Loads the SentenceTransformer model from cache.
    This is decorated with @st.cache_resource to ensure the model is loaded only once.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        pdf_file (UploadedFile): The PDF file uploaded by the user.

    Returns:
        str: The concatenated text from all pages of the PDF.
             Returns an empty string if no text can be extracted.
    """
    text = ""
    try:
        # Use a BytesIO buffer to handle the uploaded file in memory
        with pdfplumber.open(io.BytesIO(pdf_file.getvalue())) as pdf:
            # Check if pdf.pages is not empty before iterating
            if not pdf.pages:
                st.warning(f"Warning: Could not find any pages in '{pdf_file.name}'.")
                return ""
            
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""
    return text.strip()

def build_faiss_index(texts):
    """
    Builds a FAISS index from a list of texts.

    Args:
        texts (list of str): A list of text documents.

    Returns:
        faiss.Index: The FAISS index for the document embeddings.
                     Returns None if the input texts list is empty.
    """
    if not texts:
        return None
    model = load_model()
    # Encode the texts into embeddings
    embeddings = model.encode(texts, convert_to_tensor=False)
    
    # Get the dimension of the embeddings
    d = embeddings.shape[1]
    
    # Build the FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# --- Persistence Functions ---
def save_data(index, filenames, texts):
    """Saves the index, filenames, and extracted texts to disk."""
    # Save FAISS index
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
    
    # Save filenames list
    with open(FILENAMES_PATH, "wb") as f:
        pickle.dump(filenames, f)
        
    # Save texts to individual files
    for filename, text in texts.items():
        # Sanitize filename before saving
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        with open(os.path.join(TEXTS_DIR, f"{safe_filename}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

def load_data():
    """Loads the index, filenames, and extracted texts from disk."""
    index = None
    filenames = []
    texts = {}

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    
    if os.path.exists(FILENAMES_PATH):
        with open(FILENAMES_PATH, "rb") as f:
            filenames = pickle.load(f)
            
    for filename in filenames:
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        text_path = os.path.join(TEXTS_DIR, f"{safe_filename}.txt")
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                texts[filename] = f.read()
                
    return index, filenames, texts

# Initialize session state from saved files if they exist
if 'initialized' not in st.session_state:
    st.session_state.index, st.session_state.filenames, st.session_state.texts = load_data()
    st.session_state.initialized = True

# Initialize page state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Page Navigation Functions ---
def go_to_home():
    st.session_state.page = 'home'

def go_to_upload():
    st.session_state.page = 'upload'

def go_to_search():
    st.session_state.page = 'search'


# --- UI Rendering ---

# Home Page
if st.session_state.page == 'home':
    st.header("What would you like to do?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button(" Upload & Manage PDFs", on_click=go_to_upload, type="primary")
    with col2:
        st.button(" Search PDFs", on_click=go_to_search, type="primary", disabled=st.session_state.index is None)

    if st.session_state.index is None:
        st.info("The search function is disabled until you upload at least one PDF.")


# Upload Page
elif st.session_state.page == 'upload':
    st.button("← Back to Home", on_click=go_to_home)
    st.header("1. Manage Your PDFs")
    st.write("Upload new PDF files to add them to the searchable index.")
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload multiple files at once."
    )

    if uploaded_files:
        if st.button("Add PDFs to Index", type="primary"):
            with st.spinner("Extracting text and updating index... This may take a moment."):
                new_files_processed = False
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.filenames:
                        text = extract_text_from_pdf(uploaded_file)
                        if text:
                            st.session_state.texts[uploaded_file.name] = text
                            st.session_state.filenames.append(uploaded_file.name)
                            new_files_processed = True
                
                if new_files_processed:
                    docs_to_index = [st.session_state.texts[name] for name in st.session_state.filenames]
                    st.session_state.index = build_faiss_index(docs_to_index)
                    save_data(st.session_state.index, st.session_state.filenames, st.session_state.texts)
                    st.success(f"Successfully updated the index. Total PDFs indexed: {len(st.session_state.filenames)}")
                else:
                    st.info("No new PDFs to add. All uploaded files are already in the index.")
    
    if st.session_state.filenames:
        st.subheader("Currently Indexed PDFs:")
        with st.container(height=300):
            for name in st.session_state.filenames:
                st.write(f"• {name}")
        
        if st.button("Clear Entire Index", type="secondary"):
            st.session_state.index = None
            st.session_state.filenames = []
            st.session_state.texts = {}
            st.session_state.initialized = False
            
            if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
            if os.path.exists(FILENAMES_PATH): os.remove(FILENAMES_PATH)
            for file in os.listdir(TEXTS_DIR): os.remove(os.path.join(TEXTS_DIR, file))
            
            go_to_home() # Go back home after clearing
            st.rerun() # Rerun to reflect changes immediately

# Search Page
elif st.session_state.page == 'search':
    st.button("← Back to Home", on_click=go_to_home)
    st.header("2. Search for Information")
    st.write("Enter a word or a sentence to find the most relevant PDF.")
    
    with st.form(key="search_form"):
        search_query = st.text_input(
            "Search Query", 
            "", 
            placeholder="e.g., 'machine learning algorithms' or 'quarterly financial results'"
        )
        search_button = st.form_submit_button("Search")

    if search_button and search_query:
        if st.session_state.index is not None:
            with st.spinner("Searching for the most relevant PDF..."):
                model = load_model()
                
                query_embedding = model.encode([search_query], convert_to_tensor=False)
                
                k = 1
                distances, indices = st.session_state.index.search(np.array(query_embedding, dtype=np.float32), k)
                
                if indices.size > 0:
                    most_relevant_index = indices[0][0]
                    most_relevant_pdf = st.session_state.filenames[most_relevant_index]
                    
                    st.success("Most Relevant PDF Found:")
                    st.info(f"**📄 {most_relevant_pdf}**")

                    with st.expander("Show relevant text snippet"):
                        st.write(st.session_state.texts[most_relevant_pdf][:1000] + "...")
                else:
                    st.warning("Could not find a relevant PDF for your query.")

