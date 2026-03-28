# 📚 AI Librarian

An AI-powered web application for managing, searching, and summarizing
PDF documents efficiently.

------------------------------------------------------------------------

## 🚀 Problem Statement

Managing and searching through multiple PDF documents is time-consuming
and inefficient. Traditional systems lack: - Semantic understanding -
Fast retrieval - Automated summarization

AI Librarian solves this by enabling intelligent document processing and
retrieval.

https://drive.google.com/file/d/1sZh7i9GnXN1AO5VzEHiY-GH9uXCIlhFq/view?usp=drive_link

------------------------------------------------------------------------


## ✨ Features

-   🔐 User authentication (signup/login)
-   📤 Upload and manage PDFs
-   🔎 Semantic search using FAISS
-   🧠 Extractive summarization
-   📥 Download summaries

## 🤖 Models Used

-   Sentence Embeddings: all-MiniLM-L6-v2 (SentenceTransformers)
-   Summarization: distilgpt2 (used for lightweight text generation)
-   Retrieval: FAISS (vector similarity search)

## 🧠 Tech Stack

-   Streamlit
-   pdfplumber
-   FAISS
-   SentenceTransformers
-   Transformers

## ⚙️ Installation

``` bash
python -m venv venv
source venv/bin/activate
pip install streamlit pdfplumber faiss-cpu sentence-transformers numpy transformers torch
```

## ▶️ Run

``` bash
streamlit run app.py
```

## 📁 Project Structure

    users/
      ├── users_db.pkl
      ├── <username>/
          ├── uploaded_pdfs/
          ├── faiss_index.index
          ├── doc_chunks.pkl
    app.py
    README.md

## 📌 Usage

1.  Login or Signup
2.  Upload PDFs
3.  Search or summarize documents

## ⚡ Notes

-   Works best with text-based PDFs
-   First run may be slow due to model loading

## 👨‍💻 Author

Merin George P
