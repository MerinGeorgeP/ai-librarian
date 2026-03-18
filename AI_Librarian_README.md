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

------------------------------------------------------------------------

## 🧠 Overview

AI Librarian is a web-based system that allows users to:

-   📤 Upload PDFs\
-   🗂 Manage PDFs\
-   🔍 Perform semantic search\
-   📝 Generate summaries

------------------------------------------------------------------------

## 🏗 Architecture

### 🔹 Frontend

-   Built with **Streamlit**
-   Provides an interactive UI
-   Features:
    -   User Registration & Login
    -   PDF Upload
    -   Search Interface
    -   Summary Display

### 🔹 Backend

-   Built with **FastAPI**
-   Handles:
    -   Authentication (JWT-based)
    -   PDF processing
    -   AI model execution
    -   Data storage and indexing

------------------------------------------------------------------------

## ⚙️ Workflow

### 1. User Authentication

-   Users sign up with username & password
-   Passwords are securely encrypted
-   JWT token is generated on login

### 2. Dashboard Access

-   Upload PDFs
-   Search documents
-   Manage files
-   Summarize content

### 3. PDF Upload

-   Files are sent to backend
-   Text is extracted
-   Converted into embeddings
-   Stored using FAISS for fast retrieval

### 4. Managing PDFs

-   View uploaded files
-   Delete documents
-   Automatically updates search index

### 5. Semantic Search

-   User query → converted to embedding
-   FAISS finds relevant matches
-   Returns documents with excerpts

### 6. Summarization

-   Text is split into chunks
-   Each chunk summarized using AI (BART)
-   Final combined summary displayed

### 7. Logout

-   JWT token removed
-   User session ends securely

------------------------------------------------------------------------

## 🛠 Tech Stack

-   **Frontend:** Streamlit\
-   **Backend:** FastAPI\
-   **AI Models:** BART (Summarization), Embeddings\
-   **Search Engine:** FAISS\
-   **Authentication:** JWT\
-   **Storage:** File system + metadata

------------------------------------------------------------------------

## 🔐 Security Features

-   Password encryption\
-   JWT-based authentication\
-   User-specific document isolation

------------------------------------------------------------------------

## 📈 Key Advantages

-   Faster document search\
-   Intelligent summarization\
-   User-friendly interface\
-   Secure and scalable

------------------------------------------------------------------------

## 📌 Future Improvements

-   Multi-format support (Word, PPT)
-   Advanced filtering & tagging
-   Chat with documents (RAG)
-   Cloud deployment

------------------------------------------------------------------------

## 🙌 Acknowledgment

This project demonstrates the integration of AI with document management
to improve productivity and accessibility.

------------------------------------------------------------------------

## 📎 Source

Based on project documentation: DOC-20260110-WA0042.pdf
