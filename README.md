# **LangChain RAG (Retrieval-Augmented Generation)**

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Hugging Face Embeddings**, and **Pinecone**. It processes a PDF document, splits it into chunks, embeds it, stores it in Pinecone, and enables semantic search-based querying.

---

## **📘 Table of Contents**

1. [🔍 Project Overview](#project-overview)
2. [🧰 Requirements](#requirements)
3. [⚙️ Step-by-Step Workflow](#step-by-step-workflow)
    - [1️⃣ Document Loading](#1-document-loading)
    - [2️⃣ Cleaning Text](#2-cleaning-text)
    - [3️⃣ Text Splitting](#3-text-splitting)
    - [4️⃣ Metadata Preprocessing](#4-metadata-preprocessing)
    - [5️⃣ Embedding Generation](#5-embedding-generation)
    - [6️⃣ Pinecone Integration](#6-pinecone-integration)
    - [7️⃣ Querying with RAG](#7-querying-with-rag)
4. [💬 Example Queries](#example-queries)
5. [🧠 Key Concepts](#key-concepts)
6. [🚀 Future Work](#future-work)

---

## 🔍 **Project Overview**

> Build a semantic search pipeline from a PDF using:
> - `PyPDFLoader` for document loading
> - `RecursiveCharacterTextSplitter` for chunking
> - `BAAI/bge-small-en-v1.5` for embedding
> - `Pinecone` for vector storage
> - `LangChain` for orchestration

---

## 🧰 **Requirements**

- Python 3.8+
- LangChain
- Pinecone
- Transformers
- SentenceTransformers
- PyPDF

**Install dependencies:**

```bash
pip install langchain pinecone-client transformers sentence-transformers pypdf
```

---

## ⚙️ **Step-by-Step Workflow**

### 1️⃣ **Document Loading**

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ai_policy.pdf")
pages = loader.load()
```

---

### 2️⃣ **Cleaning Text**

```python
cleaned_pages = [page.page_content.replace("\n", " ").strip() for page in pages]
```

---

### 3️⃣ **Text Splitting**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(pages)
```

---

### 4️⃣ **Metadata Preprocessing**

```python
from langchain_core.documents import Document

doc_list = []

for page in pages:
    for chunk in splitter.split_text(page.page_content):
        metadata = {"source": "AI policy", "page_no": page.metadata["page"] + 1}
        doc_list.append(Document(page_content=chunk, metadata=metadata))
```

---

### 5️⃣ **Embedding Generation**

```python
from langchain.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
```

---

### 6️⃣ **Pinecone Integration**

**Initialize and create index:**

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")
pinecone.create_index("rag-index", dimension=384)
```

**Upsert chunks:**

```python
from langchain.vectorstores import Pinecone

vector_store = Pinecone.from_documents(doc_list, embed_model, index_name="rag-index")
```

---

### 7️⃣ **Querying with RAG**

```python
query = "What is the objective of Pakistan's AI Policy?"
query_vector = embed_model.embed_query(query)

results = index.query(vector=query_vector, top_k=3, include_metadata=True)

for match in results["matches"]:
    print("Score:", match['score'])
    print("Metadata:", match['metadata'])
```

---

## 💬 **Example Queries**

- "What are the objectives of the National AI Policy?"
- "Who contributed to the AI policy?"
- "What is the vision of Pakistan’s AI Policy?"

---

## 🧠 **Key Concepts**

| **Term**        | **Description**                                                                 |
|----------------|----------------------------------------------------------------------------------|
| **RAG**         | Combines retrieval from external documents with LLM responses.                 |
| **Embeddings**  | Vector representations of text used for similarity-based retrieval.            |
| **Chunking**    | Breaking large documents into manageable, overlapping sections.                |
| **Metadata**    | Additional info like page number, source, etc., attached to each chunk.        |
| **Pinecone**    | Vector database for storing and querying high-dimensional vectors.             |

---

## 🚀 **Future Work**

- Integrate QA chains using LangChain's RetrievalQA.
- Add Streamlit/Gradio interface.
- Support multi-document ingestion.
- Use prompt templates and memory.

---
