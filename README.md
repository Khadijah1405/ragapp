# 📦 Google Drive Uploaded Vectors RAG API

This project provides a **FastAPI-powered Retrieval-Augmented Generation (RAG)** interface designed for question-answering over vectorized content. The system loads prebuilt FAISS indexes from a shared drive and an API endpoint for semantic search and natural language answers using **OpenAI GPT-4**.

---

## ✅ Features

- 🔍 Semantic document search using FAISS vector indexes
- 🤖 Natural language responses powered by GPT-4 (via LangChain)
- 📦 Prebuilt vector indexes loaded from **OneDrive** at runtime
- 🚀 Easy-to-deploy REST API using FastAPI

---

## 📁 Project Structure

```
.
├── app.py                      # FastAPI main app (import above code here)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/evergabe-vector-rag.git
cd evergabe-vector-rag
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File
```bash
OPENAI_API_KEY=your_openai_key
```

---

## ▶️ Run the API

```bash
uvicorn app:app --reload
```
Then go to: [http://localhost:8000/docs](http://localhost:8000/docs) to test the API.

---

## 🔁 API Usage

### POST `/query`
**Request:**
```json
{
  "query": "What sustainability certifications are supported?"
}
```

**Response:**
```json
{
  "question": "What sustainability certifications are supported?",
  "answer": "The system supports certifications such as Blue Angel, EU Ecolabel, EMAS..."
}
```

---

## 📦 Prebuilt Index Sources

At startup, the application automatically downloads and unzips FAISS vector indexes from a **OneDrive** link:

- `source_1` → Index A
- `source_2` → Index B
- `source_3` → YouTube transcript index

These are merged and used as the searchable corpus.

---

## 📚 Use Cases

- Intelligent search for procurement and sustainability documentation
- Answering FAQs from uploaded vectorized company documents
- Semantic retrieval from multi-source document knowledge base

---
