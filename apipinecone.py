import os
import faiss
import pickle
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore

# Load API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")
os.environ["OPENAI_API_KEY"] = api_key

logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="evergabe.de RAG Model", description="API to answer procurement-related queries.")

# Custom Docstore for LangChain FAISS
class CustomDocstore(InMemoryStore):
    def search(self, doc_id: str):
        doc = self.mget([doc_id])
        return doc[0] if doc else None

# Load a FAISS index and metadata
def load_faiss_index(index_path, metadata_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    index = faiss.read_index(index_path)
    return index, metadata

# Define your FAISS index locations
index_paths = {
    "source_1": ("/mnt/data/faiss_index_m/index.faiss", "/mnt/data/faiss_index_m/index.pkl"),
    "source_2": ("/mnt/data/faiss_index_m1/index.faiss", "/mnt/data/faiss_index_m1/index.pkl"),
    "source_3": ("/mnt/data/faiss_transcripts.index", "/mnt/data/transcripts.pkl"),
}

# Merge FAISS indexes
merged_index = None
all_docs = []

docstore = CustomDocstore()
index_to_docstore_id = {}

print("\n🔧 Loading and Merging Indexes:")
for source, (idx_path, pkl_path) in index_paths.items():
    try:
        index, metadata = load_faiss_index(idx_path, pkl_path)
        print(f"📚 {source} contains: {index.ntotal} vectors | {len(metadata)} metadata items")

        if merged_index is None:
            merged_index = index
            all_docs = metadata
        else:
            vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
            merged_index.add(vectors)
            all_docs += metadata

    except Exception as e:
        print(f"⚠️ Skipping {source} due to error: {e}")

# Prepare documents
corrected_docs = []
for i, item in enumerate(all_docs):
    if isinstance(item, tuple) and len(item) == 2:
        meta, content = item
    elif isinstance(item, dict):
        meta, content = item, item.get("page_content", "No content")
    else:
        meta, content = {"source": "unknown"}, str(item)
    corrected_docs.append((str(i), Document(page_content=content, metadata=meta)))

docstore.mset(corrected_docs)

# LangChain vectorstore setup
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS(
    embedding_function=embeddings,
    index=merged_index,
    docstore=docstore,
    index_to_docstore_id={i: str(i) for i in range(len(corrected_docs))}
)

# LLM and QA chain
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 18})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# FastAPI request model
class QueryModel(BaseModel):
    query: str

# FastAPI endpoint
@app.post("/query")
async def query_rag(request: QueryModel):
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"question": request.query, "answer": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
