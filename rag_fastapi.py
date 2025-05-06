import os
import zipfile
import requests
import faiss
import pickle
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
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

# Download and unzip vector files from OneDrive
VECTORS_URL = "https://sdvvg-my.sharepoint.com/:u:/g/personal/khadijah-ali_shah_evergabe_de/EdZ2vqUtutJNndmfh-HrTQYBanvuhXBTLQUn3c_pFgN2gA?download=1"
VECTOR_PATH = "vectorstore"

@app.on_event("startup")
def download_and_setup():
    os.makedirs(VECTOR_PATH, exist_ok=True)
    zip_path = os.path.join(VECTOR_PATH, "vectors.zip")

    print("⬇️ Downloading vector ZIP from OneDrive...")
    r = requests.get(VECTORS_URL)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    print("📦 Extracting vectors...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(VECTOR_PATH)
    os.remove(zip_path)

    print("✅ Vectors extracted. Initializing index...")
    initialize_rag()

# FastAPI request model
class QueryModel(BaseModel):
    query: str

# Global QA chain
qa_chain = None

def initialize_rag():
    global qa_chain
    index_paths = {
        "source_1": (f"{VECTOR_PATH}/faiss_index_m/index.faiss", f"{VECTOR_PATH}/faiss_index_m/index.pkl"),
        "source_2": (f"{VECTOR_PATH}/faiss_index_m1/index.faiss", f"{VECTOR_PATH}/faiss_index_m1/index.pkl"),
        "source_3": (f"{VECTOR_PATH}/youtubevectors/transcripts.index", f"{VECTOR_PATH}/youtubevectors/transcripts.pkl"),
    }

    merged_index = None
    all_docs = []
    docstore = CustomDocstore()

    print("\n🔧 Loading and Merging Indexes:")
    for source, (idx_path, pkl_path) in index_paths.items():
        try:
            if not os.path.exists(idx_path) or not os.path.exists(pkl_path):
                print(f"⚠️ Skipping {source} (missing files)")
                continue

            index = faiss.read_index(idx_path)
            with open(pkl_path, "rb") as f:
                metadata = pickle.load(f)

            print(f"📚 {source}: {index.ntotal} vectors | {len(metadata)} metadata items")

            if merged_index is None:
                merged_index = index
                all_docs = metadata
            else:
                vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
                merged_index.add(vectors)
                all_docs += metadata

        except Exception as e:
            print(f"⚠️ Skipping {source} due to error: {e}")

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

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=merged_index,
        docstore=docstore,
        index_to_docstore_id={i: str(i) for i in range(len(corrected_docs))}
    )

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 18})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.post("/query")
async def query_rag(request: QueryModel):
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"question": request.query, "answer": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
