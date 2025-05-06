import os
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

qa_chain = None  # global holder

class QueryModel(BaseModel):
    query: str

class CustomDocstore(InMemoryStore):
    def search(self, doc_id: str):
        doc = self.mget([doc_id])
        return doc[0] if doc else None

def load_index(source_folder):
    idx_path = os.path.join(source_folder, "index.faiss") if "youtube" not in source_folder else os.path.join(source_folder, "transcripts.index")
    pkl_path = os.path.join(source_folder, "index.pkl") if "youtube" not in source_folder else os.path.join(source_folder, "transcripts.pkl")

    if not os.path.exists(idx_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing files for {source_folder}")

    index = faiss.read_index(idx_path)
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)

    corrected_docs = []
    for i, item in enumerate(metadata):
        if isinstance(item, tuple) and len(item) == 2:
            meta, content = item
        elif isinstance(item, dict):
            meta, content = item, item.get("page_content", "No content")
        else:
            meta, content = {"source": "unknown"}, str(item)
        corrected_docs.append((str(i), Document(page_content=content, metadata=meta)))

    docstore = CustomDocstore()
    docstore.mset(corrected_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: str(i) for i in range(len(corrected_docs))}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 18})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def create_app(source_folder: str, source_key: str):
    app = FastAPI(title=f"evergabe.de RAG API – {source_key}")

    @app.on_event("startup")
    def startup():
        global qa_chain
        logging.info(f"Loading index for {source_key}...")
        qa_chain = load_index(source_folder)
        logging.info(f"✅ Ready: {source_key}")

    @app.post("/query")
    async def query(request: QueryModel):
        try:
            response = qa_chain.invoke({"query": request.query})
            return {"source": source_key, "question": request.query, "answer": response["result"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
