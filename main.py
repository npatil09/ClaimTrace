from fastapi import FastAPI
from pydantic import BaseModel

from src.embeddings import Embedder
from src.store import VectorStore
from src.llm import LLM
from src.corpus import ingest
from src.verify import verify_claim

app = FastAPI(title="ClaimTrace")

embedder = Embedder()
store = VectorStore(dim=embedder.dim)
llm = LLM()


class Document(BaseModel):
    text: str
    source: str
    published: str | None = None


class IngestRequest(BaseModel):
    documents: list[Document]


class VerifyRequest(BaseModel):
    claim: str
    top_k: int = 4


@app.post("/ingest")
def ingest_documents(req: IngestRequest):
    docs = [d.model_dump() for d in req.documents]
    n_chunks = ingest(embedder, store, docs)
    return {"documents_added": len(docs), "chunks_created": n_chunks, "total_chunks": store.count()}


@app.post("/verify")
def verify(req: VerifyRequest):
    return verify_claim(llm, embedder, store, req.claim, top_k=req.top_k)


@app.get("/stats")
def stats():
    return {"total_chunks": store.count()}
