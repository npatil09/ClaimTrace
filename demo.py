import os
import glob
import json

from src.embeddings import Embedder
from src.store import VectorStore
from src.llm import LLM
from src.corpus import ingest
from src.verify import verify_claim


def load_sample_docs():
    docs = []
    for path in sorted(glob.glob("data/*.txt")):
        with open(path) as f:
            docs.append({"text": f.read(), "source": os.path.basename(path)})
    return docs


def main():
    embedder = Embedder()
    store = VectorStore(path="./demo_qdrant_data", dim=embedder.dim)
    llm = LLM()

    if store.count() == 0:
        docs = load_sample_docs()
        n = ingest(embedder, store, docs)
        print(f"ingested {len(docs)} documents into {n} chunks\n")

    claims = [
        "The NovaCell 4 battery will ship in March 2027",
        "The NovaCell 4 has an energy density of 420 Wh/kg",
        "Nova Dynamics has never missed a product launch date before",
        "Reno's water treatment plant expansion costs $40 million",
    ]

    for claim in claims:
        result = verify_claim(llm, embedder, store, claim)
        print(f"CLAIM: {claim}")
        print(f"  verdict: {result['verdict']} (confidence: {result['confidence']})")
        print(f"  reasoning: {result['reasoning']}")
        print()


if __name__ == "__main__":
    main()
