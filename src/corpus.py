from .chunking import chunk_corpus


def ingest(embedder, store, documents):
    chunks = chunk_corpus(documents)
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)
    store.add(chunks, vectors)
    return len(chunks)
