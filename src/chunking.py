import re
import uuid


SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(text, source, window=4, stride=2, published=None):
    sentences = split_sentences(text)
    chunks = []
    i = 0
    while i < len(sentences):
        window_sents = sentences[i:i + window]
        if not window_sents:
            break
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": " ".join(window_sents),
            "source": source,
            "published": published,
            "sentence_range": (i, i + len(window_sents)),
        })
        if i + window >= len(sentences):
            break
        i += stride
    return chunks


def chunk_corpus(documents):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(
            doc["text"], doc["source"], published=doc.get("published")
        ))
    return all_chunks
