from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    def __init__(self, path="./qdrant_data", collection="claimtrace", dim=384):
        self.client = QdrantClient(path=path)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def add(self, chunks, vectors):
        points = [
            PointStruct(
                id=chunk["id"],
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "published": chunk.get("published"),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
        ).points
        return [
            {
                "text": r.payload["text"],
                "source": r.payload["source"],
                "published": r.payload.get("published"),
                "score": r.score,
            }
            for r in results
        ]

    def count(self):
        return self.client.count(collection_name=self.collection).count
