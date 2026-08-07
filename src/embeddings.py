from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors)

    def encode_one(self, text):
        return self.encode([text])[0]
