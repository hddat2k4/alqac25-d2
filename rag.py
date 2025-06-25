import weaviate

client = weaviate.connect_to_local()

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# ==== Embedding Model ====
class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_id='BAAI/bge-m3'):
        self.model = SentenceTransformer(model_id, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

embedding_model = CustomSentenceTransformerEmbeddings()

names = "alqac"
collection = client.collections.get(names)