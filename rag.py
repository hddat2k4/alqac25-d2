import weaviate

client = weaviate.connect_to_local()

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker", use_fast=False)
# model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker").to(device)

# def rerank_documents(query, docs, top_n=None, threshold=None, batch_size=8):
#     inputs = []
#     for doc in docs:
#         text = doc.properties.get("content", "")
#         combined = f"{query} [SEP] {text}"
#         inputs.append(combined)

#     encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**encodings)
#         scores = torch.sigmoid(outputs.logits).squeeze().tolist()


#     scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)


#     if threshold is not None:
#         scored = [pair for pair in scored if pair[0] >= threshold]
#     if top_n is not None:
#         scored = scored[:top_n]

#     return [doc for score, doc in scored]

model_id = "AITeamVN/Vietnamese_Embedding"
# ==== Embedding Model ====
class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_id=model_id):
        self.model = SentenceTransformer(model_id, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

embedding_model = CustomSentenceTransformerEmbeddings()

names = "alqac_aivn"  # Change this to the name of your collection
collection = client.collections.get(names)


#python evaluate.py --retrieved_file=./data/AITeamVN_Vietnamese_Embedding_bm25_rm3_topk1_ver1.json --ground_truth_file=./data/alqac25_train.json --error_output_file=err.json