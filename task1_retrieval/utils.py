import json
import re
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances, 
)
from langchain.embeddings.base import Embeddings
from sentence_transformers.cross_encoder import CrossEncoder
import time
import os

hf_token = os.getenv("HF_TOKEN")

### Rechunking Utilities
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_article_length(article):
    words = re.findall(r'\w+', article)  # Đếm đúng cả từ và số
    return len(words)

def split_article(article_text, threshold):
    # Bước 1: Tách khoản theo regex
    # Mẫu regex tìm các khoản bắt đầu bằng số + dấu chấm + khoảng trắng (ví dụ: "1. ", "2. ", ...)
    clause_pattern = r'(\d+\.\s.*?)((?=\n\s*\d+\.\s)|$)'
    clauses = re.split(clause_pattern, article_text)
    clauses = [clause.strip() for clause in clauses if clause.strip()]  # Xóa khoảng trắng thừa
    
    # Nếu tổng số từ không vượt threshold thì giữ nguyên
    total_words = get_article_length(article_text)
    if total_words <= threshold or len(clauses) == 1:
        return [article_text]
        
    # Bước 2: Tính số chunk cần có
    n_chunks = int(total_words / threshold) + 1

    target_words_per_chunk = total_words / n_chunks

    merged_chunks = []
    current_chunk = clauses[0]  # Bắt đầu với clause đầu tiên
    current_word_count = 0
    chunk_count = 0

    for clause in clauses[1:]:
        clause_word_count = get_article_length(clause)

        # Nếu là chunk cuối cùng thì gom hết
        if chunk_count == n_chunks - 1:
            if current_chunk:
                current_chunk += '\n' + clause
            else:
                current_chunk = clause
            continue

        # Nếu cộng thêm clause vẫn dưới ngưỡng → thêm vào chunk hiện tại
        if current_word_count + clause_word_count <= target_words_per_chunk:
            if current_chunk:
                current_chunk += '\n' + clause
            else:
                current_chunk = clause
            current_word_count += clause_word_count
        else:
            # Đóng chunk hiện tại nếu không rỗng
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
                chunk_count += 1
            # Bắt đầu chunk mới với clause này
            current_chunk = clause
            current_word_count = clause_word_count

    # Thêm phần còn lại vào chunk cuối
    if current_chunk:
        merged_chunks.append(current_chunk.strip())

    return merged_chunks

def rechunk_data(data, threshold):
    rechunked_data = []
    for item in data:
        rechunked_law = []
        for article in item['articles']:
            chunks = split_article(article['text'], threshold)
            for i, chunk in enumerate(chunks):
                rechunked_law.append({
                    'id': article['id'],
                    'split': i,
                    'text': chunk
                })
        rechunked_data.append({
            'id': item['id'],
            'articles': rechunked_law
        })
    print(f"Rechunked {len(data)} laws into {sum(len(law['articles']) for law in rechunked_data)} chunks.")
    return rechunked_data

### Embedding Utilities
def data_to_dict(data):
    data_dict = []

    for item in data:
        law_id = item["id"]
        articles = item["articles"]

        for article in articles:
            article_id = article["id"]
            split = article["split"]
            text = article["text"]
            data_dict.append({
                "page_content": text,
                "metadata": {"law_id": law_id, "article_id": article_id, "split": split}
            })

    print(f"Converted data to dictionary with {len(data_dict)} entries.")

    return data_dict

def embed_data(data, model_name="AITeamVN/Vietnamee_Embedding", batch_size=32):

    # Check if GPU (CUDA) is available and set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the SentenceTransformer model and move it to the selected device
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 2048

    batch_size = 32
    texts = []
    items = []

    # Loop over the data and prepare text inputs
    for item in data:
        text = f"{item['metadata']['law_id']} {item['page_content']}"
        texts.append(text)
        items.append(item)

        # If batch is full, compute embeddings
        if len(texts) == batch_size:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=device  # Ensure encoding is done on GPU
            )
            # Assign computed embeddings back to original items
            for i, emb in enumerate(embeddings):
                items[i]['embedding'] = emb.tolist()
            texts = []
            items = []

    # Process remaining texts if total number is not divisible by batch_size
    if texts:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device  # Ensure encoding is done on GPU
        )
        for i, emb in enumerate(embeddings):
            items[i]['embedding'] = emb.tolist()

    print(f"Embedded {len(data)} items into {len(items)} embeddings.")

    return data

### Weaviate Utilities
def create_hybrid_collection(name):
    """
    Creates a Weaviate collection with hybrid search (BM25 + vector).
    
    Args:
        client: The Weaviate client instance.
        name (str): The name of the collection to create.

    This function:
    - Deletes the existing collection (if any)
    - Creates a new collection with:
        - BM25 inverted index
        - No automatic vectorizer
        - Flat vector index using cosine distance
        - Custom schema properties
    """
    client = weaviate.connect_to_local(skip_init_checks=True)

    # Delete existing collection if it exists
    client.collections.delete(name)

    # Create a new collection schema with hybrid search support
    client.collections.create(
        name=name,
        description="ALQAC",
        inverted_index_config=Configure.inverted_index(bm25_b=0.75, bm25_k1=1.2),
        vectorizer_config=Configure.Vectorizer.none(),  # We manually provide vectors
        vector_index_config=Configure.VectorIndex.flat(distance_metric=VectorDistances.COSINE),
        properties=[
            Property(name="page_content", data_type=DataType.TEXT),
            Property(name="law_id", data_type=DataType.TEXT),
            Property(name="article_id", data_type=DataType.TEXT),
            Property(name="split", data_type=DataType.NUMBER),
        ]
    )

    print(f"✅ Created schema '{name}' successfully.")

    # Close the client connection
    client.close()

def insert_data_with_vectors(name, embedding_data):
    """
    Inserts documents with precomputed vectors into a Weaviate collection using dynamic batching.

    Args:
        name (str): The name of the Weaviate collection to insert data into.
        embedding_data (list): A list of data. Each data must contain:
            - 'page_content': The text content
            - 'embedding': A list of floats representing the vector
            - 'metadata': A dict with keys: 'law_id', 'article_id', and 'split'
    """
    client = weaviate.connect_to_local(skip_init_checks=True)
    collection = client.collections.get(name)

    total = len(embedding_data)
    success_count = 0
    fail_count = 0

    # Use dynamic batching for efficient upload
    with collection.batch.dynamic() as batch:
        for obj in embedding_data:
            try:
                # Extract metadata and vector
                metadata = {
                    "page_content": obj["page_content"],
                    "law_id": obj["metadata"]["law_id"],
                    "article_id": obj["metadata"]["article_id"],
                    "split": obj["metadata"]["split"]
                }

                # Add object to batch with associated vector
                batch.add_object(
                    properties=metadata,
                    vector=obj["embedding"]
                )
                success_count += 1

            except Exception as e:
                # Handle and log insertion errors
                print(f"Error inserting object '{obj['page_content']}': {e}")
                fail_count += 1

    # Print summary
    print(f"Total vectors in file: {total}")
    print(f"Successfully inserted: {success_count}")
    print(f"Failed to insert: {fail_count}")
    print(f"Total processed: {success_count + fail_count}")

    # Close the client connection
    client.close()

### Retrieval Utilities
def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_id='AITeamVN/Vietnamee_Embedding'):
        self.model = SentenceTransformer(model_id, token=hf_token, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    
def retrieve(input_file: str, 
             output_file: str,
             name: str,
             mode: str, 
             alpha: float, 
             top_k: int):
    """
    Performs document retrieval using Weaviate based on the specified mode (hybrid, dense, or bm25).
    
    Args:
        input_file (str): Path to the input JSON file containing questions.
        output_file (str): Path to save the retrieved results.
        name (str): Not used in this function (placeholder for future use or override).
        mode (str): Retrieval mode - "hybrid", "dense", or "bm25".
        alpha (float): Hybrid weight between BM25 and vector score (only used in hybrid mode).
        top_k (int): Number of top candidates to retrieve.
    """
    
    print("Initializing embedding model...")
    embedding_model = CustomSentenceTransformerEmbeddings()
    print("Embedding model is ready.")

    try:
        with weaviate.connect_to_local() as client:
            print("Successfully connected to Weaviate.")
            collection = client.collections.get(name)

            results_for_reranking = []

            # Load the input questions
            with open(input_file, "r", encoding="utf8") as f:
                data = json.load(f)

            # Iterate through each question in the input file
            for item in tqdm(data, desc=f"Retrieving candidates ({mode})", unit="question"):
                question_id = item["question_id"]
                query = item["text"]
                query = pre_processing(query)  # Clean and normalize the query

                # Perform retrieval based on selected mode
                if mode == "hybrid":
                    res = collection.query.hybrid(
                        query=query,
                        vector=embedding_model.embed_query(query),
                        limit=top_k,
                        alpha=alpha,
                        return_metadata=["score"],
                        return_properties=["law_id", "article_id", "page_content", "split"]
                    )
                elif mode == "dense":
                    res = collection.query.near_vector(
                        near_vector=embedding_model.embed_query(query),
                        limit=top_k,
                        return_metadata=["score"],
                        return_properties=["law_id", "article_id", "page_content", "split"]
                    )
                elif mode == "bm25":
                    res = collection.query.bm25(
                        query=query,
                        limit=top_k,
                        return_metadata=["score"],
                        return_properties=["law_id", "article_id", "page_content", "split"]
                    )
                else:
                    raise ValueError(f"Invalid retrieval mode: {mode}")

                # Format the result objects for JSON output
                candidates = []
                for obj in res.objects:
                    candidates.append({
                        "law_id": obj.properties.get("law_id"),
                        "article_id": obj.properties.get("article_id"),
                        "split": obj.properties.get("split"),
                        "page_content": obj.properties.get("page_content"),
                        "retrieval_score": obj.metadata.score
                    })

                results_for_reranking.append({
                    "question_id": question_id,
                    "text": query,
                    "retrieved_candidates": candidates
                })

            # Write the retrieved results to output JSON file
            with open(output_file, "w", encoding="utf8") as f:
                json.dump(results_for_reranking, f, ensure_ascii=False, indent=4)
            print(f"\n✅ Saved {len(results_for_reranking)} retrieval results to file: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Closed connection to Weaviate.")

### Reranking Utilities
def rerank(candidate_file: str, 
           output_file: str,
           threshold_mode: str, 
           threshold_value: float,
           batch_size: int = 128):
    """
    Reranks retrieved candidates using a cross-encoder model and applies threshold-based filtering.

    Args:
        candidate_file (str): Path to file containing retrieved candidate documents.
        ground_truth_file (str): Path to ground truth data (used for potential evaluation).
        output_file (str): File to write final filtered results.
        threshold_mode (str): Either "hard" or "dynamic" to determine filtering strategy.
        threshold_value (float): Threshold value used for filtering.
        batch_size (int): Batch size for reranking using GPU acceleration.
    """

    print("Loading Cross-Encoder model for reranking...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    reranker_model = CrossEncoder('AITeamVN/Vietnamese_Reranker', device=device)
    print("Cross-Encoder model loaded successfully.")

    # Load input data
    candidate_data = load_json(candidate_file)

    # === STEP 1: Prepare all (query, candidate) pairs ===
    print("Preparing sentence pairs for batch processing...")
    all_sentence_pairs = []
    processing_info = []

    for item in candidate_data:
        query = item["text"]
        candidates = item["retrieved_candidates"]
        processing_info.append({"item_data": item, "num_candidates": len(candidates)})

        if candidates:
            # Create all query-document pairs
            pairs = [(query, cand["page_content"]) for cand in candidates]
            all_sentence_pairs.extend(pairs)
    
    print(f"Total sentence pairs to rerank: {len(all_sentence_pairs)}")

    # === STEP 2: Run reranking in batches to utilize GPU efficiently ===
    print("Starting batched reranking...")
    start_time = time.time()
    all_rerank_scores = reranker_model.predict(
        all_sentence_pairs,
        batch_size=batch_size,
        show_progress_bar=True
    )
    end_time = time.time()
    print(f"✅ Reranking completed in {end_time - start_time:.2f} seconds.")

    # === STEP 3: Assign rerank scores back to candidates and apply filtering ===
    print("Processing reranked results and applying filtering...")
    final_results = []
    current_score_index = 0

    for info in tqdm(processing_info, desc="Post-processing rerank results"):
        item_data = info["item_data"]
        num_candidates = info["num_candidates"]

        if num_candidates == 0:
            final_results.append({
                "question_id": item_data["question_id"],
                "text": item_data["text"],
                "retrieved_articles": []
            })
            continue

        # Slice scores for current query
        item_scores = all_rerank_scores[current_score_index : current_score_index + num_candidates]
        current_score_index += num_candidates

        candidates = item_data["retrieved_candidates"]
        # Assign rerank score to each candidate
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(item_scores[i])

        # Sort candidates by rerank score (descending)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Filter based on threshold strategy
        filtered_articles = []
        if threshold_mode and candidates:
            if threshold_mode == "dynamic":
                # Dynamic threshold is a percentage of the top score
                threshold = candidates[0]['rerank_score'] * threshold_value
            elif threshold_mode == "hard":
                # Hard threshold is a fixed score
                threshold = threshold_value
            else:
                raise ValueError(f"Invalid threshold_mode: {threshold_mode}")

            # Keep only candidates with score >= threshold
            for cand in candidates:
                if cand['rerank_score'] >= threshold:
                    filtered_articles.append({
                        "law_id": cand["law_id"],
                        "article_id": cand["article_id"]
                    })
        else:
            # If no filtering, return all reranked candidates
            filtered_articles = [{
                "law_id": c["law_id"],
                "article_id": c["article_id"]
            } for c in candidates]

        # Append final result for this question
        final_results.append({
            "question_id": item_data["question_id"],
            "relevant_articles": filtered_articles,
        })

    # === STEP 4: Save results to file ===
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Final results saved to: {output_file}")
