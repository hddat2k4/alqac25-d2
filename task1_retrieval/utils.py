import json, re, math, torch, os, time
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
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
from dotenv import load_dotenv


load_dotenv()
hf_token = os.getenv("HF_TOKEN")

### Rechunking Utilities
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_article_length(article):
    words = re.findall(r'\w+', article)
    return len(words)

def split_article(article_text, threshold):
    """
    Splits an article into chunks based on clauses and word count threshold.
    
    Args:
        article_text (str): The article text to split
        threshold (int): Maximum words per chunk
        
    Returns:
        list: List of article chunks
    """
    # === Step 1: Split clauses using regex
    # Regex pattern to find clauses starting with number + dot + space (e.g., "1. ", "2. ", ...)
    clause_pattern = r'(\d+\.\s.*?)((?=\n\s*\d+\.\s)|$)'
    clauses = re.split(clause_pattern, article_text)
    clauses = [clause.strip() for clause in clauses if clause.strip()]  # Xóa khoảng trắng thừa
    
    # If total words doesn't exceed threshold, keep original
    total_words = get_article_length(article_text)
    if total_words <= threshold or len(clauses) == 1:
        return [article_text]
        
    # === Step 2: Calculate required number of chunks
    n_chunks = math.ceil(total_words / threshold)

    target_words_per_chunk = total_words / n_chunks

    merged_chunks = []
    current_chunk = clauses[0]  # Start with first clause
    current_word_count = 0
    chunk_count = 0

    for clause in clauses[1:]:
        clause_word_count = get_article_length(clause)

        # If this is the last chunk, merge everything remaining
        if chunk_count == n_chunks - 1:
            if current_chunk:
                current_chunk += '\n' + clause
            else:
                current_chunk = clause
            continue

        # If adding clause still under threshold → add to current chunk
        if current_word_count + clause_word_count <= target_words_per_chunk:
            if current_chunk:
                current_chunk += '\n' + clause
            else:
                current_chunk = clause
            current_word_count += clause_word_count
        else:
            # Close current chunk if not empty
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
                chunk_count += 1
            # Start new chunk with this clause
            current_chunk = clause
            current_word_count = clause_word_count

    # Add remaining content to final chunk
    if current_chunk:
        merged_chunks.append(current_chunk.strip())

    return merged_chunks


def split_clause(text):
    split_law = re.split(r'\n(?=\d+\.)', text)
    split_law = [s.strip() for s in split_law if s.strip()]
    return split_law

def chunk_clause(data, output_file):
    """
    Splits articles in legal data into smaller chunks based on clauses.
    
    Args:
        data (list): List of laws, each containing articles
        output_file (str): Path to save chunked data
        
    Returns:
        list: Chunked data with split articles
    """
    
    # Initialize container for chunked results
    chunked_data = []
    
    # Process each law in the dataset
    for item in data:
        chunked_law = []  # Container for current law's chunked articles
        for article in item['articles']:
            # Split article text into chunks based on clauses
            chunks = split_clause(article['text'])
            # Create separate entries for each chunk
            for i, chunk in enumerate(chunks):
                chunked_law.append({
                    'id': article['id'],
                    'split': i,  # Add split index (0, 1, 2, ...)
                    'text': chunk
                })
        chunked_data.append({
            'id': item['id'],
            'articles': chunked_law
        })
    
    print(f"Chunked {len(data)} laws into {sum(len(law['articles']) for law in chunked_data)} chunks.")

    # Save chunked data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Chunked data saved to {output_file}")

    return chunked_data


def rechunk_data(data, threshold, output_file):
    """
    Rechunks legal data by splitting articles that exceed word threshold.
    
    Args:
        data (list): List of laws, each containing articles
        threshold (int): Maximum word count per chunk
        output_file (str): Path to save rechunked data
        
    Returns:
        list: Rechunked data with split articles
    """

    # Initialize container for rechunked results
    rechunked_data = []
    
    # Process each law in the dataset
    for item in data:
        rechunked_law = [] # Container for current law's rechunked articles
        for article in item['articles']:
            # Split article text into chunks based on threshold
            chunks = split_article(article['text'], threshold)
            # Create separate entries for each chunk
            for i, chunk in enumerate(chunks):
                rechunked_law.append({
                    'id': article['id'],
                    'split': i, # Add split index (0, 1, 2, ...)
                    'text': chunk
                })
        rechunked_data.append({
            'id': item['id'],
            'articles': rechunked_law
        })
    print(f"Rechunked {len(data)} laws into {sum(len(law['articles']) for law in rechunked_data)} chunks.")

    # Save rechunked data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rechunked_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Rechunked data saved to {output_file}")

    return rechunked_data

### Embedding Utilities
def data_to_dict(data):
    """
    Converts a nested legal dataset into a flat list of dictionaries, 
    each containing the text content of an article and its associated metadata.

    Args:
        data (list): A list of dictionaries, where each dictionary represents a legal document
                     with the structure:
                     {
                         "id": <law_id>,
                         "articles": [
                             {
                                 "id": <article_id>,
                                 "split": <section_name_or_type>,
                                 "text": <article_text>
                             },
                             ...
                         ]
                     }

    Returns:
        list: A flattened list of dictionaries in the format:
              {
                  "page_content": <article_text>,
                  "metadata": {
                      "law_id": <law_id>,
                      "article_id": <article_id>,
                      "split": <section_name_or_type>
                  }
              }
    """
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


def embed_data(data, model_name="AITeamVN/Vietnamese_Embedding", batch_size=16, output_file=None):
    """
    Generates vector embeddings for a list of input documents using a specified SentenceTransformer model.
    Embeddings are computed in batches, assigned back to the original data, and saved to a file.

    Args:
        data (list): A list of dictionaries, each containing 'page_content' (text) and 'metadata'.
        model_name (str): The name or path of the SentenceTransformer model to be used for embedding.
        batch_size (int): Number of items to process per batch. Smaller values reduce GPU memory usage.
        output_file (str): Optional. Path to the output JSON file where embedded data will be saved.

    Returns:
        list: The input data with an additional 'embedding' field added to each item.
    """
    # Force CUDA GPU usage and check specific GPU
    if torch.cuda.is_available():
        device = "cuda:0"  # Explicitly use first CUDA GPU
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Load model với device explicit
    model = SentenceTransformer(
        model_name, 
        token=hf_token, 
        trust_remote_code=True,
        device=device
    )
    model.max_seq_length = 2048
    
    # Verify model is on GPU
    print(f"Model device: {next(model.parameters()).device}")
    
    # Process in smaller batches to avoid memory issues
    batch_size = min(batch_size, 16)  # Reduce batch size
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_items = data[i:i+batch_size]
        texts = [f"{item['metadata']['law_id']} {item['page_content']}" 
                for item in batch_items]
        
        # Clear GPU cache before processing
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        embeddings = model.encode(
            texts,
            batch_size=len(texts),  # Process all texts in batch at once
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Assign embeddings
        for j, emb in enumerate(embeddings):
            batch_items[j]['embedding'] = emb.tolist()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Embeddings saved to {output_file}")
    
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
    def __init__(self, model_id='AITeamVN/Vietnamese_Embedding'):
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
           batch_size: int = 12): 
    """
    Reranks retrieved candidates using a cross-encoder model and applies threshold-based filtering.
    Processes each question individually with its candidates in parallel.

    Args:
        candidate_file (str): Path to file containing retrieved candidate documents.
        output_file (str): File to write final filtered results.
        threshold_mode (str): Either "hard" or "dynamic" to determine filtering strategy.
        threshold_value (float): Threshold value used for filtering.
        batch_size (int): Batch size for processing candidates per question (default: 12).
    """

    print("Loading Cross-Encoder model for reranking...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    reranker_model = CrossEncoder('AITeamVN/Vietnamese_Reranker', device=device)
    print("Cross-Encoder model loaded successfully.")

    # Load input data
    candidate_data = load_json(candidate_file)
    print(f"Total questions to process: {len(candidate_data)}")

    # Process each question individually
    final_results = []
    
    for epoch, item in enumerate(tqdm(candidate_data, desc="Processing questions"), 1):
        query = item["text"]
        candidates = item["retrieved_candidates"]
        
        print(f"\nEpoch {epoch}/{len(candidate_data)} - Question ID: {item['question_id']}")
        print(f"Number of candidates: {len(candidates)}")
        
        if not candidates:
            final_results.append({
                "question_id": item["question_id"],
                "text": item["text"],
                "relevant_articles": []
            })
            continue

        # === Step 1: Prepare sentence pairs for current question ===
        sentence_pairs = [(query, cand["page_content"]) for cand in candidates]
        
        # === Step 2: Run reranking for this question's candidates ===
        start_time = time.time()
        rerank_scores = reranker_model.predict(
            sentence_pairs,
            batch_size=min(batch_size, len(sentence_pairs)),  # Use actual number of candidates or batch_size
            show_progress_bar=False  # Disable inner progress bar since we have outer one
        )
        end_time = time.time()
        print(f"Reranking completed in {end_time - start_time:.3f} seconds")

        # === Step 3: Assign scores and apply filtering ===
        # Assign rerank score to each candidate
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(rerank_scores[i])

        # Sort candidates by rerank score (descending)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        print(f"Top score: {candidates[0]['rerank_score']:.4f}, Lowest score: {candidates[-1]['rerank_score']:.4f}")

        # Filter based on threshold strategy
        filtered_articles = []
        if threshold_mode and candidates:
            if threshold_mode == "dynamic":
                # Dynamic threshold is a percentage of the top score
                threshold = candidates[0]['rerank_score'] * threshold_value
                print(f"Dynamic threshold: {threshold:.4f}")
            elif threshold_mode == "hard":
                # Hard threshold is a fixed score
                threshold = threshold_value
                print(f"Hard threshold: {threshold}")
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

        print(f"Filtered articles: {len(filtered_articles)}/{len(candidates)}")

        # Append final result for this question
        final_results.append({
            "question_id": item["question_id"],
            "relevant_articles": filtered_articles,
        })

        # Clear GPU cache after each question to prevent memory buildup
        if device == 'cuda':
            torch.cuda.empty_cache()

    # === Step 4: Save results to file ===
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Final results saved to: {output_file}")
    print(f"Processed {len(final_results)} questions total")
    
    # Summary statistics
    total_articles = sum(len(result["relevant_articles"]) for result in final_results)
    avg_articles = total_articles / len(final_results) if final_results else 0
    print(f"Total relevant articles: {total_articles}")
    print(f"Average articles per question: {avg_articles:.2f}")


def get_json(data):
    if isinstance(data, list):
        data = ''.join(data)

    # Tìm phần sau ### Kết quả: rồi lấy giá trị "query"
    match = re.search(r'### Kết quả:.*?"query"\s*:\s*"([^"]+)"', data, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print("❌ Không tìm thấy trường 'query' sau phần Kết quả.")
        return None
    
rule_map = {
    r'\bcó quyền\b': 'có thẩm quyền',
    r'\bai \b': 'chủ thể',
    r'\báp dụng [^\s]+\s+luật\b': 'hiệu lực thi hành của luật',
    r'\bcó bị xử phạt không nếu\b': 'quy định xử phạt trong trường hợp',
    r'\bcó thể[^,\.!?]* xử phạt\b': 'thẩm quyền xử phạt',
    r'\bđược phép[^,\.!?]* nếu\b': 'trường hợp được phép theo quy định',
    r'\bcó bắt buộc phải\b': 'trách nhiệm bắt buộc theo quy định',
    r'\blà gì\??': 'định nghĩa pháp lý của',
    r'\bthuộc quyền [^?]* nào\??': 'thẩm quyền quản lý',
    r'\bđược thực hiện như thế nào\??': 'trình tự, thủ tục thực hiện',
    r'\bcó thời hạn bao lâu\??': 'thời hạn theo quy định',
    r'\bthực hiện [^,\.!?]* khi nào\b': 'thời điểm thực hiện theo quy định',
    r'\bđược coi là\b': 'khái niệm pháp lý của',
}

def apply_rules_with_overlap(text, rules):
    # Collect all matches with positions and replacement text
    matches = []
    for pattern, replacement in rules.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            matches.append((start, end, replacement))

    # Sort matches and eliminate overlaps
    matches.sort()
    non_overlapping = []
    last_end = -1
    for start, end, replacement in matches:
        if start >= last_end:
            non_overlapping.append((start, end, replacement))
            last_end = end

    # Apply replacements in reverse order to preserve indexes
    for start, end, replacement in reversed(non_overlapping):
        text = text[:start] + replacement + text[end:]

    return text

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()