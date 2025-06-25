import argparse, json, os
from rag import collection, embedding_model
from tqdm import tqdm

def retrieve_articles(item, mode: str, alpha: float = 0.2, top_k: int = 5):
    query = item["text"]
    id = item["question_id"]
    if mode == "dense":
        res = collection.query.near_vector(
            near_vector=embedding_model.embed_query(query),
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "bm25":
        res = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "hybrid":
        res = collection.query.hybrid(
            query=query,
            vector=embedding_model.embed_query(query),
            limit=top_k,
            alpha=alpha,
            return_metadata=["score"],
        )
    else:
        raise ValueError("Unsupported retrieval mode. Choose from: dense, bm25, hybrid.")
    
    ref = []
    for i in res.objects:
        ref.append({
            "law_id": i.properties.get("law_id"),
            "article_id": i.properties.get("article_id"),
        })
    result = {
        "question_id": id,
        "text": query,
        "retrieved_articles": ref,
    }
    return result

def main(input_file: str, output_file: str, mode: str, alpha: float = 0.2, top_k: int = 5):
    result = []
    with open(input_file, "r", encoding="utf8") as f:
        data = json.load(f)
    
    for item in tqdm(data, desc=f"Retrieving ({mode})", unit="question"):
        res = retrieve_articles(item, mode=mode, alpha=alpha, top_k=top_k)
        result.append(res)
    
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Article Retrieval CLI")

    parser.add_argument("--mode", type=str, choices=["dense", "bm25", "hybrid"], required=True,
                        help="Retrieval mode: dense, bm25, or hybrid")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha blending factor for hybrid mode (between 0 and 1)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top documents to retrieve")
    parser.add_argument("--input_file", type=str, default="./data/alqac25_train.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str,
                        help="Path to save the output JSON file")

    args = parser.parse_args()

    # ✅ Gán giá trị mặc định cho output_file nếu chưa chỉ định
    if not args.output_file:
        args.output_file = f"./data/retrieved_articles_{args.mode}.json"

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        mode=args.mode,
        alpha=args.alpha,
        top_k=args.top_k
    )
