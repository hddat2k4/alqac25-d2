import argparse
import json
import os
import re
from collections import Counter
from tqdm import tqdm
from rag import collection, embedding_model, model_id

# Load stopword list
with open("./vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(word.strip().lower() for word in f if word.strip())



def remove_meta_choices(choices):
    """
    Lọc bỏ các lựa chọn meta như 'Cả A, B, C đều đúng', 'Tất cả các ý trên', v.v.
    
    Args:
        choices (List[str]): Danh sách các lựa chọn (chuỗi)

    Returns:
        List[str]: Danh sách chỉ chứa các lựa chọn có nội dung cụ thể
    """
    meta_patterns = [
        r"cả\s*(3|ba)?\s*(phương án|đáp án)?\s*(a|b|c|d)(,?\s*(a|b|c|d))*\s*(và\s*(a|b|c|d))?\s*(đều\s*)?(đúng|sai)?\.?$",
        r"cả\s*(a|b|c|d)\s*và\s*(a|b|c|d)\s*(là\s*)?(đúng|sai)?\.?$",
        r"[a-d](,?\s*[a-d])*\s*đều\s*(đúng|sai)\.?$",
        r"cả\s*(a|b|c|d)\s*,\s*(a|b|c|d)\s*(và\s*(a|b|c|d))?\s*(đều\s*)?(đúng|sai)?\.?$",
        r"đáp án\s*(a|b|c|d)\s*và\s*(a|b|c|d)\s*là\s*(đúng|sai)?\.?$",
        r"cả\s*(3|ba)?\s*(phương án|đáp án)?\s*trên\s*(đều\s*)?(đúng|sai)?\.?$",
        r"cả.*trên\s*(đều\s*)?(đúng|sai)?\.?$",
        r"tất\s*cả(\s*các)?(\s*ý)?(\s*kiến)?(\s*phương án)?(\s*trên)?\s*(đều\s*)?(đúng|sai)?\.?$",
        r"không\s*có\s*lựa\s*chọn\s*đúng\.?$",
        r"cả\s*(a|b|c|d)\s*và\s*(a|b|c|d)(\s*và\s*(a|b|c|d))?\s*(đều\s*)?(đúng|sai)?\.?$",
    ]
    id = ["A", "B", "C", "D"]
    result = []
    for idx in id:
        choices_lower = choices[idx].lower()
        if any(re.fullmatch(p, choices_lower.strip()) for p in meta_patterns):
            continue
        result.append(choices[idx])
    return result

def clean_yesno_question(question):
    # Xoá cụm như "đúng hay sai?", "đúng hoặc sai.", "sai hoặc đúng?", ... ở cuối câu
    return re.sub(r'\s*(đúng|sai)\s*(hoặc|hay)?\s*(đúng|sai)?[.?\s]*$', '', question.strip(), flags=re.IGNORECASE)

def expand_query_with_rm3(top_docs, original_query, stopwords, top_n=8):
    all_tokens = []
    for doc in top_docs:
        content = doc.properties.get("page_content", "")
        tokens = re.findall(r'\b\w+\b', content.lower())
        filtered_tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        all_tokens.extend(filtered_tokens)

    term_counts = Counter(all_tokens)
    top_terms = [term for term, _ in term_counts.most_common(top_n)]
    return original_query.strip() + " " + " ".join(top_terms)

def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())


def retrieve_articles(item, mode: str, alpha: float = 0.2, top_k: int = 5, use_rm3: bool = False, use_rerank: bool = False):
    if item['question_type'] == "Trắc nghiệm":
        query = item["text"]+ ":" + " " + ";".join(remove_meta_choices(item["choices"]))
    elif item['question_type'] == "Đúng/Sai":
        query = clean_yesno_question(item["text"])
    else:
        query = item["text"]
    query = pre_processing(query)
    qid = item["question_id"]
    if use_rm3:
        # initial_res = collection.query.hybrid(
        #     query=query,
        #     vector=embedding_model.embed_query(query),
        #     limit=5,
        #     alpha=alpha,
        #     return_metadata=["score"],
        # )
        
        initial_res = collection.query.bm25(
            query=query,
            limit=10,
            return_metadata=["score"],
        )
        query_1 = expand_query_with_rm3(initial_res.objects, query, stopwords, top_n=5)
    else: 
        query_1 = query


    if mode == "dense":
        res = collection.query.near_vector(
            near_vector=embedding_model.embed_query(query_1),
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "bm25":
        res = collection.query.bm25(
            query=query_1,
            limit=top_k,
            return_metadata=["score"],
        )
    elif mode == "hybrid":
        res = collection.query.hybrid(
            query=query_1,
            vector=embedding_model.embed_query(query),
            limit=top_k,
            alpha=alpha,
            return_metadata=["score"],
        )
    else:
        raise ValueError("Unsupported retrieval mode. Choose from: dense, bm25, hybrid.")
    
    # if use_rerank:
    #     docs = rerank_documents(query, res.objects, top_n=top_k)  # có thể thêm threshold=0.5 nếu muốn lọc

    #     ref = [
    #         {
    #             "law_id": obj.properties.get("law_id"),
    #             "article_id": obj.properties.get("article_id"),
    #             "score": obj.metadata.score
    #         }
    #         for obj in docs
    #     ]

    # else:
    #     ref = [
    #         {
    #             "law_id": obj.properties.get("law_id"),
    #             "article_id": obj.properties.get("article_id"),
    #             "score": obj.metadata.score
    #         }
    #         for obj in res.objects            
    #     ]

    ref = [
        {
            "law_id": obj.properties.get("law_id"),
            "article_id": obj.properties.get("article_id"),
            "page_content": obj.properties.get("page_content"),
            "score": obj.metadata.score
        }
        for obj in res.objects            
    ]

    return {
        "query": query_1,
        "question_id": qid,
        "text": query,
        "retrieved_articles": ref,
    }

def retrieve_single_query(question_text, choices=None, question_type="Tự luận", mode="hybrid", alpha=0.5, top_k=5, use_rm3=False, use_rerank=False):
    item = {
        "question_id": "manual_test",
        "text": question_text,
        "choices": choices or [],
        "question_type": question_type
    }
    return retrieve_articles(item, mode=mode, alpha=alpha, top_k=top_k, use_rm3=use_rm3, use_rerank=use_rerank)




def main(input_file: str, output_file: str, mode: str, alpha: float = 0.2, top_k: int = 5, use_rm3: bool = False):
    result = []
    with open(input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Retrieving ({mode})", unit="question"):
        res = retrieve_articles(item, mode=mode, alpha=alpha, top_k=top_k, use_rm3=use_rm3)
        result.append(res)

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Article Retrieval CLI")
    parser.add_argument("--mode", type=str, choices=["dense", "bm25", "hybrid"], required=True,
                        help="Retrieval mode: dense, bm25, or hybrid")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha blending factor for hybrid mode")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top documents to retrieve")
    parser.add_argument("--input_file", type=str, default="./data/alqac25_train.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str,
                        help="Path to save the output JSON file")
    parser.add_argument("--rm3", action="store_true",
                        help="Enable RM3 query expansion before retrieval")
    #parser.add_argument("--rerank", action="store_true", help="Use CrossEncoder reranking after retrieval")

    args = parser.parse_args()

    if not args.output_file:
        model_id = model_id.replace("/", "_")
        args.output_file = f"./data/{model_id}_{args.mode}{'_rm3' if args.rm3 else ''}_topk{args.top_k}_ver1.json"

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        mode=args.mode,
        alpha=args.alpha,
        top_k=args.top_k,
        use_rm3=args.rm3
    )
