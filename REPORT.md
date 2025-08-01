# D2 Team - ALQAC 2025 Technical Report

This repository contains the code and experiments for our participation in **ALQAC 2025** (Automated Legal Question Answering Competition).

## Results

- **Task 1: Legal Document Retrieval**  
  Achieved **7th place** using a hybrid retrieval pipeline with BM25 + dense embeddings (Weaviate) and AITeamVN/Vietnamese-Reranker.

- **Task 2: Legal Question Answering**  
  Achieved **8th place** using fine-tuned open-source models (AITeamVN/GRPO-VI-Qwen2-7B-RAG and others), with step-by-step prompting and ensemble strategies.

## Methodology

- **Task 1**: Hybrid retrieval  
  - BM25 over raw legal texts  
  - Dense retrieval with `AITeamVN/Vietnamese-Embedding`  
  - Query rewriting with `AITeamVN/GRPO-VI-Qwen2-7B-RAG`  
  - Reranking with `AITeamVN/Vietnamese-Reranker`  

- **Task 2**: Question Answering  
  - Essay questions: step-by-step prompting with few-shot examples  
  - Yes/No & Multiple Choice: zero-shot prompting with ensemble & majority voting  
  - Final submission combined results from multiple runs for robustness

## Reproduction

- See the [technical report PDF](./D2Team_TechnicalReport.pdf) for full details.  
- Instructions to reproduce Task 1 and Task 2 are included in the report.  

## Acknowledgment
We thank our mentor **Thin Dang Van** for invaluable support during this project.
