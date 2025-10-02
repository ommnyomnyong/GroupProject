# -*- coding: utf-8 -*-
"""
GMP 청킹 JSONL 파일 임베딩 → OpenAI 임베딩 모델 → 로컬 저장

python gmp_vectordb.py --input "./chunks/fda_semantic_chunks.jsonl" --output "gmp_embeddings.pkl" --batch-size 10
"""

import argparse
import json
import os
import time
import pickle
from typing import List, Dict, Any
from langchain.schema import Document
from tqdm import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

BATCH_SIZE = 50
MAX_RETRIES = 3
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI text small 임베딩 모델

def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def load_jsonl_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    log(f"Loading JSONL file: {jsonl_path}")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                log(f"JSON decode error at line {line_num}: {e}")
                continue
    log(f"Loaded {len(chunks)} chunks")
    return chunks

def filter_chunks_by_jurisdiction(chunks: List[Dict[str, Any]], target_filter: str = None) -> List[Dict[str, Any]]:
    if not target_filter:
        log("No jurisdiction filtering applied")
        return chunks

    log(f"Filtering chunks by jurisdiction '{target_filter}'")
    filtered = [c for c in chunks if target_filter.lower() in c.get('jurisdiction', '').lower()]
    log(f"Filtered: {len(filtered)} out of {len(chunks)}")
    return filtered

def prepare_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    log("Converting chunks to Document objects")
    documents = []
    for chunk in chunks:
        metadata = {k: str(v) for k, v in chunk.items() if v is not None}
        documents.append(Document(page_content=chunk.get('text', ''), metadata=metadata))
    log(f"{len(documents)} Documents prepared")
    return documents

def embed_and_save_locally(documents: List[Document], embedding_model: str, batch_size: int, output_path: str):
    log(f"Embedding {len(documents)} documents using '{embedding_model}'")
    embedder = OpenAIEmbeddings(
        model=embedding_model,
        # 따로 encode_kwargs/model_kwargs 없음
    )

    vectors = []
    metadatas = []
    ids = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch_docs = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        meta_batch = [doc.metadata for doc in batch_docs]
        for attempt in range(MAX_RETRIES):
            try:
                batch_vecs = embedder.embed_documents(texts)
                for idx, vec in enumerate(batch_vecs):
                    vec_id = meta_batch[idx].get('id', f'doc_{i+idx}')
                    ids.append(vec_id)
                    vectors.append(vec)
                    metadatas.append(meta_batch[idx])
                break
            except Exception as e:
                log(f"Embedding batch failed attempt {attempt+1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

    save_data = [{'id': id_, 'vector': vec, 'metadata': meta} for id_, vec, meta in zip(ids, vectors, metadatas)]

    log(f"Saving embedded vectors locally to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    log("Local embedding save complete")

def main():
    parser = argparse.ArgumentParser(description="OpenAI Embeddings Local Save Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Input JSONL file path")
    parser.add_argument('--output', type=str, required=True, help="Output pickle file path")
    parser.add_argument('--filter', type=str, default=None, help="Jurisdiction filter (optional)")
    parser.add_argument('--model', type=str, default=EMBEDDING_MODEL, help="Embedding model")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")

    args = parser.parse_args()

    log("Start pipeline")

    chunks = load_jsonl_chunks(args.input)
    if not chunks:
        log("No chunks found to process")
        return

    filtered = filter_chunks_by_jurisdiction(chunks, args.filter)
    if not filtered:
        log("No filtered chunks to process")
        return

    documents = prepare_documents(filtered)
    embed_and_save_locally(documents, args.model, args.batch_size, args.output)

    log("Pipeline finished successfully")

if __name__ == '__main__':
    main()
