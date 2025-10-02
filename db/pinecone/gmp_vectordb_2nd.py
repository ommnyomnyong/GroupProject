# -*- coding: utf-8 -*-
"""
재청킹된 GMP JSONL → 임베딩 → 로컬 파일 저장 파이프라인

python gmp_vectordb_2nd.py --input "./chunks/fda_2nd_semantic_chunks.jsonl" --output "gmp_embeddings_2nd.pkl" --batch-size 10
"""

import argparse
import json
import os
import time
import pickle
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.schema import Document
    from tqdm import tqdm
except ImportError:
    print("필요한 라이브러리를 설치하세요:")
    print("pip install langchain openai tqdm")
    exit(1)


BATCH_SIZE = 50
MAX_RETRIES = 3
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI 텍스트 임베딩 소형 모델


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_rechunked_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    log(f"Loading rechunked JSONL file: {jsonl_path}")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL 파일이 없습니다: {jsonl_path}")

    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                log(f"JSON parse error at line {line_num}: {e}")
    log(f"Loaded {len(chunks)} chunks")
    return chunks


def filter_rechunked_chunks(chunks: List[Dict[str, Any]],
                            jurisdiction_filter: Optional[str] = None,
                            chunk_type_filter: Optional[str] = None,
                            doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    filtered = chunks

    if jurisdiction_filter:
        log(f"Filtering by jurisdiction: {jurisdiction_filter}")
        filtered = [
            c for c in filtered
            if c.get("jurisdiction") and jurisdiction_filter.lower() in c.get("jurisdiction").lower()
        ]
    if chunk_type_filter and chunk_type_filter != "auto":
        log(f"Filtering by chunk_type: {chunk_type_filter}")
        if chunk_type_filter == "large":
            filtered = [
                c for c in filtered
                if not c.get("parent_chunk_id") or c.get("semantic_level") == 1
            ]
        elif chunk_type_filter == "small":
            filtered = [
                c for c in filtered
                if c.get("parent_chunk_id") and (c.get("semantic_level", 0) >= 2 or c.get("sub_chunk_index") is not None)
            ]
        elif chunk_type_filter == "original":
            filtered = [c for c in filtered if not c.get("parent_chunk_id")]
    if doc_type_filter:
        log(f"Filtering by doc_type: {doc_type_filter}")
        filtered = [c for c in filtered if c.get("doc_type", "GMP").upper() == doc_type_filter.upper()]

    log(f"Filtered chunks: {len(filtered)} / {len(chunks)}")
    if len(filtered) == 0:
        log(f"⚠ No chunks after filtering.")
    return filtered


def prepare_rechunked_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    log("Converting to Documents")
    documents = []
    for chunk in chunks:
        metadata = {}
        known_keys = [
            "id", "doc_id", "source_path", "title", "jurisdiction",
            "doc_date", "doc_version", "section_id", "section_title",
            "normative_strength", "page_start", "page_end", "chunk_index",
            "text", "parent_chunk_id", "sub_chunk_index", "semantic_level",
            "doc_type", "sop_number", "procedure_step"
        ]
        for k in known_keys:
            v = chunk.get(k)
            if v is not None:
                metadata[k] = str(v)
        if chunk.get('parent_chunk_id'):
            if chunk.get('semantic_level'):
                metadata['chunk_hierarchy'] = f"semantic_level_{chunk['semantic_level']}"
            elif chunk.get('sub_chunk_index') is not None:
                metadata['chunk_hierarchy'] = f"recursive_sub_{chunk['sub_chunk_index']}"
            else:
                metadata['chunk_hierarchy'] = "rechunked_unknown"
        else:
            metadata['chunk_hierarchy'] = "original"

        documents.append(Document(page_content=chunk.get("text", ""), metadata=metadata))
    log(f"{len(documents)} Documents created")
    return documents


def embed_and_save_locally(documents: List[Document], embedding_model: str, batch_size: int, output_path: str):
    log(f"Embedding {len(documents)} documents using {embedding_model}")
    embedder = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
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

    log(f"Saving to local pickle file: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    log("Saved embedded vectors locally")


def main():
    parser = argparse.ArgumentParser(description="Embed rechunked GMP/SOP JSONL and save locally using OpenAI")
    parser.add_argument("--input", type=str, required=True, help="Input rechunked JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument("--filter", type=str, help="Jurisdiction filter (optional)")
    parser.add_argument("--chunk-type", choices=["large", "small", "original", "auto"], default="auto", help="Chunk type filter")
    parser.add_argument("--doc-type", choices=["GMP", "SOP"], help="Document type filter")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL, help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for embedding")
    args = parser.parse_args()

    log("Rechunk embed pipeline started")
    chunks = load_rechunked_jsonl(args.input)
    filtered_chunks = filter_rechunked_chunks(chunks, args.filter, args.chunk_type, args.doc_type)

    if not filtered_chunks:
        log("No chunks after filtering, exiting")
        return

    documents = prepare_rechunked_documents(filtered_chunks)
    embed_and_save_locally(documents, args.model, args.batch_size, args.output)
    log("Pipeline completed successfully")

if __name__ == "__main__":
    main()