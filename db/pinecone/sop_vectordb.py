# -*- coding: utf-8 -*-
"""
SOP 청크 리스트 → OpenAI 임베딩 → 로컬 저장 파이프라인
"""
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
EMBEDDING_MODEL = "text-embedding-3-small"


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def filter_sop_chunks(chunks: List[Dict[str, Any]], jurisdiction_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    if not jurisdiction_filter:
        log("No jurisdiction filtering applied")
        return chunks
    log(f"Filtering SOP chunks by jurisdiction: {jurisdiction_filter}")
    filtered = [c for c in chunks if jurisdiction_filter.lower() in c.get('jurisdiction', '').lower()]
    log(f"Filtered {len(filtered)} chunks out of {len(chunks)}")
    if len(filtered) == 0:
        log("⚠ No SOP chunks found after filtering")
    return filtered


def prepare_sop_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    log("Converting SOP chunks to Documents")
    documents = []
    for chunk in chunks:
        metadata = {k: str(v) for k, v in chunk.items() if v is not None}
        documents.append(Document(page_content=chunk.get('text', ''), metadata=metadata))
    log(f"{len(documents)} Documents prepared")
    return documents


def embed_documents_in_memory(documents: List[Document], embedding_model: str, batch_size: int):
    log(f"Embedding {len(documents)} documents with model {embedding_model}")
    embedder = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectors = []
    metadatas = []
    ids = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding SOP batches"):
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

    # 결과를 리스트로 반환
    return [{'id': id_, 'vector': vec, 'metadata': meta} for id_, vec, meta in zip(ids, vectors, metadatas)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embed SOP chunks to OpenAI embeddings and save locally")
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL, help="OpenAI embedding model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size")
    args = parser.parse_args()

    # 예시: 외부에서 chunks를 받아오는 부분을 main.py에서 처리해야 함
    # chunks = ...
    # filtered = filter_sop_chunks(chunks, jurisdiction_filter=None)
    # documents = prepare_sop_documents(filtered)
    # embed_and_save_locally(documents, args.model, args.batch_size, args.output)
    print("이제는 main.py 등에서 청킹 결과를 받아서 위 함수들을 직접 호출하세요.")

if __name__ == "__main__":
    main()


# # -*- coding: utf-8 -*-
# """
# SOP 청크 JSONL → OpenAI 임베딩 → 로컬 저장 파이프라인
# python sop_vectordb.py --input "../chunking/chunks/sop_regex_chunks.jsonl" --output "sop_embeddings.pkl"
# """

# import argparse
# import json
# import os
# import time
# import pickle
# from typing import List, Dict, Any, Optional
# from dotenv import load_dotenv
# load_dotenv()

# try:
#     from langchain.embeddings.openai import OpenAIEmbeddings
#     from langchain.schema import Document
#     from tqdm import tqdm
# except ImportError:
#     print("필요한 라이브러리를 설치하세요:")
#     print("pip install langchain openai tqdm")
#     exit(1)

# BATCH_SIZE = 50
# MAX_RETRIES = 3
# EMBEDDING_MODEL = "text-embedding-3-small"

# def log(msg: str) -> None:
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {msg}", flush=True)

# def load_sop_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
#     log(f"Loading SOP JSONL file: {jsonl_path}")
#     if not os.path.exists(jsonl_path):
#         raise FileNotFoundError(f"File not found: {jsonl_path}")

#     chunks = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             try:
#                 chunk = json.loads(line.strip())
#                 chunks.append(chunk)
#             except json.JSONDecodeError as e:
#                 log(f"JSON decode error at line {line_num}: {e}")
#                 continue
#     log(f"Loaded {len(chunks)} SOP chunks")
#     return chunks

# def filter_sop_chunks(chunks: List[Dict[str, Any]], jurisdiction_filter: Optional[str] = None) -> List[Dict[str, Any]]:
#     if not jurisdiction_filter:
#         log("No jurisdiction filtering applied")
#         return chunks

#     log(f"Filtering SOP chunks by jurisdiction: {jurisdiction_filter}")
#     filtered = [c for c in chunks if jurisdiction_filter.lower() in c.get('jurisdiction', '').lower()]
#     log(f"Filtered {len(filtered)} chunks out of {len(chunks)}")
#     if len(filtered) == 0:
#         log("⚠ No SOP chunks found after filtering")
#     return filtered

# def prepare_sop_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
#     log("Converting SOP chunks to Documents")
#     documents = []
#     for chunk in chunks:
#         metadata = {k: str(v) for k, v in chunk.items() if v is not None}
#         documents.append(Document(page_content=chunk.get('text', ''), metadata=metadata))
#     log(f"{len(documents)} Documents prepared")
#     return documents

# def embed_and_save_locally(documents: List[Document], embedding_model: str, batch_size: int, output_path: str):
#     log(f"Embedding {len(documents)} documents with model {embedding_model}")
#     embedder = OpenAIEmbeddings(
#         model=embedding_model,
#         openai_api_key=os.getenv("OPENAI_API_KEY")
#     )

#     vectors = []
#     metadatas = []
#     ids = []

#     for i in tqdm(range(0, len(documents), batch_size), desc="Embedding SOP batches"):
#         batch_docs = documents[i:i + batch_size]
#         texts = [doc.page_content for doc in batch_docs]
#         meta_batch = [doc.metadata for doc in batch_docs]

#         for attempt in range(MAX_RETRIES):
#             try:
#                 batch_vecs = embedder.embed_documents(texts)
#                 for idx, vec in enumerate(batch_vecs):
#                     vec_id = meta_batch[idx].get('id', f'doc_{i+idx}')
#                     ids.append(vec_id)
#                     vectors.append(vec)
#                     metadatas.append(meta_batch[idx])
#                 break
#             except Exception as e:
#                 log(f"Embedding batch failed attempt {attempt+1}: {e}")
#                 if attempt == MAX_RETRIES - 1:
#                     raise
#                 time.sleep(2 ** attempt)

#     save_data = [{'id': id_, 'vector': vec, 'metadata': meta} for id_, vec, meta in zip(ids, vectors, metadatas)]

#     log(f"Saving to local pickle file: {output_path}")
#     with open(output_path, 'wb') as f:
#         pickle.dump(save_data, f)

#     log("SOP embedding saved locally")

# def main():
#     parser = argparse.ArgumentParser(description="Embed SOP JSONL chunks to OpenAI embeddings and save locally")
#     parser.add_argument("--input", type=str, required=True, help="Input SOP JSONL file path")
#     parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
#     parser.add_argument("--filter", type=str, help="Jurisdiction filter (optional)")
#     parser.add_argument("--model", type=str, default=EMBEDDING_MODEL, help="OpenAI embedding model")
#     parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size")
#     args = parser.parse_args()

#     log("Starting SOP embed pipeline")
#     chunks = load_sop_jsonl(args.input)
#     filtered = filter_sop_chunks(chunks, args.filter)

#     if not filtered:
#         log("No SOP chunks after filtering, exiting")
#         return

#     documents = prepare_sop_documents(filtered)
#     embed_and_save_locally(documents, args.model, args.batch_size, args.output)
#     log("SOP embed pipeline completed successfully")

# if __name__ == "__main__":
#     main()
