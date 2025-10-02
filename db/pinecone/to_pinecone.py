# -*- coding: utf-8 -*-
"""
메모리 내 임베딩 결과와 메타데이터를 Pinecone에 업서트하는 도구
"""
import os
import time
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

INDEX_DIMENSION = 1536
BATCH_SIZE = 50
MAX_RETRIES = 3

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def initialize_pinecone(index_name: str, dimension: int = INDEX_DIMENSION, reset: bool = False):
    log(f"Initializing Pinecone index: {index_name}")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = pc.list_indexes().names()

    if index_name in existing_indexes:
        idx_info = pc.describe_index(index_name)
        if idx_info.dimension != dimension or reset:
            log(f"Deleting existing index '{index_name}' due to dimension mismatch or reset")
            pc.delete_index(index_name)
            time.sleep(10)
            existing_indexes.remove(index_name)

    if index_name not in existing_indexes:
        log(f"Creating new index '{index_name}' with dimension {dimension}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(15)

    index = pc.Index(index_name)
    log(f"Pinecone index '{index_name}' ready")
    return index

def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
    norm_meta = {}
    for k, v in meta.items():
        if v is None:
            norm_meta[k] = ""
        else:
            norm_meta[k] = str(v)
    return norm_meta

def prepare_vectors(embeddings: List[Any], metadata_list: List[Dict[str, Any]], namespace_prefix: str):
    vectors = []
    for i, (embedding, meta) in enumerate(zip(embeddings, metadata_list)):
        vector_id = f"{namespace_prefix}_{meta.get('id', f'auto_{i:08d}') }"
        norm_meta = normalize_metadata(meta)
        vector_values = embedding.get('vector') if isinstance(embedding, dict) else embedding
        vectors.append({
            'id': vector_id,
            'values': vector_values,
            'metadata': norm_meta,
        })
    return vectors

def upload_vectors(index, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = BATCH_SIZE):
    total = len(vectors)
    log(f"Uploading {total} vectors to namespace '{namespace}' in batches of {batch_size}")
    for i in tqdm(range(0, total, batch_size), desc=f"Uploading {namespace}"):
        batch = vectors[i:i + batch_size]
        for attempt in range(MAX_RETRIES):
            try:
                index.upsert(vectors=batch, namespace=namespace)
                break
            except Exception as e:
                log(f"Upload failed on batch {i}-{i + len(batch)} attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
    log(f"Completed uploading namespace '{namespace}'")

# 메모리 내 embedding/metadata를 바로 Pinecone에 업서트하는 함수

def upsert_embeddings_to_pinecone(
    index_name: str,
    embeddings: List[Any],
    metadata: List[Dict[str, Any]],
    namespace: str,
    dimension: int = INDEX_DIMENSION,
    reset: bool = False,
    batch_size: int = BATCH_SIZE
):
    index = initialize_pinecone(index_name, dimension=dimension, reset=reset)
    if len(embeddings) != len(metadata):
        log(f"Error: Embedding count ({len(embeddings)}) doesn't match metadata count ({len(metadata)})")
        return
    namespace_prefix = namespace.replace(" ", "_").lower()
    vectors = prepare_vectors(embeddings, metadata, namespace_prefix)
    upload_vectors(index, vectors, namespace=namespace_prefix, batch_size=batch_size)
    log("Pinecone 업서트 완료")

# main.py에서 아래처럼 사용하면 됩니다:
# upsert_embeddings_to_pinecone(
#     index_name="gmp-sop-vectordb",
#     embeddings=embedding_results,  # 임베딩 결과 리스트
#     metadata=metadata_list,        # 메타데이터 리스트
#     namespace="sop-2nd"
# )


# # -*- coding: utf-8 -*-
# """
# 로컬 임베딩.pkl와 metadata.json, ids.json을 Pinecone에 업로드하는 도구.

# 동일 인덱스 내 여러 namespace에 구분하여 업로드 가능.
# 데이터마다 메타데이터 차이 고려하여 적절 필터링 및 정제 수행.

# 실행 예:
# python to_pinecone.py --index-name gmp-sop-vectordb --embeddings-pkl .\embeddings\gmp_embeddings.pkl --metadata-jsonl .\chunks\fda_semantic_chunks.jsonl --namespace gmp-1st --batch-size 50

# python to_pinecone.py --index-name gmp-sop-vectordb --embeddings-pkl .\embeddings\gmp_embeddings_2nd.pkl --metadata-jsonl .\chunks\fda_2nd_semantic_chunks.jsonl --namespace gmp-2nd --batch-size 50

# python to_pinecone.py --index-name gmp-sop-vectordb --embeddings-pkl .\embeddings\sop_embeddings.pkl --metadata-jsonl ..\chunking\chunks\sop_regex_chunks.jsonl --namespace sop --batch-size 50

# python to_pinecone.py --index-name gmp-sop-vectordb --embeddings-pkl .\embeddings\old_gmp_embeddings.pkl --metadata-jsonl .\chunks\old_fda_semantic_chunks.jsonl --namespace old-gmp-1st --batch-size 50

# python to_pinecone.py --index-name gmp-sop-vectordb --embeddings-pkl .\embeddings\old_gmp_embeddings_2nd.pkl --metadata-jsonl .\chunks\old_fda_2nd_semantic_chunks.jsonl --namespace old-gmp-2nd --batch-size 50


# """

# """
# 파인콘 업서트 데이터 정보

# - 임베딩 벡터 (Embeddings)
#   - 문서 내 텍스트 조각(chunk)을 벡터화한 실수 배열
#   - 벡터 차원은 1536 (사용하는 임베딩 모델에 따라 달라질 수 있음)
#   - 벡터는 문서 의미 검색의 핵심 데이터

# - 메타데이터 (Metadata)
#   - 원본 텍스트 조각과 관련된 부가정보
#   - 주요 필드
#     - id: 고유 식별자, 네임스페이스 접두어로 구분 포함
#     - doc_id: 문서 고유 ID
#     - title: 문서 제목
#     - page_start, page_end: 해당 텍스트 조각이 포함된 문서 내 페이지 범위
#     - chunk_index: 문서 내 조각 순서 인덱스
#     - text: 원문 텍스트 조각 전체 (검색 결과에 보여줄 컨텍스트)
#   - 메타데이터 값들은 모두 문자열로 저장
#   - 텍스트는 최대한 원문 그대로 포함하지만, 너무 크면 추후 원본 텍스트는 따로 DB로 관리를 고려해야 함.

# - 네임스페이스 (Namespace)
#   - 같은 인덱스 내에서 GMP 1차, 2차, SOP로 데이터 그룹 구분
#   - 구분: "gmp-1st", "gmp-2nd", "sop-1st", "sop-2nd", "gmp-old-1st", "gmp-old-2nd"

# - 데이터 관리 및 활용
#   - 메타데이터의 id 또는 doc_id로 검색 결과 원문 참조 및 위치 하이라이팅 가능
#   - 벡터 유사도 검색 시 반환되는 메타데이터 활용해 사용자에게 문서 관련 정보를 제공
#   - 대규모 데이터에도 효율적인 품질과 속도 유지

# """

# import os
# import json
# import pickle
# import time
# from pathlib import Path
# from typing import List, Dict, Any
# from pinecone import Pinecone, ServerlessSpec
# from tqdm import tqdm
# from dotenv import load_dotenv
# load_dotenv()

# INDEX_DIMENSION = 1536
# BATCH_SIZE = 50
# MAX_RETRIES = 3

# def log(msg: str):
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# def initialize_pinecone(index_name: str, dimension: int = INDEX_DIMENSION, reset: bool = False):
#     log(f"Initializing Pinecone index: {index_name}")
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     existing_indexes = pc.list_indexes().names()

#     if index_name in existing_indexes:
#         idx_info = pc.describe_index(index_name)
#         if idx_info.dimension != dimension or reset:
#             log(f"Deleting existing index '{index_name}' due to dimension mismatch or reset")
#             pc.delete_index(index_name)
#             time.sleep(10)
#             existing_indexes.remove(index_name)

#     if index_name not in existing_indexes:
#         log(f"Creating new index '{index_name}' with dimension {dimension}")
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#         time.sleep(15)

#     index = pc.Index(index_name)
#     log(f"Pinecone index '{index_name}' ready")
#     return index

# def load_embeddings(pkl_path: Path) -> List[Any]:
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
#     log(f"Loaded {len(data)} embedding vectors from {pkl_path}")
#     return data

# def load_metadata_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#     log(f"Loaded {len(data)} metadata records from {jsonl_path}")
#     return data

# def normalize_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
#     # 모든 메타데이터 값을 문자열로 변환, None은 빈 문자열로
#     norm_meta = {}
#     for k, v in meta.items():
#         if v is None:
#             norm_meta[k] = ""
#         else:
#             norm_meta[k] = str(v)
#     return norm_meta

# def prepare_vectors(embeddings: List[Any], metadata_list: List[Dict[str, Any]], namespace_prefix: str):
#     vectors = []
#     for i, (embedding, meta) in enumerate(zip(embeddings, metadata_list)):
#         vector_id = f"{namespace_prefix}_{meta.get('id', f'auto_{i:08d}')}"
#         norm_meta = normalize_metadata(meta)
#         vector_values = embedding.get('vector') if isinstance(embedding, dict) else embedding
#         vectors.append({
#             'id': vector_id,
#             'values': vector_values,
#             'metadata': norm_meta,
#         })
#     return vectors

# def upload_vectors(index, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = BATCH_SIZE):
#     total = len(vectors)
#     log(f"Uploading {total} vectors to namespace '{namespace}' in batches of {batch_size}")
#     for i in tqdm(range(0, total, batch_size), desc=f"Uploading {namespace}"):
#         batch = vectors[i:i + batch_size]
#         for attempt in range(MAX_RETRIES):
#             try:
#                 index.upsert(vectors=batch, namespace=namespace)
#                 break
#             except Exception as e:
#                 log(f"Upload failed on batch {i}-{i + len(batch)} attempt {attempt + 1}: {e}")
#                 if attempt == MAX_RETRIES - 1:
#                     raise
#                 time.sleep(2 ** attempt)
#     log(f"Completed uploading namespace '{namespace}'")

# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Upload embeddings and full metadata (including full text) to Pinecone")
#     parser.add_argument("--index-name", required=True, help="Pinecone index name")
#     parser.add_argument("--embeddings-pkl", required=True, type=Path, help="Path to embeddings pickle file")
#     parser.add_argument("--metadata-jsonl", required=True, type=Path, help="Path to metadata JSONL file with full text")
#     parser.add_argument("--namespace", required=True, help="Namespace for Pinecone")
#     parser.add_argument("--reset", action="store_true", help="Reset Pinecone index before upload")
#     parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
#     args = parser.parse_args()

#     index = initialize_pinecone(args.index_name, reset=args.reset)
#     embeddings = load_embeddings(args.embeddings_pkl)
#     metadata = load_metadata_jsonl(args.metadata_jsonl)

#     if len(embeddings) != len(metadata):
#         log(f"Error: Embedding count ({len(embeddings)}) doesn't match metadata count ({len(metadata)})")
#         return

#     namespace_prefix = args.namespace.replace(" ", "_").lower()
#     vectors = prepare_vectors(embeddings, metadata, namespace_prefix)

#     upload_vectors(index, vectors, namespace=namespace_prefix, batch_size=args.batch_size)

# if __name__ == "__main__":
#     main()
