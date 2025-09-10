# -*- coding: utf-8 -*-
"""
SOP JSONL → 임베딩 → 벡터 DB 저장 파이프라인
JSONL 파일의 청킹된 SOP 문서들을 임베딩하고 Chroma 벡터 DB에 저장
"""
from pathlib import Path
import re
import hashlib
import argparse
import json
import os
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    from tqdm import tqdm
except ImportError:
    print("필요한 라이브러리를 설치하세요:")
    print("pip install langchain langchain-openai langchain-community faiss-cpu tqdm")
    exit(1)


# ==================== 설정 ====================
DB_PATH = "./sop_vector_db"
COLLECTION_NAME = "sop_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 10
MAX_RETRIES = 3

def log(msg: str) -> None:
    """로깅 함수"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ==================== 1. JSONL 데이터 로드 ====================
def load_jsonl_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일에서 청킹된 데이터 로드"""
    log(f"JSONL 파일 로드: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL 파일이 없습니다: {jsonl_path}")
    
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                log(f"JSON 파싱 오류 (라인 {line_num}): {e}")
                continue
    
    log(f"총 {len(chunks)}개 청크 로드 완료")
    return chunks


# ==================== 2. 임베딩 생성 ====================
def create_embeddings_batch(texts: List[str], model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    log(f"임베딩 생성 시작: {len(texts)}개 텍스트, 모델: {model_name}")
    
    embedder = OpenAIEmbeddings(model=model_name)
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="임베딩 생성"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # 재시도 로직
        for attempt in range(MAX_RETRIES):
            try:
                batch_embeddings = embedder.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                log(f"임베딩 실패 (시도 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(2 ** attempt)  # 지수 백오프
    
    log(f"임베딩 생성 완료: {len(all_embeddings)}개")
    return all_embeddings

def prepare_sop_documents_with_metadata(chunks: List[Dict[str, Any]]) -> List[Document]:
    documents = []
    for chunk in chunks:
        metadata = {
            "id": chunk.get("id"),
            "document_id": chunk.get("document_id"),  # SOP 문서번호
            "document_title": chunk.get("document_title"),
            "version": chunk.get("version"),
            "effective_date": chunk.get("effective_date"),
            "confidentiality": chunk.get("confidentiality"),
            "distribution_scope": chunk.get("distribution_scope"),
            "author": chunk.get("author"),
            "approval": chunk.get("approval"),
            "section_id": chunk.get("section_id"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "change_history": chunk.get("change_history"),
            "key_terms": chunk.get("key_terms"),
            "related_guidelines": chunk.get("related_guidelines"),
            "compliance_areas": chunk.get("compliance_areas"),
            "created_at": chunk.get("created_at"),
        }
        doc = Document(
            page_content=chunk.get("text", ""),
            metadata=metadata
        )
        documents.append(doc)
    return documents

# ==================== 3. 벡터 DB 저장 ====================
def create_faiss_collection(documents: List[Document], embedding_model: str, reset: bool = False) -> FAISS:
    """FAISS 벡터 DB 생성 및 저장 (배치 처리 적용)"""
    log(f"벡터 DB 생성: {DB_PATH}")
    
    # 임베딩 객체 생성
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    # 리셋 옵션 처리
    if reset and os.path.exists(DB_PATH):
        log("기존 벡터 DB 삭제")
        import shutil
        shutil.rmtree(DB_PATH)
    
    # 첫 번째 배치로 FAISS 초기화
    first_batch = documents[:BATCH_SIZE]
    log(f"첫 번째 배치 ({len(first_batch)}개)로 FAISS 초기화...")
    
    for attempt in range(MAX_RETRIES):
        try:
            vectordb = FAISS.from_documents(
                documents=first_batch,
                embedding=embeddings
            )
            break
        except Exception as e:
            log(f"초기화 실패 (시도 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                raise e
            time.sleep(2 ** attempt)
    
    # 나머지 배치들 추가
    remaining_docs = documents[BATCH_SIZE:]
    for i in tqdm(range(0, len(remaining_docs), BATCH_SIZE), desc="FAISS 배치 추가"):
        batch_docs = remaining_docs[i:i + BATCH_SIZE]
        
        for attempt in range(MAX_RETRIES):
            try:
                vectordb.add_documents(batch_docs)
                break
            except Exception as e:
                log(f"배치 추가 실패 (시도 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise e
                time.sleep(2 ** attempt)
    
    # 로컬 저장
    vectordb.save_local(DB_PATH)
    
    log(f"FAISS 벡터 DB 저장 완료: {len(documents)}개 문서")
    return vectordb


# ==================== 5. 메인 실행 ====================
def main():
    global DB_PATH, BATCH_SIZE
    """전체 파이프라인 실행"""
    parser = argparse.ArgumentParser(
        description="SOP JSONL 파일을 임베딩하여 벡터 DB로 저장"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="입력 JSONL 파일 경로"
    )
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="기존 벡터 DB 초기화"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=EMBEDDING_MODEL, 
        help="임베딩 모델명 (기본: text-embedding-3-small)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=DB_PATH, 
        help="벡터 DB 저장 디렉토리"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=BATCH_SIZE, 
        help="임베딩 배치 크기"
    )
    # parser.add_argument(
    #     "--skip-validation", 
    #     action="store_true", 
    #     help="품질 검증 건너뛰기"
    # )
    
    args = parser.parse_args()
    
    DB_PATH = args.output_dir
    BATCH_SIZE = args.batch_size
    
    try:
        log("SOP 벡터 DB 생성 파이프라인 시작")
        
        # 1단계: JSONL 로드
        chunks = load_jsonl_chunks(args.input)
        if not chunks:
            log("처리할 데이터가 없습니다.")
            return
        
        # 2단계: Document 형식 변환
        documents = prepare_sop_documents_with_metadata(chunks)
        
        # 3단계: 벡터 DB 생성
        vectordb = create_faiss_collection(
            documents=documents, 
            embedding_model=args.model,
            reset=args.reset
        )
        
        # # 4단계: 품질 검증 (옵션)
        # if not args.skip_validation:
        #     validation_results = validate_vector_store(vectordb)
        
        # log("모든 작업이 완료되었습니다!")
        # log(f"벡터 DB 위치: {DB_PATH}")
        # log(f"저장된 문서 수: {len(documents)}")
        
    except Exception as e:
        log(f"오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()