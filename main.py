from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

app = FastAPI(
    title="GMP-SOP API",
    description="SOP 문서와 GMP 요약을 연결하는 API 서버",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SOPRequest(BaseModel):
    sop_id: str

class GMPResponse(BaseModel):
    gmp: List[str]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gmp-sop-vectordb"
ALLOWED_NAMESPACES = ["sop", "old-gmp-2nd", "gmp-1st", "old-gmp-1st", "gmp-2nd"]
VECTOR_DIM = 1536

def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    return index

pinecone_index = None

@app.on_event("startup")
def startup_event():
    global pinecone_index
    pinecone_index = get_pinecone_index()

@app.get("/")
def read_root():
    return {
        "message": "GMP-SOP API 서버가 정상적으로 실행 중입니다!",
        "version": "1.0.0",
        "status": "running"
    }

# 네임스페이스별 페이지네이션 적용 데이터 조회 API
@app.get("/get_data")
def get_data(
    namespace: str,
    top_k: int = Query(10, ge=1, le=100),  # 한번에 가져올 최대 개수 (1~100 제한)
    offset: Optional[int] = Query(0, ge=0)  # 페이징을 위한 오프셋, 기본 0
):
    if namespace not in ALLOWED_NAMESPACES:
        raise HTTPException(status_code=400, detail=f"지원되는 네임스페이스는 {ALLOWED_NAMESPACES} 입니다.")

    # Pinecone query는 offset이 없으므로, limit + offset을 구현하려면
    # 실제로는 REST API 또는 반복 쿼리 필요함. (아래는 임시 구현 - top_k개만 반환)
    dummy_vector = [0.0] * VECTOR_DIM
    results = pinecone_index.query(
        vector=dummy_vector,
        top_k=top_k + (offset or 0),
        namespace=namespace,
        include_metadata=True
    )
    # offset 적용
    matches = results.get("matches", [])[offset:]
    data = [match.get("metadata", {}) for match in matches]

    return {
        "namespace": namespace,
        "count": len(data),
        "data": data
    }

@app.get("/test_connection")
def test_connection():
    try:
        stats = pinecone_index.describe_index_stats()
        return {
            "status": "연결 성공",
            "message": "Pinecone 인덱스 연결 정상",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "연결 실패",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )
