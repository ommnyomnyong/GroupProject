from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="GMP-SOP API",
    description="SOP 문서와 GMP 요약을 연결하는 API 서버",
    version="1.0.0"
)

# CORS 설정 - React 앱에서 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # React 개발 서버 포트가 다를 수 있음
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class SOPRequest(BaseModel):
    sop_id: str

class GMPResponse(BaseModel):
    gmp: List[str]

# Pinecone 연결 및 인덱스 준비
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gmp-sop-vectordb"
NAMESPACE_LIST = ["sop", "gmp-1st", "gmp-2nd", "old-gmp-1st", "old-gmp-2nd"]
VECTOR_DIM = 1536

# Pinecone 인덱스 연결 함수
def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    return index

# 서버 시작 시 Pinecone 연결 및 데이터 캐싱
data_cache = {}

@app.on_event("startup")
def startup_event():
    global pinecone_index, data_cache
    pinecone_index = get_pinecone_index()
    data_cache = {}
    for ns in NAMESPACE_LIST:
        # 임의의 벡터로 최대 top_k로 쿼리 (실제 데이터가 많으면 REST API로 id 목록을 받아서 fetch해야 함)
        dummy_vector = [0.0] * VECTOR_DIM
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=10000,  # 최대 10,000개까지
            namespace=ns,
            include_metadata=True
        )
        data_cache[ns] = [match.get("metadata", {}) for match in results.get("matches", [])]

@app.get("/")
def read_root():
    return {
        "message": "GMP-SOP API 서버가 정상적으로 실행 중입니다!",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/get_gmp/", response_model=GMPResponse)
def get_gmp(request: SOPRequest):
    sop_id = request.sop_id
    if not sop_id:
        raise HTTPException(status_code=400, detail="sop_id는 필수 입력값입니다.")
    gmp_list = []
    for ns in NAMESPACE_LIST:
        for meta in data_cache.get(ns, []):
            if meta.get("id") == sop_id:
                text = meta.get("text", "")
                if text:
                    gmp_list.append(text)
    if not gmp_list:
        raise HTTPException(status_code=404, detail=f"SOP ID '{sop_id}'에 대한 GMP 데이터를 찾을 수 없습니다.")
    return GMPResponse(gmp=gmp_list)

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

@app.get("/sop_list")
def get_sop_list():
    available_ids = set()
    for ns in NAMESPACE_LIST:
        for meta in data_cache.get(ns, []):
            if "id" in meta:
                available_ids.add(meta["id"])
    return {
        "available_sop_ids": list(available_ids),
        "total_count": len(available_ids)
    }

@app.get("/get_data/{namespace}")
def get_data(namespace: str):
    if namespace not in data_cache:
        raise HTTPException(status_code=404, detail="해당 namespace 데이터가 없습니다.")
    return {"data": data_cache[namespace]}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )