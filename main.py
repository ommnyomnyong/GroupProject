from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import pymysql
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수 및 상수
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "gmp-sop-vectordb"
VECTOR_DIM = 1536
ALLOWED_NAMESPACES = ["sop", "gmp-1st", "old-gmp-1st", "gmp-2nd"]

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3307,
    'user': os.getenv('DB_USER', 'ommnyomnyong'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'database': os.getenv('DB_NAME', 'gmp_sop'),
    'charset': 'utf8mb4'
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

pinecone_index = None

@app.on_event("startup")
def startup_event():
    global pinecone_index
    try:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY 환경변수가 필요합니다.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Pinecone 초기화 실패: {e}")
        pinecone_index = None

def test_connection():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        conn.close()
        if result:
            return {"status": "연결 성공"}
        else:
            raise Exception("쿼리 결과 없음")
    except Exception as e:
        return {"status": "연결 실패", "error": str(e)}


@app.get("/")
def read_root():
    return {
        "message": "GMP-SOP API 서버 정상 실행 중",
        "version": "1.0.0",
        "status": "running"
    }

# Pinecone에서 SOP 전문 데이터 페이징 조회
@app.get("/get_sop_text")
def get_sop_text(
    namespace: str = Query("sop"),
    top_k: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    if pinecone_index is None:
        raise HTTPException(status_code=500, detail="Pinecone 인덱스가 초기화되어 있지 않습니다.")
    if namespace not in ALLOWED_NAMESPACES:
        raise HTTPException(status_code=400, detail=f"지원되는 네임스페이스는 {ALLOWED_NAMESPACES} 입니다.")

    try:
        dummy_vector = [0.0] * VECTOR_DIM
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=top_k + offset,
            namespace=namespace,
            include_metadata=True
        )
        matches = results.get("matches", [])[offset:]
        data = [match.get("metadata", {}) for match in matches]
        return {
            "namespace": namespace,
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone 쿼리 오류: {e}")

# 수정해야 할 SOP 목록 조회 (Mariadb)
@app.get("/get_sop_to_update")
def get_sop_to_update():
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT sop_id, sop_pinecone_id, sop_title, sop_content, created_at, updated_at
                FROM SOP
                ORDER BY updated_at DESC
            """
            cursor.execute(sql)
            sop_data = cursor.fetchall()
        return {"sop_data": sop_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {e}")
    finally:
        conn.close()

# 변경된 GMP 조회 (Mariadb)
@app.get("/get_changed_gmp")
def get_changed_gmp():
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT gmp_id, topic, gmp_content, similarity_score, created_at
                FROM GMP
                ORDER BY created_at DESC
            """
            cursor.execute(sql)
            gmp_data = cursor.fetchall()
        return {"gmp_data": gmp_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {e}")
    finally:
        conn.close()

# 변경 근거 조회
@app.get("/get_sop_gmp_link")
def get_sop_gmp_link():
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT sop_id, gmp_id, match_score, change_rationale, key_changes, update_recommendation, completed, created_at
                FROM SOP_GMP_LINK
                ORDER BY created_at DESC
            """
            cursor.execute(sql)
            link_data = cursor.fetchall()
        return {"link_data": link_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=8000,
        reload=True, log_level="info"
    )
