from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pymysql
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json


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
ALLOWED_NAMESPACES = ["sop", "gmp-1st", "gmp-2nd", "old-gmp-1st", "old-gmp-2nd"]

# DB 접속 정보 (Cloudtype 환경)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'mariadb'),  # Cloudtype 환경에서는 'mariadb'
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': 'utf8mb4'
}

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

# ---- Pydantic 모델 정의 ----
class AnalysisSummaryModel(BaseModel):
    총_gmp_변경점: int = Field(..., alias="total_gmp_changes")
    영향받는_sop_섹션: int = Field(..., alias="affected_sop_sections")
    분석완료시각: str = Field(..., alias="analyzed_at")

class SopInfoModel(BaseModel):
    sop_id: str
    sop_title: Optional[str] = None
    sop_content: Optional[str] = None
    match_score: Optional[float] = 0

class GmpChangeInfoModel(BaseModel):
    change_id: str
    topic: Optional[str] = None
    old_gmp_content: Optional[str] = ""
    new_gmp_content: Optional[str] = ""
    similarity_score: Optional[float] = 0

class DetailedAnalysisModel(BaseModel):
    sop_info: SopInfoModel
    gmp_change_info: Optional[GmpChangeInfoModel] = None
    change_rationale: Optional[Dict[str, Any]] = {}
    key_changes: Optional[List[Any]] = []
    update_recommendation: Optional[str] = ""

class SaveAllRequestModel(BaseModel):
    summary: AnalysisSummaryModel
    detailed_analysis: List[DetailedAnalysisModel]

# ---- DB 저장 함수 ----
def insert_analysis_summary(summary: AnalysisSummaryModel):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ANALYSIS_SUMMARY (
                    total_gmp_changes INT,
                    affected_sop_sections INT,
                    created_at VARCHAR(50)
                )
            ''')
            sql = """
                INSERT INTO ANALYSIS_SUMMARY (
                    total_gmp_changes,
                    affected_sop_sections,
                    created_at
                ) VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (
                summary.총_gmp_변경점,
                summary.영향받는_sop_섹션,
                summary.분석완료시각
            ))
        conn.commit()
    finally:
        conn.close()


def insert_sop_data(sop_list: List[DetailedAnalysisModel]):
    create_sql = '''
        CREATE TABLE IF NOT EXISTS SOP (
            sop_id VARCHAR(100) PRIMARY KEY,
            sop_title VARCHAR(255),
            sop_content TEXT,
            created_at DATETIME,
            updated_at DATETIME
        )
    '''
    insert_sql = """
        INSERT INTO SOP (
            sop_id,
            sop_title,
            sop_content,
            created_at,
            updated_at
        ) VALUES (%s, %s, %s, NOW(), NOW())
        ON DUPLICATE KEY UPDATE
            sop_title = VALUES(sop_title),
            sop_content = VALUES(sop_content),
            updated_at = NOW()
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)
            for item in sop_list:
                sop_info = item.sop_info
                if not sop_info.sop_id or not sop_info.sop_content:
                    continue
                cursor.execute(insert_sql, (
                    sop_info.sop_id,
                    sop_info.sop_title,
                    sop_info.sop_content
                ))
            conn.commit()
    finally:
        conn.close()


def insert_gmp_data(sop_list: List[DetailedAnalysisModel]):
    create_sql = '''
        CREATE TABLE IF NOT EXISTS GMP (
            gmp_id VARCHAR(100) PRIMARY KEY,
            topic VARCHAR(255),
            gmp_content TEXT,
            similarity_score FLOAT,
            created_at DATETIME
        )
    '''
    insert_sql = """
        INSERT INTO GMP (
            gmp_id,
            topic,
            gmp_content,
            similarity_score,
            created_at
        ) VALUES (%s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            topic = VALUES(topic),
            gmp_content = VALUES(gmp_content),
            similarity_score = VALUES(similarity_score)
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)
            for item in sop_list:
                gmp_info = item.gmp_change_info
                if not gmp_info or not gmp_info.change_id:
                    continue
                old_content = gmp_info.old_gmp_content or ""
                new_content = gmp_info.new_gmp_content or ""
                gmp_content = f"OLD:\n{old_content}\n\nNEW:\n{new_content}"
                cursor.execute(insert_sql, (
                    gmp_info.change_id,
                    gmp_info.topic,
                    gmp_content,
                    gmp_info.similarity_score
                ))
            conn.commit()
    finally:
        conn.close()


def insert_sop_gmp_link(sop_list: List[DetailedAnalysisModel]):
    create_sql = '''
        CREATE TABLE IF NOT EXISTS SOP_GMP_LINK (
            sop_id VARCHAR(100),
            gmp_id VARCHAR(100),
            match_score FLOAT,
            change_rationale TEXT,
            key_changes TEXT,
            update_recommendation TEXT,
            completed VARCHAR(20),
            created_at DATETIME,
            PRIMARY KEY (sop_id, gmp_id)
        )
    '''
    insert_sql = """
        INSERT INTO SOP_GMP_LINK (
            sop_id,
            gmp_id,
            match_score,
            change_rationale,
            key_changes,
            update_recommendation,
            completed,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            match_score = VALUES(match_score),
            change_rationale = VALUES(change_rationale),
            key_changes = VALUES(key_changes),
            update_recommendation = VALUES(update_recommendation),
            completed = VALUES(completed)
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)
            for item in sop_list:
                sop_info = item.sop_info
                gmp_info = item.gmp_change_info
                if not sop_info.sop_id or not gmp_info or not gmp_info.change_id:
                    continue
                rationale_json = json.dumps(item.change_rationale, ensure_ascii=False)
                key_changes_json = json.dumps(item.key_changes, ensure_ascii=False)
                completed_status = '미처리'
                cursor.execute(insert_sql, (
                    sop_info.sop_id,
                    gmp_info.change_id,
                    sop_info.match_score or 0,
                    rationale_json,
                    key_changes_json,
                    item.update_recommendation,
                    completed_status
                ))
            conn.commit()
    finally:
        conn.close()

# ---- FastAPI 엔드포인트 ----
@app.post("/save_all")
def save_all(request: SaveAllRequestModel):
    try:
        insert_analysis_summary(request.summary)
        insert_sop_data(request.detailed_analysis)
        insert_gmp_data(request.detailed_analysis)
        insert_sop_gmp_link(request.detailed_analysis)
        return {"result": "모든 데이터 저장 성공!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                SELECT sop_id, sop_title, sop_content, created_at, updated_at
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
