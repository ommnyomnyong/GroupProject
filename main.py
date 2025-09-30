from fastapi import FastAPI, HTTPException, Query, File, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pymysql
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import requests
from datetime import datetime
from fastapi.responses import FileResponse
from docx import Document
import tempfile

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
    'host': os.getenv('DB_HOST', 'mariadb'),
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
# ---- 파일 업로드 엔드포인트 ----
@app.post("/upload_json")
async def upload_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents.decode('utf-8'))
        summary = data.get('summary', {})
        pipeline_info = data.get('pipeline_info', {})

        # pipeline_info에서 analyzed_at을 summary로 복사
        if '분석완료시각' in pipeline_info:
            summary['analyzed_at'] = pipeline_info['분석완료시각']

        # 한글 키가 있으면 영문 키로 복사
        if '총_gmp_변경점' in summary:
            summary['total_gmp_changes'] = summary['총_gmp_변경점']
        if '영향받는_sop_섹션' in summary:
            summary['affected_sop_sections'] = summary['영향받는_sop_섹션']

        # 필수 키가 없으면 에러 메시지 반환
        required_keys = ['total_gmp_changes', 'affected_sop_sections', 'analyzed_at']
        for key in required_keys:
            if key not in summary:
                raise HTTPException(status_code=400, detail=f"'{key}' 필드가 summary에 없습니다.")

        summary_model = AnalysisSummaryModel(**summary)
        detailed = [DetailedAnalysisModel(**item) for item in data['detailed_analysis']]
        insert_analysis_summary(summary_model)
        insert_sop_data(detailed)
        insert_gmp_data(detailed)
        insert_sop_gmp_link(detailed)
        return {"result": "파일 업로드 및 데이터 저장 성공!"}
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

@app.patch("/sop/{sop_id}")
def update_sop(sop_id: str, update: SopUpdateModel):
    update_data = update.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="수정할 내용이 없습니다.")
    update_data['updated_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_clause = ", ".join([f"{k}=%s" for k in update_data.keys()])
    values = list(update_data.values())
    values.append(sop_id)
    sql = f"UPDATE SOP SET {set_clause} WHERE sop_id=%s"
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)
        conn.commit()
    finally:
        conn.close()
    return {"result": "SOP 수정 성공"}

@app.get("/export_sop_docx")
def export_sop_docx():
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT sop_title, sop_content, updated_at
                FROM SOP
                ORDER BY updated_at DESC
            """)
            sop_data = cursor.fetchall()
    finally:
        conn.close()

    # 워드 문서 생성
    doc = Document()
    doc.add_heading("SOP 전문", 0)
    for sop in sop_data:
        doc.add_heading(sop['sop_title'] or "(제목 없음)", level=1)
        doc.add_paragraph(sop['sop_content'] or "(내용 없음)")
        doc.add_paragraph(f"수정일: {sop['updated_at']}")
        doc.add_paragraph("")

    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name

    # 파일 응답 후 임시 파일 삭제
    filename = f"SOP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    response = FileResponse(tmp_path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    # FastAPI의 background task로 파일 삭제
    from fastapi import BackgroundTasks
    def cleanup():
        os.remove(tmp_path)
    response.background = BackgroundTasks()
    response.background.add_task(cleanup)
    return response

def export_gmp_docx(sop_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # SOP_GMP_LINK에서 sop_id로 연결된 gmp_id, 근거 등 조회
            cursor.execute("""
                SELECT gmp_id, change_rationale, key_changes, update_recommendation
                FROM SOP_GMP_LINK
                WHERE sop_id=%s AND completed='처리'
            """, (sop_id,))
            link_data = cursor.fetchall()
            gmp_ids = [row['gmp_id'] for row in link_data]
            # GMP 테이블에서 전문 조회
            if gmp_ids:
                format_strings = ','.join(['%s'] * len(gmp_ids))
                cursor.execute(f"""
                    SELECT gmp_id, topic, gmp_content
                    FROM GMP
                    WHERE gmp_id IN ({format_strings})
                """, tuple(gmp_ids))
                gmp_data = {row['gmp_id']: row for row in cursor.fetchall()}
            else:
                gmp_data = {}
    finally:
        conn.close()

    # 워드 문서 생성
    doc = Document()
    doc.add_heading(f"SOP({sop_id})와 연결된 GMP 전문", 0)
    for link in link_data:
        gmp = gmp_data.get(link['gmp_id'])
        doc.add_heading(gmp['topic'] if gmp else link['gmp_id'], level=1)
        doc.add_paragraph(gmp['gmp_content'] if gmp else "(GMP 내용 없음)")
        doc.add_paragraph(f"근거: {link['change_rationale']}")
        doc.add_paragraph(f"주요 변경사항: {link['key_changes']}")
        doc.add_paragraph(f"업데이트 권장사항: {link['update_recommendation']}")
        doc.add_paragraph("")

    # 임시 파일로 저장 및 반환
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        tmp_path = tmp.name
    filename = f"GMP_{sop_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    response = FileResponse(tmp_path, filename=filename, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    from fastapi import BackgroundTasks
    def cleanup():
        os.remove(tmp_path)
    response.background = BackgroundTasks()
    response.background.add_task(cleanup)
    return response

if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=8000,
        reload=True, log_level="info"
    )
