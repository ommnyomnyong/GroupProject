import pymysql
import json
import os
from dotenv import load_dotenv

load_dotenv()

# DB 접속 정보 (Cloudtype 환경)
db_config = {
    'host': 'localhost',  # Cloudtype에서 MariaDB 서비스명
    'port': 3306,
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'charset': 'utf8mb4'
}

def insert_analysis_summary(summary_dict, db_config):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO ANALYSIS_SUMMARY (
                    total_gmp_changes,
                    affected_sop_sections,
                    created_at
                ) VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (
                summary_dict.get('총_gmp_변경점', 0),
                summary_dict.get('영향받는_sop_섹션', 0),
                summary_dict.get('분석완료시각')
            ))
        conn.commit()
    finally:
        conn.close()

def insert_sop_data(sop_list, db_config):
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
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            for item in sop_list:
                sop_info = item.get('sop_info', {})
                sop_id = sop_info.get('sop_id')
                sop_title = sop_info.get('sop_title')
                sop_content = sop_info.get('sop_content')
                if sop_id is None or sop_content is None:
                    continue
                cursor.execute(insert_sql, (
                    sop_id,
                    sop_title,
                    sop_content
                ))
            conn.commit()
    finally:
        conn.close()

def insert_gmp_data(sop_list, db_config):
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
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            for item in sop_list:
                gmp_info = item.get('gmp_change_info', {})
                gmp_id = gmp_info.get('change_id')
                topic = gmp_info.get('topic')
                old_content = gmp_info.get('old_gmp_content', '')
                new_content = gmp_info.get('new_gmp_content', '')
                gmp_content = f"OLD:\n{old_content}\n\nNEW:\n{new_content}"
                similarity_score = gmp_info.get('similarity_score', 0)
                if gmp_id is None or not gmp_content:
                    continue
                cursor.execute(insert_sql, (
                    gmp_id,
                    topic,
                    gmp_content,
                    similarity_score
                ))
            conn.commit()
    finally:
        conn.close()

def insert_sop_gmp_link(sop_list, db_config):
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
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            for item in sop_list:
                sop_info = item.get('sop_info', {})
                gmp_info = item.get('gmp_change_info', {})
                rationale = item.get('change_rationale', {})
                key_changes = item.get('key_changes', [])
                update_recommendation = item.get('update_recommendation', '')
                completed_status = '미처리'
                sop_id = sop_info.get('sop_id')
                gmp_id = gmp_info.get('change_id')
                match_score = sop_info.get('match_score', 0)
                if sop_id is None or gmp_id is None:
                    continue
                change_rationale_json = json.dumps(rationale, ensure_ascii=False)
                key_changes_json = json.dumps(key_changes, ensure_ascii=False)
                cursor.execute(insert_sql, (
                    sop_id,
                    gmp_id,
                    match_score,
                    change_rationale_json,
                    key_changes_json,
                    update_recommendation,
                    completed_status
                ))
            conn.commit()
    finally:
        conn.close()

# 메인 실행 부분
if __name__ == "__main__":
    with open('../professional_reranker_cohere_gmp_sop_analysis_formatted.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    summary = data['summary']
    summary['분석완료시각'] = data.get('pipeline_info', {}).get('분석완료시각')
    insert_analysis_summary(summary, db_config)
    insert_sop_data(data.get('detailed_analysis', []), db_config)
    insert_gmp_data(data.get('detailed_analysis', []), db_config)
    insert_sop_gmp_link(data.get('detailed_analysis', []), db_config)
