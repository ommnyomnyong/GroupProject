-- 분석 결과 테이블
CREATE TABLE ANALYSIS_SUMMARY (
  analysis_id BIGINT AUTO_INCREMENT PRIMARY KEY, -- 분석 세션 고유 ID
  total_gmp_changes INT,                         -- 총 GMP 변경점
  affected_sop_sections INT,                     -- 영향받는 SOP 섹션 수
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP  -- 분석 완료 시각
);

-- SOP 수정 조항 테이블
CREATE TABLE SOP (
  sop_id VARCHAR(50) PRIMARY KEY,   -- RAG 결과의 sop_id 그대로 사용
  sop_title VARCHAR(255),           -- SOP 제목
  sop_content TEXT NOT NULL,        -- SOP 전문
  modified BOOLEAN DEFAULT FALSE, -- SOP 수정 시 True로 변경되도록
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- GMP 변경 조항 테이블
CREATE TABLE GMP (
  gmp_id VARCHAR(50) PRIMARY KEY, -- RAG 결과의 change_id 등 사용
  topic VARCHAR(100),             -- GMP 변경 주제 / topic별로 요약해줄 예정
  gmp_content TEXT,               -- 근거 GMP 원문 / sop_content
  similarity_score FLOAT,         -- 유사도 점수
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 수정 필요 SOP와 변경 GMP 연결
CREATE TABLE SOP_GMP_LINK (
  link_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  sop_id VARCHAR(50) NOT NULL,
  gmp_id VARCHAR(50) NOT NULL,
  match_score FLOAT,              -- SOP-GMP 매칭 점수
  change_rationale JSON,          -- 수정 사유, 법적 근거 등 (JSON으로 저장)
  key_changes JSON,               -- 주요 변경사항 (JSON 배열)
  update_recommendation TEXT,     -- 업데이트 권장사항
  completed VARCHAR(20) NOT NULL DEFAULT '미처리', -- '처리', '보류', '미처리'
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (sop_id) REFERENCES SOP(sop_id) ON DELETE CASCADE,
  FOREIGN KEY (gmp_id) REFERENCES GMP(gmp_id) ON DELETE CASCADE,
  UNIQUE KEY unique_sop_gmp (sop_id, gmp_id)
);

-- 수정한 SOP 관리
CREATE TABLE SOP_MODIFIED (
  modified_content_id BIGINT AUTO_INCREMENT PRIMARY KEY,  -- 수정 내용 고유 ID
  sop_id VARCHAR(50) NOT NULL,                            -- 수정된 SOP ID (SOP 테이블 참조)
  sop_title VARCHAR(255) NOT NULL,                        -- 수정 제목
  sop_content TEXT NOT NULL,                              -- 수정 내용 전문
  educated BOOLEAN DEFAULT FALSE,                         -- 교육 생성 시 True로 업데이트 -> 이력 관리 
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,          -- 생성 시각
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (sop_id) REFERENCES SOP(sop_id) ON DELETE CASCADE
);