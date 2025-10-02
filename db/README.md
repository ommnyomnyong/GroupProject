GMP 벡터 데이터베이스 생성기
AI 기반 GMP(Good Manufacturing Practice) 문서 검색 시스템을 위한 벡터 데이터베이스 생성 도구입니다.

🎯개요
create_vector_db.py는 GMP 관련 문서들을 벡터화하여 FAISS(Facebook AI Similarity Search) 데이터베이스를 생성하는 스크립트입니다. 이를 통해 AI 기반 문서 검색 및 질의응답 시스템을 구축할 수 있습니다.

주요 특징
📄 대용량 문서 처리: 39,000+ 문서 청크 처리 가능

⚡ 배치 처리: 메모리 효율적인 배치별 벡터화

🔍 FAISS 통합: 고성능 유사도 검색을 위한 FAISS 벡터 스토어

📊 진행상황 추적: tqdm을 통한 실시간 처리 현황 표시

💾 자동 저장: 생성된 벡터 DB 자동 저장 및 로드

🚀 주요 기능
1. 문서 로드 및 변환
python
# JSONL 파일에서 청크 데이터 로드
chunks = load_chunks_from_jsonl(input_file)
documents = convert_to_documents(chunks)
2. 벡터 임베딩 생성
OpenAI 임베딩 모델 사용 (text-embedding-3-small)

배치 처리: 메모리 효율성을 위한 10개씩 배치 처리

오류 복구: 실패한 배치 자동 재시도

3. FAISS 벡터 스토어 구축
python
# FAISS 벡터 스토어 생성
vectordb = FAISS.from_documents(
    documents[:10], 
    embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

# 나머지 문서들 배치별 추가
for batch in tqdm(batches):
    vectordb.add_documents(batch)
4. 자동 저장 및 검증
생성된 벡터 DB를 로컬에 저장

기본 검색 테스트로 정상 동작 확인

💻 설치
1. 필수 패키지 설치
bash
pip install -r requirements.txt
2. requirements.txt 내용
text
# Core dependencies
python-dotenv>=1.0.0
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.29
openai>=1.0.0

# Vector store and embeddings
faiss-cpu>=1.7.4
numpy>=1.21.0

# Utilities
tqdm>=4.64.0
3. 환경 변수 설정
.env 파일에 OpenAI API 키 설정:

bash
OPENAI_API_KEY=your_openai_api_key_here
📖 사용법
기본 사용법
bash
python create_vector_db.py --input ./chunks.jsonl
고급 옵션
bash
python create_vector_db.py \
    --input ./data/chunks.jsonl \
    --output ./my_vector_db \
    --batch-size 20
매개변수 설명
--input: 입력 JSONL 파일 경로 (필수)

--output: 출력 벡터 DB 디렉토리 경로 (선택, 기본값: ./gmp_vector_db)

--batch-size: 배치 크기 (선택, 기본값: 10)

📁 파일 구조
입력 파일 형식 (chunks.jsonl)
json
{"content": "GMP 규정에 따른 품질관리...", "metadata": {"source": "doc1.pdf", "page": 1}}
{"content": "제조공정의 검증은...", "metadata": {"source": "doc2.pdf", "page": 5}}
출력 디렉토리 구조
text
gmp_vector_db/
├── index.faiss          # FAISS 인덱스 파일
├── index.pkl            # 메타데이터 파일
└── docstore.pkl         # 문서 저장소
🛠 기술 스택
핵심 라이브러리
LangChain: 문서 처리 및 벡터 스토어 통합

FAISS: 고성능 유사도 검색

OpenAI: 텍스트 임베딩 생성

tqdm: 진행상황 시각화

임베딩 모델
모델: text-embedding-3-small

차원: 1,536차원 벡터

거리 측정: 코사인 유사도

🔧 실행 결과 예시
bash
PS C:\workspace\GMP-PROJECT> python create_vector_db.py --input "./chunks.jsonl"
[2025-09-09 13:53:57] GMP 벡터 DB 생성 파이프라인 시작
[2025-09-09 13:53:57] JSONL 파일 로드: ./chunks.jsonl
[2025-09-09 13:54:00] 총 39744개 청크 로드 완료
[2025-09-09 13:54:00] Document 형식으로 변환 중...
[2025-09-09 13:54:01] 39744개 Document 변환 완료
[2025-09-09 13:54:01] 벡터 DB 생성: ./gmp_vector_db
[2025-09-09 13:54:05] 첫 번째 배치 (10개)로 FAISS 초기화...
FAISS 배치 추가: 100%|████████████| 3974/3974 [42:01<00:00,  1.58it/s]
[2025-09-09 14:36:12] FAISS 벡터 DB 저장 완료: 39744개 문서
[2025-09-09 14:36:12] 벡터 DB 생성 파이프라인 완료!
❗ 문제 해결
일반적인 오류 및 해결법
1. OpenAI API 키 오류
bash
Error: OpenAI API key not found
해결법: .env 파일에 올바른 API 키 설정 확인

2. 메모리 부족
bash
MemoryError: Unable to allocate array
해결법: --batch-size 값을 더 작게 설정 (예: 5)

3. FAISS 설치 오류
bash
ImportError: No module named 'faiss'
해결법:

bash
# CPU 버전
pip install faiss-cpu

# GPU 버전 (CUDA 필요)
pip install faiss-gpu
4. 검증 함수 오류
bash
NameError: name 'validate_vector_store' is not defined
해결법: 스크립트에 검증 함수 추가 또는 해당 라인 주석 처리

성능 최적화 팁
배치 크기 조정: 메모리와 속도 균형 맞추기

GPU 가속: faiss-gpu 사용으로 속도 향상

병렬 처리: 대용량 데이터셋의 경우 멀티프로세싱 고려

📈 성능 정보
처리 속도: 약 1.58 문서/초 (39,744개 문서 기준)

총 처리 시간: 약 42분 (OpenAI API 호출 포함)

메모리 사용량: 배치 크기에 따라 가변적

디스크 사용량: 문서 수에 비례하여 증가