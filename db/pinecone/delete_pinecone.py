import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
"""
pinecone에서 원하는 namespace의 데이터 삭제하기
"""
# 환경 변수에서 API 키 불러오기
api_key = os.getenv("PINECONE_API_KEY")
index_name = "gmp-sop-vectordb"

# Pinecone 연결
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# 삭제할 네임스페이스 리스트
namespaces_to_delete = ["sop", "sop-1st", "sop-2nd"]

for ns in namespaces_to_delete:
    index.delete(delete_all=True, namespace=ns)
    print(f"✅ 네임스페이스 '{ns}'의 모든 데이터 삭제 완료")
