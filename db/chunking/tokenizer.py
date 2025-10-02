"""
'알기쉬운+GMP+용어집(민원인+안내서).pdf'에서 추출한 단어 토큰화 -> 사용자 설정 사전으로 등록
RAG가 GMP와 SOP 탐색 시 사용자 사전을 탐색하도록 하기 위함
ex.)'출하 승인 감독' vs '승인' 구분하도록
"""

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained('some-model')

# 용어 리스트 로드
with open("gmp_all_terms_cleaned.txt", "r", encoding="utf-8") as f:
    new_tokens = [line.strip() for line in f if line.strip()]

# 사용자 사전(토큰) 추가
num_added = tokenizer.add_tokens(new_tokens)
print(f"{num_added}개의 토큰이 사용자 사전으로 추가되었습니다.")

"""
RAG를 돌리기 전에 위 코드를 실행하여 사용자 사전에 GMP 용어를 추가,
이후 사전을 불러와서 RAG가 사전을 참고하도록 해야 함.
"""