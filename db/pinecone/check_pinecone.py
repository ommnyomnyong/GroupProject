# -*- coding: utf-8 -*-
"""
Pinecone GMP 벡터 DB 검증 및 테스트 통합 도구
- 기본 통계 확인
- 샘플 데이터 조회
- 실제 검색 테스트
- 메타데이터 분석
- 네임스페이스 지원

검색 테스트
python check_pinecone.py --mode search --query "품질" --namespace new-gmp-fda/fda-sop 중 1개
"""

import os
import random
import argparse
from collections import Counter
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ==================== 설정 ====================
INDEX_NAME = "gmp-sop-vectordb"
EMBEDDING_MODEL = "text-embedding-3-small"  # 또는 text-embedding-ada-002
EMBEDDING_DIMENSION = 1536  # OpenAI 임베딩 차원


def log(msg: str) -> None:
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_pinecone_connection():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(INDEX_NAME)


def get_embedder():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ==================== 1. 기본 통계 확인 ====================
def check_basic_stats():
    log("기본 통계 확인 시작")
    index = get_pinecone_connection()
    stats = index.describe_index_stats()
    print("\n" + "="*50)
    print("📊 PINECONE 인덱스 기본 통계")
    print("="*50)
    print(f"총 벡터 수: {stats['total_vector_count']:,}개")
    print(f"차원: {stats['dimension']}")
    print(f"메트릭: {stats['metric']}")
    print(f"인덱스 사용률: {stats['index_fullness']:.2%}")
    if 'namespaces' in stats and stats['namespaces']:
        print(f"\n📁 네임스페이스별 통계:")
        for ns, ns_stats in stats['namespaces'].items():
            ns_name = ns if ns else "기본"
            print(f"  {ns_name}: {ns_stats['vector_count']:,}개 벡터")
    else:
        print("\n📁 네임스페이스: 기본 네임스페이스만 사용")
    return stats

# ==================== 2. 샘플 데이터 확인 ====================
def check_sample_data(num_samples: int = 5, namespace: str = None):
    log(f"샘플 데이터 {num_samples}개 확인 시작")
    if namespace:
        log(f"네임스페이스: {namespace}")
    index = get_pinecone_connection()
    random_vector = [random.random() for _ in range(EMBEDDING_DIMENSION)]
    results = index.query(
        vector=random_vector,
        top_k=num_samples,
        namespace=namespace,
        include_metadata=True,
        include_values=False
    )
    print("\n" + "="*50)
    print(f"🔍 샘플 데이터 ({len(results['matches'])}개)")
    if namespace:
        print(f"🏷️ 네임스페이스: {namespace}")
    print("="*50)
    for i, match in enumerate(results['matches'], 1):
        print(f"\n{i}. ID: {match['id']}")
        print(f"   유사도 점수: {match['score']:.4f}")
        if 'metadata' in match:
            metadata = match['metadata']
            print(f"   📄 제목: {metadata.get('title', 'N/A')}")
            print(f"   📂 출처: {metadata.get('source_path', 'N/A')}")
            print(f"   📋 섹션: {metadata.get('section_title', 'N/A')}")
            print(f"   🏛️ 관할: {metadata.get('jurisdiction', 'N/A')}")
            if 'text' in metadata:
                text = metadata['text']
                display_text = text[:200] + "..." if len(text) > 200 else text
                print(f"   📝 내용: {display_text}")
        print("-" * 40)
    return results

# ==================== 3. 실제 검색 테스트 ====================
def test_search(queries: list = None, top_k: int = 5, namespace: str = None):
    log("검색 기능 테스트 시작")
    if namespace:
        log(f"네임스페이스: {namespace}")
    if queries is None:
        queries = [
            "GMP 품질관리",
            "밸리데이션",
            "문서관리", 
            "변경관리",
            "위험평가"
        ]
    index = get_pinecone_connection()
    embedder = get_embedder()
    print("\n" + "="*50)
    print("🔎 실제 검색 테스트")
    if namespace:
        print(f"🏷️ 네임스페이스: {namespace}")
    print("="*50)
    for query_idx, query in enumerate(queries, 1):
        print(f"\n🔍 검색 {query_idx}: '{query}'")
        print("-" * 30)
        try:
            query_vector = embedder.embed_query(query)
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            print(f"📊 검색 결과: {len(results['matches'])}개 발견")
            for i, match in enumerate(results['matches'], 1):
                print(f"\n  {i}. 점수: {match['score']:.4f}")
                if 'metadata' in match:
                    metadata = match['metadata']
                    title = metadata.get('title', 'N/A')
                    section = metadata.get('section_title', 'N/A')
                    jurisdiction = metadata.get('jurisdiction', 'N/A')
                    print(f"     제목: {title}")
                    print(f"     섹션: {section}")
                    print(f"     관할: {jurisdiction}")
                    if 'text' in metadata:
                        text = metadata['text']
                        display_text = text[:120] + "..." if len(text) > 120 else text
                        print(f"     내용: {display_text}")
                if i < len(results['matches']):
                    print("     " + "-" * 25)
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
        if query_idx < len(queries):
            print("\n" + "="*40)
    return True

# ==================== 4. 메타데이터 분석 ====================
def analyze_metadata(sample_count: int = 100, namespace: str = None):
    log(f"메타데이터 분석 시작 (샘플: {sample_count}개)")
    if namespace:
        log(f"네임스페이스: {namespace}")
    index = get_pinecone_connection()
    all_metadata = []
    batch_size = 50
    num_batches = (sample_count + batch_size - 1) // batch_size
    for batch in range(num_batches):
        random_vector = [random.random() for _ in range(EMBEDDING_DIMENSION)]
        current_batch_size = min(batch_size, sample_count - len(all_metadata))
        results = index.query(
            vector=random_vector,
            top_k=current_batch_size,
            namespace=namespace,
            include_metadata=True
        )
        for match in results['matches']:
            if 'metadata' in match:
                all_metadata.append(match['metadata'])
        if len(all_metadata) >= sample_count:
            break
    print("\n" + "="*50)
    print(f"📈 메타데이터 분석 (샘플: {len(all_metadata)}개)")
    if namespace:
        print(f"🏷️ 네임스페이스: {namespace}")
    print("="*50)
    jurisdictions = [m.get('jurisdiction') for m in all_metadata if m.get('jurisdiction')]
    if jurisdictions:
        jurisdiction_counts = Counter(jurisdictions)
        print(f"\n🏛️ 관할 지역별 문서 수:")
        for jurisdiction, count in jurisdiction_counts.most_common():
            percentage = (count / len(jurisdictions)) * 100
            print(f"  {jurisdiction}: {count}개 ({percentage:.1f}%)")
    titles = [m.get('title') for m in all_metadata if m.get('title')]
    if titles:
        title_counts = Counter(titles)
        print(f"\n📄 주요 문서 제목 (상위 10개):")
        for i, (title, count) in enumerate(title_counts.most_common(10), 1):
            percentage = (count / len(titles)) * 100
            print(f"  {i:2d}. {title}: {count}개 ({percentage:.1f}%)")
    versions = [m.get('doc_version') for m in all_metadata if m.get('doc_version')]
    if versions:
        version_counts = Counter(versions)
        print(f"\n📋 문서 버전별 분포:")
        for version, count in version_counts.most_common():
            percentage = (count / len(versions)) * 100
            print(f"  {version}: {count}개 ({percentage:.1f}%)")
    sections = [m.get('section_title') for m in all_metadata if m.get('section_title')]
    if sections:
        section_counts = Counter(sections)
        print(f"\n📑 주요 섹션 (상위 10개):")
        for i, (section, count) in enumerate(section_counts.most_common(10), 1):
            percentage = (count / len(sections)) * 100
            print(f"  {i:2d}. {section}: {count}개 ({percentage:.1f}%)")
    doc_types = []
    for m in all_metadata:
        source = m.get('source_path', '')
        if '.pdf' in source.lower():
            doc_types.append('PDF')
        elif '.docx' in source.lower() or '.doc' in source.lower():
            doc_types.append('Word')
        elif '.txt' in source.lower():
            doc_types.append('Text')
        else:
            doc_types.append('기타')
    if doc_types:
        type_counts = Counter(doc_types)
        print(f"\n📎 문서 형태별 분포:")
        for doc_type, count in type_counts.most_common():
            percentage = (count / len(doc_types)) * 100
            print(f"  {doc_type}: {count}개 ({percentage:.1f}%)")
    return all_metadata

# ==================== 네임스페이스별 검색 ====================
def search_all_namespaces(query: str, top_k: int = 3):
    log(f"모든 네임스페이스에서 '{query}' 검색")
    index = get_pinecone_connection()
    embedder = get_embedder()
    stats = index.describe_index_stats()
    namespaces = list(stats.get('namespaces', {}).keys()) if stats.get('namespaces') else [None]
    if not namespaces or (len(namespaces) == 1 and namespaces[0] is None):
        print("\n📁 사용 가능한 네임스페이스가 없습니다. 기본 네임스페이스에서 검색합니다.")
        namespaces = [None]
    query_vector = embedder.embed_query(query)
    print(f"\n🔍 전체 네임스페이스 검색: '{query}'")
    print("="*60)
    for ns in namespaces:
        ns_name = ns if ns else "기본"
        print(f"\n📁 네임스페이스: {ns_name}")
        print("-" * 30)
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=ns,
                include_metadata=True,
                include_values=False
            )
            print(f"📊 결과: {len(results['matches'])}개")
            for i, match in enumerate(results['matches'], 1):
                print(f"  {i}. 점수: {match['score']:.4f}")
                if 'metadata' in match:
                    metadata = match['metadata']
                    title = metadata.get('title', 'N/A')
                    jurisdiction = metadata.get('jurisdiction', 'N/A')
                    print(f"     제목: {title[:50]}..." if len(title) > 50 else f"     제목: {title}")
                    print(f"     관할: {jurisdiction}")
        except Exception as e:
            print(f"❌ 오류: {e}")

# ==================== 메인 실행 함수 ====================
def run_all_checks(namespace: str = None):
    print("🚀 Pinecone GMP 벡터 DB 종합 검증 시작")
    if namespace:
        print(f"🏷️ 네임스페이스: {namespace}")
    print("="*60)
    try:
        stats = check_basic_stats()
        if stats['total_vector_count'] == 0:
            print("\n❌ 벡터가 저장되어 있지 않습니다!")
            return False
        check_sample_data(num_samples=3, namespace=namespace)
        test_search(top_k=3, namespace=namespace)
        analyze_metadata(sample_count=50, namespace=namespace)
        print("\n" + "="*60)
        print("✅ 모든 검증 작업이 완료되었습니다!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n❌ 검증 중 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pinecone GMP 벡터 DB 검증 도구 (OpenAI 임베딩)")
    parser.add_argument("--mode", choices=['all', 'stats', 'sample', 'search', 'metadata', 'search-all'], 
                       default='all', help="실행할 검증 모드")
    parser.add_argument("--query", type=str, help="검색할 쿼리 (search 모드용)")
    parser.add_argument("--samples", type=int, default=5, help="샘플 수")
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 수")
    parser.add_argument("--namespace", type=str, help="검색할 네임스페이스 (예: fda-new-gmp)")
    args = parser.parse_args()
    try:
        if args.mode == 'all':
            run_all_checks(namespace=args.namespace)
        elif args.mode == 'stats':
            check_basic_stats()
        elif args.mode == 'sample':
            check_sample_data(args.samples, namespace=args.namespace)
        elif args.mode == 'search':
            queries = [args.query] if args.query else None
            test_search(queries, args.top_k, namespace=args.namespace)
        elif args.mode == 'search-all':
            if not args.query:
                args.query = "GMP 품질관리"
            search_all_namespaces(args.query, args.top_k)
        elif args.mode == 'metadata':
            analyze_metadata(args.samples, namespace=args.namespace)
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")

if __name__ == "__main__":
    main()
