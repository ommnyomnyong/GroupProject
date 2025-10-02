import json

def load_gmp_terms(filepath):
    terms = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 용어가 , 등으로 구분돼 있으면 분리할 수 있게 처리
            for term in line.strip().split(','):
                term = term.strip()
                if term:
                    terms.add(term)
    return terms

def check_terms_in_chunks(jsonl_path, terms):
    term_counts = {term: 0 for term in terms}
    chunk_matches = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '').lower()
            matched_terms = [term for term in terms if term.lower() in text]
            chunk_matches.append({'chunk_id': data.get('id'), 'matched_terms': matched_terms})
            for term in matched_terms:
                term_counts[term] += 1

    return term_counts, chunk_matches

if __name__ == "__main__":
    gmp_terms_file = "./GMP 용어집/gmp_all_terms.txt"
    jsonl_file = "./chunking/chunks/fda_2nd_semantic_chunks.jsonl"

    gmp_terms = load_gmp_terms(gmp_terms_file)
    counts, matches = check_terms_in_chunks(jsonl_file, gmp_terms)

    print("GMP 용어별 출현 횟수:")
    for term, count in counts.items():
        print(f"{term}: {count}")



    print(f"총 {len(matches)}개의 청크를 검사했습니다.")
