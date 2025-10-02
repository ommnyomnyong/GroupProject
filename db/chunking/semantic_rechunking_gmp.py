# -*- coding: utf-8 -*-
"""
GMP 의미 기반 재청킹 도구 (FDA 필터링 버전)
기존 SemanticChunker로 생성된 청크를 더 작은 의미 단위로 재청킹

실행 코드:
python semantic_rechunk.py --input fda_semantic_chunks.jsonl --output fda_2nd_semantic_chunks.jsonl --filter FDA
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# LangChain imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

@dataclass
class ChunkRecord:
    """청크 레코드 구조"""
    id: str
    doc_id: str
    source_path: str
    title: str
    jurisdiction: Optional[str]
    doc_date: Optional[str] 
    doc_version: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    parent_chunk_id: Optional[str] = None  # 원본 청크 ID
    sub_chunk_index: Optional[int] = None   # 서브 청크 인덱스
    semantic_level: Optional[int] = None    # 의미 분할 레벨

class SemanticRechunker:
    """의미 기반 재청킹기"""
    
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 device: str = 'cpu',
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: int = 50,
                 min_chunk_length: int = 100):
        self.model_name = model_name
        self.device = device
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_length = min_chunk_length
        
        # 지연 초기화
        self._embeddings = None
        self._splitter = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"[PID {os.getpid()}] Loading embeddings model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={'normalize_embeddings': False},
                model_kwargs={'device': self.device},
            )
        return self._embeddings
    
    @property
    def splitter(self):
        if self._splitter is None:
            print(f"[PID {os.getpid()}] Initializing SemanticChunker")
            print(f"[PID {os.getpid()}] Threshold: {self.breakpoint_threshold_type} = {self.breakpoint_threshold_amount}")
            
            self._splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            )
        return self._splitter

# 전역 재청킹기 변수
_global_semantic_rechunker = None

def init_semantic_worker(model_name, device, breakpoint_threshold_type, 
                        breakpoint_threshold_amount, min_chunk_length):
    """워커 프로세스 초기화"""
    global _global_semantic_rechunker
    _global_semantic_rechunker = SemanticRechunker(
        model_name=model_name,
        device=device,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_length=min_chunk_length
    )

def semantic_rechunk_single_record(chunk_record_dict: Dict[str, Any]) -> List[ChunkRecord]:
    """단일 청크를 의미 기반으로 재청킹"""
    global _global_semantic_rechunker
    
    try:
        # 원본 청크 레코드 생성
        original_record = ChunkRecord(**chunk_record_dict)
        
        # 텍스트가 너무 짧으면 재청킹 안함
        if len(original_record.text) <= _global_semantic_rechunker.min_chunk_length:
            original_record.parent_chunk_id = original_record.id
            original_record.sub_chunk_index = 0
            original_record.semantic_level = 1  # 1차 의미 레벨
            return [original_record]
        
        # 의미 기반 재청킹 수행
        try:
            sub_chunks = _global_semantic_rechunker.splitter.split_text(original_record.text)
        except Exception as e:
            print(f"[PID {os.getpid()}] SemanticChunker error: {e}, falling back to original")
            original_record.parent_chunk_id = original_record.id
            original_record.sub_chunk_index = 0
            original_record.semantic_level = 1
            return [original_record]
        
        # 서브청크가 1개이거나 원본과 크게 다르지 않으면 원본 반환
        if len(sub_chunks) <= 1:
            original_record.parent_chunk_id = original_record.id
            original_record.sub_chunk_index = 0
            original_record.semantic_level = 1
            return [original_record]
        
        # 재청킹 효과가 미미하면 원본 반환 (평균 길이 차이가 20% 이하)
        original_length = len(original_record.text)
        avg_sub_length = sum(len(chunk) for chunk in sub_chunks) / len(sub_chunks)
        
        if avg_sub_length > original_length * 0.8:  # 80% 이상이면 의미있는 분할이 아님
            original_record.parent_chunk_id = original_record.id
            original_record.sub_chunk_index = 0
            original_record.semantic_level = 1
            return [original_record]
        
        # 새로운 의미 기반 청크 레코드들 생성
        new_records = []
        for sub_idx, sub_text in enumerate(sub_chunks):
            # 너무 짧은 서브청크는 제외
            if len(sub_text.strip()) < 50:
                continue
                
            new_record = ChunkRecord(
                id=f"{original_record.id}-sem{sub_idx:03d}",
                doc_id=original_record.doc_id,
                source_path=original_record.source_path,
                title=original_record.title,
                jurisdiction=original_record.jurisdiction,
                doc_date=original_record.doc_date,
                doc_version=original_record.doc_version,
                section_id=original_record.section_id,
                section_title=original_record.section_title,
                page_start=original_record.page_start,
                page_end=original_record.page_end,
                chunk_index=original_record.chunk_index,
                text=sub_text.strip(),
                parent_chunk_id=original_record.id,
                sub_chunk_index=sub_idx,
                semantic_level=2  # 2차 의미 레벨
            )
            new_records.append(new_record)
        
        # 유효한 서브청크가 없으면 원본 반환
        if not new_records:
            original_record.parent_chunk_id = original_record.id
            original_record.sub_chunk_index = 0
            original_record.semantic_level = 1
            return [original_record]
        
        return new_records
        
    except Exception as e:
        print(f"[PID {os.getpid()}] Error in semantic rechunking: {e}")
        # 오류 시 원본 반환
        original_record = ChunkRecord(**chunk_record_dict)
        original_record.parent_chunk_id = original_record.id
        original_record.sub_chunk_index = 0
        original_record.semantic_level = 1
        return [original_record]

def load_jsonl_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일에서 청크 로드"""
    print(f"📖 Loading chunks from: {jsonl_path}")
    
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk_data = json.loads(line.strip())
                chunks.append(chunk_data)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error at line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks

def filter_chunks_by_jurisdiction(chunks: List[Dict[str, Any]], target_filter: str = None) -> List[Dict[str, Any]]:
    """관할 지역별로 청크 필터링"""
    if not target_filter:
        log("필터링 없이 전체 데이터 사용")
        return chunks
    
    print(f"🔍 '{target_filter}' 관할 지역으로 필터링 시작...")
    
    filtered_chunks = []
    for chunk in chunks:
        jurisdiction = chunk.get("jurisdiction", "")
        
        # 대소문자 구분 없이 포함 여부 확인
        if jurisdiction and target_filter.lower() in jurisdiction.lower():
            filtered_chunks.append(chunk)
    
    print(f"✅ 필터링 결과: {len(filtered_chunks)}개 청크 (전체: {len(chunks)}개)")
    
    if len(filtered_chunks) == 0:
        print(f"⚠️ '{target_filter}' 관할 지역에 해당하는 문서가 없습니다!")
        
        # 사용 가능한 관할 지역 표시
        unique_jurisdictions = set()
        for chunk in chunks:
            if chunk.get("jurisdiction"):
                unique_jurisdictions.add(chunk["jurisdiction"])
        
        if unique_jurisdictions:
            print(f"사용 가능한 관할 지역: {sorted(unique_jurisdictions)}")
    
    return filtered_chunks

def save_jsonl_chunks(records: List[ChunkRecord], output_path: Path) -> None:
    """청크 레코드들을 JSONL 파일로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            record_dict = asdict(record)
            f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved {len(records)} semantic rechunked records to: {output_path}")

def semantic_rechunk_parallel(input_chunks: List[Dict[str, Any]], 
                             model_name: str = "jhgan/ko-sroberta-multitask",
                             device: str = 'cpu',
                             breakpoint_threshold_type: str = "percentile",
                             breakpoint_threshold_amount: int = 50,
                             min_chunk_length: int = 100,
                             max_workers: int = None) -> List[ChunkRecord]:
    """병렬로 의미 기반 재청킹"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count() // 2, len(input_chunks), 4)
    
    print(f"🔄 Semantic rechunking {len(input_chunks)} chunks with {max_workers} workers")
    print(f"🤖 Model: {model_name}")
    print(f"📊 Threshold: {breakpoint_threshold_type} = {breakpoint_threshold_amount}")
    print(f"📏 Min chunk length: {min_chunk_length}")
    
    start_time = time.time()
    all_records = []
    
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_semantic_worker,
        initargs=(model_name, device, breakpoint_threshold_type,
                 breakpoint_threshold_amount, min_chunk_length)
    ) as executor:
        
        # 작업 제출
        future_to_chunk = {
            executor.submit(semantic_rechunk_single_record, chunk_data): i
            for i, chunk_data in enumerate(input_chunks)
        }
        
        # 결과 수집
        completed_count = 0
        rechunked_count = 0
        kept_original_count = 0
        
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                records = future.result()
                all_records.extend(records)
                completed_count += 1
                
                # 재청킹 통계
                if len(records) > 1:
                    rechunked_count += 1
                else:
                    kept_original_count += 1
                
                if completed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    rechunk_ratio = rechunked_count / completed_count * 100
                    print(f"📈 Progress: {completed_count}/{len(input_chunks)} chunks processed")
                    print(f"   Generated: {len(all_records)} sub-chunks")
                    print(f"   Rechunked: {rechunked_count} ({rechunk_ratio:.1f}%)")
                    print(f"   Kept original: {kept_original_count}")
                    print(f"   Elapsed: {elapsed:.1f}s")
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
    
    total_time = time.time() - start_time
    expansion_ratio = len(all_records) / len(input_chunks) if input_chunks else 0
    rechunk_ratio = rechunked_count / len(input_chunks) * 100 if input_chunks else 0
    
    print(f"\n✅ Semantic rechunking completed!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📊 Original chunks: {len(input_chunks)}")
    print(f"📊 New chunks: {len(all_records)}")
    print(f"📊 Expansion ratio: {expansion_ratio:.2f}x")
    print(f"📊 Rechunked ratio: {rechunk_ratio:.1f}%")
    print(f"📊 Average processing time: {total_time/len(input_chunks):.2f}s per chunk")
    
    return all_records

def semantic_rechunk_jsonl_file(input_path: str,
                               output_path: str,
                               jurisdiction_filter: str = None,  # ⭐ 필터 매개변수 추가 ⭐
                               model_name: str = "jhgan/ko-sroberta-multitask",
                               device: str = 'cpu',
                               breakpoint_threshold_type: str = "percentile",
                               breakpoint_threshold_amount: int = 50,
                               min_chunk_length: int = 100,
                               max_workers: int = None,
                               batch_size: int = None):
    """JSONL 파일 의미 기반 재청킹 메인 함수 (필터링 포함)"""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return
    
    print(f"🚀 Starting semantic rechunking process")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    if jurisdiction_filter:
        print(f"🔍 Filter: {jurisdiction_filter}")
    
    # 청크 로드
    input_chunks = load_jsonl_chunks(input_path)
    if not input_chunks:
        print("❌ No chunks to process")
        return
    
    # ⭐ 관할 지역 필터링 추가 ⭐
    if jurisdiction_filter:
        filtered_chunks = filter_chunks_by_jurisdiction(input_chunks, jurisdiction_filter)
        if not filtered_chunks:
            print("❌ 필터링 결과 처리할 데이터가 없습니다.")
            return
        input_chunks = filtered_chunks
    
    # 샘플 청크 분석
    sample_lengths = [len(chunk.get('text', '')) for chunk in input_chunks[:100]]
    avg_length = sum(sample_lengths) / len(sample_lengths)
    max_length = max(sample_lengths)
    min_length = min(sample_lengths)
    
    print(f"📏 Sample chunk analysis (first 100):")
    print(f"   Average length: {avg_length:.0f} chars")
    print(f"   Max length: {max_length} chars")
    print(f"   Min length: {min_length} chars")
    
    # 배치 처리 여부
    if batch_size and len(input_chunks) > batch_size:
        all_records = []
        total_batches = (len(input_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(input_chunks), batch_size):
            batch = input_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            batch_records = semantic_rechunk_parallel(
                input_chunks=batch,
                model_name=model_name,
                device=device,
                breakpoint_threshold_type=breakpoint_threshold_type,
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                min_chunk_length=min_chunk_length,
                max_workers=max_workers
            )
            
            all_records.extend(batch_records)
            print(f"✅ Batch {batch_num} completed. Total sub-chunks: {len(all_records)}")
    
    else:
        # 전체 처리
        all_records = semantic_rechunk_parallel(
            input_chunks=input_chunks,
            model_name=model_name,
            device=device,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_length=min_chunk_length,
            max_workers=max_workers
        )
    
    # 결과 저장
    save_jsonl_chunks(all_records, output_path)
    
    # 최종 통계
    semantic_level_1 = sum(1 for r in all_records if r.semantic_level == 1)
    semantic_level_2 = sum(1 for r in all_records if r.semantic_level == 2)
    
    print(f"\n🎉 Semantic rechunking completed!")
    print(f"📁 Input: {input_path} ({len(input_chunks)} chunks)")
    print(f"📁 Output: {output_path} ({len(all_records)} chunks)")
    if jurisdiction_filter:
        print(f"🔍 Filter: {jurisdiction_filter}")
    print(f"📊 Semantic Level 1 (original): {semantic_level_1}")
    print(f"📊 Semantic Level 2 (rechunked): {semantic_level_2}")

def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description="GMP 의미 기반 재청킹 도구 (필터링 지원)")
    
    parser.add_argument("--input", type=str, required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="출력 JSONL 파일 경로")
    
    # ⭐ 필터링 옵션 추가 ⭐
    parser.add_argument("--filter", type=str, help="관할 지역 필터 (예: FDA, EMA, KFDA)")
    
    # 의미 청킹 파라미터
    parser.add_argument("--model", type=str, default="jhgan/ko-sroberta-multitask", help="임베딩 모델명")
    parser.add_argument("--device", type=str, default="cpu", help="디바이스 (cpu/cuda)")
    parser.add_argument("--threshold-type", 
                       choices=["percentile", "standard_deviation", "interquartile"],
                       default="percentile", help="의미 분할 기준")
    parser.add_argument("--threshold-amount", type=int, default=50,
                       help="의미 분할 임계값 (작을수록 더 세분화)")
    parser.add_argument("--min-length", type=int, default=100,
                       help="최소 청크 길이 (이보다 짧으면 재청킹 안함)")
    
    # 병렬 처리 파라미터
    parser.add_argument("--max-workers", type=int, default=None, help="최대 워커 수")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기 (메모리 절약용)")
    
    args = parser.parse_args()
    
    semantic_rechunk_jsonl_file(
        input_path=args.input,
        output_path=args.output,
        jurisdiction_filter=args.filter,  # ⭐ 필터 매개변수 전달 ⭐
        model_name=args.model,
        device=args.device,
        breakpoint_threshold_type=args.threshold_type,
        breakpoint_threshold_amount=args.threshold_amount,
        min_chunk_length=args.min_length,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
