# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import hashlib
from dataclasses import dataclass, asdict

"""
실행
python semantic_chunks_gmp.py --pdf_dir "PDF_폴더_경로" --output "semantic_chunks.jsonl" --max_workers 4 <- 병렬 처리
"""

from langchain_community.document_loaders import PyPDFLoader

# 최신 langchain text splitter 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter

# langchain_experimental 패키지 별도 설치 필요
try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    raise ImportError("langchain_experimental 패키지를 설치하고 임포트하세요: pip install langchain_experimental")

# 최신 권장 Embeddings 임포트 (설치 확인 필요)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings



@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    source_path: str
    title: str
    jurisdiction: str
    doc_date: str
    doc_version: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str

class SemanticPDFProcessor:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask",
                 device="cpu", breakpoint_threshold_type="percentile", breakpoint_threshold_amount=70):
        self.model_name = model_name
        self.device = device
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self._embeddings = None
        self._text_splitter = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"Loading embedding model on {self.device}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": False},
                model_kwargs={"device": self.device}
            )
        return self._embeddings

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            print(f"Initializing SemanticChunker threshold={self.breakpoint_threshold_amount}")
            self._text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount)
        return self._text_splitter

_global_processor = None

def init_worker(model_name, device, breakpoint_threshold_type, breakpoint_threshold_amount):
    global _global_processor
    _global_processor = SemanticPDFProcessor(model_name, device, breakpoint_threshold_type, breakpoint_threshold_amount)

def clean_text(text):
    text = text.replace('\ufeff', '').replace('\t', ' ')
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[ \u00A0]{2,}', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_metadata(text, filename):
    date_pattern = r'(20\d{2}[./\-]\d{1,2}[./\-]\d{1,2}|[0-3]?\d\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})'
    date_match = re.search(date_pattern, text[:1000], re.IGNORECASE)
    doc_date = date_match.group(0) if date_match else None
    if not doc_date:
        filename_date = re.search(r'(20\d{2}[._-]\d{1,2}[._-]\d{1,2}|20\d{4})', filename)
        doc_date = filename_date.group(0) if filename_date else None
    version_pattern = r'\b(Rev(?:ision)?|Version|Ver\.?)\s*[:\-]?\s*([A-Za-z]?\d+(?:\.\d+)*)'
    version_match = re.search(version_pattern, text[:1000], re.IGNORECASE)
    doc_version = f"{version_match.group(1)} {version_match.group(2)}" if version_match else None
    if not doc_version:
        filename_version = re.search(version_pattern, filename, re.IGNORECASE)
        doc_version = f"{filename_version.group(1)} {filename_version.group(2)}" if filename_version else None
    return {'doc_date': doc_date, 'doc_version': doc_version}

def infer_jurisdiction(path_str):
    path_lower = path_str.lower()
    if any(x in path_lower for x in ['eu', 'ema', 'european']):
        return 'EU'
    if any(x in path_lower for x in ['fda', 'usfda', 'cfr', '21cfr']):
        return 'US-FDA'
    if 'who' in path_lower:
        return 'WHO'
    if 'pic' in path_lower:
        return 'PIC/S'
    if any(x in path_lower for x in ['mfds', 'kfda', 'korea']):
        return 'KR-MFDS'
    if any(x in path_lower for x in ['ich', 'international']):
        return 'ICH'
    return None

def create_doc_id(title, path):
    slug = re.sub(r'[^\w가-힣\-_. ]+', '', title).strip()
    slug = re.sub(r'\s+', '_', slug)[:50]
    path_hash = hashlib.sha1(path.encode('utf-8')).hexdigest()[:12]
    return f"{slug}-{path_hash}"

def estimate_page_range(chunk_text, full_text, total_pages, chunk_index, all_chunks):
    chunk_pos = full_text.find(chunk_text[:100])
    if chunk_pos == -1:
        total = len(all_chunks)
        start = max(1, int(chunk_index / total * total_pages) + 1)
        end = min(total_pages, int((chunk_index +1)/ total * total_pages) + 1)
        return start, end
    start_ratio = chunk_pos / len(full_text)
    end_ratio = (chunk_pos + len(chunk_text))/len(full_text)
    start = max(1, int(start_ratio * total_pages) + 1)
    end = min(total_pages, int(end_ratio * total_pages) + 1)
    return start, end

def process_single_pdf(pdf_path):
    global _global_processor
    pdf_path = Path(pdf_path)
    print(f"Processing {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if not pages:
        print(f"Warning: No pages found in {pdf_path.name}")
        return []
    full_text = '\n\n'.join(clean_text(p.page_content) for p in pages)
    if not full_text.strip():
        print(f"Warning: No text content in {pdf_path.name}")
        return []
    title = pdf_path.stem
    doc_id = create_doc_id(title, str(pdf_path))
    metadata = extract_metadata(full_text, pdf_path.name)
    jurisdiction = infer_jurisdiction(str(pdf_path))
    chunks = _global_processor.text_splitter.split_text(full_text)
    records = []
    for idx, chunk in enumerate(chunks):
        start, end = estimate_page_range(chunk, full_text, len(pages), idx, chunks)
        rec = ChunkRecord(
            id=f"{doc_id}-{idx:04}",
            doc_id=doc_id,
            source_path=str(pdf_path),
            title=title,
            jurisdiction=jurisdiction,
            doc_date=metadata['doc_date'],
            doc_version=metadata['doc_version'],
            section_id=None,
            section_title=None,
            page_start=start,
            page_end=end,
            chunk_index=idx,
            text=chunk.strip()
        )
        records.append(rec)
    return records

def find_pdfs(directory):
    directory = Path(directory)
    return sorted(directory.rglob('*.pdf'))

def write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf8') as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
    print(f"Saved {len(records)} chunks to {output_path}")

def process_pdfs_parallel(pdf_dir, output_path,
                          model_name="jhgan/ko-sroberta-multitask",
                          device="cpu",
                          breakpoint_threshold_type="percentile",
                          breakpoint_threshold_amount=70,
                          max_workers=None):
    pdf_files = find_pdfs(pdf_dir)
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
    global _global_processor
    _global_processor = SemanticPDFProcessor(
        model_name=model_name,
        device=device,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(pdf_files))
    records = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                             initargs=(model_name, device, breakpoint_threshold_type, breakpoint_threshold_amount)) as executor:
        future_to_path = {executor.submit(process_single_pdf, str(pdf)): pdf for pdf in pdf_files}
        total_processed = 0
        for future in as_completed(future_to_path):
            pdf_path = future_to_path[future]
            try:
                new_records = future.result()
                records.extend(new_records)
                total_processed += 1
                print(f"Processed {total_processed}/{len(pdf_files)} PDFs, total chunks so far: {len(records)}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
    write_jsonl(records, output_path)
    print(f"Processing complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="폴더 내 PDF 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="출력 JSONL 파일 경로")
    parser.add_argument("--model_name", type=str, default="jhgan/ko-sroberta-multitask")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold_type", type=str, default="percentile")
    parser.add_argument("--threshold_amount", type=int, default=70)
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    process_pdfs_parallel(pdf_dir=args.pdf_dir,
                          output_path=args.output,
                          model_name=args.model_name,
                          device=args.device,
                          breakpoint_threshold_type=args.threshold_type,
                          breakpoint_threshold_amount=args.threshold_amount,
                          max_workers=args.max_workers)
