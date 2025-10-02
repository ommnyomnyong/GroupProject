# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

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
    parent_section_id: str
    section_level: int
    page_start: int
    page_end: int
    chunk_index: int
    text: str

class SemanticPDFProcessor:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask",
                 device="cpu", breakpoint_threshold_type="percentile", breakpoint_amount=70):
        self.model_name = model_name
        self.device = device
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_amount = breakpoint_amount
        self._embeddings = None
        self._text_splitter = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"Loading embedding model on {self.device}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": False},
                model_kwargs={"device": self.device},
            )
        return self._embeddings

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            print(f"Initializing SemanticChunker with threshold={self.breakpoint_amount}")
            self._text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_amount,
            )
        return self._text_splitter

def clean_text(text):
    text = text.replace('\ufeff', '').replace('\t', ' ')
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[ \u00A0]{2,}', ' ', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_document_title(text):
    pattern = r"표준작업절차서 \(SOP\)[\s\S]{0,100}?제목:\s*(.+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).split('\n')[0].strip()
    return None

def extract_metadata(text, filename):
    date_pattern = r'(20\d{2}[./\-]\d{1,2}[./\-]\d{1,2}|[0-3]\d*\s+[A-Za-z]+\s+[0-9]{4})'
    date_match = re.search(date_pattern, text[:1000], re.IGNORECASE)
    doc_date = date_match.group(0) if date_match else None
    if not doc_date:
        filename_date = re.search(r'(20\d{2}[._-]\d{1,2}[._-]\d{1,2}|20\d{4})', filename)
        doc_date = filename_date.group(0) if filename_date else None
    version_pattern = r'\b(Rev(?:ision)?|Version|Ver\.?)\s*[:\-]?\s*([A-Za-z0-9]+(?:\.\d+)*)'
    version_match = re.search(version_pattern, text[:1000], re.IGNORECASE)
    doc_version = None
    if version_match:
        doc_version = version_match.group(1) + ' ' + version_match.group(2)
    else:
        filename_version = re.search(version_pattern, filename, re.IGNORECASE)
        if filename_version:
            doc_version = filename_version.group(1) + ' ' + filename_version.group(2)
    return {'doc_date': doc_date, 'doc_version': doc_version}

def infer_jurisdiction(path):
    path_lower = path.lower()
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

def create_doc_id(path):
    base_name = Path(path).stem
    path_hash = hashlib.sha1(path.encode('utf-8')).hexdigest()[:12]
    return f"{base_name}-{path_hash}"

def estimate_page_range(chunk_text, full_text, total_pages, curr_idx, all_chunks):
    chunk_pos = full_text.find(chunk_text[:100])
    if chunk_pos == -1:
        total = len(all_chunks)
        start = max(1, int(curr_idx / total * total_pages) + 1)
        end = min(total_pages, int((curr_idx + 1) / total * total_pages) + 1)
        return start, end
    start_ratio = chunk_pos / len(full_text)
    end_ratio = (chunk_pos + len(chunk_text)) / len(full_text)
    start = max(1, int(start_ratio * total_pages) + 1)
    end = min(total_pages, int(end_ratio * total_pages))
    return start, end

def parse_sections(text):
    pattern = re.compile(r'(?:^|\n)(\d+(?:\.\d+)*\.)\s+([^\n]+(?:\n(?!\d+(?:\.\d+)*\.)[^\n]*)*)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"section_number": "1", "section_title": "전체 내용", "parent_section_id": None, "section_level": 1, "content": text.strip()}]
    chunks = []
    for i, match in enumerate(matches):
        section_number_raw = match.group(1)
        section_number = section_number_raw.rstrip('.')
        full_match = match.group(0).strip()
        level = section_number.count('.') + 1
        if level == 1:
            parent_section_id = None
        else:
            parts = section_number.split('.')
            parent_parts = parts[:-1]
            parent_section_id = '.'.join(parent_parts)
        first_line = match.group(2).split('\n')[0].strip()
        if len(first_line) > 50:
            title_part = first_line[:47] + "..."
        else:
            title_part = first_line
        section_title = f"{section_number}. {title_part}"
        chunks.append({
            'section_number': section_number,
            'section_title': section_title,
            'parent_section_id': parent_section_id,
            'section_level': level,
            'content': full_match
        })
    return chunks

def process_single_pdf(pdf_path):
    global _global_processor
    pdf_path = Path(pdf_path)
    print(f"Processing {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    full_text = '\n'.join([clean_text(page.page_content) for page in pages])
    doc_title = extract_document_title(full_text)
    if not doc_title:
        doc_title = pdf_path.stem
    if not full_text.strip():
        print(f"Warning: No text content in {pdf_path.name}")
        return []
    doc_id = create_doc_id(str(pdf_path))
    metadata = extract_metadata(full_text, pdf_path.name)
    jurisdiction = infer_jurisdiction(str(pdf_path))
    sections = parse_sections(full_text)
    records = []
    for idx, section_data in enumerate(sections):
        start_page, end_page = estimate_page_range(section_data['content'], full_text, len(pages), idx, [s['content'] for s in sections])
        record = ChunkRecord(
            id=f"{doc_id}-{idx:04}",
            doc_id=doc_id,
            source_path=str(pdf_path),
            title=doc_title,
            jurisdiction=jurisdiction,
            doc_date=metadata['doc_date'],
            doc_version=metadata['doc_version'],
            section_id=section_data['section_number'],
            section_title=section_data['section_title'],
            parent_section_id=section_data['parent_section_id'],
            section_level=section_data['section_level'],
            page_start=start_page,
            page_end=end_page,
            chunk_index=idx,
            text=section_data['content']
        )
        records.append(record)
    return records

def find_pdfs(directory):
    directory = Path(directory)
    return sorted(directory.rglob('*.pdf'))

def process_pdfs_parallel(pdf_dir,
                         model_name="jhgan/ko-sroberta-multitask",
                         device="cpu",
                         breakpoint_threshold_type="percentile",
                         breakpoint_amount=70,
                         max_workers=None,
                         missing_files=None):
    pdf_files = list(Path(pdf_dir).glob('*.pdf'))
    print(f"▶ 전체 PDF 파일명 목록: {[p.name for p in pdf_files]}")
    print(f"▶ 넘겨받은 누락 파일 리스트: {missing_files}")
    if missing_files is not None:
        pdf_files = [p for p in pdf_files if p.name in missing_files]
        print(f"▶ 필터링 후 처리할 PDF 파일명: {[p.name for p in pdf_files]}")
    if not pdf_files:
        print("처리할 PDF 파일이 없습니다.")
        return []
    global _global_processor
    _global_processor = SemanticPDFProcessor(model_name, device,
                                             breakpoint_threshold_type,
                                             breakpoint_amount)
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(pdf_files))
    records = []
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=init_worker,
                             initargs=(model_name, device,
                                       breakpoint_threshold_type,
                                       breakpoint_amount)) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, str(pdf)): pdf for pdf in pdf_files}
        processed_count = 0
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                new_records = future.result()
                records.extend(new_records)
                processed_count += 1
                print(f"Processed {processed_count}/{len(pdf_files)} PDFs. Total chunks: {len(records)}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
    print("처리 완료.")
    return records

def init_worker(model_name, device, breakpoint_threshold_type, breakpoint_amount):
    global _global_processor
    _global_processor = SemanticPDFProcessor(model_name, device, breakpoint_threshold_type, breakpoint_amount)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="폴더 내 PDF 파일 경로")
    parser.add_argument("--model_name", type=str, default="jhgan/ko-sroberta-multitask")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold_type", type=str, default="percentile")
    parser.add_argument("--threshold_amount", type=int, default=70)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--missing_files", type=str, default=None,
                        help="콤마(,)로 구분된 누락 PDF 파일명 리스트")
    args = parser.parse_args()
    if args.missing_files:
        missing_files = [f.strip() for f in args.missing_files.split(',') if f.strip()]
    else:
        missing_files = None
    # 청킹 결과를 파일로 저장하지 않고, 바로 반환
    all_chunks = process_pdfs_parallel(
        pdf_dir=args.pdf_dir,
        model_name=args.model_name,
        device=args.device,
        breakpoint_threshold_type=args.threshold_type,
        breakpoint_amount=args.threshold_amount,
        max_workers=args.max_workers,
        missing_files=missing_files
    )
    print(f"총 {len(all_chunks)}개의 청크가 생성되었습니다.")
    # 필요하다면 아래처럼 파일로 저장 가능
    # with open("chunks.jsonl", "w", encoding="utf-8") as f:
    #     for rec in all_chunks:
    #         f.write(json.dumps(asdict(rec), ensure_ascii=False) + '\n')


# """
# python regex_chunks_sop.py --pdf_dir "pdf 파일 경로" --output "sop_regex_chunks.jsonl" --max_workers 4
# """
# import argparse
# import json
# import os
# import re
# from pathlib import Path
# import hashlib
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from dataclasses import dataclass, asdict

# import pdfplumber
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.embeddings import HuggingFaceEmbeddings

# @dataclass
# class ChunkRecord:
#     id: str
#     doc_id: str
#     source_path: str
#     title: str  # 문서 제목
#     jurisdiction: str
#     doc_date: str
#     doc_version: str
#     section_id: str
#     section_title: str  # 조항 제목
#     parent_section_id: str  # 상위 조항 ID
#     section_level: int      # 조항 레벨 (1=제목, 2,3,4=내용)
#     page_start: int
#     page_end: int
#     chunk_index: int
#     text: str  # 조항 내용

# class SemanticPDFProcessor:
#     def __init__(self, model_name="jhgan/ko-sroberta-multitask",
#                  device="cpu", breakpoint_threshold_type="percentile", breakpoint_amount=70):
#         self.model_name = model_name
#         self.device = device
#         self.breakpoint_threshold_type = breakpoint_threshold_type
#         self.breakpoint_amount = breakpoint_amount
#         self._embeddings = None
#         self._text_splitter = None

#     @property
#     def embeddings(self):
#         if self._embeddings is None:
#             print(f"Loading embedding model on {self.device}")
#             self._embeddings = HuggingFaceEmbeddings(
#                 model_name=self.model_name,
#                 encode_kwargs={"normalize_embeddings": False},
#                 model_kwargs={"device": self.device},
#             )
#         return self._embeddings

#     @property
#     def text_splitter(self):
#         if self._text_splitter is None:
#             print(f"Initializing SemanticChunker with threshold={self.breakpoint_amount}")
#             self._text_splitter = SemanticChunker(
#                 self.embeddings,
#                 breakpoint_threshold_type=self.breakpoint_threshold_type,
#                 breakpoint_threshold_amount=self.breakpoint_amount,
#             )
#         return self._text_splitter

# def clean_text(text):
#     text = text.replace('\ufeff', '').replace('\t', ' ')
#     text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
#     text = re.sub(r'[ \u00A0]{2,}', ' ', text)
#     text = text.replace('\r\n', '\n').replace('\r', '\n')
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# def extract_document_title(text):
#     # '표준작업절차서 (SOP)' 이후에 '제목:'으로 시작하는 줄을 찾아 제목 추출
#     pattern = r"표준작업절차서 \(SOP\)[\s\S]{0,100}?제목:\s*(.+)"
#     match = re.search(pattern, text)
#     if match:
#         # 제목 줄 끝에 다른 정보가 붙어있을 수 있으니 줄바꿈 기준으로 자름
#         return match.group(1).split('\n')[0].strip()
#     return None

# def extract_metadata(text, filename):
#     date_pattern = r'(20\d{2}[./\-]\d{1,2}[./\-]\d{1,2}|[0-3]\d*\s+[A-Za-z]+\s+[0-9]{4})'
#     date_match = re.search(date_pattern, text[:1000], re.IGNORECASE)
#     doc_date = date_match.group(0) if date_match else None
#     if not doc_date:
#         filename_date = re.search(r'(20\d{2}[._-]\d{1,2}[._-]\d{1,2}|20\d{4})', filename)
#         doc_date = filename_date.group(0) if filename_date else None
#     version_pattern = r'\b(Rev(?:ision)?|Version|Ver\.?)\s*[:\-]?\s*([A-Za-z0-9]+(?:\.\d+)*)'
#     version_match = re.search(version_pattern, text[:1000], re.IGNORECASE)
#     doc_version = None
#     if version_match:
#         doc_version = version_match.group(1) + ' ' + version_match.group(2)
#     else:
#         filename_version = re.search(version_pattern, filename, re.IGNORECASE)
#         if filename_version:
#             doc_version = filename_version.group(1) + ' ' + filename_version.group(2)
#     return {'doc_date': doc_date, 'doc_version': doc_version}

# def infer_jurisdiction(path):
#     path_lower = path.lower()
#     if any(x in path_lower for x in ['eu', 'ema', 'european']):
#         return 'EU'
#     if any(x in path_lower for x in ['fda', 'usfda', 'cfr', '21cfr']):
#         return 'US-FDA'
#     if 'who' in path_lower:
#         return 'WHO'
#     if 'pic' in path_lower:
#         return 'PIC/S'
#     if any(x in path_lower for x in ['mfds', 'kfda', 'korea']):
#         return 'KR-MFDS'
#     if any(x in path_lower for x in ['ich', 'international']):
#         return 'ICH'
#     return None

# def create_doc_id(path):
#     base_name = Path(path).stem
#     path_hash = hashlib.sha1(path.encode('utf-8')).hexdigest()[:12]
#     return f"{base_name}-{path_hash}"

# def estimate_page_range(chunk_text, full_text, total_pages, curr_idx, all_chunks):
#     chunk_pos = full_text.find(chunk_text[:100])
#     if chunk_pos == -1:
#         total = len(all_chunks)
#         start = max(1, int(curr_idx / total * total_pages) + 1)
#         end = min(total_pages, int((curr_idx + 1) / total * total_pages) + 1)
#         return start, end
#     start_ratio = chunk_pos / len(full_text)
#     end_ratio = (chunk_pos + len(chunk_text)) / len(full_text)
#     start = max(1, int(start_ratio * total_pages) + 1)
#     end = min(total_pages, int(end_ratio * total_pages))
#     return start, end

# def parse_sections(text):
#     pattern = re.compile(r'(?:^|\n)(\d+(?:\.\d+)*\.)\s+([^\n]+(?:\n(?!\d+(?:\.\d+)*\.)[^\n]*)*)', re.MULTILINE)
#     matches = list(pattern.finditer(text))
#     if not matches:
#         return [{"section_number": "1", "section_title": "전체 내용", "parent_section_id": None, "section_level": 1, "content": text.strip()}]
#     chunks = []
#     for i, match in enumerate(matches):
#         section_number_raw = match.group(1)
#         section_number = section_number_raw.rstrip('.')
#         full_match = match.group(0).strip()
#         level = section_number.count('.') + 1
#         if level == 1:
#             parent_section_id = None
#         else:
#             parts = section_number.split('.')
#             parent_parts = parts[:-1]
#             parent_section_id = '.'.join(parent_parts)
#         first_line = match.group(2).split('\n')[0].strip()
#         if len(first_line) > 50:
#             title_part = first_line[:47] + "..."
#         else:
#             title_part = first_line
#         section_title = f"{section_number}. {title_part}"
#         chunks.append({
#             'section_number': section_number,
#             'section_title': section_title,
#             'parent_section_id': parent_section_id,
#             'section_level': level,
#             'content': full_match
#         })
#     return chunks

# def process_single_pdf(pdf_path):
#     global _global_processor
#     pdf_path = Path(pdf_path)
#     print(f"Processing {pdf_path.name}...")
#     loader = PyPDFLoader(str(pdf_path))
#     pages = loader.load()
#     full_text = '\n'.join([clean_text(page.page_content) for page in pages])
#     doc_title = extract_document_title(full_text)
#     if not doc_title:
#         doc_title = pdf_path.stem
#     if not full_text.strip():
#         print(f"Warning: No text content in {pdf_path.name}")
#         return []
#     doc_id = create_doc_id(str(pdf_path))
#     metadata = extract_metadata(full_text, pdf_path.name)
#     jurisdiction = infer_jurisdiction(str(pdf_path))
#     sections = parse_sections(full_text)
#     records = []
#     for idx, section_data in enumerate(sections):
#         start_page, end_page = estimate_page_range(section_data['content'], full_text, len(pages), idx, [s['content'] for s in sections])
#         record = ChunkRecord(
#             id=f"{doc_id}-{idx:04}",
#             doc_id=doc_id,
#             source_path=str(pdf_path),
#             title=doc_title,
#             jurisdiction=jurisdiction,
#             doc_date=metadata['doc_date'],
#             doc_version=metadata['doc_version'],
#             section_id=section_data['section_number'],
#             section_title=section_data['section_title'],
#             parent_section_id=section_data['parent_section_id'],
#             section_level=section_data['section_level'],
#             page_start=start_page,
#             page_end=end_page,
#             chunk_index=idx,
#             text=section_data['content']
#         )
#         records.append(record)
#     return records

# def find_pdfs(directory):
#     directory = Path(directory)
#     return sorted(directory.rglob('*.pdf'))

# def write_jsonl(records, output_path, append=False):
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     mode = 'a' if append else 'w'
#     with open(output_path, mode, encoding='utf-8') as f:
#         for record in records:
#             f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
#     print(f"{'Appended' if append else 'Saved'} {len(records)} chunks to {output_path}")

# def process_pdfs_parallel(pdf_dir, output_path,
#                          model_name="jhgan/ko-sroberta-multitask",
#                          device="cpu",
#                          breakpoint_threshold_type="percentile",
#                          breakpoint_amount=70,
#                          max_workers=None,
#                          missing_files=None):
#     pdf_files = list(Path(pdf_dir).glob('*.pdf'))
#     print(f"▶ 전체 PDF 파일명 목록: {[p.name for p in pdf_files]}")
#     print(f"▶ 넘겨받은 누락 파일 리스트: {missing_files}")
#     if missing_files is not None:
#         pdf_files = [p for p in pdf_files if p.name in missing_files]
#         print(f"▶ 필터링 후 처리할 PDF 파일명: {[p.name for p in pdf_files]}")
#     if not pdf_files:
#         print("처리할 PDF 파일이 없습니다.")
#         return
#     global _global_processor
#     _global_processor = SemanticPDFProcessor(model_name, device,
#                                              breakpoint_threshold_type,
#                                              breakpoint_amount)
#     if max_workers is None:
#         max_workers = min(mp.cpu_count(), len(pdf_files))
#     records = []
#     with ProcessPoolExecutor(max_workers=max_workers,
#                             initializer=init_worker,
#                             initargs=(model_name, device,
#                                       breakpoint_threshold_type,
#                                       breakpoint_amount)) as executor:
#         future_to_pdf = {executor.submit(process_single_pdf, str(pdf)): pdf for pdf in pdf_files}
#         processed_count = 0
#         for future in as_completed(future_to_pdf):
#             pdf_path = future_to_pdf[future]
#             try:
#                 new_records = future.result()
#                 records.extend(new_records)
#                 processed_count += 1
#                 print(f"Processed {processed_count}/{len(pdf_files)} PDFs. Total chunks: {len(records)}")
#             except Exception as e:
#                 print(f"Error processing {pdf_path}: {e}")
#     append_mode = missing_files is not None
#     write_jsonl(records, output_path, append=append_mode)
#     print("처리 완료.")

# def init_worker(model_name, device, breakpoint_threshold_type, breakpoint_amount):
#     global _global_processor
#     _global_processor = SemanticPDFProcessor(model_name, device, breakpoint_threshold_type, breakpoint_amount)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pdf_dir", type=str, required=True, help="폴더 내 PDF 파일 경로")
#     parser.add_argument("--output", type=str, required=True, help="출력 JSONL 파일 경로")
#     parser.add_argument("--model_name", type=str, default="jhgan/ko-sroberta-multitask")
#     parser.add_argument("--device", type=str, default="cpu")
#     parser.add_argument("--threshold_type", type=str, default="percentile")
#     parser.add_argument("--threshold_amount", type=int, default=70)
#     parser.add_argument("--max_workers", type=int, default=None)
#     parser.add_argument("--missing_files", type=str, default=None,
#                         help="콤마(,)로 구분된 누락 PDF 파일명 리스트")
#     args = parser.parse_args()
#     if args.missing_files:
#         missing_files = [f.strip() for f in args.missing_files.split(',') if f.strip()]
#     else:
#         missing_files = None
#     process_pdfs_parallel(pdf_dir=args.pdf_dir,
#                          output_path=args.output,
#                          model_name=args.model_name,
#                          device=args.device,
#                          breakpoint_threshold_type=args.threshold_type,
#                          breakpoint_amount=args.threshold_amount,
#                          max_workers=args.max_workers,
#                          missing_files=missing_files)
