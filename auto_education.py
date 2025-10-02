"""
GMP 교육자료 및 평가문항 생성 모듈 - 백엔드 통합용

사용법:
    from auto_education import GMPTrainingService
    
    service = GMPTrainingService()
    result = service.generate_training_package(
        sop_content=sop_변수,
        guideline_content=가이드라인_변수,
        num_questions=15
    )
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import openai

# .env 파일에서 API KEY 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class GMPTrainingService:
    """
    백엔드 API에서 사용할 GMP 교육자료 생성 서비스
    
    특징:
    - 파일 I/O 없이 메모리에서만 작동
    - 딕셔너리 형태로 결과 반환 (JSON 직렬화 가능)
    - 에러 핸들링 포함
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Args:
            model: 사용할 OpenAI 모델
        """
        if not openai.api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                "환경변수 또는 .env 파일에서 설정하세요."
            )
        
        self.model = model
        self.client = openai.OpenAI(api_key=openai.api_key)
    
    def generate_training_material(
        self, 
        sop_content: str, 
        guideline_content: str,
        target_audience: str = "GMP 실무자"
    ) -> Dict:
        """
        교육자료 생성
        
        Args:
            sop_content: SOP 텍스트
            guideline_content: 가이드라인 텍스트
            target_audience: 교육 대상자
        
        Returns:
            교육자료 딕셔너리
        """
        
        prompt = f"""
당신은 제약업계 GMP 전문 교육 콘텐츠 개발자입니다.

아래 SOP와 21 CFR Part 211 가이드라인을 분석하여 {target_audience}를 위한 교육자료를 생성하세요.

=== SOP 내용 ===
{sop_content}

=== 가이드라인 내용 ===
{guideline_content}

다음 JSON 형식으로 작성하세요:

{{
    "title": "교육 과정 제목",
    "overview": "교육 과정 개요 (3-4문장)",
    "learning_objectives": [
        "학습목표 1", "학습목표 2", "학습목표 3", 
        "학습목표 4", "학습목표 5", "학습목표 6"
    ],
    "regulatory_background": {{
        "regulation": "21 CFR Part 211 관련 조항",
        "key_requirements": ["요구사항 1", "요구사항 2", "요구사항 3", "요구사항 4", "요구사항 5"]
    }},
    "key_concepts": [
        {{
            "section": "섹션명",
            "concept": "개념 제목",
            "description": "상세 설명",
            "importance": "중요도",
            "practical_tips": ["팁 1", "팁 2", "팁 3"]
        }}
    ],
    "procedures": [
        {{
            "procedure_name": "절차명",
            "steps": [
                {{
                    "step_number": 1,
                    "action": "작업",
                    "details": "상세 설명",
                    "cautions": "주의사항",
                    "related_regulation": "관련 규정"
                }}
            ]
        }}
    ],
    "quality_points": ["포인트 1", "포인트 2", "포인트 3", "포인트 4", "포인트 5"],
    "common_mistakes": [
        {{
            "mistake": "실수 내용",
            "consequence": "결과",
            "prevention": "예방법"
        }}
    ],
    "case_studies": [
        {{
            "scenario": "시나리오",
            "correct_action": "올바른 조치",
            "rationale": "근거"
        }}
    ],
    "summary": "요약 (4-5문장)",
    "references": ["참조 1", "참조 2"]
}}

요구사항:
- SOP의 모든 섹션 커버
- key_concepts 최소 5개
- procedures 단계별 상세 기술
- common_mistakes 최소 3-4개
- case_studies 최소 2-3개
- quality_points 최소 5개
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "GMP 전문 교육자료 개발자"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise Exception(f"교육자료 생성 실패: {str(e)}")
    
    def generate_assessment_questions(
        self,
        training_material: Dict,
        sop_content: str,
        guideline_content: str,
        num_questions: int = 15
    ) -> List[Dict]:
        """
        평가문항 생성
        
        Args:
            training_material: 생성된 교육자료
            sop_content: SOP 텍스트
            guideline_content: 가이드라인 텍스트
            num_questions: 문항 수
        
        Returns:
            평가문항 리스트
        """
        
        # 난이도 자동 계산
        easy_count = max(1, round(num_questions * 0.3))
        medium_count = max(1, round(num_questions * 0.5))
        hard_count = max(1, num_questions - easy_count - medium_count)
        
        # 교육자료 핵심 내용 추출
        training_summary = {
            "title": training_material.get("title", ""),
            "learning_objectives": training_material.get("learning_objectives", []),
            "key_concepts": [
                {"concept": kc.get("concept", ""), "importance": kc.get("importance", "")} 
                for kc in training_material.get("key_concepts", [])[:5]
            ]
        }
        
        prompt = f"""
GMP 전문 평가문항 개발자로서 총 {num_questions}개의 평가문항을 생성하세요.

=== 교육자료 ===
{json.dumps(training_summary, ensure_ascii=False, indent=2)}

=== SOP 참조 ===
{sop_content[:5000]}

=== 가이드라인 참조 ===
{guideline_content[:5000]}

난이도 분배:
- 쉬움: {easy_count}개 (기본 개념, 용어 정의)
- 보통: {medium_count}개 (절차 적용, 상황 판단)
- 어려움: {hard_count}개 (실무 시나리오, 문제 해결)

커버할 주제:
1. 포장·표시 자재 수령·검수·승인
2. 라벨 발행·사용·회수
3. 유효기간 관리
4. 반품·회수 절차 (211.204, 211.208)
5. 혼입·교차오염 방지
6. 라벨 부착 장치 검증
7. QA/QC/QCU 역할
8. 포장기록 검토·승인

JSON 형식:
{{
    "questions": [
        {{
            "question_number": 1,
            "difficulty": "easy|medium|hard",
            "question_type": "multiple_choice|true_false|scenario_based",
            "question_text": "질문",
            "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
            "correct_answer": "정답",
            "explanation": "해설",
            "related_concept": "관련 개념",
            "regulation_reference": "규정 조항",
            "points": 10,
            "learning_objective_mapped": "학습목표"
        }}
    ]
}}

요구사항:
- 교육자료 기반 출제
- 모든 주제 골고루 커버
- 상세한 해설 포함
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "GMP 전문 평가문항 개발자"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])
            
        except Exception as e:
            raise Exception(f"평가문항 생성 실패: {str(e)}")
    
    def generate_training_package(
        self,
        sop_content: str,
        guideline_content: str,
        target_audience: str = "GMP 실무자",
        num_questions: int = 15
    ) -> Dict:
        """
        교육 패키지 생성 (교육자료 + 평가문항)
        
        백엔드 API에서 직접 호출할 메인 메서드
        
        Args:
            sop_content: SOP 텍스트 (변수에서 가져온 내용)
            guideline_content: 가이드라인 텍스트 (변수에서 가져온 내용)
            target_audience: 교육 대상자
            num_questions: 평가문항 수
        
        Returns:
            {
                "status": "success" | "error",
                "data": {
                    "training_material": {...},
                    "assessment": {
                        "questions": [...],
                        "metadata": {...}
                    }
                },
                "error": "에러 메시지 (실패 시)"
            }
        """
        
        try:
            # 1. 교육자료 생성
            training_material = self.generate_training_material(
                sop_content=sop_content,
                guideline_content=guideline_content,
                target_audience=target_audience
            )
            
            # 2. 평가문항 생성
            questions = self.generate_assessment_questions(
                training_material=training_material,
                sop_content=sop_content,
                guideline_content=guideline_content,
                num_questions=num_questions
            )
            
            # 3. 메타데이터 계산
            difficulty_dist = {
                "easy": sum(1 for q in questions if q.get("difficulty") == "easy"),
                "medium": sum(1 for q in questions if q.get("difficulty") == "medium"),
                "hard": sum(1 for q in questions if q.get("difficulty") == "hard")
            }
            
            question_types = {
                "multiple_choice": sum(1 for q in questions if q.get("question_type") == "multiple_choice"),
                "true_false": sum(1 for q in questions if q.get("question_type") == "true_false"),
                "scenario_based": sum(1 for q in questions if q.get("question_type") == "scenario_based")
            }
            
            total_points = sum(q.get("points", 10) for q in questions)
            
            # 4. 결과 패키징
            return {
                "status": "success",
                "data": {
                    "training_material": {
                        "id": f"training_{hash(training_material['title']) % 10000}",
                        "title": training_material["title"],
                        "overview": training_material.get("overview", ""),
                        "target_audience": target_audience,
                        "learning_objectives": training_material["learning_objectives"],
                        "regulatory_background": training_material.get("regulatory_background", {}),
                        "key_concepts": training_material["key_concepts"],
                        "procedures": training_material["procedures"],
                        "quality_points": training_material.get("quality_points", []),
                        "common_mistakes": training_material.get("common_mistakes", []),
                        "case_studies": training_material.get("case_studies", []),
                        "summary": training_material["summary"],
                        "references": training_material["references"]
                    },
                    "assessment": {
                        "id": f"assessment_{hash(training_material['title']) % 10000}",
                        "total_questions": len(questions),
                        "total_points": total_points,
                        "passing_score": total_points * 0.7,
                        "difficulty_distribution": difficulty_dist,
                        "question_types": question_types,
                        "questions": questions
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": None
            }


# ============================================================================
# FastAPI 통합 예제
# ============================================================================

"""
FastAPI 사용 예제:

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from auto_education import GMPTrainingService

app = FastAPI()
service = GMPTrainingService(model="gpt-3.5-turbo")

class TrainingRequest(BaseModel):
    sop_content: str
    guideline_content: str
    target_audience: str = "GMP 실무자"
    num_questions: int = 15

@app.post("/api/training/generate")
def generate_training(request: TrainingRequest):
    result = service.generate_training_package(
        sop_content=request.sop_content,
        guideline_content=request.guideline_content,
        target_audience=request.target_audience,
        num_questions=request.num_questions
    )
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result["data"]

# 사용법:
# POST /api/training/generate
# {
#     "sop_content": "SOP 내용...",
#     "guideline_content": "가이드라인 내용...",
#     "num_questions": 15
# }
"""


# ============================================================================
# Flask 통합 예제
# ============================================================================

"""
Flask 사용 예제:

from flask import Flask, request, jsonify
from auto_education import GMPTrainingService

app = Flask(__name__)
service = GMPTrainingService(model="gpt-3.5-turbo")

@app.route('/api/training/generate', methods=['POST'])
def generate_training():
    data = request.get_json()
    
    result = service.generate_training_package(
        sop_content=data.get('sop_content'),
        guideline_content=data.get('guideline_content'),
        target_audience=data.get('target_audience', 'GMP 실무자'),
        num_questions=data.get('num_questions', 15)
    )
    
    if result["status"] == "error":
        return jsonify({"error": result["error"]}), 500
    
    return jsonify(result["data"])

# 사용법:
# POST /api/training/generate
# Content-Type: application/json
# {
#     "sop_content": "SOP 내용...",
#     "guideline_content": "가이드라인 내용...",
#     "num_questions": 15
# }
"""


# ============================================================================
# 간단한 사용 예제
# ============================================================================

if __name__ == "__main__":
    # 백엔드 작업자가 사용할 예제
    
    service = GMPTrainingService(model="gpt-3.5-turbo")
    
    # 변수에 저장된 SOP와 가이드라인
    sop_text = "여기에 SOP 내용..."
    guideline_text = "여기에 가이드라인 내용..."
    
    # 교육 패키지 생성
    result = service.generate_training_package(
        sop_content=sop_text,
        guideline_content=guideline_text,
        target_audience="제조부서 실무자",
        num_questions=15
    )
    
    # 결과 확인
    if result["status"] == "success":
        print("성공!")
        print(f"교육자료 제목: {result['data']['training_material']['title']}")
        print(f"평가문항 수: {result['data']['assessment']['total_questions']}")
    else:
        print(f"실패: {result['error']}")