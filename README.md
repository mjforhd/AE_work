# A Chat QA Automation Suite

A사 **A Chat** 서비스(GPT-4o · Claude · Gemini 통합 AI)의 품질을 자동 검증하는 테스트 스위트입니다.  
**PART 1 품질 기준 정의서**의 지표 체계를 코드로 구현했으며, CI/CD 파이프라인 연계까지 포함합니다.

> **Note** 본 프로젝트에 등장하는 회사명 및 서비스명(A사, A Chat 등)은 보안상의 이유로 실제 명칭을 임의로 변경한 것입니다.

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [빠른 시작](#2-빠른-시작)
3. [테스트 실행](#3-테스트-실행)
4. [벤치마크 리포트](#4-벤치마크-리포트)
5. [LLM Judge 채점](#5-llm-judge-채점-가산점)
6. [프롬프트 관리](#6-프롬프트-관리)
7. [CI/CD 연계](#7-cicd-연계)
8. [출력 해석](#8-출력-해석)
9. [PART 1 품질 기준 대응표](#9-part-1-품질-기준-대응표)

---

## 1. 프로젝트 구조

```
AE_work/
│
├── .github/workflows/
│   └── qa-pipeline.yml      # GitHub Actions (PR·main·nightly 자동 실행)
│
├── ci/
│   ├── gitlab-ci.yml        # GitLab CI 파이프라인
│   └── CICD_DESIGN.md       # CI/CD 연계 설계 문서
│
├── config/
│   └── settings.py          # API 키·성능 임계값(TTFT SLA 등) 설정
│
├── core/
│   ├── client.py            # AChatClient — GPT/Claude/Gemini 통합 어댑터
│   └── validator.py         # ResponseValidator — 스키마·성능·품질 검증
│
├── tests/
│   ├── conftest.py          # 픽스처, API 키 자동 스킵, 크레딧 부족 자동 스킵
│   ├── test_basic.py        # 기본 검증 17개
│   ├── test_edge.py         # 경계값·예외 검증 18개
│   ├── test_streaming.py    # 스트리밍 검증 12개
│   └── test_performance.py  # 성능 검증 8개
│
├── judge/
│   └── llm_judge.py         # LLM Judge — Cross-evaluation 자동 채점 (가산점)
│
├── reports/
│   └── reporter.py          # 5개 프롬프트 × 3개 모델 비교 벤치마크
│
├── docs/
│   └── prompt_set.json      # 벤치마크·Judge 프롬프트 셋 (코드 분리 관리)
│
├── .gitignore               # .env·자동생성 파일 제외
├── pytest.ini
├── README.md
└── requirements.txt
```

### 지원 모델

| A Chat 모델명 | API 모델 ID | 벤더 |
|---|---|---|
| `gpt-4o` | `gpt-4o` | OpenAI |
| `claude-sonnet-4-6` | `claude-sonnet-4-6` | Anthropic |
| `gemini-2.0-flash` | `gemini-2.5-flash` | Google |

---

## 2. 빠른 시작

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) `.env` 파일 생성

프로젝트 루트에 `.env`를 만들고 API 키를 입력합니다.

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

> 키가 없거나 크레딧이 부족한 모델의 테스트는 **FAIL 대신 SKIP**으로 자동 처리됩니다.

### 3) 실행

```bash
python -m pytest          # 전체 실행
python -m pytest -v -s    # 상세 출력
```

> `pytest` 명령어가 안 된다면 `python -m pytest`를 사용하세요. (Windows PATH 미등록 시 발생)

---

## 3. 테스트 실행

### 분류별 실행

```bash
python -m pytest -m "not api"           # API 없는 단위 테스트 (빠름)
python -m pytest -m "api and not slow"  # 통합 테스트 (API 호출)
python -m pytest -m "api"              # 전체 (slow 포함)
```

### 파일별 실행

```bash
python -m pytest tests/test_basic.py        # 기본 검증        17개
python -m pytest tests/test_edge.py         # 경계값·예외      18개
python -m pytest tests/test_streaming.py    # 스트리밍         12개
python -m pytest tests/test_performance.py  # 성능             8개
```

### 테스트 항목 상세

#### `test_basic.py` — 기본 검증 (17개)

| 테스트 함수 | 검증 내용 |
|---|---|
| `test_normal_response_schema` | 3개 모델 정상 응답 + A Chat 스키마 일치 확인 |
| `test_auth_failure` | 잘못된 API 키 → 401/403 오류 처리 |
| `test_missing_messages_empty_list` | 빈 messages 리스트 → 400 오류 |
| `test_missing_messages_none` | messages=None → 400 오류 |
| `test_unsupported_model` | `gpt-99` 등 5종 미지원 모델 → 명확한 오류 반환 |

#### `test_edge.py` — 경계값·예외 검증 (18개)

| 테스트 함수 | 검증 내용 |
|---|---|
| `test_max_tokens_1_finish_reason_length` | `max_tokens=1` → `finish_reason: "length"` 반드시 반환 |
| `test_max_tokens_small_values` | 소규모 max_tokens(5·10·50) 경계 동작 |
| `test_empty_message_content_behavior` | `content: ""` 빈 메시지 → 모델별 동작 비교 |
| `test_whitespace_only_content_behavior` | 공백만 있는 content → 모델별 동작 |
| `test_no_system_prompt_default_behavior` | 시스템 프롬프트 없을 때 모델별 기본 동작 |
| `test_system_prompt_vs_no_system_prompt` | 시스템 프롬프트 유무에 따른 응답 차이 비교 |

#### `test_streaming.py` — 스트리밍 검증 (12개)

| 테스트 함수 | TC ID | 검증 내용 |
|---|---|---|
| `test_stream_chunk_structure` | TC-STR-001/002 | SSE 청크 순서·구조, 마지막 청크 `finish_reason` |
| `test_stream_merged_semantic_equivalence` | TC-STR-004 | 병합↔비스트리밍 **임베딩 기반** Semantic Stability ≥ 95% |
| `test_stream_abort_then_next_request` | TC-STR-003 | abort 후 동일 클라이언트 다음 요청 정상 처리 |
| `test_stream_chunk_count_increases` | — | 청크 2개 이상 수신 (콘텐츠 청크 + finish 청크) |

#### `test_performance.py` — 성능 검증 (8개)

| 테스트 함수 | 검증 내용 |
|---|---|
| `test_concurrent_model_requests` | 3개 모델 동시 요청 → 응답시간 분포 + 병렬 효율 |
| `test_temperature_variation` | temperature 0.0/0.7/1.5 → 응답시간·토큰 변화 측정 |
| `test_ttft_per_model` | 모델별 TTFT 단일 측정 + SLA 검증 |
| `test_ttft_comparison` | 5회 반복 → p50/p95/p99 분위수 SLA 비교 (PART 1 §2.1) |

---

## 4. 벤치마크 리포트

5개 프롬프트를 3개 모델에 전송하고 성능·품질 수치를 자동 비교합니다.

```bash
python -m reports.reporter

# 옵션
python -m reports.reporter --output results/report.json
python -m reports.reporter --max-tokens 300
```

### 측정 항목

| 항목 | SLA | 비고 |
|---|---|---|
| 평균 응답시간 ± std | — | 참고 지표 |
| **응답시간 p95** | ≤ 5s | PART 1 §2.1 |
| **TTFT p50 / p95 / p99** | ≤ 1s / 2s / 3s | PART 1 §2.1 |
| 평균 완료 토큰 ± std | — | `completion_tokens` 기준 |
| finish_reason 분포 | `stop` ≥ 98% | Blocking 게이트 |
| 한국어 응답 비율 | ≥ 95% | P3·P5는 낮음이 정상 |

콘솔에 Rich 테이블 + Blocking 게이트 패널이 출력되고, `reports/benchmark_results.json`이 자동 저장됩니다.

---

## 5. LLM Judge 채점 (가산점)

3개 모델 응답을 **Cross-evaluation** 방식으로 자동 채점합니다.  
Judge를 3회 실행해 **기준별 Median**을 반환하고, `stdev > 0.5` 기준은 Human Review Required로 로깅합니다.

```bash
python -m judge.llm_judge
python -m judge.llm_judge --prompts 3 --consistency-runs 3
```

### Self-evaluation 편향 방지 (4중 전략)

| 방법 | 내용 |
|---|---|
| **Cross-evaluation** | 피평가 모델 ≠ 심사 모델. Claude 응답 → GPT 심사 / GPT·Gemini 응답 → Claude 심사 |
| **Blind Review** | 모델명을 Judge 프롬프트에서 제거해 텍스트 품질만으로 채점 |
| **Few-shot 앵커링** | 우수·오류·장황 예시 3개로 채점 기준 고정. 스타일이 아닌 품질에 집중 |
| **비결정성 통제** | n=3회 실행 → Median 채택. stdev > 0.5 시 `Human Review Required` 로깅 |

### 채점 기준 (각 1~5점)

정확성 · 유용성 · 자연스러움 · 간결성

---

## 6. 프롬프트 관리

벤치마크·Judge 평가용 프롬프트는 **`docs/prompt_set.json`** 에서 관리합니다.  
코드를 수정하지 않고 프롬프트를 추가·변경할 수 있습니다.

```jsonc
{
  "benchmark_prompts": [
    {
      "id": "P1",
      "label": "지시 이행률·시스템 프롬프트 준수",
      "qa_intent": "...",
      "tags": ["instruction_following"],
      "expected_korean_low": false,
      "text": "프롬프트 내용..."
    }
  ],
  "judge_prompts": [{ "id": "J1", "text": "..." }]
}
```

### 현재 벤치마크 프롬프트 5개 — A Chat 고유 리스크 탐지 목적

| ID | QA 검증 의도 | `expected_korean_low` |
|---|---|---|
| P1 | 지시 이행률 + 시스템 프롬프트 준수 (한국어·하십시오체·3문장 강제) | ✗ |
| P2 | 복합 제약 조건 동시 충족 (제목/본문·핵심가치·마크다운) | ✗ |
| P3 | 비결정성 환경 JSON 출력 포맷 파싱 안정성 | **✓** (JSON이라 낮음이 정상) |
| P4 | 결함 A 시나리오 — 파일 파싱·요약 스트레스 테스트 | ✗ |
| P5 | 결함 B 시나리오 — Prompt Injection 및 보안성 검증 | **✓** (거부 응답이라 낮음이 정상) |

---

## 7. CI/CD 연계

설계 상세는 **`ci/CICD_DESIGN.md`** 를 참고하세요.

### 파이프라인 파일

| 플랫폼 | 파일 |
|---|---|
| **GitHub Actions** | `.github/workflows/qa-pipeline.yml` (push 시 자동 인식) |
| **GitLab CI** | `ci/gitlab-ci.yml` (Settings › CI/CD › Custom config path 지정 필요) |

### 파이프라인 흐름

```
PR/MR          →  단위 테스트만   (API 없음, 빠름)
main 병합      →  단위 → 통합(Blocking 게이트) → 벤치마크 리포트
nightly 01:00  →  전체(slow 포함) → 벤치마크 → LLM Judge
```

- **통합 테스트 실패** → Slack 알림 + 배포 자동 차단 (NO-GO)
- **벤치마크 이상** → Slack 알림 + HOLD 후보로 추적

### GitHub Secrets 설정

| Secret | 내용 |
|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 |
| `ANTHROPIC_API_KEY` | Anthropic API 키 |
| `GOOGLE_API_KEY` | Google AI Studio API 키 |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL (알림용, 선택) |

---

## 8. 출력 해석

### 테스트 결과 레벨

| 레벨 | 의미 | 대응 |
|---|---|---|
| `PASS` 🟢 | 기준 충족 | — |
| `WARNING` 🟡 | 기준 초과이나 허용 범위 내 | 추적 필요 |
| `FAIL` 🔴 | 기준 명백 초과 | 즉각 대응 |
| `SKIP` ⚪ | API 키 없거나 크레딧 부족 | 충전·설정 후 재실행 |

### Blocking 게이트 판정

벤치마크 리포트 하단에 표시됩니다.

| 판정 | 조건 |
|---|---|
| **GO** | Blocking 전체 통과 + Non-Blocking 기준 충족 |
| **HOLD** | Blocking 통과, Non-Blocking 일부 미달 → 원인 분석 후 재평가 |
| **NO-GO** | Blocking 항목 1건이라도 위반 → 즉시 배포 차단 |

### TTFT 분위수 해석

```
  Model                    p50(s)  p95(s)  p99(s)
  ─────────────────────────────────────────────
  claude-sonnet-4-6         0.41    0.89    0.95   ✓ 전체 SLA 충족
  gpt-4o                    0.62    1.24    1.38   ⚠ p95·p99 초과
  gemini-2.0-flash          0.88    1.90    2.10   ✗ p99 초과
```

SLA 기준: p50 ≤ 1s / p95 ≤ 2s / p99 ≤ 3s (PART 1 §2.1)

### Semantic Stability 해석

```
[TC-STR-004][gpt-4o] Semantic Stability 100.0% (method=openai_embedding)
```

- `openai_embedding`: OpenAI `text-embedding-3-small` 기반 측정 (정밀)
- `keyword_overlap_fallback`: OpenAI API 불가 시 자동 대체

합격 기준: ≥ 95% (PART 1 §5.1)

---

## 9. PART 1 품질 기준 대응표

### Layer 3 — Platform Reliability (자동화)

| 지표 | 합격 기준 | 구현 | 게이트 |
|---|---|---|---|
| 스트리밍 정합성 | 100% | `validate_schema(is_chunk=True)` | **Blocking** |
| 응답 완결성 | `stop` ≥ 98% | `reporter._display_blocking_gate()` | **Blocking** |
| TTFT p50/p95/p99 | 1s / 2s / 3s | `validate_performance_distribution()` | Non-Blocking |
| 전체 응답 p95 | ≤ 5s | `validate_performance_distribution()` | Non-Blocking |

### Layer 2 — Context Compliance (자동화 + Judge)

| 지표 | 합격 기준 | 구현 |
|---|---|---|
| Semantic Stability | ≥ 95% | `validate_semantic_stability()` |
| finish_reason 정합성 | content 길이 일관 | `validate_finish_reason(max_tokens=...)` |
| 한국어 응답 비율 | ≥ 95% | `validate_content_quality()` |

### Layer 1 — Base Model Quality (Judge + Human)

| 지표 | 구현 |
|---|---|
| 정확성·유용성·자연스러움·간결성 (1~5점) | `LLMJudge.score()` — 3회 Median |
| Judge 일관성 stdev ≤ 0.5 | `LLMJudge.validate_consistency()` |
| Self-evaluation 편향 방지 | `LLMJudge._select_judge()` — Cross-evaluation |
