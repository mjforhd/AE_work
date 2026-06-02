# CI/CD 연계 설계 (PART 2 요구사항 ⑤)

본 테스트 스위트를 GitHub Actions / GitLab CI에 연계하기 위한 설계 문서.

**워크플로 파일 위치:**

| 플랫폼 | 파일 경로 | 비고 |
|---|---|---|
| GitHub Actions | `.github/workflows/qa-pipeline.yml` | push 시 GitHub이 자동 인식 |
| GitLab CI | `ci/gitlab-ci.yml` | Settings › CI/CD › Custom CI config 경로 지정 필요 |

---

## 1. 파이프라인 구조

```
[PR / MR]           [main 병합 / 배포 전]    [nightly 정기 스케줄]
    │                       │                         │
unit-test ──────────► unit-test ──────────────► unit-test
    │                       │                         │
    ✓ (여기서 종료)    integration-test ──────► integration-test
                            │                   integration-test-full (slow)
                            │                         │
                       benchmark ──────────────► benchmark
                                                      │
                                                 llm-judge
```

### 트리거 조건

| 트리거 | 실행 범위 | 목적 |
|---|---|---|
| **PR / Merge Request** | `unit-test` 만 | 빠른 피드백. API 비용 없이 프레임워크 회귀 차단 |
| **main 병합 / 배포 전** | `unit-test` → `integration-test` → `benchmark` | 실제 LLM 호출 회귀 검증 + 비교 리포트 |
| **정기 스케줄 (nightly)** | 전체 (slow 포함) + `llm-judge` | 벤더 무음 업데이트로 인한 Drift 조기 감지 (PART 1 §8.3) |

> **LLM 호출 비용 분리 원칙**: API 호출(`@pytest.mark.api`)은 main 병합·nightly에서만 실행.
> PR 단계에서는 API 없는 단위 테스트(`-m "not api"`)만 돌려 속도·비용을 절감한다.

---

## 2. 테스트 실패 시 배포 차단 조건 (Quality Gate)

PART 1 §7(품질 게이트)을 CI 차단 규칙으로 매핑한다.

| 결과 | 조치 |
|---|---|
| **Blocking 항목 FAIL** (보안, 스트리밍 무결성, 응답 성공률, finish_reason 정합, Critical Hallucination) | `integration-test` job 실패 → exit code ≠ 0 → **배포 단계 자동 차단 (NO-GO)** |
| **Non-Blocking 항목 WARNING** (TTFT 초과, 자연스러움 추세 등) | 파이프라인 통과 + 아티팩트에 기록 → **HOLD 후보로 추적** |
| 벤치마크 지표 직전 Baseline 대비 **−10% 이상 하락** | `benchmark` job 실패 → Drift 의심 알림 발송 |

구현 방식: `pytest` exit code가 0이 아니면 GitHub Actions / GitLab CI job이 실패(빨간불)로 표시되고, 이후 `deploy` 스테이지가 자동 차단된다. 별도의 커스텀 스크립트 없이 pytest의 기본 동작으로 게이트를 구현한다.

---

## 3. 테스트 결과 알림 채널 및 형식

| 채널 | 트리거 | 내용 |
|---|---|---|
| **Slack Webhook** | `integration-test` 실패 (NO-GO) | 브랜치·커밋·파이프라인 링크 포함 실패 요약 |
| **Slack Webhook** | `benchmark` 실패 (HOLD 후보) | 이상 감지 경고 + 리포트 링크 |
| **JUnit XML 리포트** | 모든 실행 | `reports/junit-*.xml` → CI UI에 테스트별 Pass/Fail 표시 |
| **benchmark_results.json** | main 병합·nightly | `reports/benchmark_results.json` → 아티팩트 90일 보관 |
| **llm-judge.log** | nightly | `reports/llm-judge.log` → Judge 점수 + Human Review 플래그 기록 |

### Slack 알림 설정 방법

| 플랫폼 | 설정 위치 |
|---|---|
| GitHub | Repository › Settings › Secrets and variables › Actions 에서 `SLACK_WEBHOOK_URL` 추가 |
| GitLab | Settings › CI/CD › Variables 에서 `SLACK_WEBHOOK_URL` 추가 (Masked 체크) |

---

## 4. 비밀(Secret) 관리

API 키와 Webhook URL은 코드에 포함하지 않고 CI/CD Secrets에서 환경 변수로 주입한다.

| 변수명 | 용도 | 설정 위치 |
|---|---|---|
| `OPENAI_API_KEY` | GPT-4o 호출 | CI Secrets |
| `ANTHROPIC_API_KEY` | Claude 호출 | CI Secrets |
| `GOOGLE_API_KEY` | Gemini 호출 | CI Secrets |
| `SLACK_WEBHOOK_URL` | 알림 Webhook | CI Secrets (Masked) |

로컬 개발 환경은 `.env` 파일(`.gitignore`에 포함)을 사용한다.

---

## 5. 아티팩트 보관 정책

| 아티팩트 | 보관 기간 | 목적 |
|---|---|---|
| JUnit XML (`junit-*.xml`) | 30일 | CI UI 테스트 결과 확인 |
| 벤치마크 JSON (`benchmark_results.json`) | **90일** | 시계열 Baseline/Drift 추적 (PART 1 §8) |
| Judge 로그 (`llm-judge.log`) | 30일 | Human Review 플래그 확인 |

> 벤치마크 JSON을 90일 보관하는 이유: PART 1 §8.3의 "Drift 자동 감지"를 위해 직전 측정값과 비교해야 하기 때문이다. CI에서는 아티팩트 히스토리를 Baseline으로 활용한다.

---

## 6. 확장 고려사항

- **모델 추가**: `config/settings.py`의 `SUPPORTED_MODELS`와 `MODEL_API_ID`에 한 줄 추가하면 parametrize 테스트·비교 리포트가 자동 확장된다.
- **프롬프트 추가**: `docs/prompt_set.json`만 수정. 코드 변경 불필요.
- **LLM Judge 비용 절감**: Judge는 nightly 스케줄에서만 실행. 배포 게이트에는 포함하지 않는다.
- **다중 환경(dev/staging/prod)**: GitLab의 `environment` 키워드 또는 GitHub Environments를 활용해 각 환경별 임계값을 다르게 적용할 수 있다.
