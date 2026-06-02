from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    @field_validator("openai_api_key", "anthropic_api_key", "google_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    # ── 단일 샘플 응답 시간 임계값 (per-request validate_performance 용) ──
    response_time_warning_s: float = 5.0   # p95 SLA 초과 → Warning
    response_time_error_s: float = 8.0     # 명백한 초과 → Error
    ttft_warning_s: float = 1.0            # p50 SLA 초과 → Warning (단일 샘플)
    ttft_error_s: float = 2.0             # p95 SLA 초과 → Error  (단일 샘플)

    # ── TTFT 분위수 SLA (PART1 2.1) — validate_performance_distribution 용 ──
    ttft_p50_s: float = 1.0   # TTFT p50 ≤ 1s
    ttft_p95_s: float = 2.0   # TTFT p95 ≤ 2s
    ttft_p99_s: float = 3.0   # TTFT p99 ≤ 3s

    # ── 전체 응답 시간 분위수 SLA (PART1 2.1) ──
    response_p95_s: float = 5.0   # total response p95 ≤ 5s

    # ── Blocking 게이트 임계값 (PART1 7.1) ──
    response_success_rate_min: float = 0.995  # 응답 성공률 ≥ 99.5%  (Blocking)
    finish_reason_stop_min: float = 0.98      # finish_reason:stop 비율 ≥ 98%

    # ── 비결정성·품질 임계값 (PART1 5.1 / 2.x) ──
    semantic_stability_min: float = 0.95   # Semantic Stability ≥ 95%
    semantic_similarity_threshold: float = 0.85  # 클러스터 묶음 기준 코사인 유사도
    prompt_adherence_min: float = 0.95     # 시스템 프롬프트 준수율 ≥ 95%
    hallucination_rate_max: float = 0.05   # 환각률 ≤ 5%

    # ── 콘텐츠 품질 임계값 ──
    korean_ratio_min: float = 0.95


settings = Settings()

SUPPORTED_MODELS = ["gpt-4o", "claude-sonnet-4-6", "gemini-2.0-flash"]

# Maps A Chat internal model name → actual vendor API model ID
MODEL_API_ID = {
    "gpt-4o": "gpt-4o",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    # gemini-2.0-flash is deprecated for new API keys; map to current equivalent
    "gemini-2.0-flash": "gemini-2.5-flash",
}
