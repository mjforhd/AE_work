"""
test_basic.py — 기본 검증

테스트 항목:
  1. 3개 모델 정상 응답 수신 및 A Chat 스키마 검증
  2. 잘못된 API 키 → 인증 오류(401/403) 발생 확인
  3. messages 파라미터 누락(빈 리스트) → AChatError(400)
  4. 지원하지 않는 모델명 → AChatError(400)
"""

import pytest
import openai
import anthropic

from core.client import AChatClient, AChatError
from core.validator import ResponseValidator, ValidationLevel
from tests.conftest import MODELS


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _is_auth_error(exc: Exception) -> bool:
    """401/403 인증 오류인지 벤더별로 확인한다."""
    # OpenAI
    if isinstance(exc, openai.AuthenticationError):
        return True
    # Anthropic
    if isinstance(exc, anthropic.AuthenticationError):
        return True
    # Google (google-generativeai 는 google.api_core 예외를 사용)
    try:
        from google.api_core import exceptions as gex
        if isinstance(exc, (gex.PermissionDenied, gex.Unauthenticated, gex.InvalidArgument)):
            return True
    except ImportError:
        pass
    # 상태 코드 기반 폴백
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (401, 403):
        return True
    # 메시지 기반 폴백 (Google API KEY_INVALID 등)
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ["api key", "api_key", "key_invalid", "permission", "unauthenticated", "unauthorized", "401", "403"]
    )


# ──────────────────────────────────────────────
# 1. 정상 응답 및 스키마 검증
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_normal_response_schema(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
    simple_kr_message: list,
) -> None:
    """정상 응답 수신 시 A Chat 스펙 스키마를 만족하는지 검증한다."""
    response = client.chat(model=model, messages=simple_kr_message, max_tokens=150)

    # 스키마 검증
    schema_result = validator.validate_schema(response)
    assert schema_result.status, f"[{model}] 스키마 검증 실패: {schema_result.message}"

    choice = response["choices"][0]
    content: str = choice["message"]["content"]
    finish_reason: str | None = choice["finish_reason"]

    # finish_reason 정합성
    fr_result = validator.validate_finish_reason(model, finish_reason, content)
    assert fr_result.status, f"[{model}] finish_reason 검증 실패: {fr_result.message}"

    # 기본 필드 확인
    assert content.strip(), f"[{model}] content가 비어 있음"
    assert response["model"] == model, (
        f"[{model}] 응답 model 필드 불일치: '{response['model']}'"
    )
    assert response["usage"]["total_tokens"] > 0, f"[{model}] usage.total_tokens가 0"


# ──────────────────────────────────────────────
# 2. 인증 실패
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_auth_failure(model: str, bad_client_factory) -> None:
    """잘못된 API 키로 요청 시 401/403 인증 오류가 발생해야 한다."""
    bad = bad_client_factory(model)

    with pytest.raises(Exception) as exc_info:
        bad.chat(model=model, messages=[{"role": "user", "content": "안녕"}])

    exc = exc_info.value
    assert not isinstance(exc, AChatError), (
        f"[{model}] AChatError가 아닌 벤더 인증 오류를 기대했으나 AChatError 발생: {exc}"
    )
    assert _is_auth_error(exc), (
        f"[{model}] 인증 오류를 기대했으나 '{type(exc).__name__}: {exc}' 발생"
    )


# ──────────────────────────────────────────────
# 3. 필수 파라미터 누락
# ──────────────────────────────────────────────

@pytest.mark.parametrize("model", MODELS)
def test_missing_messages_empty_list(model: str, client: AChatClient) -> None:
    """messages=[] 전송 시 AChatError(400) 발생 확인."""
    with pytest.raises(AChatError) as exc_info:
        client.chat(model=model, messages=[])

    assert exc_info.value.status_code == 400, (
        f"[{model}] 400을 기대했으나 status_code={exc_info.value.status_code}"
    )


@pytest.mark.parametrize("model", MODELS)
def test_missing_messages_none(model: str, client: AChatClient) -> None:
    """messages=None 전송 시 400 오류 발생 확인."""
    with pytest.raises((AChatError, TypeError)):
        client.chat(model=model, messages=None)  # type: ignore[arg-type]


# ──────────────────────────────────────────────
# 4. 지원하지 않는 모델명
# ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "invalid_model",
    ["gpt-99", "claude-x", "gemini-99", "unknown-model", ""],
)
def test_unsupported_model(invalid_model: str, client: AChatClient) -> None:
    """미지원 모델명 입력 시 AChatError(400) 발생 확인."""
    with pytest.raises(AChatError) as exc_info:
        client.chat(
            model=invalid_model,
            messages=[{"role": "user", "content": "테스트"}],
        )

    assert exc_info.value.status_code == 400, (
        f"[{invalid_model!r}] 400을 기대했으나 status_code={exc_info.value.status_code}"
    )
