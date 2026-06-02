"""
test_edge.py — 경계값 및 예외 검증

테스트 항목:
  1. max_tokens=1 → finish_reason: "length" 검증
  2. 빈 메시지(content: "") 전송 시 모델별 동작 검증
  3. 시스템 프롬프트 없이 요청 시 모델별 기본 동작 검증
  4. (보조) 시스템 프롬프트 유무에 따른 응답 차이 비교
"""

import pytest

from core.client import AChatClient, AChatError
from core.validator import ResponseValidator
from tests.conftest import MODELS

# 경계값 테스트용: 긴 응답을 유도해 max_tokens 제한에 확실히 걸리게 함
_LONG_PROMPT = [
    {"role": "user", "content": "한국의 역사에 대해 최대한 자세히 길게 설명해주세요."}
]
_SHORT_PROMPT = [
    {"role": "user", "content": "당신의 역할은 무엇인가요?"}
]


# ──────────────────────────────────────────────
# 1. max_tokens=1 → finish_reason: "length"
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_max_tokens_1_finish_reason_length(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """max_tokens=1로 응답이 강제로 잘릴 때 finish_reason이 반드시 "length"여야 한다."""
    response = client.chat(
        model=model,
        messages=_LONG_PROMPT,
        max_tokens=1,
    )

    schema_result = validator.validate_schema(response)
    assert schema_result.status, f"[{model}] 스키마 검증 실패: {schema_result.message}"

    finish_reason = response["choices"][0]["finish_reason"]
    assert finish_reason == "length", (
        f"[{model}] max_tokens=1인데 finish_reason={finish_reason!r} (expected 'length')"
    )

    # 완료 토큰이 max_tokens 이하인지 확인
    completion_tokens = response["usage"]["completion_tokens"]
    assert completion_tokens <= 1, (
        f"[{model}] max_tokens=1인데 completion_tokens={completion_tokens}"
    )


@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_max_tokens_small_values(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """max_tokens=5, 10, 50에서 항상 finish_reason이 "length"여야 한다."""
    for max_tokens in [5, 10, 50]:
        response = client.chat(
            model=model,
            messages=_LONG_PROMPT,
            max_tokens=max_tokens,
        )
        schema_result = validator.validate_schema(response)
        assert schema_result.status, (
            f"[{model}] max_tokens={max_tokens} 스키마 실패: {schema_result.message}"
        )
        finish_reason = response["choices"][0]["finish_reason"]
        assert finish_reason == "length", (
            f"[{model}] max_tokens={max_tokens}인데 finish_reason={finish_reason!r}"
        )
        tokens = response["usage"]["completion_tokens"]
        assert tokens <= max_tokens, (
            f"[{model}] max_tokens={max_tokens}인데 completion_tokens={tokens}"
        )


# ──────────────────────────────────────────────
# 2. 빈 메시지(content: "") 전송 시 모델별 동작
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_empty_message_content_behavior(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    빈 content("")를 가진 메시지 전송 시 모델별 동작을 검증한다.

    모델마다 동작이 다를 수 있으므로 두 가지 결과 모두 허용한다:
      - 응답 성공:  A Chat 스키마를 만족해야 함
      - API 오류:   처리 가능한 예외여야 하며 의미 있는 오류 메시지가 있어야 함
    """
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": ""}],
            max_tokens=100,
        )
        # 응답을 받은 경우 — 스키마 유효성 확인
        result = validator.validate_schema(response)
        assert result.status, (
            f"[{model}] 빈 메시지 허용 응답 스키마 실패: {result.message}"
        )
        finish_reason = response["choices"][0]["finish_reason"]
        assert finish_reason in {"stop", "length", "content_filter"}, (
            f"[{model}] 유효하지 않은 finish_reason: {finish_reason!r}"
        )
        print(f"\n  [{model}] ACCEPT — finish_reason={finish_reason!r}")

    except (AChatError, Exception) as exc:
        # 오류가 발생한 경우 — AssertionError나 빈 메시지 오류는 허용 안 됨
        assert not isinstance(exc, AssertionError), (
            f"[{model}] AssertionError 발생 (구현 버그): {exc}"
        )
        err_msg = str(exc).strip()
        assert err_msg, f"[{model}] 오류 메시지가 비어 있음"
        print(f"\n  [{model}] REJECT — {type(exc).__name__}: {err_msg[:80]}")


@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_whitespace_only_content_behavior(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """공백만 포함된 content(" ") 전송 시 모델별 동작을 검증한다."""
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "   "}],
            max_tokens=100,
        )
        result = validator.validate_schema(response)
        assert result.status, (
            f"[{model}] 공백 메시지 응답 스키마 실패: {result.message}"
        )
        print(f"\n  [{model}] ACCEPT — whitespace content 허용")
    except Exception as exc:
        assert not isinstance(exc, AssertionError), str(exc)
        print(f"\n  [{model}] REJECT — {type(exc).__name__}: {str(exc)[:80]}")


# ──────────────────────────────────────────────
# 3. 시스템 프롬프트 없이 요청 — 모델별 기본 동작
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_no_system_prompt_default_behavior(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    시스템 프롬프트 없이 user 메시지만 전송 시 모델 기본 동작을 검증한다.

    모든 모델은 시스템 프롬프트 없이도 유효한 응답을 반환해야 한다.
    """
    response = client.chat(
        model=model,
        messages=_SHORT_PROMPT,  # user message only, no system
        max_tokens=150,
    )

    schema_result = validator.validate_schema(response)
    assert schema_result.status, (
        f"[{model}] 시스템 없는 응답 스키마 실패: {schema_result.message}"
    )

    choice = response["choices"][0]
    content: str = choice["message"]["content"]
    finish_reason = choice["finish_reason"]

    assert content.strip(), f"[{model}] 시스템 프롬프트 없는 응답 content가 비어 있음"
    assert finish_reason in {"stop", "length", "content_filter"}, (
        f"[{model}] 유효하지 않은 finish_reason: {finish_reason!r}"
    )
    assert choice["message"]["role"] == "assistant", (
        f"[{model}] 응답 role이 'assistant'가 아님: {choice['message']['role']!r}"
    )

    print(f"\n  [{model}] 기본 응답 (앞 60자): {content[:60]}…")


# ──────────────────────────────────────────────
# 4. (보조) 시스템 프롬프트 유무 응답 차이 비교
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_system_prompt_vs_no_system_prompt(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """시스템 프롬프트 유무에 따른 응답을 비교한다. 양쪽 모두 유효해야 한다."""
    user_msg = {"role": "user", "content": "당신의 역할은 무엇인가요?"}

    resp_without = client.chat(
        model=model,
        messages=[user_msg],
        max_tokens=100,
    )
    resp_with = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 친절한 한국어 QA 도우미입니다."},
            user_msg,
        ],
        max_tokens=100,
    )

    for label, resp in [("without_sys", resp_without), ("with_sys", resp_with)]:
        result = validator.validate_schema(resp)
        assert result.status, f"[{model}][{label}] 스키마 검증 실패: {result.message}"
        assert resp["choices"][0]["message"]["content"].strip(), (
            f"[{model}][{label}] 응답이 비어 있음"
        )

    without_content = resp_without["choices"][0]["message"]["content"]
    with_content = resp_with["choices"][0]["message"]["content"]

    print(
        f"\n  [{model}] without_sys : {without_content[:60]}…"
        f"\n  [{model}]    with_sys : {with_content[:60]}…"
    )
