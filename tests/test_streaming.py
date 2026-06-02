"""
test_streaming.py — 스트리밍 검증

테스트 항목:
  1. stream=True 전환 시 SSE 청크 구조 검증 (모델별)          [TC-STR-001/002]
  2. 전체 청크 병합 결과와 비스트리밍 응답의 의미적 동일성 검증  [TC-STR-004]
     - validate_semantic_stability()로 임베딩 기반 코사인 유사도 ≥ 95% 판정
     - OpenAI 임베딩 불가 시 키워드 오버랩 폴백 자동 적용
  3. 스트리밍 중 연결 강제 종료(abort) 후 다음 요청 정상 동작 확인  [TC-STR-003]
"""

import pytest

from core.client import AChatClient
from core.validator import ResponseValidator
from tests.conftest import MODELS  # noqa: F401 (conftest에서 fixture로도 주입됨)

# ── 팩트 질문: 예측 가능한 핵심 키워드로 의미적 동일성 판단 ──
_FACTUAL_MESSAGES = [
    {"role": "user", "content": "한국의 수도가 어디인지 한 단어로만 답해주세요."}
]
_FACTUAL_KEYWORDS = {"서울", "Seoul", "seoul"}

# ── abort 시나리오용: 복수 청크를 유도하되 타임아웃을 피할 적당한 길이 ──
_LONG_MESSAGES = [
    {"role": "user", "content": "인공지능의 장점 3가지를 각각 한 문장으로 설명해주세요."}
]


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _collect_stream(gen) -> tuple[list[dict], str]:
    """제너레이터를 모두 소비해 청크 목록과 병합 텍스트를 반환한다."""
    chunks = list(gen)
    merged = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    return chunks, merged


def _has_keyword(text: str) -> bool:
    return any(kw in text for kw in _FACTUAL_KEYWORDS)


# ──────────────────────────────────────────────
# 1. SSE 청크 구조 검증
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_stream_chunk_structure(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
    simple_kr_message: list,
) -> None:
    """chat_stream()이 반환하는 모든 청크가 A Chat SSE 스펙을 만족하는지 검증한다."""
    chunks, merged = _collect_stream(
        client.chat_stream(model=model, messages=simple_kr_message, max_tokens=150)
    )

    assert chunks, f"[{model}] 청크가 하나도 수신되지 않음"

    # ── 모든 청크 스키마 검증 ──
    for i, chunk in enumerate(chunks):
        result = validator.validate_schema(chunk, is_chunk=True)
        assert result.status, f"[{model}] 청크 #{i} 스키마 실패: {result.message}\n  chunk={chunk}"

    # ── 모든 청크의 model 필드 일치 ──
    for i, chunk in enumerate(chunks):
        assert chunk["model"] == model, (
            f"[{model}] 청크 #{i} model 필드 불일치: '{chunk['model']}'"
        )

    # ── 중간 청크: finish_reason=None ──
    for i, chunk in enumerate(chunks[:-1]):
        mid_finish = chunk["choices"][0]["finish_reason"]
        assert mid_finish is None, (
            f"[{model}] 중간 청크 #{i}에 finish_reason 있음: '{mid_finish}'"
        )

    # ── 마지막 청크: finish_reason 존재 및 유효값 ──
    last_finish = chunks[-1]["choices"][0]["finish_reason"]
    assert last_finish is not None, (
        f"[{model}] 마지막 청크에 finish_reason 없음"
    )
    assert last_finish in {"stop", "length", "content_filter"}, (
        f"[{model}] 유효하지 않은 finish_reason: '{last_finish}'"
    )

    # ── 전체 병합 콘텐츠 비어있지 않아야 함 ──
    assert merged.strip(), f"[{model}] 전체 병합 결과가 비어 있음"


# ──────────────────────────────────────────────
# 2. TC-STR-004 — 청크 병합 vs 비스트리밍 의미적 동일성
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_stream_merged_semantic_equivalence(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """TC-STR-004: 스트리밍 병합 결과와 비스트리밍 응답의 의미적 동일성을 검증한다.

    PART1 5.2 절차 적용:
      - 두 응답을 OpenAI 임베딩으로 벡터화해 코사인 유사도 기반 Semantic Stability를 측정한다.
      - Stability ≥ 95%(settings.semantic_stability_min) → Pass
      - OpenAI API 불가 시 키워드 오버랩 폴백이 자동 적용된다.

    보조 검증(1차):
      - 팩트 질문에 대해 예측 가능한 키워드("서울"/"Seoul") 포함 여부를 먼저 확인한다.
      - 키워드 검증 통과 후 임베딩 유사도로 정합성을 수치화한다.
    """
    max_tok = 500
    ns_resp = client.chat(model=model, messages=_FACTUAL_MESSAGES, max_tokens=max_tok)
    ns_content: str = ns_resp["choices"][0]["message"]["content"]

    chunks, merged = _collect_stream(
        client.chat_stream(model=model, messages=_FACTUAL_MESSAGES, max_tokens=max_tok)
    )

    assert ns_content.strip(), f"[{model}] 비스트리밍 응답이 비어 있음"
    assert merged.strip(), f"[{model}] 스트리밍 병합 결과가 비어 있음"

    # ── 1차: 키워드 포함 여부 (빠른 필터) ──
    assert _has_keyword(ns_content), (
        f"[{model}] 비스트리밍 응답에 핵심 키워드 {_FACTUAL_KEYWORDS} 없음: {ns_content!r}"
    )
    assert _has_keyword(merged), (
        f"[{model}] 스트리밍 병합 응답에 핵심 키워드 {_FACTUAL_KEYWORDS} 없음: {merged!r}"
    )

    # ── 2차: 임베딩 기반 Semantic Stability (PART1 5.2, TC-STR-004) ──
    stability = validator.validate_semantic_stability(
        model=model,
        texts=[ns_content, merged],
    )
    print(
        f"\n  [TC-STR-004][{model}] Semantic Stability "
        f"{stability.value.get('stability_score', 0):.1%} "
        f"(method={stability.value.get('method', '?')})"
    )
    assert stability.status, (
        f"[{model}] 스트리밍↔비스트리밍 의미적 동일성 미달: {stability.message}"
    )

    # ── finish_reason 유효성 ──
    stream_finish = chunks[-1]["choices"][0]["finish_reason"]
    ns_finish = ns_resp["choices"][0]["finish_reason"]
    assert stream_finish in {"stop", "length", "content_filter"}, (
        f"[{model}] 스트리밍 finish_reason 유효하지 않음: '{stream_finish}'"
    )
    assert ns_finish in {"stop", "length", "content_filter"}, (
        f"[{model}] 비스트리밍 finish_reason 유효하지 않음: '{ns_finish}'"
    )


# ──────────────────────────────────────────────
# 3. 스트리밍 abort 후 다음 요청 정상 동작
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_stream_abort_then_next_request(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    스트리밍 중 강제 종료 후 동일 클라이언트로 다음 요청이 정상 동작하는지 확인한다.

    Python generator.close()로 연결 abort를 시뮬레이션한다.
    Anthropic의 context manager(__exit__), OpenAI/Gemini의 HTTP 연결도 정상 해제되어야 한다.
    """
    gen = client.chat_stream(model=model, messages=_LONG_MESSAGES, max_tokens=200)

    # 최대 3청크만 읽고 중단 (abort 시뮬레이션)
    chunks_read = 0
    for chunk in gen:
        chunks_read += 1
        if chunks_read >= 3:
            break

    # 명시적으로 generator를 닫아 GeneratorExit → 연결 해제
    gen.close()

    # abort 전에 청크를 최소 1개 읽었는지 확인 (테스트 유효성)
    assert chunks_read >= 1, f"[{model}] abort 전 청크를 하나도 읽지 못함"

    # ── abort 후 동일 클라이언트로 다음 요청 ──
    # gemini-2.5-flash는 thinking 모델이라 내부 추론 토큰을 소비하므로
    # max_tokens를 충분히 줘야 실제 출력이 생성된다.
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": "안녕"}],
        max_tokens=300,
    )
    result = validator.validate_schema(response)
    assert result.status, (
        f"[{model}] abort 후 다음 요청 스키마 검증 실패: {result.message}"
    )

    content: str = response["choices"][0]["message"]["content"]
    assert content.strip(), f"[{model}] abort 후 다음 응답 content가 비어 있음"


# ──────────────────────────────────────────────
# 4. 스트리밍과 비스트리밍의 usage 필드 비교 (보조 검증)
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_stream_chunk_count_increases(
    model: str,
    client: AChatClient,
    simple_kr_message: list,
) -> None:
    """스트리밍 청크는 2개 이상이어야 한다 (최소 콘텐츠 청크 + 최종 finish 청크)."""
    chunks, _ = _collect_stream(
        client.chat_stream(model=model, messages=simple_kr_message, max_tokens=150)
    )
    # 최소: 콘텐츠 청크 1개 + finish_reason 청크 1개
    assert len(chunks) >= 2, (
        f"[{model}] 청크 수가 너무 적음: {len(chunks)}개"
    )
