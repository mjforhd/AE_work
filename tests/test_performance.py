"""
test_performance.py — 성능 비교 검증

테스트 항목:
  1. 동일 프롬프트를 3개 모델에 asyncio.gather로 동시 요청 → 응답 시간 분포 비교
  2. temperature 0.0/0.7/1.5 변화에 따른 응답 시간·완료 토큰 수 변화 측정
  3. TTFT(첫 토큰 수신 시간) 모델별 측정 및 임계값(≤1s) 검증
"""

import asyncio
import time
from typing import Optional

import pytest

from config.settings import settings
from core.client import AChatClient
from core.validator import ResponseValidator, ValidationLevel
from tests.conftest import MODELS

_PERF_MESSAGES = [
    {"role": "user", "content": "인공지능의 장점을 세 줄로 요약해주세요."}
]

_TEMPERATURES = [0.0, 0.7, 1.5]

# Anthropic은 temperature 최대 1.0 제한 (초과 시 API 오류 발생)
_MODEL_MAX_TEMP: dict[str, float] = {
    "gpt-4o": 2.0,
    "claude-sonnet-4-6": 1.0,
    "gemini-2.0-flash": 2.0,
}


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _clamp_temperature(model: str, temperature: float) -> float:
    """모델별 API 상한값으로 temperature를 클램핑한다."""
    return min(temperature, _MODEL_MAX_TEMP.get(model, 2.0))


def _measure_ttft(
    client: AChatClient,
    model: str,
    messages: list,
    max_tokens: int = 150,
) -> tuple[float, float]:
    """
    스트리밍 모드 기준 TTFT(Time To First Token)와 전체 응답 시간을 측정한다.

    첫 번째 비어있지 않은 content 청크가 도착한 시점을 TTFT로 기록한다.

    Returns:
        (ttft_seconds, total_elapsed_seconds)
    """
    start = time.perf_counter()
    ttft: Optional[float] = None
    gen = client.chat_stream(model=model, messages=messages, max_tokens=max_tokens)
    try:
        for chunk in gen:
            if ttft is None and chunk["choices"][0]["delta"].get("content", ""):
                ttft = time.perf_counter() - start
    finally:
        gen.close()
    total = time.perf_counter() - start
    return (ttft if ttft is not None else total), total


def _print_table(headers: list[str], rows: list[tuple], widths: list[int]) -> None:
    """간단한 텍스트 테이블을 출력한다."""
    fmt = "  " + "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    sep = "  " + "-" * (sum(widths) + 2 * (len(widths) - 1))
    print("\n" + fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ──────────────────────────────────────────────
# 1. 동시 요청 — asyncio.gather
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.slow
async def test_concurrent_model_requests(
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    3개 모델에 asyncio.gather로 동시 요청한 뒤 응답 시간 분포를 비교한다.

    동기 AChatClient.chat()을 run_in_executor로 감싸 ThreadPool에서 병렬 실행한다.
    총 소요 시간이 직렬 실행 대비 단축되는지(병렬 효율)도 측정한다.
    """
    loop = asyncio.get_running_loop()

    async def _achat(model: str) -> tuple[str, dict, float]:
        start = time.perf_counter()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat(model=model, messages=_PERF_MESSAGES, max_tokens=150),
        )
        return model, response, time.perf_counter() - start

    total_start = time.perf_counter()
    results: list[tuple[str, dict, float]] = await asyncio.gather(
        *(_achat(m) for m in MODELS)
    )
    total_elapsed = time.perf_counter() - total_start

    # ── 모든 모델 응답 검증 ──
    for model, response, elapsed in results:
        schema_result = validator.validate_schema(response)
        assert schema_result.status, (
            f"[{model}] 동시 요청 스키마 검증 실패: {schema_result.message}"
        )
        assert response["choices"][0]["message"]["content"].strip(), (
            f"[{model}] 동시 요청 응답 content가 비어 있음"
        )

    # ── 성능 임계값 검증 (FAIL 수준만 실패) ──
    for model, response, elapsed in results:
        perf = validator.validate_performance(model, elapsed, ttft=0.0)
        assert perf.status, f"[{model}] 동시 요청 성능 FAIL: {perf.message}"

    # ── 응답 시간 비교 테이블 ──
    sorted_results = sorted(results, key=lambda x: x[2])
    sum_sequential = sum(e for _, _, e in results)
    parallel_efficiency = total_elapsed / sum_sequential if sum_sequential else 1.0

    _print_table(
        headers=["Model", "Time(s)", "Tokens", "Level"],
        widths=[26, 8, 7, 8],
        rows=[
            (
                m,
                f"{e:.2f}",
                str(resp["usage"]["completion_tokens"]),
                validator.validate_performance(m, e, 0.0).level.value.upper(),
            )
            for m, resp, e in sorted_results
        ],
    )
    print(
        f"\n  동시 총 소요: {total_elapsed:.2f}s  /  "
        f"직렬 합계: {sum_sequential:.2f}s  /  "
        f"병렬 효율: {parallel_efficiency:.0%}"
    )


# ──────────────────────────────────────────────
# 2. Temperature 변화 측정
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_temperature_variation(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    temperature 0.0/0.7/1.5 변화에 따른 응답 시간·완료 토큰 수 변화를 측정한다.

    claude-sonnet-4-6의 temperature 상한(1.0) 초과 시 자동 클램핑한다.
    """
    rows = []
    for temperature in _TEMPERATURES:
        clamped = _clamp_temperature(model, temperature)

        start = time.perf_counter()
        response = client.chat(
            model=model,
            messages=_PERF_MESSAGES,
            temperature=clamped,
            max_tokens=200,
        )
        elapsed = time.perf_counter() - start

        # 유효성 검증
        schema_result = validator.validate_schema(response)
        assert schema_result.status, (
            f"[{model}] temperature={temperature} 스키마 검증 실패: {schema_result.message}"
        )
        assert response["choices"][0]["message"]["content"].strip(), (
            f"[{model}] temperature={temperature} 응답이 비어 있음"
        )

        tokens = response["usage"]["completion_tokens"]
        note = " *" if clamped != temperature else ""
        rows.append((f"{temperature:.1f}{note}", f"{clamped:.1f}", f"{elapsed:.2f}", str(tokens)))

    _print_table(
        headers=[f"[{model}] Temp", "Applied", "Time(s)", "Tokens"],
        widths=[16, 8, 8, 7],
        rows=rows,
    )
    print("  * Anthropic API 제한으로 1.0으로 클램핑")


# ──────────────────────────────────────────────
# 3. TTFT 모델별 측정
# ──────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("model", MODELS)
def test_ttft_per_model(
    model: str,
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """
    스트리밍 모드 기준 모델별 TTFT를 측정하고 임계값(≤1s)과 비교한다.

    ValidationLevel.WARNING은 허용(status=True), FAIL은 실패로 처리한다.
    """
    ttft, total_elapsed = _measure_ttft(client, model, _PERF_MESSAGES)
    perf = validator.validate_performance(model, total_elapsed, ttft)

    print(
        f"\n  [{model}]  TTFT={ttft:.3f}s  "
        f"Total={total_elapsed:.2f}s  "
        f"→ {perf.level.value.upper()}"
    )
    if perf.level == ValidationLevel.WARNING:
        print(f"  [WARNING] {perf.message}")

    assert perf.status, f"[{model}] TTFT 성능 임계값 FAIL: {perf.message}"


@pytest.mark.api
@pytest.mark.slow
def test_ttft_comparison(
    client: AChatClient,
    validator: ResponseValidator,
) -> None:
    """3개 모델의 TTFT를 반복 측정해 p50/p95/p99 분위수를 비교한다 (PART1 2.1).

    TTFT_SAMPLES 회 반복 측정 후 validate_performance_distribution()으로
    PART1 Latency SLA를 검증한다.
    """
    TTFT_SAMPLES = 5  # 분위수 신뢰도를 위해 5회 이상 권장

    all_ttfts: dict[str, list[float]] = {}
    all_totals: dict[str, list[float]] = {}

    for model in MODELS:
        ttfts_m: list[float] = []
        totals_m: list[float] = []
        for _ in range(TTFT_SAMPLES):
            ttft, total = _measure_ttft(client, model, _PERF_MESSAGES)
            ttfts_m.append(ttft)
            totals_m.append(total)
        all_ttfts[model] = ttfts_m
        all_totals[model] = totals_m

    # ── p50/p95/p99 비교표 출력 ──
    rows = []
    for model in MODELS:
        from core.validator import _percentile
        t = all_ttfts[model]
        rows.append((
            model,
            f"{_percentile(t, 50):.3f}",
            f"{_percentile(t, 95):.3f}",
            f"{_percentile(t, 99):.3f}",
        ))

    _print_table(
        headers=["Model", "p50(s)", "p95(s)", "p99(s)"],
        widths=[26, 8, 8, 8],
        rows=sorted(rows, key=lambda r: float(r[1])),
    )
    print(f"\n  (각 모델 TTFT {TTFT_SAMPLES}회 측정 기준)")

    # ── validate_performance_distribution() SLA 검증 ──
    for model in MODELS:
        dist_result = validator.validate_performance_distribution(
            model=model,
            elapsed_times=all_totals[model],
            ttfts=all_ttfts[model],
        )
        print(f"\n  [{model}] {dist_result.level.value.upper()}: {dist_result.message}")
        assert dist_result.status, (
            f"[{model}] TTFT 분포 SLA 초과: {dist_result.message}"
        )
