"""
ResponseValidator — A Chat 응답의 스키마·성능·품질을 검증한다.

PART1 품질 기준 정의서 반영:
  - validate_performance()            : 단일 샘플, p50/Warning/Error 판정
  - validate_performance_distribution(): 복수 샘플 → p50/p95/p99 분위수 SLA 검증 (PART1 2.1)
  - validate_finish_reason()          : finish_reason + content 길이 정합성
  - validate_content_quality()        : 한국어 비율·반복 패턴
  - validate_semantic_stability()     : OpenAI 임베딩 기반 코사인 유사도 클러스터링 (PART1 5.2)
  - generate_report()                 : Blocking / Non-Blocking 게이트 분리 집계 (PART1 7.1)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ValidationError, field_validator

from config.settings import settings


# ──────────────────────────────────────────────
# Level / Result
# ──────────────────────────────────────────────

class ValidationLevel(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationResult:
    status: bool               # True = acceptable (PASS or WARNING)
    level: ValidationLevel
    message: str
    value: Any = None
    model: Optional[str] = None
    check: str = ""
    blocking: bool = False     # True = PART1 7.1 Blocking 게이트 항목


# ──────────────────────────────────────────────
# Pydantic schemas — A Chat 응답 스펙
# ──────────────────────────────────────────────

_VALID_FINISH_REASONS = {None, "stop", "length", "content_filter"}


class _MessageSchema(BaseModel):
    role: str
    content: str


class _ChoiceSchema(BaseModel):
    index: int
    message: _MessageSchema
    finish_reason: Optional[str] = None

    @field_validator("finish_reason")
    @classmethod
    def _check_finish(cls, v: Optional[str]) -> Optional[str]:
        if v not in _VALID_FINISH_REASONS:
            raise ValueError(f"finish_reason '{v}' not in {_VALID_FINISH_REASONS}")
        return v


class _UsageSchema(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _AChatResponseSchema(BaseModel):
    id: str
    object: str
    model: str
    choices: List[_ChoiceSchema]
    usage: _UsageSchema


# ── 스트리밍 청크 스키마 ──

class _DeltaSchema(BaseModel):
    role: Optional[str] = None
    content: str = ""


class _ChunkChoiceSchema(BaseModel):
    index: int
    delta: _DeltaSchema
    finish_reason: Optional[str] = None

    @field_validator("finish_reason")
    @classmethod
    def _check_finish(cls, v: Optional[str]) -> Optional[str]:
        if v not in _VALID_FINISH_REASONS:
            raise ValueError(f"finish_reason '{v}' not in {_VALID_FINISH_REASONS}")
        return v


class _AChatChunkSchema(BaseModel):
    id: str
    object: str
    model: str
    choices: List[_ChunkChoiceSchema]


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    """선형 보간 방식으로 0~100 사이 백분위수를 계산한다."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_data[-1]
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도를 순수 파이썬으로 계산한다."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cluster_by_similarity(
    embeddings: list[list[float]], threshold: float
) -> list[list[int]]:
    """
    탐욕적 클러스터링: 아직 미배정 텍스트 중 첫 번째를 seed로 삼아,
    코사인 유사도 ≥ threshold 인 나머지를 같은 클러스터로 묶는다.
    """
    n = len(embeddings)
    assigned = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if not assigned[j]:
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    cluster.append(j)
                    assigned[j] = True
        clusters.append(cluster)
    return clusters


# ──────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────

class ResponseValidator:

    # ── 스키마 검증 ──

    def validate_schema(
        self, response: dict, is_chunk: bool = False
    ) -> ValidationResult:
        """Pydantic 모델로 A Chat 응답 스키마를 검증한다.

        스트리밍 정합성(PART1 L3)은 Blocking 게이트에 해당하므로
        is_chunk=True 인 경우 blocking=True 로 반환한다.
        """
        schema_cls = _AChatChunkSchema if is_chunk else _AChatResponseSchema
        try:
            schema_cls.model_validate(response)
            return ValidationResult(
                status=True,
                level=ValidationLevel.PASS,
                message="스키마 검증 통과",
                check="schema",
                blocking=is_chunk,
            )
        except ValidationError as exc:
            summary = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            return ValidationResult(
                status=False,
                level=ValidationLevel.FAIL,
                message=f"스키마 검증 실패: {summary}",
                value=exc.errors(),
                check="schema",
                blocking=is_chunk,  # 스트리밍 청크 실패는 Blocking
            )

    # ── 단일 샘플 성능 검증 ──

    def validate_performance(
        self, model: str, elapsed_time: float, ttft: float
    ) -> ValidationResult:
        """단일 요청의 응답 시간과 TTFT를 임계값과 비교한다.

        임계값 대응:
          - ttft > ttft_error_s  → Error  (p95 SLA 초과)
          - ttft > ttft_warning_s → Warning (p50 SLA 초과)
          - elapsed > response_time_error_s  → Error
          - elapsed > response_time_warning_s → Warning (p95 SLA 초과)
        """
        issues: list[tuple[ValidationLevel, str]] = []

        if elapsed_time > settings.response_time_error_s:
            issues.append((
                ValidationLevel.FAIL,
                f"응답시간 {elapsed_time:.2f}s > {settings.response_time_error_s}s (Error)",
            ))
        elif elapsed_time > settings.response_time_warning_s:
            issues.append((
                ValidationLevel.WARNING,
                f"응답시간 {elapsed_time:.2f}s > {settings.response_time_warning_s}s (p95 SLA)",
            ))

        if ttft > settings.ttft_error_s:
            issues.append((
                ValidationLevel.FAIL,
                f"TTFT {ttft:.3f}s > {settings.ttft_error_s}s (p95 SLA 초과)",
            ))
        elif ttft > settings.ttft_warning_s:
            issues.append((
                ValidationLevel.WARNING,
                f"TTFT {ttft:.3f}s > {settings.ttft_warning_s}s (p50 SLA 초과)",
            ))

        if not issues:
            return ValidationResult(
                status=True,
                level=ValidationLevel.PASS,
                message=f"성능 정상 (응답 {elapsed_time:.2f}s, TTFT {ttft:.3f}s)",
                value={"elapsed_time": elapsed_time, "ttft": ttft},
                model=model,
                check="performance",
            )

        _priority = {ValidationLevel.FAIL: 2, ValidationLevel.WARNING: 1, ValidationLevel.PASS: 0}
        worst_level = max(issues, key=lambda t: _priority[t[0]])[0]
        return ValidationResult(
            status=worst_level != ValidationLevel.FAIL,
            level=worst_level,
            message="; ".join(msg for _, msg in issues),
            value={"elapsed_time": elapsed_time, "ttft": ttft},
            model=model,
            check="performance",
        )

    # ── 분포 기반 성능 검증 (PART1 2.1) ──

    def validate_performance_distribution(
        self,
        model: str,
        elapsed_times: list[float],
        ttfts: list[float],
    ) -> ValidationResult:
        """복수 측정값의 p50/p95/p99 분위수로 PART1 2.1 Latency SLA를 검증한다.

        TTFT SLA : p50 ≤ 1s / p95 ≤ 2s / p99 ≤ 3s
        Total SLA: p95 ≤ 5s

        단일 샘플 수가 적을수록 분위수 신뢰도가 낮아지므로, 측정값이 5개 미만이면
        Warning 메시지에 소표본 주의를 첨부한다.
        """
        n_elapsed = len(elapsed_times)
        n_ttft = len(ttfts)

        issues: list[tuple[ValidationLevel, str]] = []
        percentiles: dict[str, float] = {}

        if ttfts:
            tp50 = _percentile(ttfts, 50)
            tp95 = _percentile(ttfts, 95)
            tp99 = _percentile(ttfts, 99)
            percentiles.update({"ttft_p50": tp50, "ttft_p95": tp95, "ttft_p99": tp99})

            if tp99 > settings.ttft_p99_s:
                issues.append((
                    ValidationLevel.FAIL,
                    f"TTFT p99 {tp99:.3f}s > {settings.ttft_p99_s}s",
                ))
            elif tp95 > settings.ttft_p95_s:
                issues.append((
                    ValidationLevel.WARNING,
                    f"TTFT p95 {tp95:.3f}s > {settings.ttft_p95_s}s",
                ))
            elif tp50 > settings.ttft_p50_s:
                issues.append((
                    ValidationLevel.WARNING,
                    f"TTFT p50 {tp50:.3f}s > {settings.ttft_p50_s}s",
                ))

            if n_ttft < 5:
                issues.append((
                    ValidationLevel.WARNING,
                    f"TTFT 샘플 수 부족 ({n_ttft}개 < 5) — 분위수 신뢰도 낮음",
                ))

        if elapsed_times:
            ep95 = _percentile(elapsed_times, 95)
            percentiles["elapsed_p95"] = ep95
            if ep95 > settings.response_p95_s:
                issues.append((
                    ValidationLevel.WARNING,
                    f"응답시간 p95 {ep95:.2f}s > {settings.response_p95_s}s",
                ))

            if n_elapsed < 5:
                issues.append((
                    ValidationLevel.WARNING,
                    f"응답시간 샘플 수 부족 ({n_elapsed}개 < 5) — 분위수 신뢰도 낮음",
                ))

        if not issues:
            ttft_summary = (
                f"TTFT p50={percentiles.get('ttft_p50', 0):.3f}s "
                f"p95={percentiles.get('ttft_p95', 0):.3f}s "
                f"p99={percentiles.get('ttft_p99', 0):.3f}s"
            ) if ttfts else "TTFT 샘플 없음"
            return ValidationResult(
                status=True,
                level=ValidationLevel.PASS,
                message=f"성능 분포 정상 — {ttft_summary}",
                value=percentiles,
                model=model,
                check="performance_distribution",
            )

        _priority = {ValidationLevel.FAIL: 2, ValidationLevel.WARNING: 1, ValidationLevel.PASS: 0}
        worst_level = max(issues, key=lambda t: _priority[t[0]])[0]
        return ValidationResult(
            status=worst_level != ValidationLevel.FAIL,
            level=worst_level,
            message="; ".join(msg for _, msg in issues),
            value=percentiles,
            model=model,
            check="performance_distribution",
        )

    # ── finish_reason 정합성 검증 ──

    def validate_finish_reason(
        self,
        model: str,
        finish_reason: Optional[str],
        content: str,
        max_tokens: Optional[int] = None,
    ) -> ValidationResult:
        """finish_reason 유효성과 content 길이 간 정합성을 검증한다.

        정합성 규칙 (PART2 요구사항 ②):
          - finish_reason=stop   → content가 비어 있으면 Error
          - finish_reason=length → max_tokens 제공 시, content 추정 토큰이
                                   max_tokens의 80% 미만이면 Warning
                                   (잘림 없이 length가 반환된 비정상 케이스)
          - finish_reason=content_filter → Warning (콘텐츠 제한 발생)
        """
        if finish_reason not in _VALID_FINISH_REASONS:
            return ValidationResult(
                status=False,
                level=ValidationLevel.FAIL,
                message=f"유효하지 않은 finish_reason: '{finish_reason}'",
                value=finish_reason,
                model=model,
                check="finish_reason",
                blocking=True,  # 무결성 위반 → Blocking
            )

        if finish_reason == "stop" and not content.strip():
            return ValidationResult(
                status=False,
                level=ValidationLevel.FAIL,
                message="finish_reason=stop인데 content가 비어 있음",
                value={"finish_reason": finish_reason, "content_length": len(content)},
                model=model,
                check="finish_reason",
                blocking=True,
            )

        if finish_reason == "length":
            if max_tokens is not None:
                # 한글 포함 응답은 문자 수 ÷ 2 ≈ 토큰 수 (근사)
                est_tokens = max(len(content) // 2, 1)
                ratio = est_tokens / max_tokens
                if ratio < 0.8:
                    return ValidationResult(
                        status=True,
                        level=ValidationLevel.WARNING,
                        message=(
                            f"finish_reason=length이나 추정 토큰({est_tokens}) < "
                            f"max_tokens({max_tokens})의 80% — 비정상 잘림 의심"
                        ),
                        value={"finish_reason": finish_reason, "est_tokens": est_tokens, "max_tokens": max_tokens},
                        model=model,
                        check="finish_reason",
                    )
            return ValidationResult(
                status=True,
                level=ValidationLevel.WARNING,
                message="finish_reason=length: 응답이 max_tokens에 의해 잘림",
                value=finish_reason,
                model=model,
                check="finish_reason",
            )

        if finish_reason == "content_filter":
            return ValidationResult(
                status=True,
                level=ValidationLevel.WARNING,
                message="콘텐츠 필터에 의해 응답이 제한됨",
                value=finish_reason,
                model=model,
                check="finish_reason",
            )

        return ValidationResult(
            status=True,
            level=ValidationLevel.PASS,
            message=f"finish_reason 정상: '{finish_reason}'",
            value=finish_reason,
            model=model,
            check="finish_reason",
        )

    # ── 콘텐츠 품질 검증 ──

    def validate_content_quality(self, model: str, content: str) -> ValidationResult:
        """공백·null·반복 패턴 감지 및 한국어 비율을 측정한다."""
        if not content or not content.strip():
            return ValidationResult(
                status=False,
                level=ValidationLevel.FAIL,
                message="content가 비어 있거나 공백만 포함",
                value=content,
                model=model,
                check="content_quality",
            )

        issues: list[tuple[str, str]] = []

        rep_match = re.search(r"(.{20,}?)\1{2,}", content)
        if rep_match:
            issues.append(("warning", f"반복 패턴 감지: '{rep_match.group(1)[:30]}…'"))

        non_space = [c for c in content if not c.isspace()]
        total = len(non_space)
        korean = sum(1 for c in non_space if "가" <= c <= "힣")
        ratio = korean / total if total else 0.0

        if ratio < settings.korean_ratio_min:
            lvl = "warning" if ratio > 0 else "fail"
            issues.append((lvl, f"한국어 비율 미달 ({ratio:.1%} < {settings.korean_ratio_min:.0%})"))

        if not issues:
            return ValidationResult(
                status=True,
                level=ValidationLevel.PASS,
                message=f"콘텐츠 품질 정상 (한국어 {ratio:.1%})",
                value={"korean_ratio": ratio},
                model=model,
                check="content_quality",
            )

        has_error = any(lvl == "fail" for lvl, _ in issues)
        worst = ValidationLevel.FAIL if has_error else ValidationLevel.WARNING
        return ValidationResult(
            status=not has_error,
            level=worst,
            message="; ".join(msg for _, msg in issues),
            value={"korean_ratio": ratio},
            model=model,
            check="content_quality",
        )

    # ── Semantic Stability 검증 (PART1 5.2) ──

    def validate_semantic_stability(
        self,
        model: str,
        texts: list[str],
        similarity_threshold: Optional[float] = None,
    ) -> ValidationResult:
        """N개 응답 텍스트의 Semantic Stability Score를 측정한다.

        PART1 5.2 절차:
          1. OpenAI text-embedding-3-small으로 각 응답을 벡터화
          2. 코사인 유사도 기반 탐욕적 클러스터링
          3. 최대 클러스터 비율 = Stability Score
          4. Score ≥ settings.semantic_stability_min (0.95) → Pass

        OpenAI API를 사용할 수 없으면 Keyword Overlap 기반 폴백으로 전환한다.

        합격 기준: ≥ 95% (PART1 5.1)
        주의: Stability는 정확성과 직교한다. Hallucination Rate와 반드시 함께 해석할 것.
        """
        threshold = similarity_threshold if similarity_threshold is not None else settings.semantic_similarity_threshold

        if len(texts) < 2:
            return ValidationResult(
                status=True,
                level=ValidationLevel.WARNING,
                message=f"텍스트 수 부족 ({len(texts)}개) — 최소 2개 이상 필요",
                value={"stability_score": 1.0, "n": len(texts)},
                model=model,
                check="semantic_stability",
            )

        # ── 1차: OpenAI 임베딩 기반 ──
        try:
            embeddings = self._get_openai_embeddings(texts)
            method = "openai_embedding"
        except Exception as emb_err:
            # ── 폴백: 키워드 오버랩 기반 ──
            try:
                embeddings = self._get_keyword_embeddings(texts)
                method = "keyword_overlap_fallback"
            except Exception:
                return ValidationResult(
                    status=False,
                    level=ValidationLevel.FAIL,
                    message=f"임베딩 획득 실패: {emb_err}",
                    model=model,
                    check="semantic_stability",
                )

        clusters = _cluster_by_similarity(embeddings, threshold)
        max_cluster_size = max(len(c) for c in clusters)
        stability_score = max_cluster_size / len(texts)

        passed = stability_score >= settings.semantic_stability_min
        level = ValidationLevel.PASS if passed else ValidationLevel.FAIL
        method_note = "" if method == "openai_embedding" else f" [폴백: {method}]"

        return ValidationResult(
            status=passed,
            level=level,
            message=(
                f"Semantic Stability {stability_score:.1%} "
                f"(기준 ≥ {settings.semantic_stability_min:.0%}, "
                f"n={len(texts)}, threshold={threshold:.2f}){method_note}"
            ),
            value={
                "stability_score": stability_score,
                "n": len(texts),
                "max_cluster_size": max_cluster_size,
                "n_clusters": len(clusters),
                "threshold": threshold,
                "method": method,
            },
            model=model,
            check="semantic_stability",
        )

    @staticmethod
    def _get_openai_embeddings(texts: list[str]) -> list[list[float]]:
        """OpenAI text-embedding-3-small 모델로 임베딩을 생성한다."""
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _get_keyword_embeddings(texts: list[str]) -> list[list[float]]:
        """OpenAI API 미사용 시 TF 기반 키워드 벡터를 생성한다 (폴백).

        각 텍스트를 한글 어절 집합으로 변환해 이진 벡터화한다.
        코사인 유사도는 자카드 유사도와 근사적으로 동일하게 동작한다.
        """
        import re as _re
        tokenized = [
            set(_re.findall(r"[가-힣a-zA-Z0-9]+", t.lower())) for t in texts
        ]
        vocab = sorted(set().union(*tokenized))
        if not vocab:
            raise ValueError("vocab이 비어 있음")
        return [
            [1.0 if w in tok else 0.0 for w in vocab]
            for tok in tokenized
        ]

    # ── 리포트 생성 (PART1 7.1 Blocking / Non-Blocking 분리) ──

    def generate_report(self, results: list) -> str:
        """검증 결과를 Blocking 게이트 / Non-Blocking 집계로 분리해 출력한다.

        PART1 7.1 판정 로직:
          - Blocking 항목 중 하나라도 실패 → NO-GO
          - Blocking 통과 + Non-Blocking 기준 미달 → HOLD (조건부)
          - 전체 통과 → GO
        """
        from collections import defaultdict

        flat: list[ValidationResult] = []
        for item in results:
            if isinstance(item, ValidationResult):
                flat.append(item)
            elif isinstance(item, list):
                flat.extend(item)

        if not flat:
            return "검증 결과 없음"

        blocking_results = [r for r in flat if r.blocking]
        non_blocking_results = [r for r in flat if not r.blocking]

        # ── Blocking 게이트 집계 ──
        blocking_failed = [r for r in blocking_results if not r.status]
        gate_status = "✗ NO-GO" if blocking_failed else "✓ GO (Blocking 통과)"

        # ── Non-Blocking 모델별 집계 ──
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "warning": 0, "fail": 0}
        )
        for r in non_blocking_results:
            key = r.model or "unknown"
            counts[key][r.level.value] += 1

        lines = [
            "=" * 68,
            "  A Chat QA Validation Report",
            "=" * 68,
            "",
            f"  [Blocking 게이트]  {gate_status}",
        ]

        if blocking_failed:
            lines.append("  위반 항목:")
            for r in blocking_failed:
                lines.append(
                    f"    ✗ [{r.model or '-'}] {r.check}: {r.message}"
                )
        elif blocking_results:
            lines.append(f"  검사 항목 {len(blocking_results)}건 전체 통과")

        lines += [
            "",
            "  [Non-Blocking 집계] (모델별 Pass / Warning / Fail)",
            f"  {'Model':<26} {'PASS':>6} {'WARN':>6} {'FAIL':>6}",
            "  " + "-" * 50,
        ]
        for mdl, c in sorted(counts.items()):
            lines.append(
                f"  {mdl:<26} {c['pass']:>6} {c['warning']:>6} {c['fail']:>6}"
            )

        # ── 전체 판정 ──
        non_blocking_failed = [r for r in non_blocking_results if not r.status]
        if not blocking_failed and not non_blocking_failed:
            verdict = "GO"
        elif blocking_failed:
            verdict = "NO-GO"
        else:
            verdict = "HOLD (Non-Blocking 미달 — 원인 분석 후 재평가)"

        lines += [
            "",
            f"  최종 판정: {verdict}",
            "=" * 68,
        ]

        all_failed = [r for r in flat if not r.status]
        if all_failed:
            lines.append("\n  [실패 항목 목록]")
            # Blocking 먼저, 그 다음 Non-Blocking
            for r in sorted(all_failed, key=lambda x: (not x.blocking, x.level.value)):
                tag = r.level.value.upper()
                gate = "[BLOCKING] " if r.blocking else ""
                lines.append(
                    f"    [{tag}] {gate}{r.model or '-':20s} / {r.check:24s}: {r.message}"
                )

        return "\n".join(lines)
