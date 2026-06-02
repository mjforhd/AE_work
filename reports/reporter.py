"""
reporter.py — A Chat 모델 비교 리포트 자동 생성

5개 프롬프트를 3개 모델에 전송하고 아래 항목을 수치로 비교한다.
  - 평균 응답 시간 (s, 평균±표준편차)
  - TTFT (스트리밍 모드 기준, 프롬프트 3회 평균)
  - 평균 완료 토큰 수 (completion_tokens)
  - finish_reason 분포 (stop / length / content_filter 비율)
  - 한국어 응답 비율 (유니코드 AC00-D7A3 범위 기반)

출력: 콘솔 rich 테이블 + reports/benchmark_results.json

실행:
    python -m reports.reporter
    python -m reports.reporter --output path/to/output.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config.settings import settings, SUPPORTED_MODELS
from core.client import AChatClient

# A Chat 특화 검증을 위한 벤치마크 프롬프트 셋
#
# [설계 의도 — Grounding]
# 일반 지식 검증을 의도적으로 배제하고, A Chat 고유 리스크(도메인·복합제약·보안·결함 시나리오)를
# 사전 탐지하기 위해 설계됨. 프롬프트는 코드와 분리된 docs/prompt_set.json 에서 관리한다.
# → 비개발자(QA엔지니어, PM)도 코드 변경 없이 프롬프트를 추가·수정할 수 있다.
# → expected_korean_low=true 항목(P3·P5)은 한국어 비율 낮음이 의도된 결과임.
def _load_benchmark_prompts() -> tuple[list[str], list[str]]:
    """docs/prompt_set.json 에서 benchmark_prompts를 로드한다.

    JSON 파일이 없거나 파싱 오류 시 빈 리스트를 반환하지 않고 FileNotFoundError를 그대로 올린다.
    프롬프트 셋 누락은 침묵하면 안 되는 설정 오류이기 때문이다.
    """
    prompt_set_path = Path(__file__).parent.parent / "docs" / "prompt_set.json"
    data = json.loads(prompt_set_path.read_text(encoding="utf-8"))
    entries = data["benchmark_prompts"]
    texts = [e["text"] for e in entries]
    labels = [f"{e['id']}: {e['label']}" for e in entries]
    return texts, labels


BENCHMARK_PROMPTS, BENCHMARK_LABELS = _load_benchmark_prompts()

# TTFT 측정 횟수 (비용 절감을 위해 3회)
TTFT_SAMPLE_COUNT = 3


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class PromptResult:
    prompt_index: int
    model: str
    elapsed_time: float
    completion_tokens: int
    prompt_tokens: int
    finish_reason: Optional[str]
    content: str
    korean_ratio: float
    error: Optional[str] = None


@dataclass
class ModelStats:
    model: str
    sample_count: int
    avg_elapsed: float
    std_elapsed: float
    elapsed_p95: float            # 전체 응답시간 p95 (PART1 2.1)
    avg_ttft: float
    std_ttft: float
    ttft_p50: float               # TTFT p50 (PART1 2.1)
    ttft_p95: float               # TTFT p95
    ttft_p99: float               # TTFT p99
    avg_tokens: float
    std_tokens: float
    finish_reason_dist: dict[str, float]
    avg_korean_ratio: float


# ──────────────────────────────────────────────
# Reporter
# ──────────────────────────────────────────────

class Reporter:
    def __init__(self) -> None:
        self.client = AChatClient()
        self.console = Console()

    # ── 내부 측정 유틸 ──

    @staticmethod
    def _korean_ratio(text: str) -> float:
        """유니코드 AC00-D7A3 범위(한글 음절) 기반 한국어 비율을 반환한다."""
        non_space = [c for c in text if not c.isspace()]
        if not non_space:
            return 0.0
        korean = sum(1 for c in non_space if "가" <= c <= "힣")
        return korean / len(non_space)

    def _measure_ttft(self, model: str, messages: list, max_tokens: int = 150) -> float:
        """스트리밍 모드로 TTFT(첫 토큰 수신 시간)를 측정한다."""
        start = time.perf_counter()
        ttft: Optional[float] = None
        gen = self.client.chat_stream(model=model, messages=messages, max_tokens=max_tokens)
        try:
            for chunk in gen:
                if chunk["choices"][0]["delta"].get("content", "") and ttft is None:
                    ttft = time.perf_counter() - start
        finally:
            gen.close()
        return ttft if ttft is not None else (time.perf_counter() - start)

    # ── 벤치마크 실행 ──

    def _run_nonstream(
        self,
        progress: Progress,
        max_tokens: int,
    ) -> list[PromptResult]:
        """5개 프롬프트 × 3개 모델 비스트리밍 호출로 기본 지표를 수집한다."""
        results: list[PromptResult] = []
        task = progress.add_task(
            "[cyan]비스트리밍 측정[/]",
            total=len(BENCHMARK_PROMPTS) * len(SUPPORTED_MODELS),
        )
        for model in SUPPORTED_MODELS:
            for i, prompt in enumerate(BENCHMARK_PROMPTS):
                label = BENCHMARK_LABELS[i] if i < len(BENCHMARK_LABELS) else f"P{i+1}"
                progress.update(task, description=f"[cyan]{model}[/] {label}")
                messages = [{"role": "user", "content": prompt}]
                try:
                    start = time.perf_counter()
                    response = self.client.chat(model=model, messages=messages, max_tokens=max_tokens)
                    elapsed = time.perf_counter() - start
                    content = response["choices"][0]["message"]["content"]
                    results.append(PromptResult(
                        prompt_index=i,
                        model=model,
                        elapsed_time=elapsed,
                        completion_tokens=response["usage"]["completion_tokens"],
                        prompt_tokens=response["usage"]["prompt_tokens"],
                        finish_reason=response["choices"][0]["finish_reason"],
                        content=content,
                        korean_ratio=self._korean_ratio(content),
                    ))
                except Exception as exc:
                    results.append(PromptResult(
                        prompt_index=i, model=model,
                        elapsed_time=0, completion_tokens=0, prompt_tokens=0,
                        finish_reason=None, content="", korean_ratio=0.0,
                        error=str(exc),
                    ))
                progress.advance(task)
        return results

    def _run_ttft(self, progress: Progress) -> dict[str, list[float]]:
        """첫 N개 프롬프트를 스트리밍 모드로 호출해 모델별 TTFT 목록을 수집한다."""
        ttft_map: dict[str, list[float]] = {m: [] for m in SUPPORTED_MODELS}
        total = len(SUPPORTED_MODELS) * TTFT_SAMPLE_COUNT
        task = progress.add_task("[magenta]TTFT 측정[/]", total=total)
        for model in SUPPORTED_MODELS:
            for i in range(TTFT_SAMPLE_COUNT):
                progress.update(task, description=f"[magenta]TTFT {model}[/] [{i+1}/{TTFT_SAMPLE_COUNT}]")
                messages = [{"role": "user", "content": BENCHMARK_PROMPTS[i]}]
                try:
                    ttft = self._measure_ttft(model, messages)
                    ttft_map[model].append(ttft)
                except Exception:
                    pass
                progress.advance(task)
        return ttft_map

    def run(self, max_tokens: int = 200) -> tuple[list[PromptResult], dict[str, list[float]]]:
        """벤치마크 전체를 실행해 원시 측정값을 반환한다."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        with progress:
            results = self._run_nonstream(progress, max_tokens)
            ttft_map = self._run_ttft(progress)
        return results, ttft_map

    # ── 집계 ──

    def aggregate(
        self,
        results: list[PromptResult],
        ttft_map: dict[str, list[float]],
    ) -> list[ModelStats]:
        from core.validator import _percentile

        stats_list: list[ModelStats] = []
        for model in SUPPORTED_MODELS:
            valid = [r for r in results if r.model == model and not r.error]
            if not valid:
                continue

            times = [r.elapsed_time for r in valid]
            tokens = [r.completion_tokens for r in valid]
            ratios = [r.korean_ratio for r in valid]
            ttfts = ttft_map.get(model, [])

            finish_counts: dict[str, int] = {}
            for r in valid:
                k = r.finish_reason or "unknown"
                finish_counts[k] = finish_counts.get(k, 0) + 1
            n = len(valid)

            stats_list.append(ModelStats(
                model=model,
                sample_count=n,
                avg_elapsed=statistics.mean(times),
                std_elapsed=statistics.stdev(times) if n > 1 else 0.0,
                elapsed_p95=_percentile(times, 95),
                avg_ttft=statistics.mean(ttfts) if ttfts else 0.0,
                std_ttft=statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
                ttft_p50=_percentile(ttfts, 50) if ttfts else 0.0,
                ttft_p95=_percentile(ttfts, 95) if ttfts else 0.0,
                ttft_p99=_percentile(ttfts, 99) if ttfts else 0.0,
                avg_tokens=statistics.mean(tokens),
                std_tokens=statistics.stdev(tokens) if n > 1 else 0.0,
                finish_reason_dist={k: v / n for k, v in finish_counts.items()},
                avg_korean_ratio=statistics.mean(ratios),
            ))
        return stats_list

    # ── 콘솔 출력 ──

    def display(self, stats_list: list[ModelStats]) -> None:
        if not stats_list:
            self.console.print("[red]측정 결과 없음[/]")
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n = stats_list[0].sample_count

        table = Table(
            title=(
                f"[bold]A Chat QA 모델 비교 리포트[/]\n"
                f"{ts}  (비스트리밍 n={n}, TTFT n={TTFT_SAMPLE_COUNT})"
            ),
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
            min_width=88,
        )
        table.add_column("항목", style="bold", min_width=26, no_wrap=True)
        for s in stats_list:
            table.add_column(s.model, justify="center", min_width=20)

        def _tc(v: float, warn: float, err: float) -> str:
            if v <= warn:
                return "green"
            return "yellow" if v <= err else "red"

        def _ttft_color(v: float, threshold: float) -> str:
            return "green" if v <= threshold else "yellow"

        def _kc(v: float) -> str:
            if v >= settings.korean_ratio_min:
                return "green"
            return "yellow" if v >= 0.5 else "red"

        rows: list[tuple] = [
            # ── 응답 시간 ──
            (
                "평균 응답시간 (s)",
                *[
                    f"[{_tc(s.avg_elapsed, settings.response_time_warning_s, settings.response_time_error_s)}]"
                    f"{s.avg_elapsed:.2f}±{s.std_elapsed:.2f}[/]"
                    for s in stats_list
                ],
            ),
            (
                "응답시간 p95 (s) [SLA≤5s]",
                *[
                    f"[{_tc(s.elapsed_p95, settings.response_p95_s, settings.response_time_error_s)}]"
                    f"{s.elapsed_p95:.2f}[/]"
                    for s in stats_list
                ],
            ),
            # ── TTFT 분위수 (PART1 2.1) ──
            (
                f"TTFT p50 (s) [SLA≤{settings.ttft_p50_s}s]",
                *[
                    f"[{_ttft_color(s.ttft_p50, settings.ttft_p50_s)}]{s.ttft_p50:.3f}[/]"
                    for s in stats_list
                ],
            ),
            (
                f"TTFT p95 (s) [SLA≤{settings.ttft_p95_s}s]",
                *[
                    f"[{_ttft_color(s.ttft_p95, settings.ttft_p95_s)}]{s.ttft_p95:.3f}[/]"
                    for s in stats_list
                ],
            ),
            (
                f"TTFT p99 (s) [SLA≤{settings.ttft_p99_s}s]",
                *[
                    f"[{_ttft_color(s.ttft_p99, settings.ttft_p99_s)}]{s.ttft_p99:.3f}[/]"
                    for s in stats_list
                ],
            ),
            # ── 토큰 / finish_reason / 언어 ──
            (
                "평균 완료 토큰",
                *[f"{s.avg_tokens:.0f}±{s.std_tokens:.0f}" for s in stats_list],
            ),
            *[
                (
                    f"finish: {reason}",
                    *[
                        (
                            f"[{'green' if s.finish_reason_dist.get(reason, 0) >= settings.finish_reason_stop_min else 'yellow'}]"
                            f"{s.finish_reason_dist.get(reason, 0.0):.0%}[/]"
                            if reason == "stop"
                            else f"{s.finish_reason_dist.get(reason, 0.0):.0%}"
                        )
                        for s in stats_list
                    ],
                )
                for reason in ["stop", "length", "content_filter"]
            ],
            (
                "한국어 응답 비율",
                *[f"[{_kc(s.avg_korean_ratio)}]{s.avg_korean_ratio:.1%}[/]" for s in stats_list],
            ),
            (
                "유효 샘플 수",
                *[str(s.sample_count) for s in stats_list],
            ),
        ]

        for row in rows:
            table.add_row(*row)

        self.console.print()
        self.console.print(table)

        # ── 프롬프트 QA 의도 범례 ──
        self._display_prompt_legend()

        # ── Blocking 게이트 판정 (PART1 7.1) ──
        self._display_blocking_gate(stats_list)
        self.console.print()

    def _display_prompt_legend(self) -> None:
        """벤치마크 프롬프트별 QA 검증 의도를 범례로 출력한다."""
        from rich.panel import Panel
        lines = [f"  [cyan]{lbl}[/]" for lbl in BENCHMARK_LABELS]
        lines.append("")
        lines.append("  [dim]※ P3(JSON 출력)·P5(보안 거부)는 한국어 비율 낮음이 의도된 결과입니다.[/]")
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold]벤치마크 프롬프트 QA 의도[/]",
                border_style="blue",
                padding=(0, 2),
            )
        )

    def _display_blocking_gate(self, stats_list: list[ModelStats]) -> None:
        """PART1 7.1 Blocking 게이트를 모델별로 평가해 콘솔에 출력한다."""
        from rich.panel import Panel

        lines: list[str] = []
        overall_go = True

        for s in stats_list:
            issues: list[str] = []

            stop_ratio = s.finish_reason_dist.get("stop", 0.0)
            if stop_ratio < settings.finish_reason_stop_min:
                issues.append(
                    f"응답 완결성 {stop_ratio:.1%} < {settings.finish_reason_stop_min:.0%} [Blocking]"
                )

            if s.ttft_p99 > settings.ttft_p99_s:
                issues.append(
                    f"TTFT p99 {s.ttft_p99:.3f}s > {settings.ttft_p99_s}s"
                )

            if issues:
                overall_go = False
                color = "red"
                verdict = "⚠ HOLD/REVIEW"
            else:
                color = "green"
                verdict = "✓ Blocking 통과"

            line = f"[{color}]{s.model:<28}  {verdict}[/]"
            if issues:
                line += "\n" + "\n".join(f"  [yellow]→ {i}[/]" for i in issues)
            lines.append(line)

        gate_color = "green" if overall_go else "red"
        gate_label = "GO — 전 모델 Blocking 게이트 통과" if overall_go else "HOLD — 일부 모델 Blocking 게이트 미달"
        self.console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold {gate_color}][PART1 7.1 Blocking 게이트]  {gate_label}[/]",
                border_style=gate_color,
                padding=(0, 2),
            )
        )

    # ── JSON 저장 ──

    def save_json(
        self,
        stats_list: list[ModelStats],
        results: list[PromptResult],
        output_path: str = "reports/benchmark_results.json",
    ) -> str:
        data = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "prompts": [
                    {"label": BENCHMARK_LABELS[i], "text": p}
                    for i, p in enumerate(BENCHMARK_PROMPTS)
                ],
                "ttft_sample_count": TTFT_SAMPLE_COUNT,
                "thresholds": {
                    "response_time_warning_s": settings.response_time_warning_s,
                    "response_time_error_s": settings.response_time_error_s,
                    "response_p95_s": settings.response_p95_s,
                    "ttft_p50_s": settings.ttft_p50_s,
                    "ttft_p95_s": settings.ttft_p95_s,
                    "ttft_p99_s": settings.ttft_p99_s,
                    "finish_reason_stop_min": settings.finish_reason_stop_min,
                    "korean_ratio_min": settings.korean_ratio_min,
                },
            },
            "summary": [
                {
                    "model": s.model,
                    "sample_count": s.sample_count,
                    "avg_elapsed_s": round(s.avg_elapsed, 3),
                    "std_elapsed_s": round(s.std_elapsed, 3),
                    "elapsed_p95_s": round(s.elapsed_p95, 3),
                    "avg_ttft_s": round(s.avg_ttft, 3),
                    "std_ttft_s": round(s.std_ttft, 3),
                    "ttft_p50_s": round(s.ttft_p50, 3),
                    "ttft_p95_s": round(s.ttft_p95, 3),
                    "ttft_p99_s": round(s.ttft_p99, 3),
                    "avg_completion_tokens": round(s.avg_tokens, 1),
                    "std_completion_tokens": round(s.std_tokens, 1),
                    "finish_reason_distribution": {
                        k: round(v, 3) for k, v in s.finish_reason_dist.items()
                    },
                    "avg_korean_ratio": round(s.avg_korean_ratio, 3),
                }
                for s in stats_list
            ],
            "raw_results": [
                {
                    "prompt_index": r.prompt_index,
                    "prompt_label": BENCHMARK_LABELS[r.prompt_index] if r.prompt_index < len(BENCHMARK_LABELS) else "",
                    "prompt": BENCHMARK_PROMPTS[r.prompt_index],
                    "model": r.model,
                    "elapsed_s": round(r.elapsed_time, 3),
                    "completion_tokens": r.completion_tokens,
                    "prompt_tokens": r.prompt_tokens,
                    "finish_reason": r.finish_reason,
                    "korean_ratio": round(r.korean_ratio, 3),
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    # ── 공개 진입점 ──

    def generate(self, output_path: str = "reports/benchmark_results.json") -> None:
        self.console.rule("[bold]A Chat QA 벤치마크 시작[/]")
        self.console.print(
            f"  모델: {', '.join(SUPPORTED_MODELS)}\n"
            f"  프롬프트: {len(BENCHMARK_PROMPTS)}개  |  TTFT 샘플: {TTFT_SAMPLE_COUNT}회/모델\n"
        )
        results, ttft_map = self.run()
        stats_list = self.aggregate(results, ttft_map)
        self.display(stats_list)
        saved = self.save_json(stats_list, results, output_path)
        self.console.print(f"[bold green]✓ JSON 저장 완료:[/] {saved}")
        self.console.rule()


# ──────────────────────────────────────────────
# CLI 진입점: python -m reports.reporter
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Chat 모델 비교 리포트 생성")
    parser.add_argument(
        "--output",
        default="reports/benchmark_results.json",
        help="JSON 출력 경로 (기본값: reports/benchmark_results.json)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="모델별 max_tokens (기본값: 200)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reporter = Reporter()
    reporter.generate(output_path=args.output)
