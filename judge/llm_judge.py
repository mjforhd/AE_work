"""
llm_judge.py — LLM Judge 자동 평가 (가산점)

각 모델의 응답을 4가지 기준으로 1~5점 채점한다.
  - 정확성 (Accuracy)      : 사실적 정확도
  - 유용성 (Usefulness)    : 사용자에 대한 실질적 도움
  - 자연스러움 (Naturalness): 자연스럽고 읽기 쉬운 표현
  - 간결성 (Conciseness)   : 핵심 전달의 효율성

Self-evaluation 편향(Self-evaluation Bias) 방지 전략 (PART1 6.4 반영):

  [문제 정의]
  LLM이 자신의 응답을 스스로 심사할 때 발생하는 편향.
  예: GPT-4o가 작성한 답안을 GPT-4o가 채점하면 자기 스타일에 관대하거나
      반대로 과도하게 낮게 평가하는 경향이 생겨 "동일 기준 비교"(PART1 4장) 전제가 붕괴된다.

  [이상적 해결 — 오픈소스 중립 Judge]
  Meta Llama-3-70b-Instruct 같은 평가 대상 3개 모델과 무관한 오픈소스 모델을
  Judge로 사용하면 벤더 편향을 원천 차단할 수 있다.
  단, 본 구현은 별도 GPU 인프라 없이 즉시 실행 가능해야 하므로 아래 현실적 대안을 채택.

  [채택 전략 — Cross-evaluation + 보완 장치]
  - 완화 방법 1 — Cross-evaluation (핵심): 피평가 모델 ≠ 심사 모델.
      · claude-sonnet-4-6 응답 → GPT-4o가 심사
      · gpt-4o / gemini-2.0-flash 응답 → Claude가 심사
    교차 심사가 불가능한 경우(API 키 미설정 등) 타 가용 모델로 자동 폴백.
  - 완화 방법 2 — Blind Review: Judge 프롬프트에 모델명을 노출하지 않음.
    텍스트 품질만으로 채점 → 특정 벤더 선입견 차단.
  - 완화 방법 3 — Few-shot 앵커링: 우수·오류·장황 응답 예시 3개로 채점 기준 고정.
    응답 스타일이 아닌 품질 지표(정확성·유용성 등)에 집중하도록 강제.
  - 완화 방법 4 — 비결정성 통제: score()가 Judge를 n=3회 실행 → 기준별 중앙값(Median) 반환.
    stdev > 0.5 초과 기준별로 Human Review Required를 logging.warning으로 기록.

실행:
    python -m judge.llm_judge
    python -m judge.llm_judge --prompts 3 --consistency-runs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import anthropic
import openai as _openai
from rich import box
from rich.console import Console
from rich.table import Table

from config.settings import settings, SUPPORTED_MODELS, MODEL_API_ID

# Cross-evaluation 매핑: 피평가 모델 → 심사 모델 (PART1 6.4 ②)
# 동일 모델이 자신을 심사하지 않도록 반대 벤더로 교차 배정한다.
_CROSS_JUDGE_MAP: dict[str, str] = {
    "claude-sonnet-4-6": "gpt-4o",       # Claude 응답 → GPT-4o 심사
    "gpt-4o": "claude-sonnet-4-6",        # GPT 응답 → Claude 심사
    "gemini-2.0-flash": "claude-sonnet-4-6",  # Gemini 응답 → Claude 심사
}

_CLAUDE_MODEL_ID = MODEL_API_ID["claude-sonnet-4-6"]
_GPT_MODEL_ID = MODEL_API_ID["gpt-4o"]

CRITERIA = ["accuracy", "usefulness", "naturalness", "conciseness"]
CRITERIA_KR = {
    "accuracy": "정확성",
    "usefulness": "유용성",
    "naturalness": "자연스러움",
    "conciseness": "간결성",
}

# ── Few-shot 예시: 3가지 품질 수준을 명시해 채점 앵커를 고정한다 ──
_FEW_SHOT_EXAMPLES = """
[예시 1 — 우수한 응답]
질문: "한국의 수도는 어디인가요?"
응답: "한국의 수도는 서울입니다. 서울은 약 천만 명이 거주하는 대도시로, 정치·경제·문화의 중심지입니다."
출력: {"accuracy": 5, "usefulness": 4, "naturalness": 5, "conciseness": 4, "reason": "정확하고 배경 정보도 유용하나 질문 범위보다 다소 길다"}

[예시 2 — 사실 오류]
질문: "파이썬(Python) 프로그래밍 언어란 무엇인가요?"
응답: "파이썬은 아프리카에 서식하는 대형 뱀의 일종입니다."
출력: {"accuracy": 1, "usefulness": 1, "naturalness": 3, "conciseness": 4, "reason": "질문의 맥락(프로그래밍 언어)을 완전히 오해했다"}

[예시 3 — 과도하게 장황함]
질문: "물의 끓는점은 몇 도인가요?"
응답: "물에 대해 설명드리겠습니다. 물은 H₂O 분자로 이루어져 있으며 어는점은 0°C, 끓는점은 100°C(1기압 기준)입니다. 또한 물은 지구에서 가장 중요한 물질 중 하나로 생명 유지에 필수적입니다. 바다, 강, 구름 등 다양한 형태로 존재합니다..."
출력: {"accuracy": 5, "usefulness": 3, "naturalness": 4, "conciseness": 2, "reason": "정확하지만 단순 질문에 불필요한 정보가 과도하다"}
""".strip()

_JUDGE_SYSTEM = (
    "당신은 AI 응답의 품질을 객관적으로 평가하는 전문 심사관입니다. "
    "응답을 생성한 AI 모델명은 알려주지 않으므로, 텍스트 품질만을 기준으로 공정하게 채점하세요."
)

_JUDGE_USER_TEMPLATE = """\
아래 기준으로 AI 응답을 각 1~5점으로 채점하세요.

평가 기준:
- accuracy    (정확성): 사실적·논리적으로 정확한가?
- usefulness  (유용성): 질문에 실질적으로 도움이 되는가?
- naturalness (자연스러움): 표현이 자연스럽고 읽기 쉬운가?
- conciseness (간결성): 핵심을 불필요한 내용 없이 전달하는가?

{few_shot}

이제 다음 질문과 응답을 평가하세요. 반드시 JSON 한 줄로만 출력하세요.

질문: {prompt}
응답: {response}

출력 형식 (JSON만, 설명 없음):
{{"accuracy": <1-5>, "usefulness": <1-5>, "naturalness": <1-5>, "conciseness": <1-5>, "reason": "<한 문장 이유>"}}"""


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class ScoreResult:
    model: str
    prompt: str
    accuracy: float       # n회 실행 기준별 중앙값(Median)
    usefulness: float
    naturalness: float
    conciseness: float
    reason: str
    total: float = field(init=False)
    needs_human_review: bool = False   # stdev > 0.5 기준 이상 시 True
    judge_runs: int = 1                # 실제 채점 반복 횟수
    parse_error: Optional[str] = None

    def __post_init__(self) -> None:
        self.total = (self.accuracy + self.usefulness + self.naturalness + self.conciseness) / 4


# ──────────────────────────────────────────────
# LLMJudge
# ──────────────────────────────────────────────

class LLMJudge:
    def __init__(self) -> None:
        self._anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._openai = _openai.OpenAI(api_key=settings.openai_api_key)
        self.console = Console()

    # ── 공개 API ──

    def score(
        self,
        prompt: str,
        responses: dict[str, str],
        runs: int = 3,
    ) -> dict[str, ScoreResult]:
        """여러 모델의 응답을 Cross-evaluation으로 n회 채점 후 기준별 Median을 반환한다.

        PART1 6.4 ①②③ 반영:
          - Cross-evaluation: 피평가 모델 ≠ 심사 모델 (Self-evaluation 편향 차단)
          - Blind Review: 모델명을 Judge 프롬프트에서 제거
          - 비결정성 통제: Judge를 runs=3회 실행 → 기준별 Median 채택.
            stdev > 0.5 기준은 Human Review Required로 logging.warning 기록.

        Args:
            prompt: 원본 사용자 질문
            responses: {model_name: response_text}
            runs: Judge 반복 실행 횟수 (기본 3회)

        Returns:
            {model_name: ScoreResult}  — 점수는 runs회 median 기반
        """
        results: dict[str, ScoreResult] = {}
        for model, response_text in responses.items():
            judge_model = self._select_judge(model)

            # ── runs회 채점 수집 ──
            runs_data: list[dict] = []
            last_reason = ""
            for _ in range(runs):
                raw = self._call_judge(prompt, response_text, judge_model)
                parsed = self._parse_scores(raw)
                if "_error" not in parsed:
                    runs_data.append(parsed)
                    last_reason = parsed.get("reason", last_reason)

            if not runs_data:
                results[model] = ScoreResult(
                    model=model, prompt=prompt,
                    accuracy=0.0, usefulness=0.0, naturalness=0.0, conciseness=0.0,
                    reason="", judge_runs=0, parse_error="모든 실행에서 파싱 실패",
                )
                continue

            # ── 기준별 Median 계산 + stdev 검사 ──
            median_scores: dict[str, float] = {}
            needs_review = False
            for criterion in CRITERIA:
                values = [d[criterion] for d in runs_data if criterion in d]
                if not values:
                    median_scores[criterion] = 0.0
                    continue
                median_scores[criterion] = statistics.median(values)
                if len(values) >= 2:
                    sd = statistics.stdev(values)
                    if sd > 0.5:
                        needs_review = True
                        logger.warning(
                            "[LLMJudge] Human Review Required — "
                            "model=%s  criterion=%s  stdev=%.3f  values=%s",
                            model, criterion, sd, values,
                        )

            results[model] = ScoreResult(
                model=model,
                prompt=prompt,
                accuracy=median_scores.get("accuracy", 0.0),
                usefulness=median_scores.get("usefulness", 0.0),
                naturalness=median_scores.get("naturalness", 0.0),
                conciseness=median_scores.get("conciseness", 0.0),
                reason=last_reason,
                needs_human_review=needs_review,
                judge_runs=len(runs_data),
            )
        return results

    def validate_consistency(
        self,
        prompt: str,
        response: str,
        model: str = "gpt-4o",
        runs: int = 3,
    ) -> bool:
        """동일 응답에 대해 `runs`회 채점하고 기준별 stdev가 0.5 이하인지 확인한다.

        stdev > 0.5 인 기준이 하나라도 있으면 Judge 채점이 불안정한 것으로 판단해
        Human Review Required를 logging.warning으로 기록한다.
        score()가 내부에서 이미 비결정성을 통제하므로, 이 메서드는 독립적인
        Judge 안정성 감사(Audit) 용도로 사용한다.

        Returns:
            True  = 일관성 있음 (stdev ≤ 0.5 전 기준)
            False = 불일치 감지 — Human Review Required
        """
        judge_model = self._select_judge(model)
        runs_data: list[dict[str, float]] = []
        for _ in range(runs):
            raw = self._call_judge(prompt, response, judge_model)
            parsed = self._parse_scores(raw)
            if "_error" not in parsed:
                runs_data.append(parsed)

        if len(runs_data) < 2:
            logger.warning("[LLMJudge] validate_consistency: 파싱 성공 샘플 부족 (%d/%d) — Human Review", len(runs_data), runs)
            return False

        consistent = True
        for criterion in CRITERIA:
            values = [d[criterion] for d in runs_data if criterion in d]
            if len(values) < 2:
                continue
            sd = statistics.stdev(values)
            if sd > 0.5:
                consistent = False
                logger.warning(
                    "[LLMJudge] Human Review Required — "
                    "model=%s  criterion=%s  stdev=%.3f  values=%s",
                    model, criterion, sd, values,
                )
        return consistent

    # ── 출력 ──

    def display_results(
        self,
        all_scores: list[ScoreResult],
        human_review_items: list[dict],
    ) -> None:
        """모델별 점수 비교표와 Human Review 목록을 콘솔에 출력한다."""
        self._print_score_table(all_scores)
        self._print_human_review(human_review_items)

    # ── 내부 메서드 ──

    def _select_judge(self, evaluated_model: str) -> str:
        """Cross-evaluation 매핑에서 심사 모델을 선택한다.

        preferred 모델의 API 키가 없으면 다른 가용 모델로 폴백한다.
        """
        preferred = _CROSS_JUDGE_MAP.get(evaluated_model, "claude-sonnet-4-6")

        if preferred == "gpt-4o" and settings.openai_api_key:
            return "gpt-4o"
        if preferred == "claude-sonnet-4-6" and settings.anthropic_api_key:
            return "claude-sonnet-4-6"

        # 폴백: API 키가 있는 아무 모델이나 사용 (단, 피평가 모델 제외)
        for fallback in ["claude-sonnet-4-6", "gpt-4o"]:
            if fallback == evaluated_model:
                continue
            if fallback == "claude-sonnet-4-6" and settings.anthropic_api_key:
                return fallback
            if fallback == "gpt-4o" and settings.openai_api_key:
                return fallback

        return "claude-sonnet-4-6"  # 최후 수단

    def _call_judge(self, prompt: str, response: str, judge_model: str) -> str:
        """지정된 Judge 모델 API를 호출해 원시 텍스트 응답을 반환한다.

        temperature=0 고정으로 채점 일관성을 최대화한다.
        """
        user_content = _JUDGE_USER_TEMPLATE.format(
            few_shot=_FEW_SHOT_EXAMPLES,
            prompt=prompt,
            response=response,
        )
        if judge_model == "gpt-4o":
            completion = self._openai.chat.completions.create(
                model=_GPT_MODEL_ID,
                max_tokens=256,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
            return completion.choices[0].message.content.strip()
        else:
            message = self._anthropic.messages.create(
                model=_CLAUDE_MODEL_ID,
                max_tokens=256,
                temperature=0.0,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            return message.content[0].text.strip()

    @staticmethod
    def _parse_scores(raw: str) -> dict:
        """
        Claude 응답 텍스트에서 JSON 점수를 추출한다.

        1차: 직접 json.loads()
        2차: 텍스트에서 첫 '{' ~ 마지막 '}' 구간 추출
        3차: regex로 JSON 블록 탐색
        """
        # 1차 시도
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2차 시도: 중괄호 경계 추출
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass

        # 3차 시도: regex
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {"_error": f"JSON 파싱 실패: {raw[:80]}"}

    def _print_score_table(self, all_scores: list[ScoreResult]) -> None:
        if not all_scores:
            self.console.print("[red]채점 결과 없음[/]")
            return

        # 모델별 기준별 평균 집계
        from collections import defaultdict
        model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: {c: [] for c in CRITERIA})
        for s in all_scores:
            for c in CRITERIA:
                val = getattr(s, c)
                if val > 0:
                    model_scores[s.model][c].append(val)

        # Judge 반복 횟수 확인 (대표값)
        sample_runs = next((s.judge_runs for s in all_scores if s.judge_runs > 0), 1)

        table = Table(
            title=f"[bold]LLM Judge 모델 점수 비교표[/]  [dim](Judge {sample_runs}회 실행 · Median 기준)[/]",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("모델", style="bold", min_width=24)
        for c in CRITERIA:
            table.add_column(CRITERIA_KR[c], justify="center", min_width=10)
        table.add_column("종합 평균", justify="center", min_width=10, style="bold")
        table.add_column("Human Review", justify="center", min_width=13)

        def _color(v: float) -> str:
            if v >= 4.0:
                return "green"
            return "yellow" if v >= 3.0 else "red"

        # 모델별 Human Review 필요 여부 집계
        model_needs_review: dict[str, bool] = {}
        for s in all_scores:
            model_needs_review[s.model] = model_needs_review.get(s.model, False) or s.needs_human_review

        for model in SUPPORTED_MODELS:
            if model not in model_scores:
                continue
            scores_by_criterion = model_scores[model]
            avgs = {c: statistics.mean(v) if v else 0.0 for c, v in scores_by_criterion.items()}
            total = statistics.mean(avgs.values()) if avgs else 0.0
            review_flag = (
                "[bold yellow]⚠ Required[/]" if model_needs_review.get(model)
                else "[green]✓ Pass[/]"
            )
            table.add_row(
                model,
                *[f"[{_color(avgs[c])}]{avgs[c]:.2f}[/]" for c in CRITERIA],
                f"[{_color(total)}][bold]{total:.2f}[/][/]",
                review_flag,
            )

        self.console.print()
        self.console.print(table)

    def _print_human_review(self, items: list[dict]) -> None:
        if not items:
            self.console.print("\n[green]✓ Human Review 필요 항목 없음[/]\n")
            return

        self.console.print(f"\n[bold yellow]⚠ Human Review 필요 항목 ({len(items)}건)[/]")
        for i, item in enumerate(items, 1):
            self.console.print(
                f"  {i}. [cyan]{item['model']}[/]  —  {item['prompt'][:60]}…"
            )
        self.console.print()


# ──────────────────────────────────────────────
# CLI 진입점: python -m judge.llm_judge
# ──────────────────────────────────────────────

def _load_judge_prompts() -> list[str]:
    """docs/prompt_set.json 에서 judge_prompts를 로드한다."""
    prompt_set_path = Path(__file__).parent.parent / "docs" / "prompt_set.json"
    data = json.loads(prompt_set_path.read_text(encoding="utf-8"))
    return [e["text"] for e in data["judge_prompts"]]


_JUDGE_PROMPTS = _load_judge_prompts()


def _run(n_prompts: int, consistency_runs: int) -> None:
    from core.client import AChatClient

    judge = LLMJudge()
    client = AChatClient()
    console = Console()

    prompts = _JUDGE_PROMPTS[:n_prompts]
    all_scores: list[ScoreResult] = []
    human_review_items: list[dict] = []

    console.rule("[bold]LLM Judge 평가 시작[/]")
    console.print(
        f"  프롬프트: {len(prompts)}개 × 모델: {len(SUPPORTED_MODELS)}개"
        f"  |  일관성 검증: {consistency_runs}회/응답\n"
    )

    for prompt in prompts:
        console.print(f"[bold cyan]Q:[/] {prompt}")

        # 1. 각 모델 응답 수집
        responses: dict[str, str] = {}
        for model in SUPPORTED_MODELS:
            try:
                resp = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                responses[model] = resp["choices"][0]["message"]["content"]
                console.print(f"  [dim]{model}: {responses[model][:60]}…[/]")
            except Exception as exc:
                console.print(f"  [red]{model} 오류: {exc}[/]")

        if not responses:
            continue

        # 2. 채점
        scored = judge.score(prompt, responses)
        all_scores.extend(scored.values())

        # 3. 일관성 검증 (첫 번째 모델만 대표로 실행해 비용 절감)
        first_model = next(iter(responses))
        is_consistent = judge.validate_consistency(
            prompt, responses[first_model], model=first_model, runs=consistency_runs
        )
        if not is_consistent:
            human_review_items.append({
                "model": first_model,
                "prompt": prompt,
                "reason": "채점 분산 ±0.5 초과",
            })

        console.print()

    # 4. 결과 출력
    judge.display_results(all_scores, human_review_items)
    console.rule()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Judge 평가 실행")
    parser.add_argument("--prompts", type=int, default=3, help="사용할 프롬프트 수 (최대 3)")
    parser.add_argument("--consistency-runs", type=int, default=3, help="일관성 검증 반복 횟수")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _run(n_prompts=min(args.prompts, len(_JUDGE_PROMPTS)), consistency_runs=args.consistency_runs)
