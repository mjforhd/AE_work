"""
공통 픽스처 및 pytest 설정.

마커:
  @pytest.mark.api   — 실제 API 키가 필요한 테스트. .env 미설정 시 자동 스킵.
  @pytest.mark.slow  — 실행 시간이 긴 테스트.
"""

import pytest

from config.settings import settings
from core.client import AChatClient
from core.validator import ResponseValidator

MODELS = ["gpt-4o", "claude-sonnet-4-6", "gemini-2.0-flash"]


# ──────────────────────────────────────────────
# pytest 마커 등록
# ──────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "api: 실제 API 키가 필요한 테스트 (requires .env)")
    config.addinivalue_line("markers", "slow: 실행 시간이 긴 테스트")


# ──────────────────────────────────────────────
# 결제 오류 감지
# ──────────────────────────────────────────────

def _is_billing_error(exc: BaseException) -> bool:
    """API 크레딧 부족·결제 오류인지 확인한다."""
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        "credit balance is too low",
        "insufficient_quota",
        "quota exceeded",
        "billing",
        "payment required",
        "you exceeded your current quota",
    ])


def _billing_skip_reason(exc: BaseException) -> str:
    """벤더별 결제 오류 안내 메시지를 반환한다."""
    msg = str(exc)
    if "anthropic" in type(exc).__module__:
        return f"Anthropic 크레딧 부족 — console.anthropic.com › Plans & Billing 에서 충전 필요\n  원본: {msg[:120]}"
    if "openai" in type(exc).__module__:
        return f"OpenAI 쿼터 초과 — platform.openai.com › Billing 확인 필요\n  원본: {msg[:120]}"
    return f"API 결제 오류 (SKIP) — {msg[:120]}"


# ──────────────────────────────────────────────
# API 키 자동 스킵 + 결제 오류 자동 스킵
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _skip_without_api_keys(request: pytest.FixtureRequest) -> None:
    """@pytest.mark.api가 붙은 테스트는 API 키가 없으면 자동 스킵한다."""
    if not request.node.get_closest_marker("api"):
        return

    missing = [
        name
        for name, value in [
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
            ("GOOGLE_API_KEY", settings.google_api_key),
        ]
        if not value
    ]
    if missing:
        pytest.skip(f"API 키 미설정 (스킵): {', '.join(missing)}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """테스트 실행 중 API 크레딧·결제 오류 발생 시 FAIL 대신 SKIP 처리한다.

    크레딧 부족은 코드 결함이 아니라 인프라 상태이므로 SKIP으로 분류한다.
    충전 후 재실행하면 정상 통과한다.
    """
    outcome = yield
    if outcome.excinfo is not None:
        _, exc_val, _ = outcome.excinfo
        if _is_billing_error(exc_val):
            try:
                pytest.skip(_billing_skip_reason(exc_val))
            except pytest.skip.Exception as skip_exc:
                outcome.force_exception(skip_exc)


# ──────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> AChatClient:
    """유효한 API 키로 생성한 AChatClient. 세션 전체에서 공유한다."""
    return AChatClient()


@pytest.fixture(scope="session")
def validator() -> ResponseValidator:
    return ResponseValidator()


@pytest.fixture
def bad_client_factory():
    """잘못된 API 키를 주입한 AChatClient를 생성하는 팩토리 픽스처."""
    def _make(model: str) -> AChatClient:
        bad_key = "invalid-api-key-test-xyz-000"
        if model == "gpt-4o":
            return AChatClient(openai_api_key=bad_key)
        elif model == "claude-sonnet-4-6":
            return AChatClient(anthropic_api_key=bad_key)
        else:
            return AChatClient(google_api_key=bad_key)
    return _make


@pytest.fixture
def simple_kr_message() -> list[dict]:
    """비용을 최소화하는 짧은 한국어 테스트 메시지."""
    return [{"role": "user", "content": "안녕하세요. 한 문장으로 짧게 인사해주세요."}]
