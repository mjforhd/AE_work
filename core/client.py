"""
AChatClient — 단일 인터페이스로 GPT-4o / Claude / Gemini를 통합 호출.

실제 A Chat API가 없으므로 각 벤더 API를 직접 호출하되, 응답을 A Chat 스펙
(OpenAI Chat Completions 호환)으로 변환하는 어댑터 패턴을 적용한다.

A Chat 응답 스펙 (non-streaming):
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "model": "<model-name>",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
}

A Chat 청크 스펙 (streaming):
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion.chunk",
    "model": "<model-name>",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "..."}, "finish_reason": null}]
}
"""

import uuid
from typing import Generator, Optional

import openai
import anthropic
from google import genai as google_genai
from google.genai import types as genai_types

from config.settings import settings, SUPPORTED_MODELS, MODEL_API_ID


class AChatError(Exception):
    """클라이언트 레벨 오류. status_code로 HTTP 오류 코드를 나타낸다."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class AChatClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        # 테스트에서 잘못된 키를 주입할 수 있도록 키를 파라미터로 받는다
        oa_key = openai_api_key or settings.openai_api_key
        ant_key = anthropic_api_key or settings.anthropic_api_key
        goog_key = google_api_key or settings.google_api_key

        # 키가 없으면 None으로 저장한다.
        # SDK 클라이언트는 빈 문자열("")을 거부하므로 키가 있을 때만 생성한다.
        # 실제 API 호출 시점에 None이면 AChatError(401)을 발생시킨다.
        self._openai = openai.OpenAI(api_key=oa_key) if oa_key else None
        self._anthropic = anthropic.Anthropic(api_key=ant_key) if ant_key else None
        # google.genai는 Client 인스턴스 방식으로 키 격리가 보장된다 (전역 configure 불필요)
        self._gemini = google_genai.Client(api_key=goog_key) if goog_key else None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """동기 호출. A Chat 응답 스펙 dict를 반환한다."""
        self._validate(model, messages)
        if stream:
            raise AChatError("stream=True일 때는 chat_stream()을 사용하세요.", status_code=400)

        if model == "gpt-4o":
            return self._chat_openai(messages, temperature, max_tokens)
        elif model == "claude-sonnet-4-6":
            return self._chat_anthropic(messages, temperature, max_tokens)
        else:
            return self._chat_gemini(messages, temperature, max_tokens)

    def chat_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[dict, None, None]:
        """스트리밍 호출. A Chat 청크 스펙 dict를 yield한다."""
        self._validate(model, messages)

        if model == "gpt-4o":
            yield from self._stream_openai(messages, temperature, max_tokens)
        elif model == "claude-sonnet-4-6":
            yield from self._stream_anthropic(messages, temperature, max_tokens)
        else:
            yield from self._stream_gemini(messages, temperature, max_tokens)

    # ──────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────

    def _validate(self, model: str, messages: list) -> None:
        if model not in SUPPORTED_MODELS:
            raise AChatError(
                f"지원하지 않는 모델: '{model}'. 지원 모델: {SUPPORTED_MODELS}",
                status_code=400,
            )
        if not messages:
            raise AChatError("messages는 필수입니다.", status_code=400)

    # ──────────────────────────────────────────────
    # OpenAI (gpt-4o)
    # ──────────────────────────────────────────────

    def _chat_openai(self, messages: list, temperature: float, max_tokens: int) -> dict:
        if self._openai is None:
            raise AChatError("OPENAI_API_KEY가 설정되지 않았습니다.", status_code=401)
        response = self._openai.chat.completions.create(
            model=MODEL_API_ID["gpt-4o"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._adapt_openai(response)

    def _stream_openai(
        self, messages: list, temperature: float, max_tokens: int
    ) -> Generator[dict, None, None]:
        if self._openai is None:
            raise AChatError("OPENAI_API_KEY가 설정되지 않았습니다.", status_code=401)
        stream = self._openai.chat.completions.create(
            model=MODEL_API_ID["gpt-4o"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            yield {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": choice.delta.role or "assistant",
                        "content": choice.delta.content or "",
                    },
                    "finish_reason": choice.finish_reason,
                }],
            }

    def _adapt_openai(self, response) -> dict:
        choice = response.choices[0]
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content or "",
                },
                "finish_reason": choice.finish_reason,
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    # ──────────────────────────────────────────────
    # Anthropic (claude-sonnet-4-6)
    # ──────────────────────────────────────────────

    _ANTHROPIC_FINISH_REASON = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "stop",
    }

    def _split_system(self, messages: list) -> tuple[Optional[str], list]:
        """system 역할 메시지를 분리한다. Anthropic API는 system을 별도 파라미터로 받는다."""
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    def _map_anthropic_finish(self, stop_reason: Optional[str]) -> str:
        return self._ANTHROPIC_FINISH_REASON.get(stop_reason or "", "stop")

    def _chat_anthropic(self, messages: list, temperature: float, max_tokens: int) -> dict:
        if self._anthropic is None:
            raise AChatError("ANTHROPIC_API_KEY가 설정되지 않았습니다.", status_code=401)
        system, filtered = self._split_system(messages)
        kwargs = dict(
            model=MODEL_API_ID["claude-sonnet-4-6"],
            messages=filtered,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        response = self._anthropic.messages.create(**kwargs)
        return self._adapt_anthropic(response)

    def _stream_anthropic(
        self, messages: list, temperature: float, max_tokens: int
    ) -> Generator[dict, None, None]:
        if self._anthropic is None:
            raise AChatError("ANTHROPIC_API_KEY가 설정되지 않았습니다.", status_code=401)
        system, filtered = self._split_system(messages)
        kwargs = dict(
            model=MODEL_API_ID["claude-sonnet-4-6"],
            messages=filtered,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        with self._anthropic.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": "claude-sonnet-4-6",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }],
                }
            # 스트림 완료 후 finish_reason 포함 최종 청크
            final = stream.get_final_message()
            yield {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": "claude-sonnet-4-6",
                "choices": [{
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": self._map_anthropic_finish(final.stop_reason),
                }],
            }

    def _adapt_anthropic(self, response) -> dict:
        content = response.content[0].text if response.content else ""
        return {
            "id": response.id,
            "object": "chat.completion",
            "model": "claude-sonnet-4-6",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": self._map_anthropic_finish(response.stop_reason),
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        }

    # ──────────────────────────────────────────────
    # Google Gemini (gemini-2.0-flash) — google.genai SDK
    # ──────────────────────────────────────────────

    def _convert_to_gemini_contents(self, messages: list) -> tuple[Optional[str], list]:
        """messages를 Gemini contents 형식으로 변환하고 system_instruction을 분리한다."""
        system_instruction = None
        contents = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return system_instruction, contents

    def _map_gemini_finish(self, finish_reason) -> str:
        if finish_reason is None:
            return "stop"
        name = finish_reason.name if hasattr(finish_reason, "name") else str(finish_reason).upper()
        return {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
        }.get(name, "stop")

    def _build_gemini_config(
        self, system_instruction: Optional[str], temperature: float, max_tokens: int
    ) -> "genai_types.GenerateContentConfig":
        """GenerateContentConfig를 생성한다. system_instruction은 여기서 전달한다."""
        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _chat_gemini(self, messages: list, temperature: float, max_tokens: int) -> dict:
        if self._gemini is None:
            raise AChatError("GOOGLE_API_KEY가 설정되지 않았습니다.", status_code=401)
        system_instruction, contents = self._convert_to_gemini_contents(messages)
        config = self._build_gemini_config(system_instruction, temperature, max_tokens)
        response = self._gemini.models.generate_content(
            model=MODEL_API_ID["gemini-2.0-flash"],
            contents=contents,
            config=config,
        )
        return self._adapt_gemini(response)

    def _stream_gemini(
        self, messages: list, temperature: float, max_tokens: int
    ) -> Generator[dict, None, None]:
        if self._gemini is None:
            raise AChatError("GOOGLE_API_KEY가 설정되지 않았습니다.", status_code=401)
        system_instruction, contents = self._convert_to_gemini_contents(messages)
        config = self._build_gemini_config(system_instruction, temperature, max_tokens)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        finish_reason = "stop"
        for chunk in self._gemini.models.generate_content_stream(
            model=MODEL_API_ID["gemini-2.0-flash"],
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                fr = chunk.candidates[0].finish_reason
                if fr:
                    finish_reason = self._map_gemini_finish(fr)
            try:
                text = chunk.text
            except Exception:
                continue
            if text:
                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": "gemini-2.0-flash",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }],
                }

        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": "gemini-2.0-flash",
            "choices": [{
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": finish_reason,
            }],
        }

    def _adapt_gemini(self, response) -> dict:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            content = "".join(
                p.text for p in candidate.content.parts
                if hasattr(p, "text") and p.text
            )
        else:
            try:
                content = response.text or ""
            except Exception:
                content = ""
        finish_reason = self._map_gemini_finish(candidate.finish_reason)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": "gemini-2.0-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
