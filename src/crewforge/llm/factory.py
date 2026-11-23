"""Provider-agnostic BaseChatModel factory (Ollama default, OpenAI/Anthropic optional)."""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from crewforge.config.schemas import ModelProfileConfig, ModelProvider


class LLMFactoryError(Exception):
    """Raised when an LLM client cannot be constructed from a ModelProfileConfig."""


def _require_api_key(profile: ModelProfileConfig, default_env: str) -> str:
    env_var = profile.api_key_env or default_env
    api_key = os.environ.get(env_var)
    if not api_key:
        msg = (
            f"Provider '{profile.provider.value}' requires environment variable "
            f"'{env_var}' to be set."
        )
        raise LLMFactoryError(msg)
    return api_key


def _import_optional_class(module_name: str, class_name: str, install_extra: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        msg = (
            f"Optional dependency '{module_name}' is not installed. "
            f"Install it with: uv add {module_name} (or uv sync --extra {install_extra})"
        )
        raise LLMFactoryError(msg) from exc
    return getattr(module, class_name)


def build_chat_model(
    profile: ModelProfileConfig,
    *,
    fake_responses: Sequence[str] | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain BaseChatModel from a validated ModelProfileConfig."""
    if profile.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=profile.model,
            base_url=profile.base_url or "http://localhost:11434",
            temperature=profile.temperature,
            num_predict=profile.max_tokens,
            **profile.extra_kwargs,
        )

    if profile.provider == ModelProvider.OPENAI:
        _require_api_key(profile, "OPENAI_API_KEY")
        chat_cls = _import_optional_class("langchain_openai", "ChatOpenAI", "openai")
        return cast(
            BaseChatModel,
            chat_cls(
                model=profile.model,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                timeout=profile.timeout_seconds,
                base_url=profile.base_url,
                **profile.extra_kwargs,
            ),
        )

    if profile.provider == ModelProvider.ANTHROPIC:
        _require_api_key(profile, "ANTHROPIC_API_KEY")
        chat_cls = _import_optional_class("langchain_anthropic", "ChatAnthropic", "anthropic")
        return cast(
            BaseChatModel,
            chat_cls(
                model=profile.model,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                timeout=profile.timeout_seconds,
                **profile.extra_kwargs,
            ),
        )

    if profile.provider == ModelProvider.FAKE:
        responses = list(fake_responses) if fake_responses else ["[fake-llm-response]"]
        return GenericFakeChatModel(messages=iter(AIMessage(content=r) for r in responses))

    msg = f"Unsupported model provider: '{profile.provider}'"
    raise LLMFactoryError(msg)