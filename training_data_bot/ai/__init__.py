"""
AI client module.

This module provides AI integration for text generation using
various providers (OpenAI, Anthropic, etc.).
"""

from training_data_bot.ai.client import AIClient
from training_data_bot.ai.providers import (
    BaseAIProvider,
    AIResponse,
    OpenAIProvider,
    AnthropicProvider,
)

__all__ = [
    "AIClient",
    "BaseAIProvider",
    "AIResponse",
    "OpenAIProvider",
    "AnthropicProvider",
]