"""
Simplified rate limiting functionality for LLM requests.
"""

import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import UTC, datetime

from llm.supported_llm_models import BedrockModel, OpenAIModel


class RateLimitStorage(ABC):
    """Abstract interface for rate limit storage backends."""

    @abstractmethod
    def get_current_usage(self, key: str) -> tuple[int, str]:
        """
        Get current usage count and last update timestamp for a key.

        Returns:
            Tuple of (count, last_update_timestamp)
            Returns (0, current_time) if key doesn't exist
        """
        pass

    @abstractmethod
    def update_usage(self, key: str, count: int, timestamp: str) -> bool:
        """
        Update usage count and timestamp for a key.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if storage backend is available."""
        pass


class InMemoryRateLimitStorage(RateLimitStorage):
    """Simple in-memory rate limit storage for development/testing."""

    def __init__(self):
        self._data = defaultdict(lambda: (0, datetime.now(UTC).isoformat()))

    def get_current_usage(self, key: str) -> tuple[int, str]:
        return self._data[key]

    def update_usage(self, key: str, count: int, timestamp: str) -> bool:
        self._data[key] = (count, timestamp)
        return True

    def is_available(self) -> bool:
        return True


class LLMRateLimiter:
    """
    Simplified rate limiter for LLM requests.
    """

    # Rate limits per model (requests per minute, tokens per minute)
    RATE_LIMITS = {
        # Conservative defaults
        "default": (10, 50000),
        # Bedrock models
        BedrockModel.CLAUDE_3_HAIKU: (20, 100000),
        BedrockModel.CLAUDE_3_5_SONNET: (15, 80000),
        BedrockModel.CLAUDE_4_5_HAIKU: (20, 100000),
        BedrockModel.CLAUDE_SONNET_4_5: (10, 60000),
        BedrockModel.NOVA_MICRO: (30, 150000),
        BedrockModel.NOVA_LITE: (25, 120000),
        BedrockModel.NOVA_PRO: (15, 80000),
        BedrockModel.NOVA_PREMIER: (10, 60000),
        # OpenAI models
        OpenAIModel.GPT4O_MINI: (50, 200000),
        OpenAIModel.GPT4O: (20, 100000),
        OpenAIModel.O1_MINI: (10, 50000),
        OpenAIModel.O1: (5, 30000),
    }

    def __init__(self, storage: RateLimitStorage, environment: str = "local"):
        self.storage = storage
        self.environment = environment

    def can_submit_request(
        self,
        model: BedrockModel | OpenAIModel | str,
        estimated_calls: int = 1,
        estimated_tokens: int = 1000,
    ) -> tuple[bool, float]:
        """
        Check if a request can be submitted based on rate limits.

        Args:
            model: The model to check limits for
            estimated_calls: Number of API calls
            estimated_tokens: Estimated token usage

        Returns:
            Tuple of (can_proceed, delay_seconds)
        """
        if not self.storage.is_available():
            return True, 0.0  # If storage unavailable, allow requests

        model_str = str(model)
        limits = self.RATE_LIMITS.get(model, self.RATE_LIMITS["default"])
        calls_per_minute, tokens_per_minute = limits

        # Get current usage
        key_calls = f"calls:{self.environment}:{model_str}"
        key_tokens = f"tokens:{self.environment}:{model_str}"

        calls_count, calls_timestamp = self.storage.get_current_usage(key_calls)
        tokens_count, tokens_timestamp = self.storage.get_current_usage(key_tokens)

        now = datetime.now(UTC)
        current_time = now.isoformat()

        # Reset counters if more than a minute has passed
        try:
            calls_time = datetime.fromisoformat(calls_timestamp.replace("Z", "+00:00"))
            tokens_time = datetime.fromisoformat(tokens_timestamp.replace("Z", "+00:00"))

            if (now - calls_time).total_seconds() >= 60:
                calls_count = 0
            if (now - tokens_time).total_seconds() >= 60:
                tokens_count = 0
        except Exception:
            # If timestamp parsing fails, reset counters
            calls_count = 0
            tokens_count = 0

        # Check if we would exceed limits
        if calls_count + estimated_calls > calls_per_minute:
            return False, 60.0  # Wait a full minute

        if tokens_count + estimated_tokens > tokens_per_minute:
            return False, 60.0  # Wait a full minute

        return True, 0.0

    def increment_usage(
        self,
        model: BedrockModel | OpenAIModel | str,
        calls: int = 1,
        tokens: int | None = None,
    ) -> None:
        """
        Increment usage counters after successful request.

        Args:
            model: The model used
            calls: Number of API calls to add
            tokens: Number of tokens to add (optional)
        """
        if not self.storage.is_available():
            return

        model_str = str(model)
        current_time = datetime.now(UTC).isoformat()

        # Update calls counter
        key_calls = f"calls:{self.environment}:{model_str}"
        calls_count, _ = self.storage.get_current_usage(key_calls)
        self.storage.update_usage(key_calls, calls_count + calls, current_time)

        # Update tokens counter if provided
        if tokens is not None:
            key_tokens = f"tokens:{self.environment}:{model_str}"
            tokens_count, _ = self.storage.get_current_usage(key_tokens)
            self.storage.update_usage(key_tokens, tokens_count + tokens, current_time)

    @staticmethod
    def conservative_token_estimate(text: str) -> int:
        """
        Provide a conservative estimate of token count from text.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(len(text) // 3, 1)


def create_rate_limit_storage() -> RateLimitStorage:
    """Create appropriate rate limit storage for the environment."""
    # For this standalone version, always use in-memory storage
    # In production, this could be Redis or another persistent storage
    return InMemoryRateLimitStorage()