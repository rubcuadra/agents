"""
Pricing constants for different LLM models.
"""

from dataclasses import dataclass

from llm.supported_llm_models import BedrockModel, OpenAIModel


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""

    input_price_per_1k_tokens: float
    output_price_per_1k_tokens: float


class ComputePricing:
    """
    Pricing information for different LLM providers and models.

    For a given model, the pricing is per 1K tokens. This is normalized to make it easier
    to compare costs between models.

    - Bedrock pricing is per 1K tokens.
    - OpenAI pricing is per 1M tokens.
    """

    # AWS Bedrock pricing (as of June 2025)
    # Bedrock pricing is per 1K tokens.
    # New model ids for AWS terraform: https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-permissions.html
    # https://aws.amazon.com/bedrock/pricing/
    BEDROCK_PRICING: dict[str, ModelPricing] = {
        BedrockModel.NOVA_MICRO: ModelPricing(
            input_price_per_1k_tokens=0.000035,
            output_price_per_1k_tokens=0.00014,
        ),
        BedrockModel.NOVA_PRO: ModelPricing(
            input_price_per_1k_tokens=0.0008,
            output_price_per_1k_tokens=0.0032,
        ),
        BedrockModel.NOVA_LITE: ModelPricing(
            input_price_per_1k_tokens=0.00006,
            output_price_per_1k_tokens=0.00024,
        ),
        BedrockModel.NOVA_PREMIER: ModelPricing(
            input_price_per_1k_tokens=0.0025,
            output_price_per_1k_tokens=0.0125,
        ),
        BedrockModel.CLAUDE_SONNET_4: ModelPricing(
            input_price_per_1k_tokens=0.003,
            output_price_per_1k_tokens=0.015,
        ),
        BedrockModel.CLAUDE_SONNET_4_5: ModelPricing(
            input_price_per_1k_tokens=0.0033,
            output_price_per_1k_tokens=0.0165,
        ),
        BedrockModel.CLAUDE_3_HAIKU: ModelPricing(
            input_price_per_1k_tokens=0.00025,
            output_price_per_1k_tokens=0.00125,
        ),
        BedrockModel.CLAUDE_3_5_SONNET: ModelPricing(
            input_price_per_1k_tokens=0.003,
            output_price_per_1k_tokens=0.015,
        ),
        BedrockModel.CLAUDE_3_7_SONNET: ModelPricing(
            input_price_per_1k_tokens=0.003,
            output_price_per_1k_tokens=0.015,
        ),
        BedrockModel.CLAUDE_4_5_HAIKU: ModelPricing(
            input_price_per_1k_tokens=0.001,
            output_price_per_1k_tokens=0.005,
        ),
        BedrockModel.PIXTRAL_LARGE: ModelPricing(
            input_price_per_1k_tokens=0.002,
            output_price_per_1k_tokens=0.006,
        ),
        BedrockModel.OPENAI_GPT_OSS_120B: ModelPricing(
            input_price_per_1k_tokens=0.00015,
            output_price_per_1k_tokens=0.00060,
        ),
        BedrockModel.OPENAI_GPT_OSS_20B: ModelPricing(
            input_price_per_1k_tokens=0.00007,
            output_price_per_1k_tokens=0.00020,
        ),
    }

    # OpenAI pricing (as of March 2024)
    # OpenAI pricing is per 1M tokens. Convert back to per million pricing by multiplying by 1,000.
    # https://openai.com/pricing
    OPENAI_PRICING: dict[str, ModelPricing] = {
        OpenAIModel.GPT4O: ModelPricing(
            input_price_per_1k_tokens=0.0025,
            output_price_per_1k_tokens=0.01,
        ),
        OpenAIModel.O1: ModelPricing(
            input_price_per_1k_tokens=0.015,
            output_price_per_1k_tokens=0.06,
        ),
        OpenAIModel.O1_MINI: ModelPricing(
            input_price_per_1k_tokens=0.0011,
            output_price_per_1k_tokens=0.0044,
        ),
        OpenAIModel.O3_MINI: ModelPricing(
            input_price_per_1k_tokens=0.0011,
            output_price_per_1k_tokens=0.0044,
        ),
        OpenAIModel.GPT4O_MINI: ModelPricing(
            input_price_per_1k_tokens=0.00015,
            output_price_per_1k_tokens=0.0006,
        ),
    }

    @classmethod
    def get_pricing(cls, model_id: str) -> ModelPricing:
        """
        Get pricing information for a specific model.

        Args:
            model_id: The ID of the model to get pricing for

        Returns:
            ModelPricing object with pricing information

        Raises:
            ValueError: If the model ID is not found in any pricing table
        """
        pricing: ModelPricing | None = None
        if model_id in cls.BEDROCK_PRICING:
            pricing = cls.BEDROCK_PRICING[model_id]
        elif model_id in cls.OPENAI_PRICING:
            pricing = cls.OPENAI_PRICING[model_id]

        if pricing is None:
            raise ValueError(f"Pricing not found for model: {model_id}")

        return pricing

    @classmethod
    def calculate_cost(
        cls,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate the cost for a given number of input and output tokens.

        Args:
            model_id: The ID of the model to calculate cost for
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            The total cost in dollars
        """
        pricing = cls.get_pricing(model_id)

        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k_tokens
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k_tokens

        return input_cost + output_cost