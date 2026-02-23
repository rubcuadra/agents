"""
Supported LLM models for the service.
"""

from enum import StrEnum

from llm.config import config


def get_region_prefix(region: str) -> str:
    """
    Used to determine the region prefix for the Bedrock models
    https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
    """
    if region.startswith("us-"):
        return "us"
    elif region.startswith("eu-"):
        return "eu"
    else:
        raise ValueError(f"Unsupported region: {region}")


region_prefix = get_region_prefix(config.region)
is_us_region = region_prefix == "us"


class BedrockModel(StrEnum):
    """Supported AWS Bedrock models."""

    # Nova Models
    # https://aws.amazon.com/bedrock/pricing/#Nova
    NOVA_MICRO = f"{region_prefix}.amazon.nova-micro-v1:0"
    NOVA_PRO = f"{region_prefix}.amazon.nova-pro-v1:0"
    NOVA_LITE = f"{region_prefix}.amazon.nova-lite-v1:0"
    # Fall back to NOVA_PRO for non-US regions
    NOVA_PREMIER = (
        f"{region_prefix}.amazon.nova-premier-v1:0"
        if is_us_region
        else f"{region_prefix}.amazon.nova-pro-v1:0"
    )

    # Claude Models
    # https://aws.amazon.com/bedrock/pricing/#Claude
    CLAUDE_SONNET_4_5 = f"{region_prefix}.anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_SONNET_4 = f"{region_prefix}.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_3_5_SONNET = f"{region_prefix}.anthropic.claude-3-5-sonnet-{'20241022-v2' if is_us_region else '20240620-v1'}:0"
    CLAUDE_3_7_SONNET = f"{region_prefix}.anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_3_HAIKU = f"{region_prefix}.anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_4_5_HAIKU = f"{region_prefix}.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Mistral AI Models
    # https://mistral.ai/pricing/
    PIXTRAL_LARGE = f"{region_prefix}.mistral.pixtral-large-2502-v1:0"

    # OpenAI OSS Models (via AWS Bedrock)
    # https://aws.amazon.com/bedrock/pricing/
    OPENAI_GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
    OPENAI_GPT_OSS_20B = "openai.gpt-oss-20b-1:0"

    def is_claude(self) -> bool:
        return self in [
            BedrockModel.CLAUDE_SONNET_4_5,
            BedrockModel.CLAUDE_SONNET_4,
            BedrockModel.CLAUDE_3_5_SONNET,
            BedrockModel.CLAUDE_3_HAIKU,
            BedrockModel.CLAUDE_3_7_SONNET,
            BedrockModel.CLAUDE_4_5_HAIKU,
        ]

    def supports_only_temperature(self) -> bool:
        return self in [
            BedrockModel.CLAUDE_4_5_HAIKU,
            BedrockModel.CLAUDE_SONNET_4_5,
        ]

    def is_mistral(self) -> bool:
        return self == BedrockModel.PIXTRAL_LARGE

    def is_openai(self) -> bool:
        return self in {
            BedrockModel.OPENAI_GPT_OSS_120B,
            BedrockModel.OPENAI_GPT_OSS_20B,
        }


class BedrockModelKeys(StrEnum):
    """Keys for the Bedrock models."""

    CLAUDE_3_HAIKU = "CLAUDE_3_HAIKU"
    CLAUDE_3_5_SONNET = "CLAUDE_3_5_SONNET"
    CLAUDE_3_7_SONNET = "CLAUDE_3_7_SONNET"
    CLAUDE_SONNET_4 = "CLAUDE_SONNET_4"
    CLAUDE_SONNET_4_5 = "CLAUDE_SONNET_4_5"
    NOVA_LITE = "NOVA_LITE"
    NOVA_MICRO = "NOVA_MICRO"
    NOVA_PREMIER = "NOVA_PREMIER"
    NOVA_PRO = "NOVA_PRO"
    PIXTRAL_LARGE = "PIXTRAL_LARGE"
    OPENAI_GPT_OSS_120B = "OPENAI_GPT_OSS_120B"
    OPENAI_GPT_OSS_20B = "OPENAI_GPT_OSS_20B"


class OpenAIModel(StrEnum):
    """Supported OpenAI models."""

    # GPT-4 Models
    # https://openai.com/pricing#gpt-4
    GPT4O = "gpt-4o"  # $0.03/1K input tokens, $0.06/1K output tokens
    GPT4O_MINI = "gpt-4o-mini"  # $0.01/1K input tokens, $0.03/1K output tokens

    # O1 Models
    # https://openai.com/pricing#o1
    O1 = "o1"  # $0.015/1K input tokens, $0.03/1K output tokens
    O1_MINI = "o1-mini"  # $0.005/1K input tokens, $0.015/1K output tokens
    O3_MINI = "o3-mini"  # $0.002/1K input tokens, $0.006/1K output tokens

    def is_claude(self) -> bool:
        return False

    def is_mistral(self) -> bool:
        return False

    def is_openai(self) -> bool:
        return True

    def supports_only_temperature(self) -> bool:
        return False


# Models supported by the LLM client
BEDROCK_MODELS = [model.value for model in BedrockModel]
OPENAI_MODELS = [model.value for model in OpenAIModel]

SupportedLLMModels = OpenAIModel | BedrockModel