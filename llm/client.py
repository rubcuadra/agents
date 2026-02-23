"""
LLM Client module providing a unified interface for different LLM services.
Complete copy with Bedrock support restored.
"""

import base64
import csv
import io
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from botocore.config import Config
from botocore.exceptions import ClientError
from openai import OpenAI
from PIL import Image

from llm.config import config, get_aws_session
from llm.guardrail_prompts import (
    REPHRASE_REMOVE_SUSPICIOUS_SYSTEM_PROMPT,
    REPHRASE_REMOVE_SUSPICIOUS_USER_PROMPT,
)
from llm import llm_json
from llm.logger import logger
from llm.pricing import ComputePricing
from llm.rate_limiter import LLMRateLimiter, create_rate_limit_storage
from llm.supported_llm_models import (
    BEDROCK_MODELS,
    OPENAI_MODELS,
    BedrockModel,
    OpenAIModel,
)
from llm.tracking import (
    AIInvestigationTracking,
    AIInvestigationTrackingDataAccess,
    pgpark_cursor,
)


class AWSAuthenticationError(Exception):
    """Raised when AWS authentication fails (expired tokens, invalid credentials, etc.)."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


@dataclass
class ImageData:
    """Represents an image with its encoded data and description for LLM processing."""

    encoded_data: str  # Base64 encoded image data
    description: str | None = None
    media_type: str = "image/png"  # MIME type of the image


image_media_type_format = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Default model to use if the requested model is not supported
DEFAULT_MODEL = BedrockModel.CLAUDE_4_5_HAIKU

# Fall back model used when Nova models are content filtered.
FALLBACK_MODEL = BedrockModel.CLAUDE_SONNET_4_5


class MessageRole(StrEnum):
    USER = "USER"
    LLM = "LLM"


class BedrockAgentClient(Protocol):
    """Protocol defining the interface for AWS Bedrock agent client."""

    def list_prompts(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_prompt(self, **kwargs: Any) -> dict[str, Any]: ...


LLM_AS_A_JUDGE_OUTPUT_FORMAT = """
{
    "matches": <boolean>,
    "match_percentage": <float>,
    "explanation": <string>
}
"""

# https://huggingface.co/learn/cookbook/en/llm_judge
LLM_AS_A_JUDGE_SYSTEM_PROMPT = (
    """You are a expert analyzer of question and answers pairs in the context of AML and Transaction Monitoring Alerts. Your primary purpose is to asses if a given answer is accurate and correct based on the question that was asked.

**INPUT DATA**

<question>: The question that was asked. Or general input data that was provided to the LLM.
<answer>: The answer to the question, likely output from an LLM.
<guidance>: Optional guidance to help you determine if the <answer> is correct for the given <question>.
<system_prompt>: The OPTIONAL system prompt that was used to generate the <question> and <answer>.

**Step 1: Analyze the question and the answer**

Provide a 'match_percentage' scoring how well the <answer> is sufficient for the expressed statement in the <question>.
Give your match_percentage as a float on a scale of 0 to 1, where 0 means that the <answer> is not helpful at all, and 1 means that the
<answer> completely and helpfully addresses the <question>. If the match_percentage is more than 0.70 then matches: true, otherwise matches: false.
Give a concise explanation of your reasoning for the match_percentage.

If there is provided <guidance>
  * use <guidance> to help you determine the match_percentage
  * do not overly rely on <guidance>, it is a guide, if the <question>/<answer> pair is mostly related to the <guidance> then the match_percentage should be high
  * if the <question>/<answer> pair is not related to the <guidance> then the match_percentage should be low
  * Only call out issues with Fraud or AML conclusions in an <answer> if the <guidance> explicitly states there should be no conclusions one way or another on the topic

If there is provided <system_prompt>
  * use <system_prompt> to help you determine the match_percentage
  * the <system_prompt> gives more context on why the <question> and <answer> were generated


**Step 2: Validate your analysis**

* Be cautious of assuming the <answer> is incorrect just because it has more information than implied by the <question>.
* The <question>/<answer> pairs are in the context of AML and Transaction Monitoring Alerts. This means it is reasonable for a question/answer pair to contain information related to a AML or Fraud investigation.
* Only mention issues with an <answer> including AML or Fraud conclusions if the <guidance> says it is an issue.
* Do not give a low match_percentage when the <answer> indicates commentary on Fraud or AML, unless the <guidance> says it is an issue.

**Output Formatting**
Provide your feedback in the following JSON format with no other text:

"""
    + LLM_AS_A_JUDGE_OUTPUT_FORMAT
)

LLM_AS_A_JUDGE_PROMPT = """
Now here are the question and answer.

<question>
{question}
</question>

<answer>
{answer}
</answer>
"""

EVALUATION_TRACKING_FILE = "eval_set_tracking.csv"


@dataclass
class ResponseMatch:
    matches: bool
    match_percentage: float
    question: str
    answer: str
    explanation: str
    total_judge_calls: int = 0

    _prompt_id: str = ""

    def quality_threshold_is_met(self) -> bool:
        # Using 80% here because the "made up" value of 85% was causing too many false negatives.
        return self.match_percentage >= 0.80

    def format_failed_judgement(self, prompt_id: str = "") -> str:
        prompt_id = self._prompt_id if prompt_id == "" else prompt_id

        return f"Expected the response to be of high quality, but it was not (match_percentage: {self.match_percentage}).\n\nQuestion: {self.question}\n\nAnswer: {self.answer}\n\nPrompt ID: {prompt_id}\n\nExplanation: {self.explanation}\n\n"

    def record_judgement(
        self,
        llm_env: str,
        signal_agent_id: str,
        prompt_id: str,
    ) -> None:
        """
        Record the judgement by writing to a CSV file.

        Args:
            llm_env: The LLM environment (e.g. 'local')
            signal_agent_id: The signal agent ID
            prompt_id: The prompt ID
        """
        self._prompt_id = prompt_id
        # Check if file exists and has headers
        headers = [
            "timestamp",
            "llm_env",
            "signal_agent_id",
            "prompt_id",
            "question",
            "answer",
            "explanation",
            "matches",
            "match_percentage",
            "total_judge_calls",
        ]

        # Only create headers if file doesn't exist or is empty
        try:
            with open(EVALUATION_TRACKING_FILE, newline="") as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if not first_row or first_row != headers:
                    # File is empty or headers don't match, need to create headers
                    with open(EVALUATION_TRACKING_FILE, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
        except FileNotFoundError:
            # File doesn't exist, create it with headers
            with open(EVALUATION_TRACKING_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # Append the judgement to the CSV file
        with open(EVALUATION_TRACKING_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    llm_env,
                    signal_agent_id,
                    prompt_id,
                    self.question,
                    self.answer,
                    self.explanation,
                    self.matches,
                    self.match_percentage,
                    self.total_judge_calls,
                ]
            )


class LLMClient:
    """
    Client for interacting with various LLM services.

    This client provides a unified interface for different LLM services,
    including AWS Bedrock and OpenAI.
    """

    @dataclass
    class ModelResponse:
        model: BedrockModel | OpenAIModel
        text: str
        role: MessageRole
        formatted_prompt: str
        raw_response: dict[str, Any]
        error: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        cost: float | None = None
        latency_ms: int | None = None

    class ModelPrompt:
        """
        Encapsulates a prompt to be sent to an LLM model.
        """

        def __init__(
            self,
            prompt_text: str,
            system_prompt: str = "",
            max_token_output: int = 512,
            temperature: float = 0.3,  # Lowered from 0.7 for better quality control
            top_p: float = 0.9,
            group_id: str | None = None,
            query_type: str | None = None,
            tool_name: str | None = None,
            prompt_version: str | None = None,
            prompt_name: str | None = None,
            prompt_template: str | None = None,
            prompt_variables: dict[str, Any] | None = None,
            org_id: int | None = None,
            model: BedrockModel | OpenAIModel = DEFAULT_MODEL,
            images: list[ImageData] | None = None,
            skip_guardrails: bool = False,
        ):
            """
            Initialize the model prompt.

            Args:
                prompt_text: The user prompt text.
                system_prompt: The system prompt text.
                max_token_output: The maximum number of tokens to generate.
                temperature: The temperature for sampling.
                top_p: The top-p value for sampling.
                group_id: Optional group ID for tracking related prompts. If not provided,
                    will use the LLM client's group_id.
                query_type: Type of query (e.g., tool_call, direct_query, agent_action).
                tool_name: Name of the tool if using tool-based frameworks.
                prompt_version: Version identifier for the prompt template, useful for A/B testing.
                prompt_name: Identifier for the prompt template being used.
                prompt_template: The template used for the prompt with variable placeholders.
                prompt_variables: Variables used to populate the prompt template.
                org_id: Organization ID.
                model: The model to use for the prompt.
                skip_guardrails: If True, skip guardrail checks for this prompt (default: False).
            """
            self.system_prompt = system_prompt
            self.max_token_output = max_token_output
            self.temperature = temperature
            self.top_p = top_p
            self.group_id = group_id
            self.query_type = query_type
            self.tool_name = tool_name
            self.prompt_version = prompt_version
            self.prompt_name = prompt_name
            self.prompt_template = prompt_template
            self.prompt_variables = prompt_variables
            self.org_id = org_id
            self.model = model
            self.images = images
            self.skip_guardrails = skip_guardrails
            # Initialize the message history with the initial user message
            self.messages = [{"role": MessageRole.USER, "content": prompt_text}]

        def add_message(
            self, content: str, role: MessageRole = MessageRole.USER
        ) -> "LLMClient.ModelPrompt":
            """
            Add a message to the conversation history.

            Args:
                content: The message content.
                role: The role of the message sender. Either MessageRole.USER or MessageRole.LLM.
            """
            if role not in [MessageRole.USER, MessageRole.LLM]:
                raise ValueError(f"Role must be either '{MessageRole.USER}' or '{MessageRole.LLM}'")

            self.messages.append({"role": role, "content": content})
            return self

        def to_bedrock_format(self) -> str:
            """
            Convert the prompt to the format expected by AWS Bedrock.
            For Claude models, uses the Anthropic Claude Messages API format.
            For Amazon models, uses the standard Bedrock format.
            """

            def resize_image(image_data: ImageData) -> ImageData:
                """
                Given image data, resize the image to fit the max image dimensions
                for Bedrock: 8000x8000 pixels
                If the image is smaller than the max dimensions, there is no effect.
                """
                image = Image.open(io.BytesIO(base64.b64decode(image_data.encoded_data)))
                if image.size[0] <= 8000 and image.size[1] <= 8000:
                    return image_data

                # Resize the image in-place
                image.thumbnail((8000, 8000))

                buffer = io.BytesIO()
                image.save(buffer, format=image.format)
                resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return replace(
                    image_data,
                    encoded_data=resized_base64,
                )

            # Convert the message history to Bedrock format
            bedrock_messages: list[dict[str, Any]] = []
            for msg in self.messages:
                # Map our role types to Bedrock roles
                bedrock_role = "user" if msg["role"] == MessageRole.USER else "assistant"

                if self.model.is_openai():
                    if self.images:
                        raise ValueError("OpenAI Bedrock models currently do not support images.")
                    bedrock_messages.append(
                        {
                            "role": bedrock_role,
                            "content": [
                                {
                                    "type": "text",
                                    "text": msg["content"],
                                }
                            ],
                        }
                    )
                    continue

                # Check if this is a Claude model using the enum
                if self.model.is_claude():
                    # Claude format: handle image content
                    if self.images and msg["role"] == MessageRole.USER:
                        # For Claude models with images, we need to use content array format
                        content = []

                        # Handle multiple images with descriptions
                        for image_data in self.images:
                            resized_image = resize_image(image_data)
                            content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": resized_image.media_type,
                                        "data": resized_image.encoded_data,
                                    },
                                }
                            )
                            # Add description for this image
                            if resized_image.description:
                                content.append(
                                    {
                                        "type": "text",
                                        "text": f"Image: {resized_image.description}",
                                    }
                                )

                        # Add the main prompt text
                        content.append({"type": "text", "text": msg["content"]})

                        bedrock_messages.append({"role": bedrock_role, "content": content})
                    else:
                        # Claude format: direct content string
                        bedrock_messages.append({"role": bedrock_role, "content": msg["content"]})
                elif self.model.is_mistral():
                    # Mistral format: uses different content structure with type field
                    if self.images and msg["role"] == MessageRole.USER:
                        # For Mistral models with images, use content array format with type field
                        content = []

                        # Add text content first
                        if msg["content"]:
                            content.append({"type": "text", "text": msg["content"]})

                        # Handle multiple images
                        for image_data in self.images:
                            resized_image = resize_image(image_data)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{resized_image.media_type};base64,{resized_image.encoded_data}"
                                    },
                                }
                            )
                        bedrock_messages.append({"role": bedrock_role, "content": content})
                    else:
                        # Mistral format: direct content string
                        bedrock_messages.append({"role": bedrock_role, "content": msg["content"]})
                else:
                    # Amazon model format: content array with text object
                    images = []
                    if self.images is not None:
                        for image_data in self.images:
                            resized_image = resize_image(image_data)
                            images.append(
                                {
                                    "image": {
                                        "format": image_media_type_format.get(
                                            resized_image.media_type, "jpeg"
                                        ),
                                        "source": {"bytes": resized_image.encoded_data},
                                    }
                                }
                            )
                    bedrock_message = {
                        "role": bedrock_role,
                        "content": [{"text": msg["content"]}, *images],
                    }
                    bedrock_messages.append(bedrock_message)

            if self.model.is_openai() and self.system_prompt:
                bedrock_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}],
                    },
                )

            # Base prompt data
            prompt_data: dict[str, Any] = {
                "messages": bedrock_messages,
            }

            # Add model-specific parameters
            if self.model.is_claude():
                if self.model.supports_only_temperature():
                    # See https://docs.claude.com/en/api/messages#body-top-p
                    # "top_p" and "temperature" are not supported together for Claude 4.5 Haiku and Sonnet 4.5
                    # So we will favor temperature over top_p
                    prompt_data.update(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": self.max_token_output,
                            "temperature": self.temperature,
                        }
                    )
                else:
                    prompt_data.update(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": self.max_token_output,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                        }
                    )
                if self.system_prompt:
                    prompt_data["system"] = self.system_prompt
            elif self.model.is_openai():
                prompt_data["max_tokens"] = self.max_token_output
                prompt_data["temperature"] = self.temperature
                if self.top_p is not None:
                    prompt_data["top_p"] = self.top_p
            elif self.model.is_mistral():
                # Mistral model format - uses different parameter structure
                prompt_data.update(
                    {
                        "max_tokens": self.max_token_output,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                )
                # Mistral models don't support system prompts in the same way as Claude
                # We'll include the system prompt in the first user message instead
                if self.system_prompt:
                    prompt_data["messages"].insert(
                        0, {"role": "user", "content": self.system_prompt}
                    )
            else:
                # Amazon model format
                prompt_data.update(
                    {
                        "inferenceConfig": {
                            "max_new_tokens": self.max_token_output,
                            "top_p": self.top_p,
                            "temperature": self.temperature,
                        }
                    }
                )
                if self.system_prompt:
                    prompt_data["system"] = [{"text": self.system_prompt}]

            return json.dumps(prompt_data)

        def to_openai_format(self) -> list[dict[str, Any]]:
            """
            Convert the prompt to the format expected by OpenAI.

            Returns:
                A list of message dictionaries where content can be either a string
                (for text-only messages) or a list of content parts (for multimodal messages).
            """
            messages: list[dict[str, Any]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            # Add all messages from the history, mapping our roles to OpenAI roles
            for msg in self.messages:
                openai_role = "user" if msg["role"] == MessageRole.USER else "assistant"

                if self.images and msg["role"] == MessageRole.USER:
                    # For OpenAI models with images, we need to use content array format
                    content = []

                    # Handle multiple images with descriptions
                    for image_data in self.images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_data.media_type};base64,{image_data.encoded_data}"
                                },
                            }
                        )
                        # Add description for this image
                        if image_data.description:
                            content.append(
                                {
                                    "type": "text",
                                    "text": f"Image: {image_data.description}",
                                }
                            )

                    # Add the main prompt text
                    content.append({"type": "text", "text": msg["content"]})

                    messages.append({"role": openai_role, "content": content})
                else:
                    messages.append({"role": openai_role, "content": msg["content"]})

            return messages

    def __init__(
        self,
        model: OpenAIModel | BedrockModel = DEFAULT_MODEL,
        pgpark_cursor=pgpark_cursor,
        timeout: int = config.bedrock_api_timeout_seconds,
        aws_region: str = config.region,
        environment: str = config.environment,
    ):
        """
        Initialize the LLM client.

        Args:
            model: The model to use for LLM requests.
            pgpark_cursor: Database cursor for tracking
            timeout: The timeout in seconds for API calls (defaults to config.bedrock_api_timeout_seconds)
            aws_region: AWS region to use
            environment: Environment name for rate limiting (defaults to config.environment)
        """
        self.model = self._validate_model(model)
        self._bedrock_agent_client: BedrockAgentClient | None = None
        self._bedrock_client = None
        self._bedrock_agentcore_client = None
        self._openai_client: OpenAI | None = None
        self.group_id = str(uuid.uuid4())
        self._tracking_data_access = AIInvestigationTrackingDataAccess(pgpark_cursor)
        self.timeout = timeout
        self.aws_region = aws_region

        storage = create_rate_limit_storage()
        self.rate_limiter = LLMRateLimiter(storage, environment)

    @property
    def bedrock_client(self):
        """
        Lazy-loaded AWS Bedrock client.
        """
        if self._bedrock_client is None:
            session = get_aws_session()
            boto3_config = Config(
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                retries={
                    "max_attempts": 1,  # Disable boto3 retries - we handle them manually
                    "mode": "standard",
                },
                # Add specific error handling for Bedrock
                parameter_validation=False,  # Disable parameter validation to speed up requests
                max_pool_connections=5,  # Reduce connection pool size for Lambda
                tcp_keepalive=True,  # Enable TCP keepalive
            )
            self._bedrock_client = session.client(
                "bedrock-runtime", region_name=self.aws_region, config=boto3_config
            )
        return self._bedrock_client

    @property
    def openai_client(self) -> OpenAI:
        """
        Lazy-loaded OpenAI client.
        """
        if self._openai_client is None:
            self._openai_client = OpenAI(timeout=self.timeout)
        return self._openai_client

    def _validate_model(self, model_id: OpenAIModel | BedrockModel) -> OpenAIModel | BedrockModel:
        """
        Validate that the requested model is supported.

        Args:
            model_id: The model ID to validate.

        Returns:
            The validated model ID or the default model ID if the requested model is not supported.
        """
        # Check if the model is in either enum
        if model_id in BEDROCK_MODELS or model_id in OPENAI_MODELS:
            return model_id

        logger.error(f"Model {model_id} is not supported. Using default model {DEFAULT_MODEL}")
        return DEFAULT_MODEL

    def create_prompt(
        self,
        prompt_text: str,
        system_prompt: str = "",
        max_token_output: int = 512,
        temperature: float = 0.3,  # Lowered from 0.7 for better quality control
        top_p: float = 0.9,
        group_id: str | None = None,
        query_type: str | None = None,
        tool_name: str | None = None,
        prompt_version: str | None = None,
        prompt_name: str | None = None,
        prompt_template: str | None = None,
        prompt_variables: dict[str, Any] | None = None,
        org_id: int | None = None,
        model_override: OpenAIModel | BedrockModel | None = None,
        images: list[ImageData] | None = None,
        skip_guardrails: bool = False,
    ) -> "LLMClient.ModelPrompt":
        """
        Create a model prompt.

        Args:
            prompt_text: The user prompt text.
            system_prompt: The system prompt text.
            max_token_output: The maximum number of tokens to generate.
            temperature: The temperature for sampling.
            top_p: The top-p value for sampling.

            # Tracking parameters for monitoring usage and costs
            group_id: Optional group ID for tracking related prompts. If not provided,
                will use the LLM client's group_id.
            query_type: Type of query (e.g., tool_call, direct_query, agent_action).
            tool_name: Name of the tool if using tool-based frameworks.
            prompt_version: Version identifier for the prompt template, useful for A/B testing.
            prompt_name: Identifier for the prompt template being used. (e.g. local-paa-paa)
            prompt_template: The template used for the prompt with variable placeholders.
            prompt_variables: Variables used to populate the prompt template.
            org_id: Organization ID.
            skip_guardrails: If True, skip guardrail checks for this prompt (default: False).

        Returns:
            A ModelPrompt object.
        """
        return self.ModelPrompt(
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            max_token_output=max_token_output,
            temperature=temperature,
            top_p=top_p,
            group_id=group_id or self.group_id,
            query_type=query_type,
            tool_name=tool_name,
            prompt_version=prompt_version,
            prompt_name=prompt_name,
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
            org_id=org_id,
            model=model_override or self.model,
            images=images,
            skip_guardrails=skip_guardrails,
        )

    def submit_prompt(self, model_prompt: "ModelPrompt") -> "ModelResponse":
        """
        Submit a prompt to the LLM model.

        Args:
            model_prompt: A ModelPrompt object.

        Returns:
            The response from the model.
        """
        logger.info(f"Submitting prompt {model_prompt.prompt_name} to LLM")
        start_time = time.time()

        # Process the prompt
        formatted_prompt = model_prompt.to_bedrock_format()

        try:
            # Handle based on model type
            if isinstance(model_prompt.model, BedrockModel):
                response = self._submit_to_bedrock(model_prompt, model_prompt.model)
            elif isinstance(model_prompt.model, OpenAIModel):
                response = self._submit_to_openai(model_prompt, model_prompt.model)
            else:
                # This should never happen due to model validation
                logger.error(f"Model {model_prompt.model} is not supported")
                response = LLMClient.ModelResponse(
                    model=model_prompt.model,
                    text="",
                    role=MessageRole.LLM,
                    formatted_prompt="No submission made",
                    raw_response={},
                    error=f"Model {model_prompt.model} is not supported",
                )

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            response.latency_ms = latency_ms

            # Create tracking record
            tracking = AIInvestigationTracking(
                group_id=self.group_id,
                framework="bedrock" if model_prompt.model in BEDROCK_MODELS else "openai",
                model_name=str(model_prompt.model),
                query_type=model_prompt.query_type,
                tool_name=model_prompt.tool_name,
                prompt_version=model_prompt.prompt_version,
                prompt_name=model_prompt.prompt_name,
                prompt_template=model_prompt.prompt_template,
                prompt_variables=model_prompt.prompt_variables,
                raw_prompt=json.dumps(model_prompt.messages),
                raw_response=json.dumps(response.raw_response),
                processed_response=response.text,
                tokens_used=(
                    response.input_tokens + response.output_tokens
                    if response.input_tokens and response.output_tokens
                    else None
                ),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost=response.cost,
                latency_ms=latency_ms,
                status="success" if not response.error else "error",
                error_message=response.error,
                org_id=model_prompt.org_id,
            )

            # Store tracking record
            self._tracking_data_access.insert_tracking(tracking)
            logger.info(f"Submission Successful to {model_prompt.prompt_name} in {latency_ms}ms")

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Error submitting prompt: {str(e)}")
            return LLMClient.ModelResponse(
                model=model_prompt.model,
                text="",
                role=MessageRole.LLM,
                formatted_prompt=formatted_prompt,
                raw_response={},
                error=str(e),
                latency_ms=latency_ms,
            )

    def _submit_to_bedrock(self, prompt: "ModelPrompt", model: BedrockModel) -> "ModelResponse":
        """
        Submit a prompt to a Bedrock model.

        Args:
            prompt: The ModelPrompt to submit.
            model: The model ID to use.

        Returns:
            The response from the model.
        """
        formatted_prompt = prompt.to_bedrock_format()

        try:
            response = self.bedrock_client.invoke_model(
                modelId=str(model),
                body=formatted_prompt,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 200)

            # Check for error response
            if status_code >= 400:
                error_type = "Client Error" if status_code < 500 else "Server Error"
                logger.error(f"Bedrock {error_type} (HTTP {status_code}): {response_body}")
                return LLMClient.ModelResponse(
                    model=model,
                    text="",
                    formatted_prompt=formatted_prompt,
                    role=MessageRole.LLM,
                    raw_response=response_body,
                    error=f"Bedrock {error_type} (HTTP {status_code})",
                )

            # Extract the text from the response
            try:
                if model.is_claude():
                    text = response_body.get("content", [{"text": "NO RESPONSE"}])[0].get(
                        "text", ""
                    )
                    usage = response_body.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                elif model.is_mistral():
                    text = (
                        response_body.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                    usage = response_body.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                elif model.is_openai():
                    choice = response_body.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    text = message.get("content", "") or ""
                    if not text:
                        logger.warning(
                            "OpenAI Bedrock response missing message content: %s",
                            response_body,
                        )
                    usage = response_body.get("usage", {})
                    input_tokens = (
                        usage.get("input_tokens")
                        or usage.get("inputTokens")
                        or usage.get("prompt_tokens")
                        or 0
                    )
                    output_tokens = (
                        usage.get("output_tokens")
                        or usage.get("outputTokens")
                        or usage.get("completion_tokens")
                        or 0
                    )
                else:
                    text = (
                        response_body.get("output", {})
                        .get("message", {})
                        .get("content", [{"text": ""}])[0]
                        .get("text", "")
                    )
                    # Extract token usage from Bedrock response
                    usage = response_body.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)

                # Calculate cost
                cost = ComputePricing.calculate_cost(str(model), input_tokens, output_tokens)

                return LLMClient.ModelResponse(
                    model=model,
                    text=text,
                    role=MessageRole.LLM,
                    formatted_prompt=formatted_prompt,
                    raw_response=response_body,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error extracting text from Bedrock response: {e}")
                return LLMClient.ModelResponse(
                    model=model,
                    text="",
                    formatted_prompt=formatted_prompt,
                    role=MessageRole.LLM,
                    raw_response=response_body,
                    error=str(e),
                )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(f"Bedrock ClientError: {error_code} - {str(e)}")
            return LLMClient.ModelResponse(
                model=model,
                text="",
                formatted_prompt=formatted_prompt,
                role=MessageRole.LLM,
                raw_response={},
                error=f"Bedrock API error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {str(e)}")
            return LLMClient.ModelResponse(
                model=model,
                text="",
                formatted_prompt=formatted_prompt,
                role=MessageRole.LLM,
                raw_response={},
                error=f"Bedrock API error: {str(e)}",
            )

    def _submit_to_openai(self, prompt: "ModelPrompt", model: OpenAIModel) -> "ModelResponse":
        """
        Submit a prompt to an OpenAI model.

        Args:
            prompt: The ModelPrompt to submit.
            model: The model ID to use.

        Returns:
            The response from the model.
        """
        messages = prompt.to_openai_format()

        try:
            response = self.openai_client.chat.completions.create(
                model=str(model),
                messages=messages,
                max_tokens=prompt.max_token_output,
                temperature=prompt.temperature,
                top_p=prompt.top_p,
            )

            # Convert the response to a dictionary
            response_dict = response.model_dump()

            # Extract the text from the response
            text = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract token usage from OpenAI response
            usage = response_dict.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Calculate cost
            cost = ComputePricing.calculate_cost(str(model), input_tokens, output_tokens)

            return LLMClient.ModelResponse(
                model=model,
                text=text,
                role=MessageRole.LLM,
                formatted_prompt=json.dumps(messages),
                raw_response=response_dict,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception as e:
            logger.error(f"Error extracting text from OpenAI response: {e}")
            return LLMClient.ModelResponse(
                model=model,
                text="",
                role=MessageRole.LLM,
                formatted_prompt=json.dumps(messages),
                raw_response={},
                error=str(e),
            )