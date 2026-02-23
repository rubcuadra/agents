import json
import os
from typing import Any

import boto3
from dotenv import load_dotenv
from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings

load_dotenv(override=True)

# Cache for secret values to avoid repeated AWS calls
_secret_cache: dict[str, Any] = {}


def get_secret_value(secret_manager_arn: str, field_name: str | None) -> Any:
    """
    Retrieve a secret value from AWS Secrets Manager.

    Args:
        secret_arn: The ARN of the secret to retrieve
        field_name: The name of the field to retrieve, if any

    Returns:
        The secret value as a string if successful, None otherwise
    """
    if not secret_manager_arn or not secret_manager_arn.startswith("arn:aws:secretsmanager:"):
        return None

    # Check cache first
    if secret_manager_arn not in _secret_cache:
        try:
            # Use the same AWS session management as LLM client
            session = get_aws_session()
            client = session.client("secretsmanager")
            response = client.get_secret_value(SecretId=secret_manager_arn)
            if "SecretString" in response:
                secret_dict = json.loads(response["SecretString"])
                _secret_cache[secret_manager_arn] = {
                    key.lower(): value for key, value in secret_dict.items()
                }

        except Exception as e:
            print(f"Error retrieving secret: {str(e)}")

    if field_name is None:
        return _secret_cache.get(secret_manager_arn, {})
    return _secret_cache.get(secret_manager_arn, {}).get(field_name.lower())


def get_aws_session() -> boto3.Session:
    """
    Creates an AWS session using the appropriate authentication method for the environment.

    For local development:
    - Uses named profile from AWS_PROFILE environment variable
    - Falls back to AWS_DEFAULT_PROFILE
    - Finally defaults to "default" profile

    For production:
    - In AWS environments (EC2, ECS, Lambda), uses IAM role credentials
    - Accepts explicit AWS credentials from environment variables
    - Returns default session if no explicit configuration
    """
    # Check if we're running in an AWS environment (EC2, ECS, Lambda)
    # When running in AWS, we should use IAM roles rather than profiles
    running_in_aws = "AWS_EXECUTION_ENV" in os.environ or "AWS_LAMBDA_FUNCTION_NAME" in os.environ

    if running_in_aws:
        print("Running in AWS environment, using IAM role")
        # No need to specify credentials - boto3 will use the IAM role
        return boto3.Session()

    # For local development, use profile-based authentication
    profile_name = os.getenv("AWS_PROFILE", os.getenv("AWS_DEFAULT_PROFILE", "default"))
    print(f"Running locally, using profile: {profile_name}")
    try:
        return boto3.Session(profile_name=profile_name)
    except Exception as e:
        print(f"Warning: Could not create session with profile {profile_name}: {e}")
        # Fall back to default session (uses environment variables or default credentials)
        return boto3.Session()


class Config(BaseSettings):
    tenant: str = "local"
    region: str = "us-west-2"
    environment: str = "local"

    # Timeout settings for different components
    bedrock_api_timeout_seconds: int = 45

    # OpenAI
    openai_api_key: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def resolve_secret_arns(cls, v: Any, field: ValidationInfo) -> Any:
        """Resolve AWS Secrets Manager ARNs before validation."""
        if isinstance(v, str) and v.startswith("arn:aws:secretsmanager:"):
            secret_value = get_secret_value(v, field.field_name)
            if secret_value is not None:
                return secret_value
            # Get default value from the model fields if there
            # is an error in the secret retrieval
            if field.field_name and field.field_name in cls.model_fields:
                return cls.model_fields[field.field_name].default
        return v


config = Config()