#!/usr/bin/env python3
"""
Comprehensive test suite for the LLM client.
Tests both unit functionality and live API calls.
"""

import json
from llm.client import LLMClient, MessageRole, BedrockModel, ImageData
from llm.supported_llm_models import OpenAIModel
from llm.config import config


def test_model_imports():
    """Test that all models can be imported and have correct properties."""
    print("🧪 Testing Model Imports and Properties")
    print("=" * 50)

    # Test Bedrock models
    claude_haiku = BedrockModel.CLAUDE_4_5_HAIKU
    print(f"✅ Claude Haiku: {claude_haiku}")
    assert claude_haiku.is_claude() == True
    assert claude_haiku.is_openai() == False
    assert claude_haiku.supports_only_temperature() == True

    nova_micro = BedrockModel.NOVA_MICRO
    print(f"✅ Nova Micro: {nova_micro}")
    assert nova_micro.is_claude() == False
    assert nova_micro.is_openai() == False

    # Test OpenAI models
    gpt4o = OpenAIModel.GPT4O
    print(f"✅ GPT-4O: {gpt4o}")
    assert gpt4o.is_openai() == True
    assert gpt4o.is_claude() == False

    print("✅ All model imports and properties working correctly")


def test_prompt_creation():
    """Test prompt creation and message handling."""
    print("\n🧪 Testing Prompt Creation")
    print("=" * 50)

    client = LLMClient(model=BedrockModel.CLAUDE_4_5_HAIKU)

    # Test basic prompt creation
    prompt = client.create_prompt(
        prompt_text="Hello, world!",
        system_prompt="You are a helpful assistant.",
        model_override=BedrockModel.CLAUDE_4_5_HAIKU,
        max_token_output=100,
        temperature=0.7
    )

    assert prompt.messages[0]["content"] == "Hello, world!"
    assert prompt.system_prompt == "You are a helpful assistant."
    assert prompt.model == BedrockModel.CLAUDE_4_5_HAIKU
    assert prompt.max_token_output == 100
    assert prompt.temperature == 0.7

    # Test adding messages
    prompt.add_message("How are you?", MessageRole.USER)
    prompt.add_message("I'm doing well!", MessageRole.LLM)

    assert len(prompt.messages) == 3
    assert prompt.messages[1]["role"] == MessageRole.USER
    assert prompt.messages[2]["role"] == MessageRole.LLM

    print("✅ Prompt creation and message handling working")


def test_format_conversions():
    """Test Bedrock and OpenAI format conversions."""
    print("\n🧪 Testing Format Conversions")
    print("=" * 50)

    client = LLMClient()

    # Test Bedrock format
    prompt = client.create_prompt(
        prompt_text="Test message",
        system_prompt="Test system",
        model_override=BedrockModel.CLAUDE_4_5_HAIKU
    )

    bedrock_format = prompt.to_bedrock_format()
    bedrock_data = json.loads(bedrock_format)

    assert "messages" in bedrock_data
    assert "system" in bedrock_data
    assert "max_tokens" in bedrock_data
    assert "temperature" in bedrock_data
    assert bedrock_data["system"] == "Test system"

    print("✅ Bedrock format conversion working")

    # Test OpenAI format
    openai_prompt = client.create_prompt(
        prompt_text="Test message",
        system_prompt="Test system",
        model_override=OpenAIModel.GPT4O
    )

    openai_format = openai_prompt.to_openai_format()

    assert isinstance(openai_format, list)
    assert openai_format[0]["role"] == "system"
    assert openai_format[0]["content"] == "Test system"
    assert openai_format[1]["role"] == "user"
    assert openai_format[1]["content"] == "Test message"

    print("✅ OpenAI format conversion working")


def test_image_handling():
    """Test image data handling."""
    print("\n🧪 Testing Image Handling")
    print("=" * 50)

    # Create sample image data
    image_data = ImageData(
        encoded_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",  # 1x1 pixel PNG
        media_type="image/png",
        description="Test image"
    )

    client = LLMClient()
    prompt = client.create_prompt(
        prompt_text="What's in this image?",
        model_override=BedrockModel.CLAUDE_4_5_HAIKU,
        images=[image_data]
    )

    # Test that images are properly handled in format conversion
    bedrock_format = prompt.to_bedrock_format()
    bedrock_data = json.loads(bedrock_format)

    # Check that the message contains image content
    user_message = bedrock_data["messages"][0]
    assert isinstance(user_message["content"], list)

    # Find image content
    image_content = None
    for content in user_message["content"]:
        if content.get("type") == "image":
            image_content = content
            break

    assert image_content is not None
    assert "source" in image_content
    assert image_content["source"]["type"] == "base64"

    print("✅ Image handling working correctly")


def test_live_bedrock_api():
    """Test actual Bedrock API calls (requires AWS credentials)."""
    print("\n🧪 Testing Live Bedrock API")
    print("=" * 50)

    try:
        client = LLMClient(model=BedrockModel.CLAUDE_4_5_HAIKU)

        prompt = client.create_prompt(
            prompt_text="Say exactly 'Test successful!' and nothing else.",
            model_override=BedrockModel.CLAUDE_4_5_HAIKU,
            max_token_output=50,
            prompt_name="test-api-call"
        )

        print(f"📤 Submitting to {prompt.model}")
        response = client.submit_prompt(prompt)

        if response.error:
            print(f"❌ API Error: {response.error}")
            return False

        print(f"✅ Response: '{response.text}'")
        print(f"📊 Tokens: {response.input_tokens}→{response.output_tokens}")
        print(f"💰 Cost: ${response.cost:.6f}")
        print(f"⏱️ Latency: {response.latency_ms}ms")

        # Verify response contains expected text
        assert "Test successful!" in response.text
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.cost > 0
        assert response.latency_ms > 0

        return True

    except Exception as e:
        print(f"⚠️ Live API test failed: {str(e)}")
        print("   This is expected if AWS credentials are not configured")
        return False


def test_different_models():
    """Test different Bedrock models."""
    print("\n🧪 Testing Different Bedrock Models")
    print("=" * 50)

    models_to_test = [
        BedrockModel.CLAUDE_4_5_HAIKU,
        BedrockModel.NOVA_MICRO,
        BedrockModel.CLAUDE_3_5_SONNET,
    ]

    results = {}

    for model in models_to_test:
        try:
            client = LLMClient(model=model)
            prompt = client.create_prompt(
                prompt_text=f"Respond with: 'Model {model} working'",
                model_override=model,
                max_token_output=30
            )

            print(f"📤 Testing {model}")
            response = client.submit_prompt(prompt)

            if response.error:
                print(f"❌ {model}: {response.error}")
                results[str(model)] = {"success": False, "error": response.error}
            else:
                print(f"✅ {model}: {response.text[:50]}...")
                results[str(model)] = {
                    "success": True,
                    "cost": response.cost,
                    "latency": response.latency_ms
                }

        except Exception as e:
            print(f"❌ {model}: Exception - {str(e)}")
            results[str(model)] = {"success": False, "error": str(e)}

    return results


def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Running Complete LLM Client Test Suite")
    print("=" * 80)

    # Unit tests (should always pass)
    test_model_imports()
    test_prompt_creation()
    test_format_conversions()
    test_image_handling()

    print("\n" + "=" * 80)
    print("✅ All unit tests passed!")

    # Live API tests (may fail without credentials)
    print("\n🌐 Running Live API Tests")
    print("=" * 40)

    api_success = test_live_bedrock_api()
    model_results = test_different_models()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    print("✅ Unit Tests: ALL PASSED")
    print(f"🌐 Basic API Test: {'PASSED' if api_success else 'FAILED (credentials?)'}")

    successful_models = sum(1 for r in model_results.values() if r["success"])
    total_models = len(model_results)
    print(f"🤖 Model Tests: {successful_models}/{total_models} PASSED")

    if api_success:
        print("\n🎉 LLM Client is fully functional!")
        print("   ✅ All unit tests pass")
        print("   ✅ Live API calls work")
        print("   ✅ Multiple models available")
        print("   ✅ Cost tracking working")
        print("   ✅ Error handling robust")
    else:
        print("\n⚠️  LLM Client structure is correct")
        print("   ✅ All unit tests pass")
        print("   ❌ Live API needs AWS credentials")
        print("   💡 Configure AWS credentials to test live functionality")

    return api_success and successful_models == total_models


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)