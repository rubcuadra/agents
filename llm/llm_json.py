"""
JSON parsing utilities for LLM responses.
"""

import json
import re
from typing import Any


def loads(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response text, handling common formatting issues.

    Args:
        text: The text containing JSON

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
    # Remove common markdown formatting
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        # Find the first newline after ```
        start = text.find('\n')
        if start != -1:
            text = text[start + 1:]

        # Remove trailing ```
        if text.endswith("```"):
            text = text[:-3]

    # Remove any remaining markdown json indicators
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)

    # Try to extract JSON from the text
    # Look for the first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_text = text[start_idx:end_idx + 1]
        return json.loads(json_text)

    # If no braces found, try parsing the whole text
    return json.loads(text)