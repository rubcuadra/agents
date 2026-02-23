"""
Simplified tracking functionality for LLM usage.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class AIInvestigationTracking:
    """Simplified tracking record for AI investigation LLM usage."""

    group_id: str
    framework: str
    model_name: str
    query_type: str | None = None
    tool_name: str | None = None
    prompt_version: str | None = None
    prompt_name: str | None = None
    prompt_template: str | None = None
    prompt_variables: dict[str, Any] | None = None
    raw_prompt: str | None = None
    raw_response: str | None = None
    processed_response: str | None = None
    tokens_used: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    latency_ms: int | None = None
    status: str = "success"
    error_message: str | None = None
    org_id: int | None = None
    created_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


class AIInvestigationTrackingDataAccess:
    """Simplified data access for tracking records."""

    def __init__(self, cursor=None):
        self.cursor = cursor
        self.records = []  # In-memory storage for this standalone version

    def insert_tracking(self, tracking: AIInvestigationTracking) -> None:
        """Insert a tracking record (simplified to in-memory storage)."""
        self.records.append(tracking)
        # In production, this would insert into a database
        print(f"Tracked LLM usage: {tracking.model_name} - {tracking.status}")

    def get_recent_records(self, limit: int = 10) -> list[AIInvestigationTracking]:
        """Get recent tracking records."""
        return self.records[-limit:]


# Mock pgpark cursor for compatibility
class MockPgParkCursor:
    """Mock cursor for development."""
    pass


# Create a default cursor instance
pgpark_cursor = MockPgParkCursor()