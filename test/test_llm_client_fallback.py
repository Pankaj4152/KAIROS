"""
Fallback/retry test placeholders for LLM client behavior.

Unit-level retry and multi-tier stream fallback tests need a dedicated mocked
transport fixture to avoid flaky network coupling.
"""

import pytest


class TestLlmClientFallback:
    @pytest.mark.skip(reason="Requires mocked LiteLLM transport for complete_with_tools")
    def test_complete_with_tools_tier_fallback(self):
        """Reserved for complete_with_tools fallback coverage."""
        pass

    @pytest.mark.skip(reason="Requires mocked LiteLLM transport for stream fallback")
    def test_stream_with_tier_fallback(self):
        """Reserved for stream fallback coverage."""
        pass
