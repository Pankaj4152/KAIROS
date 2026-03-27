"""
Tests for runtime/llm/client.py

Covers:
  - _resolve_model: tier 1/2/3 → correct model name
  - _resolve_model: unknown tier → warning + fallback to tier 2
  - Constructor defaults (base_url, tier_models)
  - Custom base_url / tier_models override
"""

import logging
import warnings

import pytest

logger = logging.getLogger(__name__)


class TestResolveModel:
    def test_tier_1(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(1)
        logger.info("Tier 1 → model=%r", model)
        assert model == "tier1"

    def test_tier_2(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(2)
        logger.info("Tier 2 → model=%r", model)
        assert model == "tier2"

    def test_tier_3(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(3)
        logger.info("Tier 3 → model=%r", model)
        assert model == "tier3"

    def test_unknown_tier_warns_and_fallback(self):
        from llm.client import LLMClient
        client = LLMClient()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = client._resolve_model(99)
        logger.info("Unknown tier 99 → model=%r, warnings=%d", model, len(w))
        assert model == "tier2"  # fallback
        assert len(w) == 1
        assert "Unknown tier" in str(w[0].message)


class TestConstructor:
    def test_default_base_url(self):
        from llm.client import LLMClient
        client = LLMClient()
        logger.info("Default base_url: %s", client.base_url)
        assert "localhost" in client.base_url or "4000" in client.base_url

    def test_custom_base_url(self):
        from llm.client import LLMClient
        client = LLMClient(base_url="http://my-proxy:8080")
        logger.info("Custom base_url: %s", client.base_url)
        assert client.base_url == "http://my-proxy:8080"

    def test_trailing_slash_stripped(self):
        from llm.client import LLMClient
        client = LLMClient(base_url="http://my-proxy:8080/")
        logger.info("Stripped URL: %s", client.base_url)
        assert not client.base_url.endswith("/")

    def test_custom_tier_models(self):
        from llm.client import LLMClient
        custom = {1: "phi3", 2: "haiku", 3: "sonnet"}
        client = LLMClient(tier_models=custom)
        logger.info("Custom tier models: %s", client.tier_models)
        assert client._resolve_model(1) == "phi3"
        assert client._resolve_model(3) == "sonnet"


class TestLLMError:
    def test_error_is_exception(self):
        from llm.client import LLMError
        with pytest.raises(LLMError):
            raise LLMError("test error")
        logger.info("LLMError is a proper exception ✓")
