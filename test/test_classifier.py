"""
Tests for runtime/orchestrator/classifier.py

Covers:
  - _parse: valid JSON → correct output
  - _parse: JSON in markdown fences → still parses
  - _parse: preamble text before JSON → finds it
  - _parse: unknown domain/tool names silently dropped
  - _parse: garbage → DEFAULT_RESULT
  - _parse: tier/complexity clamped to 1-3
  - classify: LLM failure → DEFAULT_RESULT
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

logger = logging.getLogger(__name__)


def _make_classifier(response_text=None):
    """Build a Classifier with a mocked LLM client."""
    from orchestrator.classifier import Classifier
    mock_llm = MagicMock()
    if response_text is not None:
        mock_llm.complete = AsyncMock(return_value=response_text)
    else:
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM down"))
    return Classifier(llm_client=mock_llm)


def run(coro):
    return asyncio.run(coro)


# ── _parse tests ──────────────────────────────────────────────────────────────

class TestParse:
    def test_valid_json(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        result = c._parse('{"intent":"task","complexity":2,"domains":["tasks"],"tools_needed":[],"tier":2}')
        logger.info("Parsed: %s", result)
        assert result["intent"] == "task"
        assert result["domains"] == ["tasks"]
        assert result["tier"] == 2

    def test_markdown_fences_stripped(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '```json\n{"intent":"search","complexity":2,"domains":[],"tools_needed":["web_search"],"tier":2}\n```'
        result = c._parse(raw)
        logger.info("Parsed from fences: %s", result)
        assert result["intent"] == "search"
        assert result["tools_needed"] == ["web_search"]

    def test_preamble_text_before_json(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = 'Here is the classification:\n{"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}'
        result = c._parse(raw)
        logger.info("Parsed with preamble: %s", result)
        assert result["intent"] == "chitchat"
        assert result["tier"] == 1

    def test_unknown_domains_dropped(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '{"intent":"question","complexity":1,"domains":["tasks","weather","unknown"],"tools_needed":[],"tier":1}'
        result = c._parse(raw)
        logger.info("Domains after filtering: %s", result["domains"])
        assert result["domains"] == ["tasks"]
        assert "weather" not in result["domains"]

    def test_unknown_tools_dropped(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '{"intent":"search","complexity":2,"domains":[],"tools_needed":["web_search","fly_drone"],"tier":2}'
        result = c._parse(raw)
        logger.info("Tools after filtering: %s", result["tools_needed"])
        assert result["tools_needed"] == ["web_search"]

    def test_garbage_returns_default(self):
        from orchestrator.classifier import Classifier, DEFAULT_RESULT
        c = Classifier.__new__(Classifier)
        result = c._parse("this is not json at all")
        logger.info("Garbage parse result: %s", result)
        assert result == DEFAULT_RESULT

    def test_tier_clamped_high(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '{"intent":"question","complexity":5,"domains":[],"tools_needed":[],"tier":10}'
        result = c._parse(raw)
        logger.info("Clamped tier: %d, complexity: %d", result["tier"], result["complexity"])
        assert result["tier"] == 3
        assert result["complexity"] == 3

    def test_tier_clamped_low(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '{"intent":"question","complexity":-1,"domains":[],"tools_needed":[],"tier":0}'
        result = c._parse(raw)
        logger.info("Clamped low tier: %d, complexity: %d", result["tier"], result["complexity"])
        assert result["tier"] == 1
        assert result["complexity"] == 1

    def test_string_tier_coerced_to_int(self):
        from orchestrator.classifier import Classifier
        c = Classifier.__new__(Classifier)
        raw = '{"intent":"question","complexity":"2","domains":[],"tools_needed":[],"tier":"3"}'
        result = c._parse(raw)
        logger.info("Coerced types: tier=%s (%s)", result["tier"], type(result["tier"]).__name__)
        assert result["tier"] == 3
        assert isinstance(result["tier"], int)


# ── classify tests ────────────────────────────────────────────────────────────

class TestClassify:
    def test_returns_parsed_result(self):
        c = _make_classifier('{"intent":"task","complexity":2,"domains":["tasks"],"tools_needed":[],"tier":2}')
        result = run(c.classify("add a task"))
        logger.info("Classified: %s", result)
        assert result["intent"] == "task"

    def test_llm_failure_returns_default(self):
        from orchestrator.classifier import DEFAULT_RESULT
        c = _make_classifier()  # LLM raises Exception
        result = run(c.classify("anything"))
        logger.info("Fallback result: %s", result)
        assert result == DEFAULT_RESULT

    def test_result_always_has_required_keys(self):
        c = _make_classifier('{"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}')
        result = run(c.classify("hey"))
        required_keys = {"intent", "complexity", "domains", "tools_needed", "tier"}
        logger.info("Result keys: %s", set(result.keys()))
        assert required_keys.issubset(result.keys())

    def test_classify_uses_tier1(self):
        """Classifier must use tier 1 (local phi) — never tier 2 for classification."""
        c = _make_classifier('{"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}')
        run(c.classify("hello"))
        _, kwargs = c.llm.complete.call_args
        assert kwargs.get("tier") == 1, f"Classifier should use tier=1, got tier={kwargs.get('tier')}"

