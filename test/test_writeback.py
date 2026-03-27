"""
Tests for runtime/memory/writeback.py

Covers:
  - _extract_facts: valid JSON → list of dicts
  - _extract_facts: JSON in markdown fences → still parses
  - _extract_facts: invalid type field → filtered out
  - _extract_facts: LLM returns garbage → empty list
  - _save_facts: merges into existing preferences.json
  - _save_facts: creates preferences.json if missing
"""

import asyncio
import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

logger = logging.getLogger(__name__)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_llm(response: str):
    """Build a mock LLM client that returns the given text."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


# ── _extract_facts ────────────────────────────────────────────────────────────

class TestExtractFacts:
    def test_valid_json(self):
        from memory.writeback import _extract_facts
        llm = _mock_llm('[{"type":"preference","key":"coffee","value":"dark roast"}]')
        facts = run(_extract_facts("I prefer dark roast coffee", "Noted!", llm))
        logger.info("Extracted facts: %s", facts)
        assert len(facts) == 1
        assert facts[0]["key"] == "coffee"
        assert facts[0]["value"] == "dark roast"

    def test_json_in_markdown_fences(self):
        from memory.writeback import _extract_facts
        response = '```json\n[{"type":"fact","key":"hometown","value":"Delhi"}]\n```'
        llm = _mock_llm(response)
        facts = run(_extract_facts("I am from Delhi", "Noted!", llm))
        logger.info("Facts from fences: %s", facts)
        assert len(facts) == 1
        assert facts[0]["key"] == "hometown"

    def test_invalid_type_filtered(self):
        from memory.writeback import _extract_facts
        response = '[{"type":"random","key":"x","value":"y"},{"type":"goal","key":"fitness","value":"run 5k"}]'
        llm = _mock_llm(response)
        facts = run(_extract_facts("user", "assistant", llm))
        logger.info("Filtered facts: %s", facts)
        assert len(facts) == 1
        assert facts[0]["type"] == "goal"

    def test_garbage_returns_empty(self):
        from memory.writeback import _extract_facts
        llm = _mock_llm("I don't understand the question")
        facts = run(_extract_facts("user", "assistant", llm))
        logger.info("Garbage result: %s", facts)
        assert facts == []

    def test_empty_array_returns_empty(self):
        from memory.writeback import _extract_facts
        llm = _mock_llm("[]")
        facts = run(_extract_facts("user", "assistant", llm))
        logger.info("Empty array: %s", facts)
        assert facts == []

    def test_llm_exception_returns_empty(self):
        from memory.writeback import _extract_facts
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("timeout"))
        facts = run(_extract_facts("user", "assistant", llm))
        logger.info("LLM exception result: %s", facts)
        assert facts == []

    def test_missing_keys_filtered(self):
        from memory.writeback import _extract_facts
        response = '[{"type":"preference","key":"coffee"},{"type":"fact","key":"age","value":"25"}]'
        llm = _mock_llm(response)
        facts = run(_extract_facts("user", "assistant", llm))
        logger.info("After key filtering: %s", facts)
        # First dict missing "value", should be dropped
        assert len(facts) == 1
        assert facts[0]["key"] == "age"


# ── _save_facts ───────────────────────────────────────────────────────────────

class TestSaveFacts:
    def test_creates_preferences_file(self, tmp_data_dir):
        from memory.writeback import _save_facts
        facts = [{"type": "preference", "key": "theme", "value": "dark"}]
        run(_save_facts(facts, tmp_data_dir))

        path = os.path.join(tmp_data_dir, "preferences.json")
        assert os.path.exists(path)
        with open(path, "r") as f:
            prefs = json.load(f)
        logger.info("Created prefs: %s", prefs)
        assert prefs["preferences"]["theme"] == "dark"

    def test_merges_into_existing(self, tmp_data_dir):
        from memory.writeback import _save_facts
        # Write initial prefs
        path = os.path.join(tmp_data_dir, "preferences.json")
        initial = {"preferences": {"color": "blue"}, "facts": {}, "goals": {}}
        with open(path, "w") as f:
            json.dump(initial, f)

        # Merge new fact
        facts = [{"type": "preference", "key": "food", "value": "pizza"}]
        run(_save_facts(facts, tmp_data_dir))

        with open(path, "r") as f:
            prefs = json.load(f)
        logger.info("Merged prefs: %s", prefs)
        assert prefs["preferences"]["color"] == "blue"   # preserved
        assert prefs["preferences"]["food"] == "pizza"     # added

    def test_overwrites_existing_key(self, tmp_data_dir):
        from memory.writeback import _save_facts
        path = os.path.join(tmp_data_dir, "preferences.json")
        initial = {"preferences": {"coffee": "light"}, "facts": {}, "goals": {}}
        with open(path, "w") as f:
            json.dump(initial, f)

        facts = [{"type": "preference", "key": "coffee", "value": "dark"}]
        run(_save_facts(facts, tmp_data_dir))

        with open(path, "r") as f:
            prefs = json.load(f)
        logger.info("Updated coffee: %s", prefs["preferences"]["coffee"])
        assert prefs["preferences"]["coffee"] == "dark"

    def test_empty_facts_does_nothing(self, tmp_data_dir):
        from memory.writeback import _save_facts
        path = os.path.join(tmp_data_dir, "preferences.json")
        run(_save_facts([], tmp_data_dir))
        assert not os.path.exists(path)
        logger.info("Empty facts correctly skipped file creation")
