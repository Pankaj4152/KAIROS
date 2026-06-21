"""
Tests for runtime/tools/notes.py

Run from the project root:
    pytest test/test_notes.py -v

Structure:
    Unit/integration-style tests against a real temp SQLite DB (FTS5 triggers
    and virtual tables are worth exercising for real, not mocked).
    semantic_search is mocked at the memory.vector_store import boundary
    since it depends on an external embeddings endpoint (LiteLLM proxy) —
    that's exactly the kind of network dependency this test suite avoids
    by default per the project's existing test conventions.

What is tested:
    _validate_link_type     — valid values, None passthrough, invalid
    _normalize_tags         — comma string, list input, None, empty/whitespace filtering
    _snippet                — short body unchanged, long body truncated with ellipsis
    notes() dispatch        — unknown action, missing required fields
    create action            — happy path, with tags, with link, empty title rejected,
                               link_type without link_id rejected
    list action                — shows recent notes, tag filter, empty state, ordering
                               by updated_at desc
    get action                   — happy path with full body, nonexistent note_id
    search action (FTS5)           — match in title, match in body, no match,
                               empty query rejected, punctuation in query doesn't crash,
                               multi-term AND search
    update action                    — single field, multiple fields, updated_at bumped,
                               nonexistent note_id, no fields specified,
                               empty title rejected
    delete action                      — happy path, FTS index cleaned up too,
                               nonexistent note_id
    link action                          — happy path, invalid link_type,
                               missing link_id, nonexistent note_id
    semantic_search (mocked)               — happy path with note-source filtering,
                               excludes conversation-source results,
                               empty results, import failure handled gracefully
    end-to-end                               — create → search → update → link →
                               get → delete
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class NotesTestCase(unittest.TestCase):
    """Base class — sets up an isolated temp DB for every test method."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_dir = self._tmpdir.name
        self._db_path = os.path.join(self._data_dir, "kairos.db")

        self._env_patch = patch.dict(os.environ, {"DATA_DIR": self._data_dir})
        self._env_patch.start()

        import importlib
        if "tools.notes" in sys.modules:
            del sys.modules["tools.notes"]
        import tools.notes as notes_mod
        importlib.reload(notes_mod)
        self.notes_mod = notes_mod
        self.notes = notes_mod.notes

        # Create schema immediately so tables exist for all tests
        with self.notes_mod._get_conn() as conn:
            self.notes_mod._ensure_schema(conn)

    def tearDown(self):
        self._env_patch.stop()
        import gc
        gc.collect()
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            # On Windows, sqlite3 + FTS5 files are sometimes held briefly by background threads.
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass

    def _row_count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        return n

    def _create_note_id(self, title="Test note", body="Some body text") -> int:
        result = run(self.notes(action="create", title=title, body=body))
        return int(result.split("#")[1].split(":")[0])


# ── unit tests: pure helpers ──────────────────────────────────────────────────

class TestValidateLinkType(NotesTestCase):

    def test_valid_values(self):
        self.assertEqual(self.notes_mod._validate_link_type("task"), "task")
        self.assertEqual(self.notes_mod._validate_link_type("habit"), "habit")

    def test_none_passthrough(self):
        self.assertIsNone(self.notes_mod._validate_link_type(None))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.notes_mod._validate_link_type("event")

    def test_case_insensitive(self):
        self.assertEqual(self.notes_mod._validate_link_type("TASK"), "task")


class TestNormalizeTags(NotesTestCase):

    def test_comma_string(self):
        result = self.notes_mod._normalize_tags("ml, research, kairos")
        self.assertEqual(result, "ml,research,kairos")

    def test_list_input(self):
        result = self.notes_mod._normalize_tags(["ml", "research"])
        self.assertEqual(result, "ml,research")

    def test_none_returns_none(self):
        self.assertIsNone(self.notes_mod._normalize_tags(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.notes_mod._normalize_tags(""))

    def test_whitespace_filtered(self):
        result = self.notes_mod._normalize_tags("ml,  , research,")
        self.assertEqual(result, "ml,research")

    def test_lowercased(self):
        result = self.notes_mod._normalize_tags("ML,Research")
        self.assertEqual(result, "ml,research")


class TestSnippet(NotesTestCase):

    def test_short_body_unchanged(self):
        result = self.notes_mod._snippet("Short text", length=100)
        self.assertEqual(result, "Short text")

    def test_long_body_truncated(self):
        long_text = "x" * 200
        result = self.notes_mod._snippet(long_text, length=100)
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 104)

    def test_newlines_collapsed(self):
        result = self.notes_mod._snippet("Line one\nLine two\nLine three")
        self.assertNotIn("\n", result)


# ── unit tests: dispatch ──────────────────────────────────────────────────────

class TestDispatch(NotesTestCase):

    def test_unknown_action(self):
        result = run(self.notes(action="teleport"))
        self.assertIn("Error", result)

    def test_create_missing_title(self):
        result = run(self.notes(action="create"))
        self.assertIn("Error", result)

    def test_get_missing_note_id(self):
        result = run(self.notes(action="get"))
        self.assertIn("Error", result)

    def test_update_missing_note_id(self):
        result = run(self.notes(action="update", title="X"))
        self.assertIn("Error", result)

    def test_delete_missing_note_id(self):
        result = run(self.notes(action="delete"))
        self.assertIn("Error", result)

    def test_search_missing_query(self):
        result = run(self.notes(action="search"))
        self.assertIn("Error", result)

    def test_link_missing_fields(self):
        result = run(self.notes(action="link", note_id=1))
        self.assertIn("Error", result)


# ── unit tests: create ────────────────────────────────────────────────────────

class TestCreate(NotesTestCase):

    def test_happy_path(self):
        result = run(self.notes(action="create", title="Idea: caching layer"))
        self.assertIn("Created note", result)
        self.assertIn("caching layer", result)
        self.assertEqual(self._row_count(), 1)

    def test_with_body(self):
        result = run(self.notes(
            action="create", title="Meeting notes", body="Discussed Q3 roadmap"
        ))
        self.assertIn("Created note", result)

    def test_with_tags(self):
        result = run(self.notes(action="create", title="ML paper", tags="ml,research"))
        self.assertIn("ml,research", result)

    def test_with_link(self):
        result = run(self.notes(
            action="create", title="Task note", link_type="task", link_id=5
        ))
        self.assertIn("Created note", result)

    def test_link_type_without_link_id_rejected(self):
        result = run(self.notes(action="create", title="Bad link", link_type="task"))
        self.assertIn("Error", result)
        self.assertEqual(self._row_count(), 0)

    def test_empty_title_rejected(self):
        result = run(self.notes(action="create", title=""))
        self.assertIn("Error", result)
        self.assertEqual(self._row_count(), 0)

    def test_empty_body_defaults_to_empty_string(self):
        result = run(self.notes(action="create", title="No body note"))
        self.assertIn("Created note", result)


# ── unit tests: list ───────────────────────────────────────────────────────────

class TestList(NotesTestCase):

    def test_empty_state(self):
        result = run(self.notes(action="list"))
        self.assertIn("No notes", result)

    def test_shows_recent_notes(self):
        run(self.notes(action="create", title="First note"))
        run(self.notes(action="create", title="Second note"))
        result = run(self.notes(action="list"))
        self.assertIn("First note", result)
        self.assertIn("Second note", result)

    def test_tag_filter(self):
        run(self.notes(action="create", title="ML note", tags="ml"))
        run(self.notes(action="create", title="Cooking note", tags="recipes"))
        result = run(self.notes(action="list", tag="ml"))
        self.assertIn("ML note", result)
        self.assertNotIn("Cooking note", result)

    def test_ordering_newest_updated_first(self):
        run(self.notes(action="create", title="Older note"))
        run(self.notes(action="create", title="Newer note"))
        result = run(self.notes(action="list"))
        idx_newer = result.find("Newer note")
        idx_older = result.find("Older note")
        self.assertLess(idx_newer, idx_older)

    def test_limit_respected(self):
        for i in range(10):
            run(self.notes(action="create", title=f"Bulk note {i}"))
        result = run(self.notes(action="list", limit=3))
        self.assertIn("Notes (3)", result)


# ── unit tests: get ────────────────────────────────────────────────────────────

class TestGet(NotesTestCase):

    def test_happy_path(self):
        note_id = self._create_note_id("Full note", "This is the complete body text.")
        result = run(self.notes(action="get", note_id=note_id))
        self.assertIn("Full note", result)
        self.assertIn("This is the complete body text.", result)

    def test_shows_tags_when_present(self):
        result = run(self.notes(action="create", title="Tagged", tags="important"))
        note_id = int(result.split("#")[1].split(":")[0])
        get_result = run(self.notes(action="get", note_id=note_id))
        self.assertIn("important", get_result)

    def test_nonexistent_note_id(self):
        result = run(self.notes(action="get", note_id=99999))
        self.assertIn("Error", result)


# ── unit tests: search (FTS5) ──────────────────────────────────────────────────

class TestSearch(NotesTestCase):

    def setUp(self):
        super().setUp()
        run(self.notes(action="create", title="Caching strategy", body="Use Redis for hot data"))
        run(self.notes(action="create", title="Meeting recap", body="Discussed caching layer design"))
        run(self.notes(action="create", title="Recipe ideas", body="Pasta with garlic"))

    def test_match_in_title(self):
        result = run(self.notes(action="search", query="caching"))
        self.assertIn("Caching strategy", result)
        self.assertIn("Meeting recap", result)   # "caching" also in body
        self.assertNotIn("Recipe ideas", result)

    def test_match_in_body(self):
        result = run(self.notes(action="search", query="Redis"))
        self.assertIn("Caching strategy", result)
        self.assertNotIn("Recipe ideas", result)

    def test_no_match(self):
        result = run(self.notes(action="search", query="xyznonexistentterm"))
        self.assertIn("No notes found", result)

    def test_empty_query_rejected(self):
        result = run(self.notes(action="search", query=""))
        self.assertIn("Error", result)

    def test_punctuation_does_not_crash(self):
        # FTS5 special characters should be sanitised, not raise a syntax error
        result = run(self.notes(action="search", query="what's caching?"))
        self.assertIsInstance(result, str)
        self.assertNotIn("OperationalError", result)

    def test_multi_term_search(self):
        result = run(self.notes(action="search", query="caching layer"))
        self.assertIn("Meeting recap", result)


# ── unit tests: update ─────────────────────────────────────────────────────────

class TestUpdate(NotesTestCase):

    def setUp(self):
        super().setUp()
        self.note_id = self._create_note_id("Original title", "Original body")

    def test_update_title(self):
        result = run(self.notes(action="update", note_id=self.note_id, title="New title"))
        self.assertIn("title", result)
        get_result = run(self.notes(action="get", note_id=self.note_id))
        self.assertIn("New title", get_result)

    def test_update_body(self):
        run(self.notes(action="update", note_id=self.note_id, body="New body text"))
        get_result = run(self.notes(action="get", note_id=self.note_id))
        self.assertIn("New body text", get_result)

    def test_update_multiple_fields(self):
        result = run(self.notes(
            action="update", note_id=self.note_id,
            title="Updated", body="Updated body", tags="new,tags"
        ))
        self.assertIn("title", result)
        self.assertIn("body", result)
        self.assertIn("tags", result)

    def test_search_reflects_update(self):
        run(self.notes(action="update", note_id=self.note_id, body="searchable_unique_term_xyz"))
        result = run(self.notes(action="search", query="searchable_unique_term_xyz"))
        self.assertIn("Original title", result)

    def test_nonexistent_note_id(self):
        result = run(self.notes(action="update", note_id=99999, title="X"))
        self.assertIn("Error", result)

    def test_no_fields_specified(self):
        result = run(self.notes(action="update", note_id=self.note_id))
        self.assertIn("No changes", result)

    def test_empty_title_rejected(self):
        result = run(self.notes(action="update", note_id=self.note_id, title=""))
        self.assertIn("Error", result)


# ── unit tests: delete ────────────────────────────────────────────────────────

class TestDelete(NotesTestCase):

    def test_happy_path(self):
        note_id = self._create_note_id("Delete me", "searchable_delete_test_term")
        self.assertEqual(self._row_count(), 1)

        result = run(self.notes(action="delete", note_id=note_id))
        self.assertIn("Deleted", result)
        self.assertEqual(self._row_count(), 0)

    def test_fts_index_cleaned_up(self):
        note_id = self._create_note_id("FTS cleanup test", "unique_fts_cleanup_marker")
        run(self.notes(action="delete", note_id=note_id))

        # Search for the deleted note's content — must not appear
        result = run(self.notes(action="search", query="unique_fts_cleanup_marker"))
        self.assertIn("No notes found", result)

    def test_nonexistent_note_id(self):
        result = run(self.notes(action="delete", note_id=99999))
        self.assertIn("Error", result)


# ── unit tests: link ────────────────────────────────────────────────────────────

class TestLink(NotesTestCase):

    def setUp(self):
        super().setUp()
        self.note_id = self._create_note_id("Note to link")

    def test_happy_path(self):
        result = run(self.notes(action="link", note_id=self.note_id, link_type="task", link_id=42))
        self.assertIn("Linked", result)
        self.assertIn("task#42", result)

    def test_invalid_link_type(self):
        result = run(self.notes(action="link", note_id=self.note_id, link_type="event", link_id=1))
        self.assertIn("Error", result)

    def test_missing_link_id(self):
        result = run(self.notes(action="link", note_id=self.note_id, link_type="task"))
        self.assertIn("Error", result)

    def test_nonexistent_note_id(self):
        result = run(self.notes(action="link", note_id=99999, link_type="task", link_id=1))
        self.assertIn("Error", result)

    def test_link_reflected_in_get(self):
        run(self.notes(action="link", note_id=self.note_id, link_type="habit", link_id=7))
        result = run(self.notes(action="get", note_id=self.note_id))
        self.assertIn("habit#7", result)


# ── unit tests: semantic_search (mocked vector_store) ──────────────────────────

class TestSemanticSearch(NotesTestCase):

    def test_empty_query_rejected(self):
        result = run(self.notes(action="semantic_search", query=""))
        self.assertIn("Error", result)

    @patch("memory.vector_store.search", new_callable=AsyncMock)
    def test_happy_path_filters_to_notes(self, mock_search):
        mock_search.return_value = [
            {
                "id": 1, "content": "A note about caching",
                "source": "note", "session_id": None,
                "created_at": "2026-06-01T00:00:00", "distance": 0.1,
            },
            {
                "id": 2, "content": "USER: hi\nASSISTANT: hello",
                "source": "conversation", "session_id": "abc",
                "created_at": "2026-06-02T00:00:00", "distance": 0.2,
            },
        ]
        result = run(self.notes(action="semantic_search", query="caching"))
        self.assertIn("caching", result)
        self.assertNotIn("hello", result)   # conversation source filtered out

    @patch("memory.vector_store.search", new_callable=AsyncMock)
    def test_no_note_results(self, mock_search):
        mock_search.return_value = [
            {
                "id": 2, "content": "USER: hi\nASSISTANT: hello",
                "source": "conversation", "session_id": "abc",
                "created_at": "2026-06-02T00:00:00", "distance": 0.2,
            },
        ]
        result = run(self.notes(action="semantic_search", query="anything"))
        self.assertIn("No semantically related notes", result)

    @patch("memory.vector_store.search", new_callable=AsyncMock)
    def test_search_exception_handled(self, mock_search):
        mock_search.side_effect = Exception("embedding endpoint down")
        result = run(self.notes(action="semantic_search", query="test"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)

    def test_import_failure_handled_gracefully(self):
        # Simulate vector_store module being unavailable entirely
        with patch.dict(sys.modules, {"memory.vector_store": None}):
            result = run(self.notes(action="semantic_search", query="test"))
            self.assertIsInstance(result, str)


# ── end-to-end ─────────────────────────────────────────────────────────────────

class TestEndToEnd(NotesTestCase):

    def test_full_lifecycle(self):
        # Create
        created = run(self.notes(
            action="create", title="Kairos architecture idea",
            body="Consider splitting the classifier into two stages",
            tags="kairos,architecture",
        ))
        self.assertIn("Created", created)
        note_id = int(created.split("#")[1].split(":")[0])

        # Search — should find it
        searched = run(self.notes(action="search", query="classifier"))
        self.assertIn("Kairos architecture idea", searched)

        # Update
        updated = run(self.notes(
            action="update", note_id=note_id, body="Updated: two-stage classifier confirmed"
        ))
        self.assertIn("body", updated)

        # Link to a task
        linked = run(self.notes(action="link", note_id=note_id, link_type="task", link_id=10))
        self.assertIn("Linked", linked)

        # Get — full content with link
        got = run(self.notes(action="get", note_id=note_id))
        self.assertIn("two-stage classifier confirmed", got)
        self.assertIn("task#10", got)

        # Delete
        deleted = run(self.notes(action="delete", note_id=note_id))
        self.assertIn("Deleted", deleted)
        self.assertEqual(self._row_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)