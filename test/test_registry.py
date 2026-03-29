"""Tests for runtime/tools/registry.py."""


def test_registry_contains_web_search_definition():
    from tools import registry

    assert "web_search" in registry.REGISTRY
    tool = registry.REGISTRY["web_search"]
    assert tool["enabled"] is True
    assert callable(tool["handler"])
    assert tool["schema"]["type"] == "object"


def test_check_eligibility_marks_disabled_tool_false(monkeypatch):
    from tools import registry

    fake_registry = {
        "disabled_tool": {
            "enabled": False,
            "requires_env": [],
            "schema": {"type": "object"},
            "description": "disabled",
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(registry, "REGISTRY", fake_registry)

    eligibility = registry.check_eligibility()
    assert eligibility == {"disabled_tool": False}


def test_check_eligibility_requires_env(monkeypatch):
    from tools import registry

    fake_registry = {
        "needs_env": {
            "enabled": True,
            "requires_env": ["MISSING_API_KEY"],
            "schema": {"type": "object"},
            "description": "env required",
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(registry, "REGISTRY", fake_registry)
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    eligibility = registry.check_eligibility()
    assert eligibility == {"needs_env": False}


def test_check_eligibility_ready_when_env_present(monkeypatch):
    from tools import registry

    fake_registry = {
        "ready_tool": {
            "enabled": True,
            "requires_env": ["HAS_API_KEY"],
            "schema": {"type": "object"},
            "description": "env required",
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(registry, "REGISTRY", fake_registry)
    monkeypatch.setenv("HAS_API_KEY", "set")

    eligibility = registry.check_eligibility()
    assert eligibility == {"ready_tool": True}


def test_get_eligibility_is_cached(monkeypatch):
    from tools import registry

    calls = {"count": 0}

    def fake_check():
        calls["count"] += 1
        return {"web_search": True}

    monkeypatch.setattr(registry, "_eligibility", None)
    monkeypatch.setattr(registry, "check_eligibility", fake_check)

    first = registry.get_eligibility()
    second = registry.get_eligibility()

    assert first == {"web_search": True}
    assert second == {"web_search": True}
    assert calls["count"] == 1


def test_get_tool_schemas_only_returns_eligible(monkeypatch):
    from tools import registry

    fake_registry = {
        "allowed": {
            "enabled": True,
            "requires_env": [],
            "description": "allowed tool",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "handler": lambda: (lambda **kwargs: "ok"),
        },
        "blocked": {
            "enabled": True,
            "requires_env": [],
            "description": "blocked tool",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "handler": lambda: (lambda **kwargs: "ok"),
        },
    }

    monkeypatch.setattr(registry, "REGISTRY", fake_registry)
    monkeypatch.setattr(registry, "get_eligibility", lambda: {"allowed": True, "blocked": False})

    schemas = registry.get_tool_schemas()

    assert len(schemas) == 1
    assert schemas[0]["name"] == "allowed"
    assert schemas[0]["description"] == "allowed tool"
    assert schemas[0]["input_schema"]["type"] == "object"
