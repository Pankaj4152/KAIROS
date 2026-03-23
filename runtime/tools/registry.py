import logging
from tools.web_search import web_search

logger = logging.getLogger(__name__)

REGISTRY: dict[str, dict] = {
    "web_search": {
        "description": "Search the web for current information",
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": web_search,
    },
}