import logging
from jsonschema import validate, ValidationError
from tools.registry import REGISTRY

logger = logging.getLogger(__name__)


async def execute(tool_name: str, tool_input: dict) -> str:
    if tool_name not in REGISTRY:
        logger.warning("Rejected unknown tool: %r", tool_name)
        return f"Error: tool '{tool_name}' is not available"

    schema = REGISTRY[tool_name]["schema"]
    try:
        validate(instance=tool_input, schema=schema)
    except ValidationError as e:
        logger.warning("Tool %r validation failed: %s", tool_name, e.message)
        return f"Error: invalid input for '{tool_name}': {e.message}"

    handler = REGISTRY[tool_name]["handler"]
    logger.info("Executing tool: %s inputs=%s", tool_name, tool_input)

    try:
        result = await handler(**tool_input)
        return str(result)
    except Exception as e:
        logger.warning("Tool %r raised: %s", tool_name, e)
        return f"Error running '{tool_name}': {e}"