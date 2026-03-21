"""
WebUI channel adapter for Kairos.

Serves a single-page browser chat interface over FastAPI + WebSocket.

Why WebSocket over HTTP:
  HTTP request-response can't push tokens to the browser as they arrive.
  WebSocket is a persistent two-way connection — we push each token immediately.
  The browser appends tokens to the current message as they come in.

Why one session per connection:
  Each browser tab gets a UUID at connection time. Refresh = new session.
  This matches how users think about conversations.

Security note:
  Default host is 127.0.0.1 (localhost only).
  For LAN access use the machine's LAN IP or Tailscale address.
  Never bind to 0.0.0.0 on a VPS — this file has no auth layer.

Protocol:
  Browser sends:  plain text message
  Server sends:   token strings (stream), then "[DONE]" (end of response)
  On error:       server sends "[ERROR] <message>" then "[DONE]"
"""

import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

load_dotenv()

from gateway.normalizer import normalize_webui
from orchestrator.orchestrator import orchestrator

logger = logging.getLogger(__name__)

# Path to the static HTML file — configurable via env so it survives refactors
_DEFAULT_STATIC = Path(__file__).parent.parent / "static"
STATIC_DIR = Path(str(_DEFAULT_STATIC))


class WebUIChannel:

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        # Default to localhost — caller must explicitly pass LAN IP for LAN access
        self.host = host
        self.port = port
        self.app  = FastAPI(title="Kairos")
        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        async def serve_ui() -> HTMLResponse:
            """Serve the single-page chat UI from static/index.html."""
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                return HTMLResponse(html_path.read_text(encoding="utf-8"))
            logger.warning("index.html not found at %s", html_path)
            return HTMLResponse(
                "<h1>Kairos</h1><p>UI not found. Check STATIC_DIR.</p>",
                status_code=404,
            )

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            session_id = str(uuid.uuid4())
            logger.info("WebUI: session %s opened", session_id[:8])

            # Flag to prevent overlapping responses on the same connection
            responding = False

            try:
                while True:
                    text = await websocket.receive_text()

                    if not text.strip():
                        continue

                    # Ignore new messages while a response is still streaming.
                    # Prevents interleaved tokens from two concurrent LLM calls
                    # on the same WebSocket connection.
                    if responding:
                        logger.debug(
                            "WebUI: ignoring message during active response (session %s)",
                            session_id[:8],
                        )
                        continue

                    responding = True
                    try:
                        event = normalize_webui(text, session_id)

                        async for token in orchestrator.process(event):
                            try:
                                await websocket.send_text(token)
                            except WebSocketDisconnect:
                                # Browser closed tab mid-stream — stop sending
                                logger.info(
                                    "WebUI: session %s disconnected mid-stream",
                                    session_id[:8],
                                )
                                return

                        await websocket.send_text("[DONE]")

                    except WebSocketDisconnect:
                        raise   # let the outer handler log and clean up

                    except Exception as e:
                        # Orchestrator or normalizer failed — tell the browser
                        logger.exception(
                            "WebUI: orchestrator error for session %s: %s",
                            session_id[:8], e,
                        )
                        try:
                            await websocket.send_text(f"[ERROR] Something went wrong.")
                            await websocket.send_text("[DONE]")
                        except Exception:
                            pass   # connection may already be gone

                    finally:
                        responding = False

            except WebSocketDisconnect:
                logger.info("WebUI: session %s disconnected", session_id[:8])

    async def run(self) -> None:
        """
        Start the web server. Runs until Ctrl+C or SIGTERM.
        Calls orchestrator.startup() before accepting connections
        so the first request never takes the profile-load hit.
        """
        logger.info(
            "Kairos WebUI starting on http://%s:%s", self.host, self.port
        )

        await orchestrator.startup()

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()