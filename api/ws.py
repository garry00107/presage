"""
WebSocket endpoint for real-time token streaming.

Protocol:
  Client → Server:
    {"type": "turn", "message": "...", "session_id": "..."}

  Server → Client:
    {"type": "signal", "intent": "DEBUG", "velocity": 0.23}
    {"type": "staged", "slot": 0, "confidence": 0.91, "tokens": 847}
    {"type": "token", "content": "H"}  (one per token during streaming)
    {"type": "done", "turn_index": 3, "latency_ms": 234.5}
    {"type": "error", "message": "..."}

The WebSocket handler manages its own session — no need to create
one via REST first. Sessions are keyed by the WebSocket session_id.
"""

import asyncio
import json
import time
import structlog
from fastapi import WebSocket, WebSocketDisconnect

from api.deps import get_factory

log = structlog.get_logger(__name__)


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Handle one WebSocket connection for a PPM session.
    One connection = one session = one user conversation.
    """
    await websocket.accept()
    factory = get_factory()

    # Create or retrieve session
    session = await factory.get_session(session_id)
    if session is None:
        session = await factory.create_session(session_id=session_id)

    log.info("ws.connected", session_id=session_id)

    try:
        async for raw_msg in _receive_messages(websocket):
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            if msg.get("type") == "turn":
                await _handle_turn(websocket, session, msg)
            elif msg.get("type") == "ping":
                await _send(websocket, {"type": "pong"})
            else:
                await _send(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg.get('type')}"
                })

    except WebSocketDisconnect:
        log.info("ws.disconnected", session_id=session_id)
    except Exception as e:
        log.error("ws.error", session_id=session_id, error=str(e))
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await factory.close_session(session_id)


async def _handle_turn(websocket: WebSocket, session, msg: dict) -> None:
    """Process one turn message over WebSocket."""
    user_message = msg.get("message", "")
    source       = msg.get("source", "")
    t0           = time.monotonic()

    # Send signal info as soon as we have it
    # (observe runs before LLM — gives user fast feedback)
    signals = await session._observer.observe(user_message)
    await _send(websocket, {
        "type": "signal",
        "intent": signals.intent.value,
        "velocity": round(session._state.current_velocity, 4),
        "switch_score": round(signals.switch_score, 4),
    })

    # Send staged slot preview
    for slot_info in session._cache.slot_summary():
        if slot_info["status"] == "ready":
            await _send(websocket, {
                "type": "staged",
                "slot": slot_info["slot"],
                "tier": slot_info.get("tier"),
                "confidence": slot_info.get("confidence"),
                "tokens": slot_info.get("tokens"),
            })

    # Run the full turn (LLM call happens here)
    result = await session.turn(user_message=user_message, source=source)

    # Send response as a single message (streaming tokens in Phase 7)
    await _send(websocket, {
        "type": "response",
        "content": result.llm_response,
    })

    await _send(websocket, {
        "type": "done",
        "turn_index": result.turn_index,
        "latency_ms": round(result.latency_ms, 1),
        "tokens_injected": result.tokens_injected,
        "memories_injected": result.memories_injected,
    })


async def _receive_messages(websocket: WebSocket):
    """Async generator that yields raw messages from WebSocket."""
    while True:
        try:
            data = await websocket.receive_text()
            yield data
        except WebSocketDisconnect:
            return


async def _send(websocket: WebSocket, data: dict) -> None:
    """Send a JSON message over WebSocket. Swallow send errors."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass

