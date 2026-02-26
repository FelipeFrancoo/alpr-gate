from __future__ import annotations

import io
import logging
from typing import Any

import websockets

from domain.models import DetectionResult

logger = logging.getLogger(__name__)


def image_to_bytes(image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class WebSocketServer:
    """Gerencia conexões WebSocket e broadcast de resultados."""

    def __init__(self, port: int):
        self._port = port
        self._connections: list[Any] = []

    async def start(self) -> None:
        server = await websockets.serve(self._handle_connection, "", self._port)
        logger.info("WebSocket servidor iniciado na porta %d", self._port)
        await server.wait_closed()

    async def broadcast_result(self, result: DetectionResult) -> None:
        car_bytes = image_to_bytes(result.car_image)
        plate_bytes = image_to_bytes(result.plate_image)
        message = result.display_string

        dead_sockets = []
        for ws in self._connections:
            try:
                await ws.send(car_bytes)
                await ws.send(plate_bytes)
                await ws.send(message)
            except Exception:
                logger.debug("Socket fechado durante envio")
                dead_sockets.append(ws)

        for ws in dead_sockets:
            self._connections.remove(ws)

    async def _handle_connection(self, websocket, path) -> None:
        self._connections.append(websocket)
        try:
            await websocket.send("echo")
            async for _ in websocket:
                pass
        finally:
            if websocket in self._connections:
                self._connections.remove(websocket)
