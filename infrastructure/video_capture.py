from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PIL import Image

from frame_buffer import FrameBuffer
from roi import ROI

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Thread de captura de vídeo que alimenta o FrameBuffer.
    Reconecta automaticamente em caso de falha.
    """

    def __init__(self, source: str, frame_buffer: FrameBuffer, roi: ROI | None = None, debug: bool = False, reconnect_interval: int = 5):
        self._source = source
        self._buffer = frame_buffer
        self._roi = roi
        self._debug = debug
        self._reconnect_interval = reconnect_interval

    def run_forever(self) -> None:
        """Loop infinito de captura. Projetado para rodar em thread separada."""
        while True:
            capture = cv2.VideoCapture(self._source)

            if not capture.isOpened():
                logger.warning("Falha ao conectar: %s. Tentando em %ds...", self._source, self._reconnect_interval)
                self._buffer.clear()
                time.sleep(self._reconnect_interval)
                continue

            logger.info("Captura aberta: %s", self._source)
            if self._source.startswith("rtsp"):
                # Reduz latência em fluxos RTSP
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self._read_frames(capture)
            capture.release()

    def _read_frames(self, capture: cv2.VideoCapture) -> None:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                logger.warning("Fim do luxo de vídeo ou falha de leitura (frame vazio), reiniciando conexão...")
                self._buffer.clear()
                return

            pil_frame = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            )
            self._buffer.update(pil_frame)

            if self._debug and self._roi and self._roi.enabled:
                debug_frame = self._roi.draw_on_frame(frame)
                cv2.imwrite("./intermediate_detection_files/latest_frame_with_roi.jpg", debug_frame)
