"""
Buffer thread-safe para o frame mais recente da câmera.

Substitui a variável global LATEST_FRAME que não tinha proteção de concorrência.
"""
from __future__ import annotations

import threading
from typing import Optional
from PIL import Image


class FrameBuffer:
    """
    Buffer thread-safe para o frame mais recente da câmera.
    
    A thread de captura de vídeo chama update() para armazenar o frame,
    e a thread de detecção chama get() para obter uma cópia segura.
    """

    def __init__(self):
        self._frame: Optional[Image.Image] = None
        self._lock = threading.Lock()

    def update(self, frame: Image.Image):
        """Atualiza o frame mais recente (chamado pela thread de captura)."""
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[Image.Image]:
        """Obtém uma cópia do frame mais recente (chamado pela thread de detecção)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def clear(self):
        """Limpa o buffer."""
        with self._lock:
            self._frame = None
