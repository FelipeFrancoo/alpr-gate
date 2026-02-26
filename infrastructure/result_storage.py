from __future__ import annotations

import logging
import os
import time

from PIL import Image

from config import StorageConfig

logger = logging.getLogger(__name__)


class ResultStorage:
    """Salva imagens de resultados em disco e faz limpeza periódica."""

    def __init__(self, config: StorageConfig):
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.save_enabled

    def ensure_directory(self) -> None:
        if self.enabled:
            os.makedirs(self._config.results_path, exist_ok=True)

    def save_images(self, plate_id: str, car_image: Image.Image, plate_image: Image.Image) -> None:
        if not self.enabled:
            return

        path = self._config.results_path
        try:
            car_image.save(os.path.join(path, f"{plate_id}_car.jpg"), "JPEG")
            plate_image.save(os.path.join(path, f"{plate_id}_lp.jpg"), "JPEG")
        except Exception as e:
            logger.error("[SAVE] Erro ao salvar imagens: %s", e)

    def cleanup_old_files(self, days_to_keep: int = 3) -> None:
        if not self.enabled:
            return

        path = self._config.results_path
        if not os.path.exists(path):
            return

        max_age = days_to_keep * 86400
        now = time.time()

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue

            age = now - os.path.getmtime(file_path)
            if age > max_age:
                try:
                    os.remove(file_path)
                    logger.info("Removido arquivo antigo: %s", file_path)
                except OSError as e:
                    logger.error("Erro ao remover %s: %s", file_path, e)
