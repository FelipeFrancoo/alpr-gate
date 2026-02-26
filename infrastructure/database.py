from __future__ import annotations

import logging
from datetime import datetime, timedelta

import psycopg2

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Responsável por toda persistência no PostgreSQL."""

    def __init__(self, config: DatabaseConfig):
        self._config = config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def save_plate(self, plate_id: str, plate_text: str) -> None:
        if not self.enabled:
            return

        try:
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO main_gate_alpr_license_plates (id, license_plate, captured_at) "
                        "VALUES (%s, %s, %s)",
                        (plate_id, plate_text, datetime.now()),
                    )
                conn.commit()
        except Exception as e:
            logger.error("[DB] Erro ao salvar placa: %s", e)

    def cleanup_old_records(self, days_to_keep: int = 3) -> None:
        if not self.enabled:
            return

        try:
            cutoff = datetime.now() - timedelta(days=days_to_keep)
            with self._connect() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM main_gate_alpr_license_plates WHERE captured_at < %s",
                        (cutoff,),
                    )
                    deleted = cursor.rowcount
                conn.commit()

            if deleted > 0:
                logger.info("[DB] Removidos %d registros antigos", deleted)
        except Exception as e:
            logger.error("[DB] Erro ao limpar registros antigos: %s", e)

    def _connect(self):
        cfg = self._config
        return psycopg2.connect(
            host=cfg.host, port=cfg.port,
            database=cfg.name, user=cfg.user, password=cfg.password,
        )
