"""
Main Gate ALPR Server — Orquestrador principal.

Responsabilidades (e somente estas):
  - Carregar configuração
  - Instanciar serviços e infraestrutura
  - Iniciar threads de captura e detecção
  - Coordenar o loop de detecção → votação → envio
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import torch
import torch.serialization

# ─── Compatibilidade PyTorch 2.6+ ────────────────────────────────────
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# ─── Path setup ──────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig, load_config
from domain.models import DetectionResult, PlateReading
from frame_buffer import FrameBuffer
from infrastructure.database import DatabaseRepository
from infrastructure.result_storage import ResultStorage
from infrastructure.video_capture import VideoCapture
from infrastructure.websocket_server import WebSocketServer
from roi import ROI
from services.detection_service import DetectionService
from services.voting_service import vote_best_plate

logger = logging.getLogger(__name__)


# ─── Detection Loop ──────────────────────────────────────────────────

class DetectionLoop:
    """
    Loop principal que coordena: captura → detecção → votação → envio.
    Respeita o princípio de responsabilidade única: não faz detecção,
    não faz OCR, não faz WebSocket — apenas orquestra.
    """

    def __init__(
        self,
        config: AppConfig,
        frame_buffer: FrameBuffer,
        detection_service: DetectionService,
        ws_server: WebSocketServer,
        db_repo: DatabaseRepository,
        storage: ResultStorage,
    ):
        self._config = config
        self._buffer = frame_buffer
        self._detection = detection_service
        self._ws = ws_server
        self._db = db_repo
        self._storage = storage
        self._sent_history: list[tuple[str, datetime]] = []

    async def run(self) -> None:
        rounds: list[list[PlateReading]] = []
        target_rounds = self._config.detection.validation_rounds
        min_occurrences = self._config.detection.occurrences_to_be_valid

        while True:
            await asyncio.sleep(0.01)

            frame = self._buffer.get()
            if frame is None:
                await asyncio.sleep(1)
                continue

            readings = self._detection.detect_plates(frame)
            if not readings:
                continue

            rounds.append(readings)
            if len(rounds) < target_rounds:
                continue

            results = vote_best_plate(rounds, min_occurrences)
            if not results:
                if rounds:
                    rounds.pop(0)
                continue

            await self._send_results(results)
            rounds.clear()

    async def _send_results(self, results: list[DetectionResult]) -> None:
        self._prune_history()

        for result in results:
            result.uuid = str(uuid.uuid4())

            if not self._config.send_duplicates and self._already_sent(result.formatted_plate):
                logger.debug('Placa já enviada, pulando: "%s"', result.formatted_plate)
                continue

            self._sent_history.append((result.formatted_plate, datetime.now()))
            logger.debug("Enviando resultado: %s", result.display_string)

            await self._ws.broadcast_result(result)
            self._persist_async(result)

    def _persist_async(self, result: DetectionResult) -> None:
        """Salva no banco e em disco em thread separada para não bloquear."""
        def _save():
            self._db.save_plate(result.uuid, result.formatted_plate)
            self._storage.save_images(result.uuid, result.car_image, result.plate_image)

        threading.Thread(target=_save, daemon=True).start()

    def _already_sent(self, plate_text: str) -> bool:
        return any(
            s[0] == plate_text or SequenceMatcher(None, s[0], plate_text).ratio() > 0.8
            for s in self._sent_history
        )

    def _prune_history(self) -> None:
        cutoff = datetime.now() - timedelta(minutes=5)
        self._sent_history = [(p, t) for p, t in self._sent_history if t > cutoff]


# ─── Cleanup Task ────────────────────────────────────────────────────

async def run_cleanup_task(db: DatabaseRepository, storage: ResultStorage) -> None:
    while True:
        logger.info("Executando limpeza periódica...")
        db.cleanup_old_records(days_to_keep=3)
        storage.cleanup_old_files(days_to_keep=3)
        await asyncio.sleep(24 * 3600)


# ─── Bootstrap ───────────────────────────────────────────────────────

def _start_async_loop(
    config: AppConfig,
    ws_server: WebSocketServer,
    detection_loop: DetectionLoop,
    db: DatabaseRepository,
    storage: ResultStorage,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    asyncio.ensure_future(ws_server.start())
    asyncio.ensure_future(detection_loop.run())
    asyncio.ensure_future(run_cleanup_task(db, storage))

    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def main() -> None:
    os.environ["OMP_THREAD_LIMIT"] = "1"

    # Garantir que o .env é carregado do diretório do server.py
    server_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(server_dir, ".env")
    config = load_config(env_path)

    # Logging
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config.log_summary()

    # Componentes
    roi = ROI.from_env(config.roi_points)
    frame_buffer = FrameBuffer()

    detection_service = DetectionService(config, roi)
    ws_server = WebSocketServer(config.ws_port)
    db_repo = DatabaseRepository(config.database)
    storage = ResultStorage(config.storage)
    storage.ensure_directory()

    video = VideoCapture(config.video_source, frame_buffer, roi=roi, debug=config.debug)

    detection_loop = DetectionLoop(
        config=config,
        frame_buffer=frame_buffer,
        detection_service=detection_service,
        ws_server=ws_server,
        db_repo=db_repo,
        storage=storage,
    )

    if roi.enabled:
        print(f"✓ ROI ativada com {len(roi.points)} pontos")
    else:
        print("⚠ ROI desativada — processando frame inteiro")

    print(f"✓ WebSocket na porta {config.ws_port}")
    print(f"✓ Captura: {config.video_source}")

    # Threads
    capture_thread = threading.Thread(target=video.run_forever, daemon=True)
    work_thread = threading.Thread(
        target=_start_async_loop,
        args=(config, ws_server, detection_loop, db_repo, storage),
        daemon=True,
    )

    capture_thread.start()
    work_thread.start()

    capture_thread.join()
    work_thread.join()


if __name__ == "__main__":
    main()