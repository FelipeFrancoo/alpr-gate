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
import time
import uuid
import queue
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import cv2
import numpy as np
import torch
import torch.serialization
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

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

# ─── FastAPI Setup ───────────────────────────────────────────────────
app = FastAPI(title="ALPR Monitor")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
video_frame_queue = queue.Queue(maxsize=1)
connected_websockets: list[WebSocket] = []

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    while True:
        try:
            frame_bytes = video_frame_queue.get(timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

async def broadcast_to_web(result: DetectionResult):
    data = {
        "plate": result.formatted_plate,
        "confidence": 1.0 # Placeholder, já que o modelo atual não expõe a confiança final facilmente
    }
    for ws in connected_websockets:
        try:
            await ws.send_json(data)
        except Exception:
            pass

def draw_results(frame: np.ndarray, roi_points: list[tuple[int, int]] | None, readings: list[PlateReading], vehicles_in_roi: bool) -> np.ndarray:
    # Garantir que o frame é um array numpy válido antes de copiar
    if not isinstance(frame, np.ndarray):
        try:
            frame = np.array(frame)
        except Exception:
            return frame
            
    display_frame = frame.copy()
    
    # Desenhar ROI
    if roi_points:
        roi_color = (0, 255, 0) if vehicles_in_roi else (0, 0, 255)
        
        # Garantir que roi_points seja uma lista de tuplas de inteiros
        try:
            if isinstance(roi_points, str):
                # Caso venha como string do .env (ex: "889,188;887,295...")
                pts_list = []
                for pt_str in roi_points.split(';'):
                    if pt_str.strip():
                        x, y = map(int, pt_str.split(','))
                        pts_list.append((x, y))
                pts = np.array(pts_list, np.int32)
            else:
                pts = np.array(roi_points, np.int32)
                
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, roi_color, 2)
            
            # Pegar o primeiro ponto para colocar o texto
            first_pt = pts[0][0]
            cv2.putText(display_frame, "ROI", (int(first_pt[0]), int(first_pt[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
        except Exception as e:
            logger.error(f"Erro ao desenhar ROI: {e}")

    # Desenhar Leituras (Como PlateReading não tem as coordenadas originais, desenhamos no canto)
    y_offset = 30
    for reading in readings:
        cv2.putText(display_frame, f"LIDO: {reading.text}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        y_offset += 40

    return display_frame

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
        
        # Controle de Sessão da ROI
        self._roi_active = False
        self._roi_last_seen: float | None = None
        self._roi_session_id = 0
        self._roi_exit_grace = 0.6
        
        # Buffer da sessão atual (acumula todas as leituras da passagem do veículo)
        self._session_rounds: list[list[PlateReading]] = []
        self._session_sent_plates: set[str] = set()

    async def run(self) -> None:
        min_occurrences = self._config.detection.occurrences_to_be_valid

        while True:
            await asyncio.sleep(0.01)

            frame = self._buffer.get()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            readings, vehicles_in_roi = self._detection.detect_plates(frame)
            self._update_roi_state(vehicles_in_roi)

            # Atualizar frame de vídeo para o dashboard
            try:
                display_frame = draw_results(frame, self._config.roi_points, readings, vehicles_in_roi)
                if isinstance(display_frame, np.ndarray):
                    if video_frame_queue.full():
                        try:
                            video_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    ret, buffer = cv2.imencode('.jpg', display_frame)
                    if ret:
                        video_frame_queue.put(buffer.tobytes())
            except Exception as e:
                logger.error(f"Erro ao processar frame para vídeo: {e}")

            # Lógica de Sessão: Acumular leituras enquanto o carro está na ROI
            if self._roi_active and readings:
                self._session_rounds.append(readings)
                
                # Votação contínua com TODO o histórico da sessão atual
                results = vote_best_plate(self._session_rounds, min_occurrences)
                
                # Filtrar apenas placas que ainda não enviamos nesta mesma sessão
                new_results = [r for r in results if r.formatted_plate not in self._session_sent_plates]
                
                if new_results:
                    for r in new_results:
                        self._session_sent_plates.add(r.formatted_plate)
                    await self._send_results(new_results)

    def _update_roi_state(self, vehicles_in_roi: bool) -> None:
        now = time.monotonic()
        if vehicles_in_roi:
            if not self._roi_active:
                self._roi_active = True
                self._roi_session_id += 1
                self._session_rounds.clear()
                self._session_sent_plates.clear()
                logger.info("Veículo entrou na ROI (Sessão %d iniciada)", self._roi_session_id)
            self._roi_last_seen = now
            return

        if not self._roi_active or self._roi_last_seen is None:
            return

        if now - self._roi_last_seen > self._roi_exit_grace:
            logger.info("Veículo saiu da ROI (Sessão %d encerrada)", self._roi_session_id)
            self._roi_active = False
            self._session_rounds.clear()
            self._session_sent_plates.clear()

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
            await broadcast_to_web(result)
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

    # Removido: asyncio.ensure_future(ws_server.start()) pois o FastAPI já cuida do WebSocket
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

    # Iniciar servidor web FastAPI
    print(f"✓ Dashboard Web disponível em: http://localhost:8765")
    try:
        # O FastAPI já está rodando o WebSocket na porta 8765, então não precisamos do WebSocketServer antigo
        uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
    except KeyboardInterrupt:
        print("\nDesligando o servidor...")
    finally:
        # O daemon=True nas threads garante que elas morram quando a thread principal (uvicorn) terminar.
        print("Servidor encerrado.")


if __name__ == "__main__":
    main()