from __future__ import annotations

import logging
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import AppConfig
from domain.models import PlateReading
from roi import ROI
from services import ocr_service

logger = logging.getLogger(__name__)

VEHICLE_LABELS = frozenset(["car", "motorcycle", "bus", "train", "truck", "boat"])

_INTERMEDIATE_DIR = "./intermediate_detection_files"


class DetectionService:
    """Detecta veículos no frame e lê placas usando YOLO + OCR."""

    def __init__(self, config: AppConfig, roi: ROI):
        self._config = config
        self._roi = roi
        self._vehicle_model = YOLO(config.yolo.vehicle_model_path)
        self._plate_model = YOLO(config.yolo.plate_model_path)

    def detect_plates(self, frame: Image.Image) -> tuple[list[PlateReading], bool]:
        """
        Pipeline completo de detecção num único frame:
          1. Detectar veículos com YOLO
          2. Filtrar por ROI e distância
          3. Detectar placas em cada veículo
          4. Ler texto via OCR
        """
        vehicle_boxes = self._detect_vehicles(frame)
        if not vehicle_boxes:
            logger.debug("Nenhum veículo detectado")
            return [], False

        self._prepare_debug_dir()
        readings: list[PlateReading] = []
        vehicles_in_roi = False

        for i, box in enumerate(vehicle_boxes):
            result = self._process_vehicle(frame, box, i)
            if result is None:
                continue

            vehicles_in_roi = True
            car_crop, vehicle_coords = result
            plate_readings = self._read_plates_from_vehicle(car_crop, i, frame, vehicle_coords)
            readings.extend(plate_readings)

        return readings, vehicles_in_roi

    # ─── Pipeline interno ─────────────────────────────────────────────

    def _detect_vehicles(self, frame: Image.Image) -> list:
        confidence = self._config.yolo.confidence
        result = self._vehicle_model.predict(frame, verbose=self._config.debug, conf=confidence)[0]
        return list(result.boxes) if len(result.boxes) > 0 else []

    def _process_vehicle(self, frame: Image.Image, box, index: int) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        label = self._vehicle_model.names[int(box.cls)].strip().lower()
        if label not in VEHICLE_LABELS:
            logger.debug('Rótulo "%s" não é veículo, pulando', label)
            return None

        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        logger.debug("Veículo %d: coords (%.0f, %.0f, %.0f, %.0f)", index, x_min, y_min, x_max, y_max)

        if not self._is_inside_roi(x_min, y_min, x_max, y_max):
            logger.debug("  ✗ Fora da ROI")
            return None

        if y_max < self._config.detection.skip_before_y_max:
            logger.debug("  ✗ Muito longe (y_max=%.0f)", y_max)
            return None

        logger.debug("  ✓ Dentro da ROI, processando placas...")
        car_crop = frame.crop((x_min, y_min, x_max, y_max))

        if self._config.debug:
            car_crop.save(self._intermediate_path(f"cropped_car_{index}.jpg"))

        return car_crop, (int(x_min), int(y_min), int(x_max), int(y_max))

    def _read_plates_from_vehicle(
        self, car_image: Image.Image, car_index: int, frame: Image.Image = None, vehicle_coords: tuple[int, int, int, int] = None,
    ) -> list[PlateReading]:
        confidence = self._config.yolo.confidence
        result = self._plate_model.predict(car_image, verbose=self._config.debug, conf=confidence)[0]
        plate_boxes = list(result.boxes)

        if not plate_boxes:
            logger.debug("  ✗ Nenhuma placa detectada")
            return []

        logger.debug("  ✓ %d placa(s) detectada(s)", len(plate_boxes))
        min_chars = self._config.detection.min_chars_for_match
        readings: list[PlateReading] = []

        for j, plate_box in enumerate(plate_boxes):
            reading = self._read_single_plate(car_image, plate_box, car_index, j, frame, vehicle_coords)
            if reading is None:
                continue

            if reading.length < min_chars:
                logger.debug("  ✗ Placa curta demais (%d < %d)", reading.length, min_chars)
                continue

            logger.debug("  ✓ LEITURA ACEITA: '%s'", reading.normalized_text)
            readings.append(reading)

        return readings

    def _read_single_plate(
        self, car_image: Image.Image, box, car_idx: int, plate_idx: int,
        frame: Image.Image = None, vehicle_coords: tuple[int, int, int, int] = None,
    ) -> PlateReading | None:
        x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
        x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)

        plate_w, plate_h = x2 - x1, y2 - y1
        logger.debug("  [Placa %d_%d] Tamanho original: %dx%d px", car_idx, plate_idx, plate_w, plate_h)

        # Se temos o frame original e a placa é pequena, extrair do frame completo
        # Isso dá uma resolução MUITO maior que extrair do cropped car
        if frame is not None and vehicle_coords is not None and plate_w < 150:
            vx1, vy1, vx2, vy2 = vehicle_coords
            # Converter coordenadas da placa (relativas ao crop) para coordenadas do frame
            abs_x1 = vx1 + x1
            abs_y1 = vy1 + y1
            abs_x2 = vx1 + x2
            abs_y2 = vy1 + y2
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            abs_plate_w = abs_x2 - abs_x1
            pad = 8 if abs_plate_w < 120 else 6
            warped = ocr_service.warp_plate(frame_cv, abs_x1, abs_y1, abs_x2, abs_y2, padding=pad)
            logger.debug("  [Placa %d_%d] Usando frame completo (coords abs: %d,%d,%d,%d)",
                         car_idx, plate_idx, abs_x1, abs_y1, abs_x2, abs_y2)
        else:
            car_cv = cv2.cvtColor(np.array(car_image), cv2.COLOR_RGB2BGR)
            pad = 6 if plate_w < 120 else 4
            warped = ocr_service.warp_plate(car_cv, x1, y1, x2, y2, padding=pad)

        if warped.size == 0:
            return None

        uid = f"{car_idx}_{plate_idx}"
        if self._config.debug:
            cv2.imwrite(self._intermediate_path(f"warped_plate_{uid}.jpg"), warped)

        enhanced = ocr_service.get_enhanced_image(warped)
        if self._config.debug:
            cv2.imwrite(self._intermediate_path(f"enhanced_plate_{uid}.jpg"), enhanced)

        text, confidence = ocr_service.read_plate_text(warped)
        raw_text = text.strip().upper().replace(" ", "").replace("-", "") if text else ""

        logger.debug("  [Placa %s] OCR: '%s' (conf: %.2f)", uid, raw_text, confidence)

        plate_pil = Image.fromarray(enhanced)
        return PlateReading(car_image=car_image, plate_image=plate_pil, text=raw_text)

    # ─── Helpers ──────────────────────────────────────────────────────

    def _is_inside_roi(self, x_min: float, y_min: float, x_max: float, y_max: float) -> bool:
        if not self._roi.enabled:
            return True
        return self._roi.contains_box(x_min, y_min, x_max, y_max, threshold=0.1)

    def _prepare_debug_dir(self) -> None:
        if not self._config.debug:
            return
        try:
            if os.path.exists(_INTERMEDIATE_DIR):
                shutil.rmtree(_INTERMEDIATE_DIR)
            os.makedirs(_INTERMEDIATE_DIR, exist_ok=True)
        except OSError:
            pass

    @staticmethod
    def _intermediate_path(filename: str) -> str:
        return os.path.join(_INTERMEDIATE_DIR, filename)
