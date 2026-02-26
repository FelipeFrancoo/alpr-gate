from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import load_config
from services import ocr_service
from services.plate_validator import validate_plate


def _iter_plate_boxes(plate_model: YOLO, image: Image.Image, confidence: float) -> Iterable:
    result = plate_model.predict(image, verbose=False, conf=confidence)[0]
    return list(result.boxes)


def _crop_plate(image_cv: np.ndarray, box, padding: int = 6) -> np.ndarray:
    x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
    h, w = image_cv.shape[:2]

    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w, x2 + padding)
    py2 = min(h, y2 + padding)

    return image_cv[py1:py2, px1:px2]


def read_plate_from_image(image_path: str, assume_plate_crop: bool, confidence: float) -> tuple[str | None, float]:
    image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if assume_plate_crop:
        warped = ocr_service.warp_plate(image_cv, 0, 0, image_cv.shape[1], image_cv.shape[0], padding=2)
        text, conf = ocr_service.read_plate_text(warped)
        validated = validate_plate(text)
        return (validated or text, conf)

    config = load_config(os.path.join(os.path.dirname(__file__), ".env"))
    plate_model = YOLO(config.yolo.plate_model_path)
    boxes = _iter_plate_boxes(plate_model, image, confidence)

    if not boxes:
        return None, 0.0

    best = max(boxes, key=lambda b: float(b.conf))
    crop = _crop_plate(image_cv, best)
    if crop.size == 0:
        return None, 0.0

    warped = ocr_service.warp_plate(crop, 0, 0, crop.shape[1], crop.shape[0], padding=2)
    text, conf = ocr_service.read_plate_text(warped)
    validated = validate_plate(text)
    return (validated or text, conf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ler placa a partir de uma imagem.")
    parser.add_argument("image", help="Caminho da imagem (carro ou placa).")
    parser.add_argument("--plate-crop", action="store_true", help="A imagem já é um recorte da placa.")
    parser.add_argument("--confidence", type=float, default=0.45, help="Confiança mínima do YOLO para placa.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"Imagem não encontrada: {args.image}")

    text, conf = read_plate_from_image(args.image, args.plate_crop, args.confidence)
    if not text:
        print("Nenhuma placa detectada.")
        return

    print(f"Placa: {text} (confiança OCR: {conf:.2f})")


if __name__ == "__main__":
    main()
