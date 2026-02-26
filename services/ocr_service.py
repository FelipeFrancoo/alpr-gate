from __future__ import annotations

import logging

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_CONFIDENCE_EARLY_STOP = 0.85

# Singleton: inicializado apenas uma vez
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        logger.info("Inicializando EasyOCR reader...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def read_plate_text(plate_image_cv: np.ndarray) -> tuple[str, float]:
    """
    Pipeline completo de leitura de placa:
      1. Upscale se a imagem for pequena
      2. Filtro bilateral para reduzir ruído preservando bordas
      3. Melhoria de imagem (contraste, nitidez moderada, brilho)
      4. CLAHE para equalização de histograma
      5. Padding branco ao redor para ajudar OCR nas bordas
      6. OCR com 7 estratégias de pré-processamento
    """
    processed = _upscale_if_small(plate_image_cv)
    processed = _denoise(processed)
    enhanced_color = _enhance_plate_image(processed)
    enhanced_gray = _apply_clahe(enhanced_color)
    padded = _add_padding(enhanced_gray, pad=15)

    text, confidence = _run_ocr_multi_strategy(padded)
    return text or "", confidence


def warp_plate(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    output_width: int = 400,
    output_height: int = 130,
    padding: int = 4,
) -> np.ndarray:
    """
    Recorta a placa com padding extra e normaliza via transformação projetiva.
    O padding ajuda a capturar caracteres nas bordas da placa.
    """
    h, w = image.shape[:2]
    # Adicionar padding ao recorte (clamped aos limites da imagem)
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w, x2 + padding)
    py2 = min(h, y2 + padding)

    plate = image[py1:py2, px1:px2]
    if plate.size == 0:
        return plate

    src = np.float32([
        [0, 0], [plate.shape[1], 0],
        [0, plate.shape[0]], [plate.shape[1], plate.shape[0]],
    ])
    dst = np.float32([
        [0, 0], [output_width, 0],
        [0, output_height], [output_width, output_height],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(plate, matrix, (output_width, output_height), flags=cv2.INTER_LANCZOS4)


def get_enhanced_image(warped: np.ndarray) -> np.ndarray:
    """Aplica pipeline de melhoria completo. Retorna imagem grayscale."""
    processed = _upscale_if_small(warped)
    processed = _denoise(processed)
    enhanced_color = _enhance_plate_image(processed)
    return _apply_clahe(enhanced_color)


# ─── Pré-processamento ──────────────────────────────────────────────────

def _upscale_if_small(image: np.ndarray, min_width: int = 400) -> np.ndarray:
    """Upscale com LANCZOS4 se a imagem for muito pequena."""
    h, w = image.shape[:2]
    if w < min_width:
        scale = min_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return image


def _denoise(image: np.ndarray) -> np.ndarray:
    """Filtro bilateral: reduz ruído preservando bordas dos caracteres."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def _enhance_plate_image(image_cv: np.ndarray) -> np.ndarray:
    """Melhoria moderada para não amplificar ruído em imagens de baixa resolução."""
    pil_img = Image.fromarray(image_cv)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.8)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(1.1)
    return np.array(pil_img)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE com clipLimit reduzido para evitar amplificação de ruído."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        enhanced_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    return clahe.apply(image)


def _add_padding(image: np.ndarray, pad: int = 15) -> np.ndarray:
    """Adiciona borda branca ao redor — ajuda o OCR a não cortar caracteres."""
    color = 255 if len(image.shape) == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ocr_attempt(image: np.ndarray) -> tuple[str | None, float]:
    """Executa OCR e concatena todos os blocos da esquerda para a direita."""
    try:
        results = _get_reader().readtext(
            image,
            allowlist=_ALLOWED_CHARS,
            paragraph=False,
            contrast_ths=0.3,
            text_threshold=0.5,
        )
        if not results:
            return None, 0.0

        results = sorted(results, key=lambda r: r[0][0][0])
        texts = [r[1].strip().upper() for r in results if r[1].strip()]
        confidences = [r[2] for r in results]

        combined = "".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return (combined or None), avg_conf
    except Exception:
        return None, 0.0


def _run_ocr_multi_strategy(image: np.ndarray) -> tuple[str | None, float]:
    """Testa 7 estratégias de pré-processamento e retorna a melhor leitura."""
    best_text: str | None = None
    best_conf = 0.0

    def try_update(img: np.ndarray) -> None:
        nonlocal best_text, best_conf
        text, conf = _ocr_attempt(img)
        if text and conf > best_conf:
            best_text = text
            best_conf = conf

    # 1. Imagem direta (já com CLAHE + padding)
    try_update(image)

    # 2. Threshold adaptativo
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            gray = _to_gray(image)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            try_update(binary)
        except Exception:
            pass

    # 3. Threshold OTSU
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            gray = _to_gray(image)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary)
        except Exception:
            pass

    # 4. Invertida
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            try_update(cv2.bitwise_not(image))
        except Exception:
            pass

    # 5. Contraste alto + OTSU
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            gray = _to_gray(image)
            contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary)
        except Exception:
            pass

    # 6. Morphological closing (fecha gaps em caracteres quebrados)
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            gray = _to_gray(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary)
        except Exception:
            pass

    # 7. Gaussian blur leve + OTSU (suaviza ruído de pixelização)
    if best_conf < _CONFIDENCE_EARLY_STOP:
        try:
            gray = _to_gray(image)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary)
        except Exception:
            pass

    return best_text, best_conf
