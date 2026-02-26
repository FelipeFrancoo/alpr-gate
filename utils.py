from __future__ import annotations

import io
import os
import shutil
import time
import cv2
import numpy as np
import psycopg2
import easyocr
import re
from typing import Optional, Tuple
from PIL import Image, ImageEnhance
from datetime import datetime, timedelta
from ultralytics import YOLO

# Inicializar apenas uma vez (fora do loop principal)
reader = easyocr.Reader(['en'], gpu=False)

# Retorna o número de resultados + resultados como caixas (boxes)
def detect_with_yolo(preloaded_model: YOLO, car_image: Image, verbose: bool, confidence: float = 0.45) -> tuple[int, any]:
    result = preloaded_model.predict(car_image, verbose=verbose, conf=confidence)[0]
    return (len(result.boxes), result.boxes)

def normalize_label(label):
    return label.strip().lower()

def warp_plate(image, x1, y1, x2, y2, output_width=400, output_height=130):
    """
    Normaliza a perspectiva da placa e redimensiona para melhor OCR.
    Aumentado de 300x100 para 400x130 para melhor resolução.
    """
    plate = image[y1:y2, x1:x2]
    
    if plate.size == 0:
        return plate

    pts1 = np.float32([
        [0, 0],
        [plate.shape[1], 0],
        [0, plate.shape[0]],
        [plate.shape[1], plate.shape[0]]
    ])

    pts2 = np.float32([
        [0, 0],
        [output_width, 0],
        [0, output_height],
        [output_width, output_height]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(plate, matrix, (output_width, output_height), flags=cv2.INTER_CUBIC)

    return warped

def apply_clahe(image):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para 
    melhorar contraste, especialmente em condições de luz ruins.
    Otimizado para placas automotivas.
    """
    if len(image.shape) == 3:
        # Aplicar em canal V do HSV para melhor resultado
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        v_clahe = clahe.apply(v)
        
        hsv_clahe = cv2.merge([h, s, v_clahe])
        enhanced = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        # Converter para grayscale no final
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(image)
    
    return enhanced


def enhance_plate_image(image_cv: np.ndarray) -> np.ndarray:
    """
    Pipeline de pré-processamento avançado para melhorar a leitura OCR.
    Aplica múltiplas técnicas para lidar com diferentes condições.
    Otimizado para placas de ônibus brasileiras.
    """
    # Converter para PIL para ajustes de cor
    pil_img = Image.fromarray(image_cv)
    
    # Aumentar contraste (mais agressivo para placas escuras)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2.2)
    
    # Aumentar nitidez (mais agressivo)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.5)
    
    # Aumentar brilho levemente para imagens muito escuras
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.15)
    
    # Aumentar saturação para melhorar caracteres coloridos
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    # Converter de volta para numpy
    enhanced = np.array(pil_img)
    return enhanced

def _ocr_attempt(image, label: str) -> tuple[str | None, float]:
    """
    Executa um único OCR numa imagem e retorna (texto_concatenado, confiança_media).
    Concatena TODOS os blocos detectados ordenados da esquerda para a direita.
    """
    try:
        results = reader.readtext(image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not results:
            return None, 0.0
        # Ordenar da esquerda para direita pelo x do bounding box
        results = sorted(results, key=lambda x: x[0][0][0])
        # Concatenar todos os textos detectados
        texts = [r[1].strip().upper() for r in results if r[1].strip()]
        confidences = [r[2] for r in results]
        combined_text = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return combined_text if combined_text else None, avg_confidence
    except Exception:
        return None, 0.0


def run_ocr(image) -> tuple[str | None, float]:
    """
    Executa OCR com múltiplas estratégias de pré-processamento.
    Concatena todos os blocos detectados (esquerda → direita) para
    montar a placa completa a partir de fragmentos.
    """
    best_text = None
    best_confidence = 0.0

    def try_update(img, label):
        nonlocal best_text, best_confidence
        text, conf = _ocr_attempt(img, label)
        if text and conf > best_confidence:
            best_text = text
            best_confidence = conf
        # Também retorna para logging
        return text, conf

    # Tentativa 1: imagem como recebida
    try_update(image, "direct")

    # Tentativa 2: threshold adaptativo
    if best_confidence < 0.85:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            try_update(binary, "adaptive_threshold")
        except Exception:
            pass

    # Tentativa 3: threshold OTSU
    if best_confidence < 0.85:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary, "otsu")
        except Exception:
            pass

    # Tentativa 4: invertida (placas escuras com caracteres claros)
    if best_confidence < 0.85:
        try:
            inverted = cv2.bitwise_not(image)
            try_update(inverted, "inverted")
        except Exception:
            pass

    # Tentativa 5: contraste aumentado + OTSU
    if best_confidence < 0.85:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            try_update(binary, "high_contrast_otsu")
        except Exception:
            pass

    return best_text, best_confidence

def validate_plate(text):
    """
    Valida se a string é uma placa brasileira válida.
    Aceita formatos:
      - Antigo: AAA1234, AAA-1234, AAA 1234
      - Mercosul: AAA1A23
    
    Também tenta corrigir erros comuns de OCR:
      - O → 0 (letra O confundida com zero)
      - I → 1 (letra I confundida com um)
      - L → 1 (letra L confundida com um em alguns fonts)
      - S → 5 (letra S confundida com cinco)
      - Z → 2 (letra Z confundida com dois)
    """
    if not text:
        return None
    
    # Limpar: remover espaços e traços
    cleaned = text.upper().replace(" ", "").replace("-", "").strip()
    
    if len(cleaned) < 7 or len(cleaned) > 7:
        return None
    
    # Correções de OCR comuns (letter-to-digit confusions)
    # Aplicar apenas em posições onde esperamos números
    cleaned_list = list(cleaned)
    
    # Posições 3-6 são dígitos em ambos os formatos
    for pos in [3, 4, 5, 6]:
        if pos < len(cleaned_list):
            c = cleaned_list[pos]
            # Número confundido com letra
            if c == 'O':
                cleaned_list[pos] = '0'
            elif c == 'I' or c == 'L':
                cleaned_list[pos] = '1'
            elif c == 'S':
                cleaned_list[pos] = '5'
            elif c == 'Z':
                cleaned_list[pos] = '2'
            elif c == 'B':
                cleaned_list[pos] = '8'
            elif c == 'G':
                cleaned_list[pos] = '9'
    
    cleaned = ''.join(cleaned_list)
    
    # Validação de padrão
    pattern_old = r'^[A-Z]{3}[0-9]{4}$'
    pattern_mercosul = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'
    
    if re.match(pattern_old, cleaned) or re.match(pattern_mercosul, cleaned):
        return cleaned
    
    return None

def gen_intermediate_file_name(filename: str, file_type: str, unique_identifier: str):
    return f"./intermediate_detection_files/{filename}_{unique_identifier}.{file_type}"

# Esta função deve ser chamada logo após obter as caixas das placas
def prepare_env_for_reading_license_plates(should_save_intermediate_files: bool):
    try:
        if should_save_intermediate_files:
            if os.path.exists("./intermediate_detection_files/"):
                shutil.rmtree("./intermediate_detection_files/")
            os.mkdir("./intermediate_detection_files/")
    except: pass

# Ler caixa de uma única placa
# Retorna a placa como string
def read_license_plate(unique_identifier: str, box: any, original_image: Image, width_boost: int, additional_white_spacing_each_side: int, debug: bool, should_try_lp_crop: bool, minimum_number_of_chars_for_match: int) -> tuple[Image.Image, str]:
    """
    Lê uma placa individual com múltiplas estratégias de pré-processamento.
    Retorna (imagem_processada, texto_da_placa).
    """
    # Cortar imagem
    x_min, y_min, x_max, y_max = box.xyxy.cpu().detach().numpy()[0]
    x1, y1, x2, y2 = int(x_min), int(y_min), int(x_max), int(y_max)
    
    original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Warp plate para normalizar perspectiva (aumentado de 300x100 para 400x130)
    warped = warp_plate(original_image_cv, x1, y1, x2, y2, output_width=400, output_height=130)
    
    if debug:
        cv2.imwrite(gen_intermediate_file_name("warped_plate", "jpg", unique_identifier), warped)
    
    # Pipeline de melhoria de imagem
    enhanced_color = enhance_plate_image(warped)
    enhanced = apply_clahe(enhanced_color)
    
    if debug:
        cv2.imwrite(gen_intermediate_file_name("enhanced_plate", "jpg", unique_identifier), enhanced)
    
    # OCR com múltiplas tentativas
    text, confidence = run_ocr(enhanced)
    
    if debug:
        status = "✓" if text and confidence > 0.3 else "✗"
        print(f"  [{status}] OCR resultado: '{text}' (confiança: {confidence:.2f})")
    
    # Normalizar texto: limpar espaços e traços
    raw_text = ""
    if text:
        raw_text = text.strip().upper().replace(" ", "").replace("-", "")
    
    if debug and raw_text:
        validated = validate_plate(raw_text)
        print(f"  [info] Texto bruto: '{raw_text}' (len={len(raw_text)}) | validate_plate: '{validated}'")
    
    enhanced_pil = Image.fromarray(enhanced)
    
    # Retorna o texto BRUTO do OCR — a validação será feita no sistema de votação
    return (enhanced_pil, raw_text)

# https://stackoverflow.com/a/55117662/16638833
def img_to_bytes(image: Image, format="JPEG"):
    bytes_io = io.BytesIO()
    image.save(bytes_io, format=format)
    return bytes_io.getvalue()

def save_validated_result(db_enabled: bool, car_id: str, license_plate: str, db_server: str, db_port: str, db_name: str, db_user: str, db_pass: str, save_results_enabled: bool, results_path: str, car_image_raw: Image, license_plate_image_raw: Image):
    if db_enabled:
        try:
            conn = psycopg2.connect(host=db_server, port=db_port, database=db_name, user=db_user, password=db_pass)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO main_gate_alpr_license_plates (id, license_plate, captured_at) VALUES (%s, %s, %s)", (car_id, license_plate, datetime.now()))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB] Erro ao salvar no banco: {e}")

    if save_results_enabled:
        try:
            car_image_raw.save(os.path.join(results_path, f"{car_id}_car.jpg"), "JPEG")
            license_plate_image_raw.save(os.path.join(results_path, f"{car_id}_lp.jpg"), "JPEG")
        except Exception as e:
            print(f"[SAVE] Erro ao salvar imagens: {e}")

def cleanup_old_results(db_enabled: bool, db_server: str, db_port: str, db_name: str, db_user: str, db_pass: str, save_results_enabled: bool, results_path: str, days_to_keep: int = 3):
    current_time = time.time()
    seconds_to_keep = days_to_keep * 24 * 60 * 60

    if save_results_enabled and os.path.exists(results_path):
        for filename in os.listdir(results_path):
            file_path = os.path.join(results_path, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > seconds_to_keep:
                    try:
                        os.remove(file_path)
                        print(f"Removed old result file: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")

    if db_enabled:
        try:
            conn = psycopg2.connect(host=db_server, port=db_port, database=db_name, user=db_user, password=db_pass)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM main_gate_alpr_license_plates WHERE captured_at < %s", (datetime.now() - timedelta(days=days_to_keep),))
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            if deleted_rows > 0:
                print(f"Deleted {deleted_rows} old records from database")
        except Exception as e:
            print(f"Error cleaning up database: {e}")