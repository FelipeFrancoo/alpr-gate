"""
Ferramenta interativa para definir a ROI da câmera.

Uso:
    python calibrate_roi.py

Instruções:
    - Clique em 4 pontos para definir o polígono da ROI
    - Pressione 'r' para resetar os pontos
    - Pressione 'q' para sair e imprimir a configuração
    - Pressione 's' para salvar uma captura com a ROI desenhada
"""

import os
import sys
import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from roi import ROI

load_dotenv()
RTSP_CAPTURE_CONFIG = os.getenv("RTSP_CAPTURE_CONFIG")

points = []
current_frame = None


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"  Ponto {len(points)}: ({x}, {y})")
        else:
            print("  Já tem 4 pontos. Pressione 'r' para resetar.")


def draw_roi_preview(frame):
    output = frame.copy()

    # Desenhar pontos existentes
    for i, (px, py) in enumerate(points):
        cv2.circle(output, (px, py), 8, (255, 0, 255), -1)
        cv2.putText(output, f"P{i} ({px},{py})", (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Se tem >= 2 pontos, conectar com linhas
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(output, points[i], points[i + 1], (255, 0, 255), 2)

    # Se tem 4 pontos, fechar o polígono e preencher
    if len(points) == 4:
        polygon = np.array(points, dtype=np.int32)
        cv2.line(output, points[3], points[0], (255, 0, 255), 2)

        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (255, 0, 255))
        cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

    # Instruções
    instructions = [
        "Clique para marcar 4 pontos da ROI",
        "'r' = resetar | 'q' = sair | 's' = salvar",
        f"Pontos: {len(points)}/4",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(output, text, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output


def main():
    global points, current_frame

    print(f"Conectando à câmera: {RTSP_CAPTURE_CONFIG}")
    capture = cv2.VideoCapture(RTSP_CAPTURE_CONFIG)

    if not capture.isOpened():
        print("Erro: não foi possível abrir a captura de vídeo.")
        sys.exit(1)

    cv2.namedWindow("Calibrar ROI")
    cv2.setMouseCallback("Calibrar ROI", mouse_callback)

    print("\n=== Ferramenta de Calibração da ROI ===")
    print("Clique em 4 pontos para definir a área de interesse.")
    print("Pressione 'r' para resetar, 'q' para sair.\n")

    while True:
        ret, frame = capture.read()
        if not ret:
            # Para arquivos de vídeo, voltar ao início
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_frame = frame.copy()
        display = draw_roi_preview(frame)
        cv2.imshow("Calibrar ROI", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            print("  Pontos resetados.")
        elif key == ord('s') and len(points) == 4:
            roi = ROI(points=points)
            output = roi.draw_on_frame(current_frame)
            cv2.imwrite("roi_calibration.jpg", output)
            print("  Salvo: roi_calibration.jpg")

    capture.release()
    cv2.destroyAllWindows()

    if len(points) == 4:
        roi_string = ";".join(f"{x},{y}" for x, y in points)
        print(f"\n{'='*50}")
        print(f"  Adicione no seu .env:")
        print(f"  ROI_POINTS={roi_string}")
        print(f"{'='*50}")
    else:
        print("\nROI não definida (menos de 4 pontos).")


if __name__ == "__main__":
    main()
