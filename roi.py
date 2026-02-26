"""
ROI (Região de Interesse) para o sistema ALPR.

Define um polígono de interesse onde a detecção de veículos será processada.
Tudo fora do polígono é ignorado, melhorando performance e reduzindo falsos positivos.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, List, Tuple


class ROI:
    """
    Região de Interesse definida como um polígono de N pontos.
    Os pontos devem ser fornecidos em ordem (sentido horário ou anti-horário).
    """

    def __init__(self, points: list[tuple[int, int]] = None):
        """
        Args:
            points: Lista de tuplas (x, y) definindo o polígono da ROI.
                    Se None, a ROI cobre a imagem inteira (desativada).
        """
        self.points = points
        self._polygon = np.array(points, dtype=np.int32) if points else None

    @property
    def enabled(self) -> bool:
        """Retorna True se a ROI está ativa (tem pelo menos 3 pontos)."""
        return self._polygon is not None and len(self._polygon) >= 3

    def contains_point(self, x: float, y: float) -> bool:
        """Verifica se um ponto (x, y) está dentro do polígono ROI."""
        if not self.enabled:
            return True
        result = cv2.pointPolygonTest(self._polygon, (float(x), float(y)), False)
        return result >= 0

    def contains_box(self, x_min: float, y_min: float, x_max: float, y_max: float, threshold: float = 0.3) -> bool:
        """
        Verifica se uma bounding box tem sobreposição suficiente com a ROI.

        Args:
            x_min, y_min, x_max, y_max: Coordenadas da bounding box.
            threshold: Fração mínima dos pontos de teste que devem estar dentro da ROI (0.0 a 1.0).
                       0.3 = pelo menos 30% dos pontos de teste devem estar dentro da ROI.
        """
        if not self.enabled:
            return True

        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 0 or box_height <= 0:
            return False

        # Verificar os 4 cantos + centro + pontos médios das bordas
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        test_points = [
            (x_min, y_min),   # top-left
            (x_max, y_min),   # top-right
            (x_min, y_max),   # bottom-left
            (x_max, y_max),   # bottom-right
            (cx, cy),         # center
            (cx, y_min),      # mid-top
            (cx, y_max),      # mid-bottom
            (x_min, cy),      # mid-left
            (x_max, cy),      # mid-right
        ]

        points_inside = sum(1 for px, py in test_points if self.contains_point(px, py))
        return (points_inside / len(test_points)) >= threshold

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica a máscara ROI no frame, escurecendo tudo fora da ROI.
        Retorna o frame mascarado (útil para visualização/debug).
        """
        if not self.enabled:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self._polygon], 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked

    def draw_on_frame(self, frame: np.ndarray, color: tuple = (255, 0, 255), thickness: int = 3, alpha: float = 0.15) -> np.ndarray:
        """
        Desenha o polígono ROI no frame com preenchimento semi-transparente.

        Args:
            frame: Frame BGR do OpenCV.
            color: Cor BGR (padrão: magenta/roxo).
            thickness: Espessura da borda.
            alpha: Transparência do preenchimento (0.0 = invisível, 1.0 = opaco).
        
        Returns:
            Frame com a ROI desenhada.
        """
        if not self.enabled:
            return frame

        output = frame.copy()

        # Preenchimento semi-transparente
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._polygon], color)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Borda sólida
        cv2.polylines(output, [self._polygon], isClosed=True, color=color, thickness=thickness)

        # Labels nos cantos
        for i, (px, py) in enumerate(self.points):
            cv2.circle(output, (px, py), 6, color, -1)
            cv2.putText(output, f"P{i} ({px},{py})", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output

    def get_bounding_rect(self) -> tuple[int, int, int, int]:
        """Retorna o retângulo delimitador (x, y, w, h) do polígono."""
        if not self.enabled:
            return (0, 0, 0, 0)
        return cv2.boundingRect(self._polygon)

    def to_dict(self) -> dict:
        """Serializa a ROI para dicionário (para salvar em JSON/config)."""
        return {
            "points": self.points,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ROI":
        """Deserializa a ROI de um dicionário."""
        return cls(points=data.get("points"))

    @classmethod
    def from_env(cls, env_value: str) -> "ROI":
        """
        Cria ROI a partir de uma string de variável de ambiente.
        Formato: "x1,y1;x2,y2;x3,y3;x4,y4"
        Exemplo: "580,230;920,150;980,520;540,600"
        
        Se a string estiver vazia ou for inválida, retorna ROI desativada.
        """
        if not env_value or env_value.strip() == "":
            return cls(points=None)

        try:
            points = []
            for pair in env_value.strip().split(";"):
                x, y = pair.strip().split(",")
                points.append((int(x.strip()), int(y.strip())))

            if len(points) < 3:
                print(f"[ROI] Precisa de pelo menos 3 pontos, recebeu {len(points)}. ROI desativada.")
                return cls(points=None)

            return cls(points=points)
        except Exception as e:
            print(f"[ROI] Erro ao parsear ROI_POINTS: {e}. ROI desativada.")
            return cls(points=None)
