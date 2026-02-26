from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image


@dataclass
class PlateReading:
    """Uma leitura individual de placa feita pelo OCR."""
    car_image: Image.Image
    plate_image: Image.Image
    text: str

    @property
    def normalized_text(self) -> str:
        return self.text.strip().upper().replace(" ", "").replace("-", "")

    @property
    def length(self) -> int:
        return len(self.normalized_text)


@dataclass
class DetectionResult:
    """Resultado final validado pelo sistema de votação, pronto para envio."""
    car_image: Image.Image
    plate_image: Image.Image
    plate_text: str
    uuid: str = ""

    @property
    def formatted_plate(self) -> str:
        """Formata placa: 'ABC1234' -> 'ABC 1234'."""
        cleaned = self.plate_text.strip().upper().replace(" ", "").replace("-", "")
        if len(cleaned) == 7:
            return f"{cleaned[:3]} {cleaned[3:]}"
        return cleaned

    @property
    def display_string(self) -> str:
        return f"{self.formatted_plate} => {self.uuid}"
