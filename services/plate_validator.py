from __future__ import annotations

import re


# Mapeamento de caracteres frequentemente confundidos pelo OCR em posições numéricas
_OCR_DIGIT_CORRECTIONS: dict[str, list[str]] = {
    "O": ["0"],
    "I": ["1"],
    "L": ["4", "1"],
    "S": ["5"],
    "Z": ["2"],
    "B": ["8"],
    "G": ["9"],
}

_PATTERN_OLD = re.compile(r"^[A-Z]{3}[0-9]{4}$")
_PATTERN_MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")

# Posições 3-6 devem ser dígitos em ambos os formatos brasileiros
_DIGIT_POSITIONS = (3, 4, 5, 6)


def validate_plate(text: str | None) -> str | None:
    """
    Valida e corrige uma string como placa brasileira.

    Formatos aceitos:
        - Antigo:   AAA1234
        - Mercosul: AAA1A23

    Aplica correções de OCR nas posições numéricas antes de validar.
    Retorna a placa corrigida ou None se inválida.
    """
    if not text:
        return None

    cleaned = text.upper().replace(" ", "").replace("-", "").strip()
    if len(cleaned) != 7:
        return None

    if _PATTERN_OLD.match(cleaned) or _PATTERN_MERCOSUL.match(cleaned):
        return cleaned

    chars = list(cleaned)
    candidates: list[list[str]] = [chars]

    for pos in _DIGIT_POSITIONS:
        next_candidates: list[list[str]] = []
        for cand in candidates:
            current = cand[pos]
            if current.isdigit():
                next_candidates.append(cand)
                continue

            options = _OCR_DIGIT_CORRECTIONS.get(current)
            if not options:
                next_candidates.append(cand)
                continue

            for replacement in options:
                new_cand = cand.copy()
                new_cand[pos] = replacement
                next_candidates.append(new_cand)

        candidates = next_candidates

    for candidate in candidates:
        corrected = "".join(candidate)
        if _PATTERN_OLD.match(corrected) or _PATTERN_MERCOSUL.match(corrected):
            return corrected

    return None
