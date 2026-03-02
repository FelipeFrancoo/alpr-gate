from __future__ import annotations

import logging
from difflib import SequenceMatcher

from domain.models import PlateReading, DetectionResult
from services.plate_validator import validate_plate

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.55


def vote_best_plate(
    rounds: list[list[PlateReading]],
    min_occurrences: int,
) -> list[DetectionResult]:
    """
    Valida placas entre rounds usando votação por similaridade.

    Agrupa leituras similares, elege o grupo dominante e reconstrói
    a placa votando caractere a caractere por posição.
    """
    all_readings = _collect_readings(rounds)
    if not all_readings:
        return []

    logger.debug("[VOTAÇÃO] Total de leituras: %d", len(all_readings))
    for r in all_readings:
        logger.debug("  - '%s'", r.normalized_text)

    groups = _group_by_similarity(all_readings)

    logger.debug("[VOTAÇÃO] Grupos formados: %d", len(groups))
    for i, g in enumerate(groups):
        logger.debug("  Grupo %d: %s (%d leituras)", i, [r.normalized_text for r in g], len(g))

    dominant = _find_dominant_group(groups, min_occurrences)
    if dominant is None:
        return []

    best_plate = _reconstruct_plate([r.normalized_text for r in dominant])
    logger.debug("[VOTAÇÃO] Placa reconstruída: '%s'", best_plate)

    validated = validate_plate(best_plate)
    if validated:
        best_plate = validated
        logger.debug("[VOTAÇÃO] ✓ Placa válida: '%s'", best_plate)
    else:
        logger.debug("[VOTAÇÃO] ⚠ Fora do padrão, enviando mesmo assim: '%s'", best_plate)

    best_reading = max(dominant, key=lambda r: r.length)
    return [DetectionResult(
        car_image=best_reading.car_image,
        plate_image=best_reading.plate_image,
        plate_text=best_plate,
    )]


def _collect_readings(rounds: list[list[PlateReading]]) -> list[PlateReading]:
    return [r for rnd in rounds for r in rnd if r.normalized_text]


def _group_by_similarity(readings: list[PlateReading]) -> list[list[PlateReading]]:
    groups: list[list[PlateReading]] = []
    for reading in readings:
        text = reading.normalized_text
        placed = False
        for group in groups:
            representative = group[0].normalized_text
            ratio = SequenceMatcher(None, text, representative).ratio()
            is_substring = text in representative or representative in text
            if ratio > _SIMILARITY_THRESHOLD or is_substring:
                group.append(reading)
                placed = True
                break
        if not placed:
            groups.append([reading])
    return groups


def _find_dominant_group(
    groups: list[list[PlateReading]],
    min_occurrences: int,
) -> list[PlateReading] | None:
    groups.sort(key=len, reverse=True)
    dominant = groups[0]

    if len(dominant) < min_occurrences:
        logger.debug(
            "[VOTAÇÃO] Grupo dominante tem %d leituras (necessário %d)",
            len(dominant), min_occurrences,
        )
        return None

    logger.debug(
        "[VOTAÇÃO] Grupo dominante: %s (%d leituras)",
        [r.normalized_text for r in dominant], len(dominant),
    )
    return dominant


def _reconstruct_plate(plates: list[str]) -> str:
    """Reconstrói a placa votando caractere a caractere por posição."""
    if not plates:
        return ""

    # Filtra por tamanho predominante para evitar ruído
    lengths = [len(p) for p in plates]
    dominant_len = max(set(lengths), key=lengths.count)
    valid_plates = [p for p in plates if len(p) == dominant_len]

    reconstructed: list[str] = []

    for pos in range(dominant_len):
        chars_at_pos = [p[pos] for p in valid_plates]
        if not chars_at_pos:
            break
        # Votação simples: caractere mais frequente vence
        winner = max(set(chars_at_pos), key=chars_at_pos.count)
        reconstructed.append(winner)

    return "".join(reconstructed)
