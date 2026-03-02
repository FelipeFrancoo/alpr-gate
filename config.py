from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseConfig:
    enabled: bool
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(frozen=True)
class YoloConfig:
    vehicle_model_path: str
    plate_model_path: str
    confidence: float


@dataclass(frozen=True)
class DetectionConfig:
    min_chars_for_match: int
    validation_rounds: int
    occurrences_to_be_valid: int
    skip_before_y_max: float
    validate_brazilian_plate: bool
    try_plate_crop: bool
    rtsp_reconnect_seconds: int


@dataclass(frozen=True)
class StorageConfig:
    save_enabled: bool
    results_path: str


@dataclass(frozen=True)
class AppConfig:
    debug: bool
    ws_port: int
    video_source: str
    roi_points: str
    send_duplicates: bool
    database: DatabaseConfig
    yolo: YoloConfig
    detection: DetectionConfig
    storage: StorageConfig

    def log_summary(self) -> None:
        print(f"{'─' * 50}")
        print(f"  Debug:         {self.debug}")
        print(f"  WebSocket:     porta {self.ws_port}")
        print(f"  Captura:       {self.video_source}")
        print(f"  YOLO veículos: {self.yolo.vehicle_model_path}")
        print(f"  YOLO placas:   {self.yolo.plate_model_path}")
        print(f"  Confiança:     {self.yolo.confidence}")
        print(f"  ROI:           {self.roi_points or 'desativada'}")
        print(f"  Validação BR:  {self.detection.validate_brazilian_plate}")
        print(f"  DB:            {'ativado' if self.database.enabled else 'desativado'}")
        print(f"  Salvar imgs:   {'sim' if self.storage.save_enabled else 'não'}")
        print(f"{'─' * 50}")


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def load_config(env_path: str | None = None) -> AppConfig:
    if env_path and os.path.isfile(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    # Validar caminhos obrigatórios
    vehicle_path = _env_str("PURE_YOLO_MODEL_PATH")
    plate_path = _env_str("LICENSE_PLATE_YOLO_MODEL_PATH")
    if not vehicle_path:
        raise ValueError("PURE_YOLO_MODEL_PATH não definido no .env")
    if not plate_path:
        raise ValueError("LICENSE_PLATE_YOLO_MODEL_PATH não definido no .env")

    return AppConfig(
        debug=_env_bool("DEBUG"),
        ws_port=_env_int("WS_PORT", 8765),
        video_source=_env_str("RTSP_CAPTURE_CONFIG", "./test.mp4"),
        roi_points=_env_str("ROI_POINTS", ""),
        send_duplicates=_env_bool("SHOULD_SEND_SAME_RESULTS"),
        database=DatabaseConfig(
            enabled=_env_bool("DB_ENABLED"),
            host=_env_str("DB_SERVER", "localhost"),
            port=_env_int("DB_PORT", 5432),
            name=_env_str("DB_NAME", "lpdb"),
            user=_env_str("DB_USER", "postgres"),
            password=_env_str("DB_PASSWORD", ""),
        ),
        yolo=YoloConfig(
            vehicle_model_path=vehicle_path,
            plate_model_path=plate_path,
            confidence=_env_float("YOLO_CONFIDENCE", 0.45),
        ),
        detection=DetectionConfig(
            min_chars_for_match=_env_int("MINIMUM_NUMBER_OF_CHARS_FOR_MATCH", 3),
            validation_rounds=_env_int("NUMBER_OF_VALIDATION_ROUNDS", 5),
            occurrences_to_be_valid=_env_int("NUMBER_OF_OCCURRENCES_TO_BE_VALID", 2),
            skip_before_y_max=_env_float("SKIP_BEFORE_Y_MAX", 0),
            validate_brazilian_plate=_env_bool("VALIDATE_BRAZILIAN_PLATE", True),
            try_plate_crop=_env_bool("SHOULD_TRY_LP_CROP"),
            rtsp_reconnect_seconds=_env_int("RTSP_RECONNECT_SECONDS", 5),
        ),
        storage=StorageConfig(
            save_enabled=_env_bool("SAVE_RESULTS_ENABLED"),
            results_path=_env_str("RESULTS_PATH", "./results"),
        ),
    )
