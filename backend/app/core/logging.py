"""Structured JSON logging.

Seluruh log aplikasi keluar sebagai satu baris JSON (mudah diparsing), dengan
request ID dari context. Kredensial/PII tidak pernah dimasukkan ke log
(aturan ini ditegakkan secara disiplin oleh pemanggil, bukan oleh formatter).
"""

import json
import logging
import sys
from datetime import UTC, datetime

from app.core.context import request_id_ctx


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Uvicorn punya handler sendiri; matikan agar output tidak dobel
    # dan mengalir lewat formatter JSON yang sama. Access line uvicorn
    # TIDAK dipakai (tanpa request ID & durasi) — RequestIdMiddleware
    # (main.py) memancarkan access line yang lebih kaya.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger_uvicorn = logging.getLogger(name)
        logger_uvicorn.handlers = []
        logger_uvicorn.propagate = name != "uvicorn.access"

    # HTTP client libraries mencatat setiap request eksternal di INFO —
    # duplikat dengan log panggilan Groq yang sudah berdurasi; bungkam.
    for name in ("httpx", "httpcore", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
