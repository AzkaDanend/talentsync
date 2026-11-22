"""Application factory FastAPI — TalentSync AI."""

import logging
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers, MutableHeaders

from app.api.v1 import api_router
from app.core.config import APP_VERSION, Settings, get_settings
from app.core.context import request_id_ctx
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.db.session import build_engine

logger = logging.getLogger(__name__)

# Jalur yang tidak dicatat access log (noise tinggi, nilai debug rendah;
# probe k8s/uptime bisa memanggil tiap detik).
_ACCESS_LOG_SKIPPED = ("/api/v1/health", "/api/v1/health/ready")

# X-Request-ID dari client dibatasi panjangnya (masuk ke log & envelope).
_REQUEST_ID_MAX_LEN = 64


def _sanitize_request_id(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    return cleaned[:_REQUEST_ID_MAX_LEN] or None


class RequestIdMiddleware:
    """Pure-ASGI middleware: request ID + access log + header X-Request-ID.

    Menggantikan @app.middleware("http") (BaseHTTPMiddleware). Alasan WAJIB
    pure-ASGI (incident 21 Nov 2026):
    1. Exception tak terduga diproses ServerErrorMiddleware di paling luar —
       BaseHTTPMiddleware melepas contextvar di `finally` SEBELUM handler error
       berjalan, sehingga envelope 500 & log-nya kehilangan request ID (" - ").
       Dengan scope["state"], ID ikut scope sampai ke handler error.
    2. BaseHTTPMiddleware menjalankan app di task terpisah — request ID lebih
       rapuh dan overhead streaming tak perlu.
    """

    def __init__(self, app) -> None:  # noqa: ANN001 — kontrak ASGI
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = _sanitize_request_id(Headers(scope=scope).get("X-Request-ID")) or uuid.uuid4().hex
        # state ikut scope sampai ke handler error paling luar (resolve_request_id).
        scope.setdefault("state", {})["request_id"] = rid
        token = request_id_ctx.set(rid)
        started = time.perf_counter()
        status_code: int | None = None

        async def send_with_request_id(message) -> None:  # noqa: ANN001
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # SET (bukan append): handler error (error_response) sudah bisa
                # memasang X-Request-ID — append akan menduplikasi header.
                MutableHeaders(scope=message)["X-Request-ID"] = rid
            await send(message)

        propagating = False
        try:
            await self.app(scope, receive, send_with_request_id)
        except BaseException:
            # Exception menembus ke ServerErrorMiddleware (paling luar) yang
            # menjalankan handler error — contextvar TIDAK direset agar handler
            # & log-nya tetap membawa request ID. Task request berakhir tepat
            # setelah ini; context salinan task dibuang otomatis (tanpa leak).
            propagating = True
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            path = scope.get("path", "")
            if path not in _ACCESS_LOG_SKIPPED:
                logger.info(
                    "%s %s -> %s (%.0f ms)",
                    scope.get("method", "-"),
                    path,
                    status_code if status_code is not None else 500,
                    duration_ms,
                )
            if not propagating:
                request_id_ctx.reset(token)


def _warm_startup_caches(settings: Settings) -> None:
    """Siapkan model embedding + SELURUH cache rekomendasi sebelum port dibuka.

    Mode sinkron (default, embedding_prewarm_sync=True): startup menunggu
    hingga model & cache siap — request PERTAMA pengguna tidak membayar
    cold-start. Tes manual 22 Nov 2026: request pertama pasca-restart =
    model (±30 dtk) + encode judul (±23 dtk) + fuzzy (±17 dtk) + transfer DB
    (±12-46 dtk) ≫ axios timeout 20 dtk di UI → "timeout of 20000ms exceeded"
    berulang. Dengan prime sinkron, request pertama ≈ 1-2 dtk.

    Mode latar (embedding_prewarm_sync=False): thread daemon — request awal
    tetap membayar sebagian cold-start sampai warm-up selesai.

    Gagal warm-up tidak fatal: jalur on-demand tetap berjalan (cache diisi
    saat pertama dibutuhkan, perilaku lama)."""

    def _load() -> None:
        started = time.perf_counter()
        try:
            from app.ai.matching.embedder import get_embedder

            embedder = get_embedder(settings.embedding_model)
            # Warm-up INFERENSI pertama: torch lazy-init terjadi di sini,
            # bukan pada request pengguna (muat model saja belum cukup — Wave 3).
            embedder.encode(["pre-warm"])
            logger.info(
                "Model embedding siap (%.1f dtk): %s",
                time.perf_counter() - started,
                settings.embedding_model,
            )
        except Exception as exc:  # pragma: no cover — lingkungan tanpa torch dsb.
            logger.warning("Pre-warm model embedding dilewati: %s", exc)
            return

        try:
            from app.db.session import SessionLocal
            from app.services.recommendation_service import prime_caches

            with SessionLocal() as db:
                stats = prime_caches(db, settings)
            logger.info(
                "Cache rekomendasi siap (%.1f dtk): %s",
                time.perf_counter() - started,
                stats,
            )
        except Exception as exc:
            logger.warning("Prime cache rekomendasi gagal — akan diisi on-demand: %s", exc)

    if settings.embedding_prewarm_sync:
        _load()
        return
    threading.Thread(target=_load, name="embedder-warmup", daemon=True).start()


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging(s.log_level)

    app = FastAPI(title=s.app_name, version=APP_VERSION)
    # Settings & engine terikat ke instance aplikasi (memudahkan pengujian terisolasi).
    app.state.settings = s
    app.state.engine = build_engine(s.database_url)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestIdMiddleware)

    register_error_handlers(app)

    if s.embeddings_enabled:
        _warm_startup_caches(s)

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {"app": s.app_name, "version": APP_VERSION, "docs": "/docs"}

    return app


app = create_app()
