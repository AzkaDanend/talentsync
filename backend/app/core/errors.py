"""Error envelope konsisten untuk seluruh API.

Bentuk respons error (sesuai blueprint §4.5):

    { "error": { "code": "...", "message": "...", "details": [...], "request_id": "..." } }

Pesan ke pengguna berbahasa Indonesia; detail teknis internal (termasuk
exception asli) hanya masuk log, tidak pernah ke respons klien.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.context import request_id_ctx, resolve_request_id
from app.core.exceptions import DomainError

logger = logging.getLogger(__name__)

_STATUS_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "FILE_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "DEPENDENCY_UNAVAILABLE",
}


def error_response(
    status_code: int,
    message: str,
    details: list | None = None,
    code: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    rid = request_id if request_id else request_id_ctx.get()
    body = {
        "error": {
            "code": code or _STATUS_CODES.get(status_code, f"HTTP_{status_code}"),
            "message": message,
            "details": details or [],
            "request_id": rid,
        }
    }
    return JSONResponse(status_code=status_code, content=body, headers={"X-Request-ID": rid})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return error_response(
            exc.status_code,
            exc.message,
            exc.details,
            code=exc.code,
            request_id=resolve_request_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(
            exc.status_code, str(exc.detail), request_id=resolve_request_id(request)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": [str(part) for part in err.get("loc", [])],
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        return error_response(
            422, "Payload tidak valid.", details, request_id=resolve_request_id(request)
        )

    @app.exception_handler(DBAPIError)
    async def dbapi_error_handler(request: Request, exc: DBAPIError) -> JSONResponse:
        # Kegagalan DB (koneksi terputus, pooler tidak terjangkau, dsb.) → 503
        # yang bisa di-retry klien, BUKAN 500 + traceback panjang. Transaksi yang
        # gagal di tengah TIDAK pernah di-retry di sini (tak idempoten).
        # Log ringkas: tipe exception + tipe driver. SQL, parameter, dan
        # kredensial DSN tidak pernah masuk log/respons.
        orig = getattr(exc, "orig", None)
        driver = type(orig).__name__ if orig else "unknown"
        logger.error(
            "Kegagalan database: %s (%s) — respons 503 DEPENDENCY_UNAVAILABLE.",
            type(exc).__name__,
            driver,
        )
        logger.debug("Detail kegagalan database:", exc_info=exc)
        return error_response(
            503,
            "Layanan database sedang tidak tersedia. Coba lagi nanti.",
            request_id=resolve_request_id(request),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Satu-satunya tempat exception tak terduga dicatat dengan traceback —
        # handler lain mencegah exception ini agar traceback tidak dobel.
        logger.exception("Unhandled exception: %s", type(exc).__name__)
        return error_response(
            500, "Terjadi kesalahan internal.", request_id=resolve_request_id(request)
        )
