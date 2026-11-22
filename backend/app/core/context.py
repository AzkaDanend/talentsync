"""Request-scoped context (request ID) yang dibagikan ke logging dan error envelope."""

from contextvars import ContextVar

from starlette.requests import Request

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def resolve_request_id(request: Request) -> str:
    """Request ID untuk envelope error — valid BAHKAN saat exception tak terduga.

    Exception tak terduga diproses ServerErrorMiddleware (paling luar), SETELAH
    middleware request ID melepas contextvar — jadi contextvar saja tidak cukup.
    Middleware menyalin ID ke `scope["state"]` yang ikut scope sampai ke handler;
    fallback ke contextvar untuk pemanggilan tanpa Request (mis. background).
    """
    request_id = getattr(request.state, "request_id", None)
    return request_id if request_id else request_id_ctx.get()
