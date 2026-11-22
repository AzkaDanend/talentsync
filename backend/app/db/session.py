"""Engine & session factory (SQLAlchemy 2.0, driver psycopg v3).

Reliabilitas koneksi Supabase (incident 21 Nov 2026: OperationalError /
"SSL unexpected eof" / timeout pooler :6543 → HTTP 500 acak):

- `pool_pre_ping=True`  : koneksi yang mati di pool (dipanen pooler setelah
  idle) dibuang saat checkout, bukan dipakai lalu gagal.
- `pool_recycle=300`    : koneksi lebih tua dari 5 menit dipensiunkan proaktif
  — di bawah idle-timeout NAT/LB/cloud, di atas biaya reconnect (±2 dtk ke
  Supabase); tidak perlu lebih agresif karena pre_ping menutup sisanya.
- `do_connect` retry    : pembentukan koneksi baru adalah operasi idempoten,
  jadi 2 percobaan ulang (backoff 0.5/1.0 dtk) hanya untuk kegagalan jaringan
  transien (timeout, SSL EOF, server restart). Kegagalan autentikasi/schema
  (punya SQLSTATE) TIDAK di-retry.
- Menulis TIDAK pernah di-retry di level aplikasi: satu transaksi yang gagal
  di tengah bisa sudah setengah-commit — kegagalan tersebut dikembalikan ke
  klien sebagai 503 DEPENDENCY_UNAVAILABLE (core.errors) untuk di-retry user.
"""

import logging
import time
from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Kegagalan koneksi di-retry saat SQLSTATE kosong (galat jaringan mentah psycopg:
# timeout, SSL EOF, connection refused) atau SQLSTATE kelas koneksi/transien.
_RETRYABLE_SQLSTATES = frozenset({"08000", "08001", "08003", "08006", "57P03"})
_CONNECT_MAX_ATTEMPTS = 3  # 1 percobaan asli + 2 retry (bounded, bukan loop tak terbatas)
_CONNECT_BACKOFF_SECONDS = (0.5, 1.0)


def _is_transient_connect_failure(error: Exception) -> bool:
    """True bila kegagalan connect layak di-retry (jaringan transien, bukan salah config)."""
    sqlstate = getattr(error, "sqlstate", None)
    if sqlstate is None:
        return True  # SSL EOF / timeout / refused — tanpa SQLSTATE dari server
    return str(sqlstate) in _RETRYABLE_SQLSTATES


def build_engine(database_url: str) -> Engine:
    # connect_timeout hanya berlaku untuk psycopg (PostgreSQL); sqlite (test/E2E)
    # akan menolak opsi asing. 10 dtk: koneksi cloud (Supabase pooler) bisa lebih
    # lambat dari localhost — lihat ADR 0005.
    #
    # prepare_threshold=None (psycopg3): JANGAN pernah prepare statement —
    # WAJIB saat memakai Supabase transaction pooler (:6543). Semantik psycopg3:
    # 0 = prepare segera setiap statement (berbahaya di sini!), None = tidak
    # pernah. Transaction pooler membagikan server backend antar klien sehingga
    # nama prepared statement otomatis (_pg3_N) saling berbenturan →
    # "prepared statement already exists" → 500 acak. Biaya: re-planning per
    # query (tak signifikan di MVP).
    #
    # keepalives=1: TCP keepalive OS-level agar NAT/firewall tidak membuang
    # koneksi idle secara diam-diam (sisi Windows & Linux libpq mendukung).
    connect_args = (
        {"connect_timeout": 10, "prepare_threshold": None, "keepalives": 1}
        if database_url.startswith("postgresql")
        else {}
    )
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        # Readiness check dan request tidak boleh menggantung saat DB down
        # (Windows dapat mem-drop SYN tanpa RST → timeout OS sangat lama).
        connect_args=connect_args,
    )

    if database_url.startswith("postgresql"):
        event.listen(engine, "do_connect", _connect_with_retry)
    return engine


def _connect_with_retry(dialect, conn_rec, cargs, cparams):
    """Hook `do_connect`: bangun koneksi psycopg dengan retry terbatas.

    Gagal koneksi transien (pooler restart, SSL EOF, timeout) di-retry dengan
    backoff pendek — request tidak langsung 500 karena satu blip jaringan.
    Melempar ulang kegagalan terakhir bila semua percobaan habis.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        try:
            return dialect.dbapi.connect(*cargs, **cparams)
        except Exception as connect_error:  # noqa: BLE001 — dipilah di bawah
            is_dbapi_or_os = isinstance(connect_error, (OSError, dialect.dbapi.Error))
            if not is_dbapi_or_os or not _is_transient_connect_failure(connect_error):
                raise
            last_exc = connect_error
            if attempt < _CONNECT_MAX_ATTEMPTS:
                delay = _CONNECT_BACKOFF_SECONDS[attempt - 1]
                logger.warning(
                    "Koneksi DB gagal (percobaan %s/%s, %s) — retry dalam %.1fs.",
                    attempt,
                    _CONNECT_MAX_ATTEMPTS,
                    type(connect_error).__name__,
                    delay,
                )
                time.sleep(delay)
    assert last_exc is not None  # loop selalu raise sebelum ke sini
    raise last_exc


# Instance default untuk penggunaan CLI/scripts (alembic membangun engine sendiri).
engine = build_engine(get_settings().database_url)

# Satu session BARU per request/pemakaian — jangan pernah berbagi instance Session.
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False
)


def get_db() -> Generator[Session, None, None]:
    """Dependency FastAPI: satu session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_factory() -> sessionmaker[Session]:
    """Dependency untuk background task (butuh session baru di luar siklus request)."""
    return SessionLocal
