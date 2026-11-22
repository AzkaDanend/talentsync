"""Service rekomendasi lowongan (M6): skor semua lowongan aktif vs CV primer.

Aturan main:
- Gate identik dengan match_service (M4): CV primer (tidak dihapus) dengan
  ekstraksi TERKONFIRMASI wajib — belum ada → MatchUnavailableError (400 kode
  MATCH_UNAVAILABLE, exception match_service yang SAMA, bukan duplikat).
- Dihitung on request TANPA persist — match_results TIDAK ditulis (keputusan
  terdokumentasi): endpoint /jobs/{id}/compatibility (M4) tetap satu-satunya
  jalur persistensi hasil match; hook invalidasi M4 tidak relevan di sini.
- cv_vec + cv_title_vecs dihitung SEKALI per permintaan; vektor judul SEMUA
  lowongan + judul pengalaman CV di-encode dalam SATU batch (mirror amandemen
  performa M4 — jangan encode ulang judul per pasangan).
- Vektor lowongan dibaca batch (satu query, bukan N+1 untuk yang sudah ada);
  yang hilang di-embed on the fly via embedding_service.embed_job (guarded);
  tetap gagal → lowongan DILEWATI (tidak ikut dinilai).
- Transfer DB dipangkas dengan cache proses ber-kesegaran otoritatif-dari-DB:
  dua query metadata kecil (id + updated_at) dijalankan tiap permintaan, lalu
  payload lowongan & vektor hanya di-fetch untuk id yang BERUBAH sejak cache
  terakhir dilihat. Steady-state (katalog stabil) → transfer multi-detik
  (full-text 849 job + vektor 384 dim ≈ MB-an) dilompati seluruhnya, tanpa
  staleness: `updated_at` adalah penanda perubahan resmi (semua tulis Job /
  job_embeddings lewat ORM `onupdate=func.now`; sweep ingestion menulisnya
  eksplisit) dan nilai cache identik dengan nilai DB untuk updated_at sama.
- `total` = jumlah lowongan aktif yang BERHASIL dinilai; `items` = top `limit`
  urut overall DESC (tie-break job_id agar deterministik).
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.matching import scorer
from app.ai.matching.explainer import label_for
from app.ai.matching.scorer import score_pair
from app.core.config import Settings
from app.models.ai import CvEmbedding, JobEmbedding
from app.models.job import Job
from app.models.user import User
from app.services import embedding_service, match_service, skill_service, taxonomy_service

logger = logging.getLogger(__name__)

# Batas item rekomendasi (kontrak API M6): default 10, cap 25. Endpoint sudah
# memvalidasi lewat Query(ge=1, le=25); service menjaga ulang agar fungsi ini
# aman dipanggil langsung (skrip/evaluasi) tanpa lapisan API.
DEFAULT_LIMIT = 10
MAX_LIMIT = 25

# Maksimal nama skill yang tampil di satu item rekomendasi.
MAX_MATCHED_SKILLS = 5

# Cache proses untuk payload lowongan & vektor (lihat docstring modul).
# Nilai plain-data (dict/list/datetime) — TANPA objek ORM terikat session.
_JOB_DATA_CACHE: dict[uuid.UUID, tuple[datetime, dict]] = {}
_JOB_VECTOR_CACHE: dict[uuid.UUID, tuple[datetime, list[float]]] = {}
_CACHE_SOFT_CAP = 20_000


def reset_recommendation_cache() -> None:
    """Kosongkan cache payload/vektor rekomendasi (test & CLI/scripts)."""
    _JOB_DATA_CACHE.clear()
    _JOB_VECTOR_CACHE.clear()


def _job_payload(job: Job) -> dict:
    """Kolom yang dibutuhkan pipeline rekomendasi dari satu Job → plain dict
    bebas-session (aman disimpan di cache proses; tanpa objek ORM)."""
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "requirements": job.requirements,
        "responsibilities": job.responsibilities,
        "seniority": job.seniority,
        "skills": copy.deepcopy(job.skills) if job.skills else [],
        "company_name": job.company.name if job.company is not None else None,
        "location_text": job.location_text,
        "employment_type": job.employment_type,
    }


def _sync_job_caches(
    db: Session, settings: Settings
) -> tuple[dict[uuid.UUID, dict], dict[uuid.UUID, list[float]], int, int]:
    """Sinkronkan cache payload & vektor dengan keadaan DB (otoritatif).

    Dua query metadata kecil (id + updated_at) dijalankan setiap permintaan;
    fetch penuh hanya untuk id yang berubah atau baru. Mengembalikan
    (payload_by_id, vectors_by_id, jobs_fetched, vectors_fetched) — payload &
    vektor berisi SELURUH lowongan aktif dan SELURUH vektor aktif model aktif
    persis seperti hasil query langsung (nilai identik untuk updated_at sama).
    """
    active_meta = db.execute(
        select(Job.id, Job.updated_at).where(Job.status == "active", Job.deleted_at.is_(None))
    ).all()
    active_by_id = {jid: upd for jid, upd in active_meta}

    changed_jobs = [
        jid
        for jid, upd in active_by_id.items()
        if (entry := _JOB_DATA_CACHE.get(jid)) is None or entry[0] != upd
    ]
    jobs_fetched = 0
    if changed_jobs:
        rows = db.scalars(select(Job).where(Job.id.in_(changed_jobs))).all()
        for job in rows:
            _JOB_DATA_CACHE[job.id] = (job.updated_at, _job_payload(job))
        jobs_fetched = len(rows)

    emb_meta = db.execute(
        select(JobEmbedding.job_id, JobEmbedding.updated_at)
        .join(Job, JobEmbedding.job_id == Job.id)
        .where(
            Job.status == "active",
            Job.deleted_at.is_(None),
            JobEmbedding.model == settings.embedding_model,
            JobEmbedding.text_kind == embedding_service.TEXT_KIND_JOB,
        )
    ).all()
    changed_vecs = [
        jid
        for jid, upd in emb_meta
        if (entry := _JOB_VECTOR_CACHE.get(jid)) is None or entry[0] != upd
    ]
    vectors_fetched = 0
    if changed_vecs:
        rows = db.execute(
            select(JobEmbedding.job_id, JobEmbedding.updated_at, JobEmbedding.embedding).where(
                JobEmbedding.job_id.in_(changed_vecs),
                JobEmbedding.model == settings.embedding_model,
                JobEmbedding.text_kind == embedding_service.TEXT_KIND_JOB,
            )
        ).all()
        for jid, upd, emb in rows:
            _JOB_VECTOR_CACHE[jid] = (upd, list(emb))
        vectors_fetched = len(rows)

    # Pengaman memori: katalog normal jauh di bawah ini; bila melesat (mis.
    # proses berumur panjang melihat jutaan id), reset — request berikutnya
    # membangun ulang cache secara penuh.
    if len(_JOB_DATA_CACHE) > _CACHE_SOFT_CAP or len(_JOB_VECTOR_CACHE) > _CACHE_SOFT_CAP:
        logger.warning("Cache rekomendasi melewati %s entri — di-reset.", _CACHE_SOFT_CAP)
        _JOB_DATA_CACHE.clear()
        _JOB_VECTOR_CACHE.clear()

    payloads = {jid: _JOB_DATA_CACHE[jid][1] for jid in active_by_id if jid in _JOB_DATA_CACHE}
    vectors = {jid: _JOB_VECTOR_CACHE[jid][1] for jid, _ in emb_meta if jid in _JOB_VECTOR_CACHE}
    return payloads, vectors, jobs_fetched, vectors_fetched


def _fetch_vectors(
    db: Session, job_ids: list[uuid.UUID], settings: Settings
) -> dict[uuid.UUID, list[float]]:
    """Vektor model aktif untuk SEKUMPUNGAN id — satu query (anti N+1).

    Baris text_kind 'job' milik model aktif; baris model lama tidak disentuh."""
    if not job_ids:
        return {}
    rows = db.execute(
        select(JobEmbedding.job_id, JobEmbedding.updated_at, JobEmbedding.embedding).where(
            JobEmbedding.job_id.in_(job_ids),
            JobEmbedding.model == settings.embedding_model,
            JobEmbedding.text_kind == embedding_service.TEXT_KIND_JOB,
        )
    ).all()
    vectors: dict[uuid.UUID, list[float]] = {}
    for jid, upd, emb in rows:
        vectors[jid] = list(emb)
        _JOB_VECTOR_CACHE[jid] = (upd, vectors[jid])
    return vectors


def prime_caches(db: Session, settings: Settings) -> dict:
    """Bangun seluruh cache rekomendasi yang TIDAK bergantung pengguna.

    Dipanggil saat startup (lihat main._warm_startup_caches) agar request
    pertama pengguna tidak membayar cold-start — tes manual 22 Nov 2026:
    request pertama pasca-restart = model (±30 dtk) + encode judul (±23 dtk) +
    fuzzy (±17 dtk) + transfer DB (±12-46 dtk) ≫ budget axios 20 dtk → UI
    selalu "timeout of 20000ms exceeded" walau request berikutnya cepat.

    Setiap langkah guarded: kegagalan tidak fatal (jalur on-demand tetap
    berjalan seperti sebelumnya). Mengembalikan rekap hitungan untuk log."""
    started = time.perf_counter()
    payloads, vectors, jobs_fetched, vectors_fetched = _sync_job_caches(db, settings)
    alias_index = taxonomy_service.load_alias_index(db)

    # Memo normalisasi untuk seluruh nama skill katalog (fuzzy dulu 17 dtk CPU).
    for payload in payloads.values():
        skill_service.normalize_job_skills(db, payload["skills"], index=alias_index)

    # Encode judul seluruh lowongan aktif → cache teks (dulu 22-23 dtk CPU).
    titles = [payload["title"] for payload in payloads.values()]
    if titles:
        embedding_service.get_embedder(settings.embedding_model).encode(titles)

    # Lowongan aktif yang belum punya vektor → embed on the fly SEKALI di sini,
    # bukan menggantung di request pengguna (90 job baru dari ingestion ≈ 90-120 dtk).
    missing = [jid for jid in payloads if jid not in vectors]
    embedded = 0
    for jid in missing:
        try:
            embedding_service.embed_job(db, jid, settings)
            embedded += 1
        except Exception as embed_error:
            db.rollback()
            logger.warning(
                "Prime embed lowongan %s gagal — dilewati: %s", jid, type(embed_error).__name__
            )
    if embedded:
        vectors.update(_fetch_vectors(db, missing, settings))

    return {
        "jobs": len(payloads),
        "vectors": len(vectors),
        "titles_encoded": len(titles),
        "jobs_fetched": jobs_fetched,
        "vectors_fetched": vectors_fetched,
        "embedded_on_the_fly": embedded,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _build_reason(
    overall: float, matched_required: int, total_required: int, missing_critical: int
) -> str:
    """Narasi dua bagian deterministik (tanpa LLM) — kontrak frontend M6.

    Bagian 1 selalu tampil; bagian 2 hanya bila ada skill wajib kritis yang
    belum ada. Input sama → string identik (mudah diuji & stabil di UI).
    """
    base = f"Skor {overall:.0f} — {matched_required} dari {total_required} skill wajib terpenuhi"
    if missing_critical > 0:
        return f"{base}; {missing_critical} skill wajib belum ada."
    return f"{base}."


def _build_item(job: dict, result: dict) -> dict:
    """Satu item rekomendasi dari hasil score_pair + payload lowongan (cache)."""
    matched = result["matched"]
    missing = result["missing"]
    matched_required = sum(1 for item in matched if item.get("required"))
    total_required = matched_required + sum(1 for item in missing if item.get("required"))
    missing_critical = sum(1 for item in missing if item.get("severity") == "kritis")
    overall = float(result["overall"])
    # Skill WAJIB lebih dulu, lalu diutamakan (urutan asal dipertahankan —
    # sort Python stabil), dipangkas ke MAX_MATCHED_SKILLS nama tampilan.
    ordered = sorted(matched, key=lambda item: not item.get("required"))
    return {
        "job_id": job["id"],
        "title": job["title"],
        "company": job["company_name"],
        "location": job["location_text"],
        "employment_type": job["employment_type"],
        "overall": overall,
        "label": label_for(overall),
        "matched_required": matched_required,
        "total_required": total_required,
        "matched_skills": [
            str(item.get("display") or item.get("name") or "")
            for item in ordered[:MAX_MATCHED_SKILLS]
        ],
        "missing_critical": missing_critical,
        "reason": _build_reason(overall, matched_required, total_required, missing_critical),
    }


def recommend_jobs(
    db: Session, user: User, settings: Settings, *, limit: int = DEFAULT_LIMIT
) -> dict:
    """Rekomendasi lowongan aktif untuk CV primer kandidat (dihitung on request).

    Mengembalikan dict kontrak API GET /me/recommendations: cv_id,
    model_version, computed_at, total (lowongan aktif yang berhasil dinilai),
    items (top `limit`, urut overall DESC). TIDAK menulis match_results dan
    TIDAK commit — satu-satunya efek tulis adalah vektor on the fly di
    cv/job_embeddings (commit internal embedding_service, pola M4).
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    request_started = time.perf_counter()

    # 1) Gate + data kandidat — reuse helper match_service agar gate & sumber
    #    slug PERSIS sama dengan /compatibility (status != rejected; explicit =
    #    extraction_method 'explicit').
    cv = match_service._get_primary_cv(db, user)
    candidate_slugs, explicit_slugs = match_service._load_candidate_slugs(db, user)

    cv_vec = match_service._load_vector(db, CvEmbedding, "cv_id", cv.id, "profile", settings)
    if cv_vec is None:
        embedding_service.embed_cv(db, cv.id, settings)
        cv_vec = match_service._load_vector(db, CvEmbedding, "cv_id", cv.id, "profile", settings)
        if cv_vec is None:
            raise match_service.MatchUnavailableError()

    payload = cv.extraction.payload or {}
    # Sejajar urutan payload experiences APA ADANYA (termasuk entri tanpa judul)
    # agar len(cv_title_vecs) == len(experiences — kontrak score_pair).
    experience_titles = [
        str(exp.get("title") or "") if isinstance(exp, dict) else ""
        for exp in payload.get("experiences") or []
    ]
    taxonomy_related = match_service.load_taxonomy_related()

    # 2+3) Lowongan aktif + vektor — via cache ber-kesegaran otoritatif DB:
    # metadata kecil tiap request, fetch penuh hanya untuk id yang berubah.
    # Payload berupa plain-dict bebas-session (kebal rollback — dulu rollback
    # jalur gagal-embed meng-expire 849 objek ORM dan memicu reload per objek).
    vectors_started = time.perf_counter()
    payloads, job_vecs, jobs_fetched, vectors_fetched = _sync_job_caches(db, settings)
    active_ids = list(payloads)

    missing = [jid for jid in active_ids if jid not in job_vecs]
    for job_id in missing:
        try:
            embedding_service.embed_job(db, job_id, settings)
        except Exception as embed_error:
            # Dilewati (lowongan ini tidak ikut dinilai) — kegagalan operasional
            # yang diharapkan: log ringkas per kejadian, TANPA traceback penuh
            # (dulu membanjiri terminal saat DB tidak stabil).
            db.rollback()  # bersihkan transaksi; lowongan ini saja yang dilewati
            logger.warning(
                "Embed on the fly lowongan %s gagal — dilewati: %s",
                job_id,
                type(embed_error).__name__,
            )
    if missing:
        # Ambil vektor yang baru saja di-embed on the fly (kecil: hanya id yang
        # tadi hilang) dan masukkan ke cache — kesegaran tetap otoritatif.
        fresh = db.execute(
            select(JobEmbedding.job_id, JobEmbedding.updated_at, JobEmbedding.embedding).where(
                JobEmbedding.job_id.in_(missing),
                JobEmbedding.model == settings.embedding_model,
                JobEmbedding.text_kind == embedding_service.TEXT_KIND_JOB,
            )
        ).all()
        for jid, upd, emb in fresh:
            _JOB_VECTOR_CACHE[jid] = (upd, list(emb))
            job_vecs[jid] = _JOB_VECTOR_CACHE[jid][1]
    embed_on_the_fly_ms = (time.perf_counter() - vectors_started) * 1000

    eligible = [payloads[jid] for jid in active_ids if jid in job_vecs]
    cv_title_vecs: list[list[float]] = []
    # Vektor judul lowongan: WAJIB didefinisikan di sini (bukan hanya di dalam
    # `if eligible`) — zip(..., strict=True) dievaluasi meski eligible kosong,
    # dan variabel tak terdefinisi berarti UnboundLocalError (500) ketika TIDAK
    # ada satu pun lowongan aktif yang bisa dinilai.
    job_title_vecs: list[list[float]] = []
    if eligible:
        # Encoder judul terikat model aktif — fallback defensif bila vektor judul
        # tak sejajar (normalnya tidak pernah dipakai, mirror match_service).
        scorer.set_title_encoder(
            lambda texts, _model=settings.embedding_model: embedding_service.get_embedder(
                _model
            ).encode(list(texts))
        )
        # SATU batch untuk semua judul: judul lowongan (per pasangan) + judul
        # pengalaman CV (sekali per request) — tanpa encode ulang per pasangan.
        encoded = embedding_service.get_embedder(settings.embedding_model).encode(
            [*(payload["title"] for payload in eligible), *experience_titles]
        )
        n_jobs = len(eligible)
        job_title_vecs = [list(vector) for vector in encoded[:n_jobs]]
        cv_title_vecs = [list(vector) for vector in encoded[n_jobs:]]

    # 4) Skor per pasangan — skill lowongan di-resolve SEKALI untuk semua
    #    lowongan (anti N+1): satu indeks alias + satu query CanonicalSkill
    #    untuk seluruh batch. Dulu _resolve_job_skills dipanggil per lowongan →
    #    ratusan query indeks/query kanon per request (incident 21 Nov 2026,
    #    1-4 dtk per query via pooler = latensi menit-an).
    alias_index = taxonomy_service.load_alias_index(db)  # cache per proses — sekali muat
    normalized_by_job = {
        payload["id"]: skill_service.normalize_job_skills(db, payload["skills"], index=alias_index)
        for payload in eligible
    }
    canon_by_id = skill_service.canonical_map_for_normalized(db, normalized_by_job)

    scored: list[tuple[dict, dict]] = []
    for job, job_title_vec in zip(eligible, job_title_vecs, strict=True):
        job_dict = {
            "title": job["title"],
            "skills": skill_service.resolve_normalized_skills(
                normalized_by_job[job["id"]], canon_by_id
            ),
            "description": job["description"],
            "requirements": job["requirements"],
            "responsibilities": job["responsibilities"],
            "seniority": job["seniority"],
        }
        result = score_pair(
            cv_payload=payload,  # payload CV (bukan job) — didefinisikan di tahap 1
            job=job_dict,
            cv_vec=cv_vec,
            job_vec=job_vecs[job["id"]],
            job_title_vec=job_title_vec,
            candidate_slugs=candidate_slugs,
            explicit_slugs=explicit_slugs,
            weights=tuple(settings.match_weights),
            taxonomy_related=taxonomy_related,
            cv_title_vecs=cv_title_vecs,
        )
        scored.append((job, result))

    # 5) Urut overall DESC (tie-break job_id → deterministik), ambil top limit.
    scored.sort(key=lambda pair: (-float(pair[1]["overall"]), pair[0]["id"]))
    items = [_build_item(job, result) for job, result in scored[:limit]]

    # Observability latensi jalur rekomendasi (Phase 5): satu baris INFO per
    # permintaan dengan durasi per tahap + hitungan — cukup untuk melihat
    # bottleneck (DB vs embed vs scoring) tanpa mencemari log request lain.
    total_ms = (time.perf_counter() - request_started) * 1000
    logger.info(
        "Rekomendasi: %s lowongan aktif, %s dinilai, %s item — total %.0f ms "
        "(embed on-the-fly %.0f ms; fetch job=%s, vektor=%s — 0 berarti cache hangat).",
        len(active_ids),
        len(scored),
        len(items),
        total_ms,
        embed_on_the_fly_ms,
        jobs_fetched,
        vectors_fetched,
    )

    return {
        "cv_id": cv.id,
        "model_version": match_service.build_model_version(settings),
        "computed_at": datetime.now(UTC),
        "total": len(scored),
        "items": items,
    }
