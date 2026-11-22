"""Test M6 — rekomendasi lowongan: service recommend_jobs & GET /me/recommendations.

Semua test memakai embedder PALSU deterministik (bag-of-words hash dari
tests.test_embedding) — TIDAK memuat model sungguhan. Seed & gate reuse dari
test_matching/test_embedding agar perilaku identik dengan matching M4 (gate
MATCH_UNAVAILABLE, normalisasi skill, sumber slug CandidateSkill).
"""

import logging
from datetime import UTC, datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.ai.matching import embedder as embedder_module
from app.ai.matching import scorer
from app.ai.matching.explainer import label_for
from app.models.ai import JobEmbedding, MatchResult
from app.models.job import Job
from app.models.skill import CandidateSkill, CanonicalSkill
from app.services import embedding_service, match_service, recommendation_service
from tests.helpers import create_job, make_candidate, make_hr_with_company
from tests.test_embedding import (
    MODEL_NAME,
    FakeTextEmbedder,
    _fake_factory,
    _seed_company,
    _seed_user,
    _svc_settings,
)
from tests.test_extraction_api import CV_LINES, get_extraction, make_multiline_pdf, upload_cv
from tests.test_matching import _make_match_client, _persist_skills, _seed_cv_primary

# Judul/deskripsi SAMA untuk semua lowongan uji → faktor semantik identik,
# urutan rekomendasi murni ditentukan kecocokan skill (embedder palsu).
JOB_TEXT = {
    "title": "Data Analyst Senior",
    "description": "Menganalisa data penjualan dan membangun dashboard.",
    "requirements": "Pengalaman 2+ tahun.",
    "responsibilities": "Laporan mingguan.",
}

SKILLS_FULL = [{"name": "SQL", "required": True}, {"name": "Python", "required": True}]
SKILLS_PARTIAL = [{"name": "SQL", "required": True}, {"name": "Tableau", "required": False}]
SKILLS_NONE = [{"name": "React", "required": True}, {"name": "Docker", "required": True}]


@pytest.fixture(autouse=True)
def _reset_title_encoder():
    """Hook encoder judul bersih antar test (service M6 memasang hook global)."""
    scorer.set_title_encoder(None)
    yield
    scorer.set_title_encoder(None)


def _seed_rec_job(
    db: Session, hr, company, *, skills: list[dict], status: str = "active", deleted: bool = False
) -> Job:
    """Lowongan uji dengan teks seragam — beda hanya skill/status."""
    job = Job(
        company_id=company.id,
        posted_by_user_id=hr.id,
        source_type="hr",
        status=status,
        deleted_at=datetime.now(UTC) if deleted else None,
        employment_type="full_time",
        work_mode="hybrid",
        seniority="junior",
        skills=skills,
        **JOB_TEXT,
    )
    db.add(job)
    db.commit()
    return job


def _setup_candidate(db: Session, email: str):
    """Kandidat + CV primer terkonfirmasi + skill payload persist (SQL & Python)."""
    user = _seed_user(db, email)
    cv = _seed_cv_primary(db, user)
    _persist_skills(db, user, cv)
    return user, cv


def _add_candidate_skill(db: Session, user, cv, slug: str, *, status: str = "accepted") -> None:
    """Baris CandidateSkill manual (kontrol penuh slug & status untuk test)."""
    canon = db.scalar(select(CanonicalSkill).where(CanonicalSkill.slug == slug))
    assert canon is not None, f"slug {slug} harus ada di seed taksonomi"
    db.add(
        CandidateSkill(
            candidate_user_id=user.id,
            canonical_skill_id=canon.id,
            source_cv_id=cv.id,
            raw_name=canon.name,
            confidence=1.0,
            extraction_method="explicit",
            status=status,
        )
    )
    db.commit()


def _seed_world(db: Session, email: str, hr_email: str):
    """Kandidat siap match + HR dengan perusahaan."""
    user, cv = _setup_candidate(db, email)
    hr = _seed_user(db, hr_email, role="hr")
    company = _seed_company(db, hr)
    return user, cv, hr, company


# =====================================================================
# 1. Service recommend_jobs
# =====================================================================


class TestRecommendJobsService:
    def test_order_counts_and_no_persistence(self, db_session, seeded_taxonomy, monkeypatch):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-flow@example.dev", "rec-hr1@example.dev"
        )
        job_full = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        job_partial = _seed_rec_job(db_session, hr, company, skills=SKILLS_PARTIAL)
        job_none = _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())

        assert set(result) == {"cv_id", "model_version", "computed_at", "total", "items"}
        assert result["cv_id"] == cv.id
        assert result["model_version"] == "m4v1|emb:fake-model-untuk-tes|w:40-30-20-10|cal:v1.1"
        assert isinstance(result["computed_at"], datetime)
        assert result["total"] == 3

        # Urutan kecocokan skill: penuh > parsial > tanpa overlap.
        assert [item["job_id"] for item in result["items"]] == [
            job_full.id,
            job_partial.id,
            job_none.id,
        ]
        overalls = [item["overall"] for item in result["items"]]
        assert overalls == sorted(overalls, reverse=True)
        assert len(set(overalls)) == 3  # strictly decreasing (embedder deterministik)

        full = result["items"][0]
        assert set(full) == {
            "job_id",
            "title",
            "company",
            "location",
            "employment_type",
            "overall",
            "label",
            "matched_required",
            "total_required",
            "matched_skills",
            "missing_critical",
            "reason",
        }
        assert full["title"] == "Data Analyst Senior"
        assert full["company"] == "PT Uji Coba"
        assert full["location"] is None
        assert full["employment_type"] == "full_time"
        assert full["matched_required"] == 2
        assert full["total_required"] == 2
        assert full["matched_skills"] == ["SQL", "Python"]
        assert full["missing_critical"] == 0
        assert full["label"] == label_for(full["overall"])
        assert full["reason"] == f"Skor {full['overall']:.0f} — 2 dari 2 skill wajib terpenuhi."

        partial = result["items"][1]
        assert partial["matched_required"] == 1
        assert partial["total_required"] == 1  # Tableau preferred tidak dihitung wajib
        assert partial["matched_skills"] == ["SQL"]
        assert partial["missing_critical"] == 0  # preferred kosong → minor, bukan kritis
        assert (
            partial["reason"] == f"Skor {partial['overall']:.0f} — 1 dari 1 skill wajib terpenuhi."
        )

        none = result["items"][2]
        assert none["matched_required"] == 0
        assert none["total_required"] == 2
        assert none["matched_skills"] == []
        assert none["missing_critical"] == 2
        assert none["reason"] == (
            f"Skor {none['overall']:.0f} — 0 dari 2 skill wajib terpenuhi; 2 skill wajib belum ada."
        )

        # Keputusan terdokumentasi: TANPA persist match_results; vektor on the fly.
        assert db_session.scalars(select(MatchResult)).all() == []
        assert len(db_session.scalars(select(JobEmbedding)).all()) == 3

    def test_limit_honored_total_counts_all_scored(self, db_session, seeded_taxonomy, monkeypatch):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, _cv, hr, company = _seed_world(
            db_session, "rec-limit@example.dev", "rec-hr2@example.dev"
        )
        job_full = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        job_partial = _seed_rec_job(db_session, hr, company, skills=SKILLS_PARTIAL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings(), limit=2)
        assert result["total"] == 3  # total = semua yang dinilai, bukan panjang items
        assert [item["job_id"] for item in result["items"]] == [job_full.id, job_partial.id]

        # Cap [1, 25] dijaga service juga (panggilan langsung tanpa lapisan API).
        result_cap = recommendation_service.recommend_jobs(
            db_session, user, _svc_settings(), limit=99
        )
        assert len(result_cap["items"]) == 3

    def test_non_active_jobs_excluded(self, db_session, seeded_taxonomy, monkeypatch):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, _cv, hr, company = _seed_world(
            db_session, "rec-status@example.dev", "rec-hr3@example.dev"
        )
        job_active = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL, status="draft")
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL, status="closed")
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL, status="paused")
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL, deleted=True)

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())

        assert result["total"] == 1
        assert [item["job_id"] for item in result["items"]] == [job_active.id]
        # Hanya lowongan eligible yang di-embed on the fly — draft/closed dilewati.
        assert len(db_session.scalars(select(JobEmbedding)).all()) == 1

    def test_match_unavailable_without_or_unconfirmed_cv(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user = _seed_user(db_session, "rec-none@example.dev")
        hr = _seed_user(db_session, "rec-hr4@example.dev", role="hr")
        company = _seed_company(db_session, hr)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        settings = _svc_settings()

        with pytest.raises(match_service.MatchUnavailableError) as exc_no_cv:
            recommendation_service.recommend_jobs(db_session, user, settings)
        assert exc_no_cv.value.code == "MATCH_UNAVAILABLE"

        # CV ada tapi ekstraksi belum dikonfirmasi → exception & kode yang sama.
        _seed_cv_primary(db_session, user, confirmed=False)
        with pytest.raises(match_service.MatchUnavailableError) as exc_unconf:
            recommendation_service.recommend_jobs(db_session, user, settings)
        assert exc_unconf.value.code == "MATCH_UNAVAILABLE"

        # Gate CV dicek SEBELUM pekerjaan embedding lowongan — tidak ada baris vektor.
        assert db_session.scalars(select(JobEmbedding)).all() == []

    def test_missing_vectors_embedded_once_no_n_plus_one(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        user, _cv, hr, company = _seed_world(
            db_session, "rec-embed@example.dev", "rec-hr5@example.dev"
        )
        job_a = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        job_b = _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)

        calls: list[list[str]] = []
        inner = FakeTextEmbedder(MODEL_NAME)

        def counting_factory(_model: str):
            class _Counting:
                def encode(self, texts):
                    calls.append(list(texts))
                    return inner.encode(texts)

            return _Counting()

        monkeypatch.setattr(embedding_service, "get_embedder", counting_factory)

        first = recommendation_service.recommend_jobs(db_session, user, _svc_settings())
        # Run pertama (semua vektor hilang): embed CV + embed 2 lowongan + 1 batch judul.
        assert len(calls) == 4
        assert len(db_session.scalars(select(JobEmbedding)).all()) == 2

        # Run kedua (vektor hangat): HANYA batch judul — tanpa N+1 embedding job.
        calls.clear()
        second = recommendation_service.recommend_jobs(db_session, user, _svc_settings())
        assert len(calls) == 1
        # Batch judul: judul lowongan (urutan eligible) + judul pengalaman CV.
        assert calls[0] == [job_a.title, job_b.title, "Data Analyst Junior"]
        assert second["items"] == first["items"]  # deterministik antar pemanggilan

    def test_job_with_failed_embed_is_skipped(self, db_session, seeded_taxonomy, monkeypatch):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-skip@example.dev", "rec-hr6@example.dev"
        )
        job_ok = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        job_bad = _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)
        embedding_service.embed_cv(db_session, cv.id, _svc_settings())  # CV hangat

        real_embed_job = embedding_service.embed_job

        def flaky_embed_job(db, job_id, settings, **kwargs):
            if job_id == job_bad.id:
                raise RuntimeError("simulasi gagal embed lowongan")
            return real_embed_job(db, job_id, settings, **kwargs)

        monkeypatch.setattr(embedding_service, "embed_job", flaky_embed_job)

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())

        # Yang gagal di-embed DILEWATI dan tidak dihitung di total.
        assert result["total"] == 1
        assert [item["job_id"] for item in result["items"]] == [job_ok.id]

    def test_embed_failure_rollback_does_not_cause_reload_storm(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        """db.rollback() pada jalur gagal-embed meng-expire SELURUH objek session —
        tanpa re-hydrate terkontrol, akses .id/.skills/.title berikutnya memicu
        reload per objek (849 SELECT via pooler saat satu embed gagal). UUID
        ditangkap sebelum loop dan objek Job di-hydrate ulang SEKALI batch."""
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-storm@example.dev", "rec-hr9@example.dev"
        )
        job_ok = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        job_bad = _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)
        embedding_service.embed_cv(db_session, cv.id, _svc_settings())

        real_embed_job = embedding_service.embed_job

        def flaky_embed_job(db, job_id, settings, **kwargs):
            if job_id == job_bad.id:
                # Gagal SETELAH transaksi aktif (meniru insiden nyata:
                # IntegrityError di flush/savepoint) agar rollback benar-benar
                # meng-expire objek session, bukan no-op.
                db.scalars(select(Job)).all()
                raise RuntimeError("simulasi gagal embed lowongan")
            return real_embed_job(db, job_id, settings, **kwargs)

        monkeypatch.setattr(embedding_service, "embed_job", flaky_embed_job)

        job_selects: list[str] = []

        def _count_jobs_select(conn, cursor, stmt, params, many, extra):
            if "FROM jobs" in stmt:
                job_selects.append(stmt)

        event.listen(db_session.get_bind(), "before_cursor_execute", _count_jobs_select)
        try:
            result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())
        finally:
            event.remove(db_session.get_bind(), "before_cursor_execute", _count_jobs_select)

        # Hasil tetap benar (lowongan gagal dilewati), dan query tabel jobs
        # tetap terbatas: 1 (lowongan aktif) + 1 (fake test) + 1 (re-hydrate).
        assert result["total"] == 1
        assert [item["job_id"] for item in result["items"]] == [job_ok.id]
        assert len(job_selects) <= 3, (
            f"reload-storm: {len(job_selects)} query jobs — {job_selects[:5]}"
        )


class TestCandidateSlugSource:
    def test_rejected_skill_excluded_from_recommendations(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user = _seed_user(db_session, "rec-reject@example.dev")
        cv = _seed_cv_primary(db_session, user)  # slug manual — tanpa persist_skills
        hr = _seed_user(db_session, "rec-hr7@example.dev", role="hr")
        company = _seed_company(db_session, hr)
        _add_candidate_skill(db_session, user, cv, "sql")
        _add_candidate_skill(db_session, user, cv, "python", status="rejected")
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())

        item = result["items"][0]
        assert item["matched_required"] == 1  # python status rejected → tidak dikreditkan
        assert item["total_required"] == 2
        assert item["matched_skills"] == ["SQL"]
        assert item["missing_critical"] == 1  # python wajib (kritis) belum ada

    def test_matched_skills_required_first_and_capped(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user = _seed_user(db_session, "rec-cap@example.dev")
        cv = _seed_cv_primary(db_session, user)
        hr = _seed_user(db_session, "rec-hr8@example.dev", role="hr")
        company = _seed_company(db_session, hr)
        for slug in ("sql", "python", "excel", "pandas", "power-bi", "git"):
            _add_candidate_skill(db_session, user, cv, slug)

        job_capped = _seed_rec_job(
            db_session,
            hr,
            company,
            skills=[
                {"name": "SQL", "required": True},
                {"name": "Python", "required": True},
                {"name": "Excel", "required": True},
                {"name": "pandas", "required": True},
                {"name": "Power BI", "required": True},
                {"name": "Git", "required": True},
            ],
        )
        job_mixed = _seed_rec_job(
            db_session,
            hr,
            company,
            skills=[
                {"name": "Excel", "required": False},
                {"name": "pandas", "required": False},
                {"name": "SQL", "required": True},
                {"name": "Python", "required": True},
            ],
        )

        result = recommendation_service.recommend_jobs(db_session, user, _svc_settings())
        by_id = {item["job_id"]: item for item in result["items"]}

        capped = by_id[job_capped.id]
        assert capped["matched_required"] == 6
        assert len(capped["matched_skills"]) == 5  # cap kontrak API
        assert capped["matched_skills"] == ["SQL", "Python", "Excel", "pandas", "Power BI"]

        mixed = by_id[job_mixed.id]
        # Skill WAJIB lebih dulu, lalu diutamakan (urutan skill lowongan dipertahankan).
        assert mixed["matched_skills"] == ["SQL", "Python", "Excel", "pandas"]


# =====================================================================
# 2. API — GET /me/recommendations
# =====================================================================


class TestRecommendationsAPI:
    def test_200_shape_sort_limit_and_rbac(self, db_engine, seeded_taxonomy, monkeypatch):
        with _make_match_client(db_engine, monkeypatch) as client:
            _reg, cand_headers = make_candidate(client)
            _hr_reg, hr_headers, company = make_hr_with_company(client)

            cv = upload_cv(client, cand_headers, make_multiline_pdf(CV_LINES))
            get_extraction(client, cand_headers, cv["id"])
            resp = client.put(f"/api/v1/cvs/{cv['id']}/extraction/confirm", headers=cand_headers)
            assert resp.status_code == 200, resp.text

            job_text = {
                "title": "Analis Data",
                "description": "Menganalisa data penjualan bersama tim produk.",
                "requirements": "Pengalaman 2+ tahun.",
            }
            job_full = create_job(
                client,
                hr_headers,
                **job_text,
                skills=[{"name": "SQL", "required": True}, {"name": "Python", "required": True}],
            )
            job_partial = create_job(
                client,
                hr_headers,
                **job_text,
                skills=[{"name": "SQL", "required": True}, {"name": "Tableau", "required": False}],
            )
            job_none = create_job(
                client,
                hr_headers,
                **job_text,
                skills=[{"name": "React", "required": True}, {"name": "Docker", "required": True}],
            )
            for job in (job_full, job_partial, job_none):
                resp = client.post(f"/api/v1/hr/jobs/{job['id']}/publish", headers=hr_headers)
                assert resp.status_code == 200, resp.text

            resp = client.get("/api/v1/me/recommendations?limit=2", headers=cand_headers)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert set(body) == {"cv_id", "model_version", "computed_at", "total", "items"}
            assert body["cv_id"] == cv["id"]
            assert body["total"] == 3  # seluruh lowongan aktif dinilai
            assert len(body["items"]) == 2  # limit dihormati

            # Urutan kecocokan skill: penuh > parsial (embedder palsu deterministik).
            assert [item["job_id"] for item in body["items"]] == [
                job_full["id"],
                job_partial["id"],
            ]
            overalls = [item["overall"] for item in body["items"]]
            assert overalls == sorted(overalls, reverse=True)

            top = body["items"][0]
            assert set(top) == {
                "job_id",
                "title",
                "company",
                "location",
                "employment_type",
                "overall",
                "label",
                "matched_required",
                "total_required",
                "matched_skills",
                "missing_critical",
                "reason",
            }
            assert top["title"] == "Analis Data"
            assert top["company"] == company["name"]
            assert top["matched_required"] == 2
            assert top["total_required"] == 2
            assert top["matched_skills"] == ["SQL", "Python"]
            assert top["missing_critical"] == 0
            assert top["label"] in ("Sangat Cocok", "Cocok", "Cukup", "Kurang")
            assert top["reason"].startswith("Skor ")
            assert "2 dari 2 skill wajib terpenuhi." in top["reason"]

            # Tanpa query limit → default 10 → semua 3 item tampil.
            resp = client.get("/api/v1/me/recommendations", headers=cand_headers)
            assert resp.status_code == 200
            assert len(resp.json()["items"]) == 3

            # RBAC: HR bukan kandidat → 403; tanpa token → 401.
            resp = client.get("/api/v1/me/recommendations", headers=hr_headers)
            assert resp.status_code == 403
            assert resp.json()["error"]["code"] == "FORBIDDEN"
            resp = client.get("/api/v1/me/recommendations")
            assert resp.status_code == 401

    def test_400_unavailable_without_cv_and_limit_validation(
        self, db_engine, seeded_taxonomy, monkeypatch
    ):
        with _make_match_client(db_engine, monkeypatch) as client:
            _reg, cand_headers = make_candidate(client)
            _hr_reg, hr_headers, _company = make_hr_with_company(client)
            job = create_job(client, hr_headers)
            resp = client.post(f"/api/v1/hr/jobs/{job['id']}/publish", headers=hr_headers)
            assert resp.status_code == 200, resp.text

            # Kandidat belum punya CV terkonfirmasi → 400 MATCH_UNAVAILABLE.
            resp = client.get("/api/v1/me/recommendations", headers=cand_headers)
            assert resp.status_code == 400
            assert resp.json()["error"]["code"] == "MATCH_UNAVAILABLE"

            # limit di luar [1, 25] → 422 validasi FastAPI.
            resp = client.get("/api/v1/me/recommendations?limit=0", headers=cand_headers)
            assert resp.status_code == 422
            resp = client.get("/api/v1/me/recommendations?limit=26", headers=cand_headers)
            assert resp.status_code == 422


# =====================================================================
# 3. Ketahanan — incident 21 Nov 2026 (500 saat koneksi DB terputus)
# =====================================================================


class TestRecommendationsResilience:
    def test_db_failure_during_recommendation_returns_503_envelope(
        self, db_engine, seeded_taxonomy, monkeypatch
    ):
        """Regression GET /me/recommendations?limit=3 (incident 21 Nov 2026).

        Dulu: psycopg.OperationalError saat query lowongan aktif
        (recommendation_service) → HTTP 500 + traceback panjang. Kini kegagalan
        DBAPI dipetakan 503 DEPENDENCY_UNAVAILABLE dengan envelope request_id —
        bukan 500, dan log tetap ringkas (tanpa SQL/traceback di level ERROR).
        """
        with _make_match_client(db_engine, monkeypatch) as client:
            _reg, cand_headers = make_candidate(client)
            _hr_reg, hr_headers, _company = make_hr_with_company(client)
            cv = upload_cv(client, cand_headers, make_multiline_pdf(CV_LINES))
            get_extraction(client, cand_headers, cv["id"])
            resp = client.put(f"/api/v1/cvs/{cv['id']}/extraction/confirm", headers=cand_headers)
            assert resp.status_code == 200, resp.text
            job = create_job(client, hr_headers, skills=[{"name": "SQL", "required": True}])
            resp = client.post(f"/api/v1/hr/jobs/{job['id']}/publish", headers=hr_headers)
            assert resp.status_code == 200, resp.text

            # Simulasikan kegagalan koneksi (SSL EOF / pooler mati) tepat pada
            # query lowongan aktif — jalur persis seperti traceback incident.
            def fail_jobs_query(state):
                if state.is_select and "FROM jobs" in str(state.statement):
                    orig = Exception("SSL connection has been closed unexpectedly")
                    raise OperationalError(str(state.statement), {}, orig)

            event.listen(Session, "do_orm_execute", fail_jobs_query)

            # Tangkap log logger app.core.errors langsung — configure_logging
            # mengganti root.handlers (create_app), sehingga fixture caplog
            # (handler di root) tidak lagi menerima record aplikasi.
            captured: list[logging.LogRecord] = []

            class _Capture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    captured.append(record)

            capture_handler = _Capture(level=logging.ERROR)
            errors_logger = logging.getLogger("app.core.errors")
            errors_logger.addHandler(capture_handler)
            try:
                resp = client.get("/api/v1/me/recommendations?limit=3", headers=cand_headers)
            finally:
                event.remove(Session, "do_orm_execute", fail_jobs_query)
                errors_logger.removeHandler(capture_handler)

            assert resp.status_code == 503, resp.text
            body = resp.json()
            assert body["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
            assert body["error"]["request_id"]
            assert "X-Request-ID" in resp.headers

            # Log ERROR ringkas: tipe exception disebut, SQL & traceback tidak.
            messages = [record.getMessage() for record in captured]
            assert any("Kegagalan database" in message for message in messages), messages
            assert any("OperationalError" in message for message in messages), messages
            all_text = " ".join(messages)
            assert "SSL connection has been closed unexpectedly" not in all_text
            assert "FROM jobs" not in all_text

    def test_no_active_jobs_returns_empty_result(self, db_engine, seeded_taxonomy, monkeypatch):
        """Edge case: tidak ada lowongan aktif → 200 dengan total=0 dan items=[]
        (bukan 500 maupun MATCH_UNAVAILABLE — CV kandidat valid)."""
        with _make_match_client(db_engine, monkeypatch) as client:
            _reg, cand_headers = make_candidate(client)
            cv = upload_cv(client, cand_headers, make_multiline_pdf(CV_LINES))
            get_extraction(client, cand_headers, cv["id"])
            resp = client.put(f"/api/v1/cvs/{cv['id']}/extraction/confirm", headers=cand_headers)
            assert resp.status_code == 200, resp.text

            resp = client.get("/api/v1/me/recommendations", headers=cand_headers)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["total"] == 0
            assert body["items"] == []
            assert body["cv_id"] == cv["id"]


class TestRecommendationsMemoStability:
    def test_ranking_identical_cold_warm_and_reset_memo(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        """Memo normalisasi TIDAK boleh mengubah hasil: ranking, skor, dan
        metadata identik untuk fixture yang sama — memo dingin, memo hangat,
        dan memo dingin setelah reset (indeks dibangun ulang)."""
        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        from app.services import taxonomy_service

        user, cv, hr, company = _seed_world(
            db_session, "rec-memo@example.dev", "rec-hr10@example.dev"
        )
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_PARTIAL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_NONE)
        settings = _svc_settings()

        def snapshot(result: dict) -> dict:
            # computed_at sengaja dikecualikan (waktu komputasi).
            return {
                "cv_id": result["cv_id"],
                "model_version": result["model_version"],
                "total": result["total"],
                "items": result["items"],
            }

        cold1 = snapshot(recommendation_service.recommend_jobs(db_session, user, settings))
        warm = snapshot(recommendation_service.recommend_jobs(db_session, user, settings))
        taxonomy_service.reset_alias_index_cache()  # memo + indeks dingin kembali
        cold2 = snapshot(recommendation_service.recommend_jobs(db_session, user, settings))

        assert warm == cold1
        assert cold2 == cold1


# =====================================================================
# 4. Cache DB ber-kesegaran otoritatif (updated_at) — optimasi tahap 3
# =====================================================================


class TestRecommendationsDbCache:
    def test_second_request_skips_heavy_fetches(self, db_session, seeded_taxonomy, monkeypatch):
        """Request kedua pada katalog yang tidak berubah TIDAK men-fetch ulang
        full-text jobs maupun vektor (transfer MB-an dilompati), dan hasilnya
        identik dengan request pertama."""
        import time as time_module

        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-dbcache@example.dev", "rec-hr11@example.dev"
        )
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_PARTIAL)
        settings = _svc_settings()

        log: list[str] = []

        def _record(conn, cursor, stmt, params, many, extra):
            log.append(" ".join(stmt.split()))

        event.listen(db_session.get_bind(), "before_cursor_execute", _record)

        first = recommendation_service.recommend_jobs(db_session, user, settings)
        heavy_first = sum(
            1 for stmt in log if "jobs.description" in stmt or "job_embeddings.embedding" in stmt
        )
        assert heavy_first >= 2  # full-text jobs + vektor di-fetch saat dingin

        log.clear()
        time_module.sleep(1.05)  # SQLite CURRENT_TIMESTAMP presisi detik: pastikan
        # timestamp "sekarang" untuk perbandingan metadata berada di detik berbeda
        second = recommendation_service.recommend_jobs(db_session, user, settings)

        heavy_second = sum(
            1 for stmt in log if "jobs.description" in stmt or "job_embeddings.embedding" in stmt
        )
        assert heavy_second == 0  # cache hangat: transfer berat dilompati
        # Hasil identik (computed_at dikecualikan).
        for key in ("cv_id", "model_version", "total", "items"):
            assert second[key] == first[key]

    def test_cache_refreshes_when_job_content_changes(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        """Perubahan konten job (updated_at naik) → payload cache diperbarui:
        judul baru muncul di rekomendasi berikutnya."""
        import time as time_module
        from datetime import UTC, datetime

        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-dbcache-upd@example.dev", "rec-hr12@example.dev"
        )
        job = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        settings = _svc_settings()
        first = recommendation_service.recommend_jobs(db_session, user, settings)
        assert first["items"][0]["title"] == "Data Analyst Senior"

        time_module.sleep(1.05)  # presisi detik SQLite: updated_at harus berubah
        job.title = "Senior Analytics Lead"
        job.updated_at = datetime.now(UTC)
        db_session.commit()

        second = recommendation_service.recommend_jobs(db_session, user, settings)
        titles = [item["title"] for item in second["items"]]
        assert "Senior Analytics Lead" in titles
        assert "Data Analyst Senior" not in titles

    def test_closed_job_excluded_with_warm_cache(self, db_session, seeded_taxonomy, monkeypatch):
        """Job ditutup → hilang dari metadata aktif (filter di evaluasi ulang tiap
        request) → tidak muncul walau cache payload masih menyimpannya."""
        import time as time_module
        from datetime import UTC, datetime

        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-dbcache-close@example.dev", "rec-hr13@example.dev"
        )
        job = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        settings = _svc_settings()
        first = recommendation_service.recommend_jobs(db_session, user, settings)
        assert first["total"] == 1

        time_module.sleep(1.05)
        job.status = "closed"
        job.updated_at = datetime.now(UTC)
        db_session.commit()

        second = recommendation_service.recommend_jobs(db_session, user, settings)
        assert second["total"] == 0
        assert second["items"] == []

    def test_cache_refreshes_when_vector_re_embedded(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        """Re-embed (updated_at embedding naik) → vektor cache diperbarui:
        request berikutnya memakai vektor BARU, bukan yang lama."""
        import time as time_module

        monkeypatch.setattr(embedding_service, "get_embedder", _fake_factory)
        user, cv, hr, company = _seed_world(
            db_session, "rec-dbcache-vec@example.dev", "rec-hr14@example.dev"
        )
        job = _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        settings = _svc_settings()
        recommendation_service.recommend_jobs(db_session, user, settings)

        old_vector = recommendation_service._JOB_VECTOR_CACHE[job.id][1]

        # Re-embed dengan konten berubah → vektor baru (embedder palsu
        # deterministik per teks → konten beda menghasilkan vektor beda).
        time_module.sleep(1.05)
        job.skills = [{"name": "React", "required": True}, {"name": "Docker", "required": True}]
        db_session.commit()
        embedding_service.embed_job(db_session, job.id, settings)

        recommendation_service.recommend_jobs(db_session, user, settings)
        new_vector = recommendation_service._JOB_VECTOR_CACHE[job.id][1]
        assert new_vector != old_vector
        assert len(new_vector) == len(old_vector)  # dimensi sama (384)


# =====================================================================
# 5. Prime cache saat startup (incident tes manual 22 Nov: cold > 20 dtk)
# =====================================================================


class TestPrimeCaches:
    def test_prime_caches_fills_all_and_second_sync_is_noop(
        self, db_session, seeded_taxonomy, monkeypatch
    ):
        """prime_caches mengisi seluruh cache user-independent (payload job,
        vektor, encode judul, memo skill); sinkronisasi kedua tanpa fetch berat;
        request pertama setelah prime = hasil identik dengan tanpa prime."""
        import sqlalchemy as sa

        # TextEmbedder ASLI (jalur cache encode nyata) dengan model chunk palsu.
        from tests.test_embedding import FakeChunkModel

        embedder = embedder_module.TextEmbedder(MODEL_NAME)
        monkeypatch.setattr(embedder, "_load_model", lambda: FakeChunkModel())
        monkeypatch.setattr(embedding_service, "get_embedder", lambda _model: embedder)
        from app.services import recommendation_service as rs

        user, cv, hr, company = _seed_world(
            db_session, "rec-prime@example.dev", "rec-hr15@example.dev"
        )
        _seed_rec_job(db_session, hr, company, skills=SKILLS_FULL)
        _seed_rec_job(db_session, hr, company, skills=SKILLS_PARTIAL)
        settings = _svc_settings()

        stats = rs.prime_caches(db_session, settings)
        assert stats["jobs"] == 2
        assert stats["vectors"] == 2
        assert stats["titles_encoded"] == 2
        assert stats["embedded_on_the_fly"] == 2  # vektor belum ada saat prime
        assert len(rs._JOB_DATA_CACHE) == 2
        assert len(rs._JOB_VECTOR_CACHE) == 2
        # encode judul ter-cache: request berikutnya tanpa panggilan model
        cached_titles = {key[1] for key in embedder_module._ENCODE_CACHE}
        assert "Data Analyst Senior" in cached_titles

        # Sinkronisasi kedua (katalog tidak berubah) → TANPA fetch berat.
        log: list[str] = []

        def _record(conn, cursor, stmt, params, many, extra):
            log.append(" ".join(stmt.split()))

        sa.event.listen(db_session.get_bind(), "before_cursor_execute", _record)
        try:
            stats2 = rs.prime_caches(db_session, settings)
        finally:
            sa.event.remove(db_session.get_bind(), "before_cursor_execute", _record)
        assert stats2["jobs_fetched"] == 0
        assert stats2["vectors_fetched"] == 0

        # Request pertama setelah prime = hasil identik dengan alur lama.
        result = recommendation_service.recommend_jobs(db_session, user, settings)
        assert result["total"] == 2
        assert [item["title"] for item in result["items"]] == [
            "Data Analyst Senior",
            "Data Analyst Senior",
        ]
