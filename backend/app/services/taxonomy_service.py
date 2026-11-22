"""Service taksonomi skill (M2): seed JSON, indeks alias, normalisasi, autocomplete.

Kontrak untuk normalizer/ekstraksi (M3, agen berikutnya):
- `seed_from_json` idempoten — jalankan berulang tanpa duplikat/error.
- `load_alias_index` satu query; kunci = alias & nama kanon (lowercase).
- `normalize_skill` DICTIONARY-FIRST: exact → alias → fuzzy terkendali; tidak
  pernah memaksa pemetaan (gagal = needs_review + saran).
- `autocomplete` untuk kolom pencarian skill di UI.
"""

import json
import uuid
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.skill import CanonicalSkill, SkillAlias

# Lokasi berkas taksonomi default (app/services/ -> app/ai/extraction/taxonomy/).
DEFAULT_TAXONOMY_JSON = (
    Path(__file__).resolve().parents[1] / "ai" / "extraction" / "taxonomy" / "skills_v1.json"
)

# Confidence per metode pencocokan (kontrak tetap untuk konsumen M3).
_CONFIDENCE_EXACT = 0.95
_CONFIDENCE_ALIAS = 0.95
_CONFIDENCE_FUZZY = 0.8

# Fuzzy terkendali: kandidat terbaik harus unggul jelas dari runner-up agar
# tidak ada pemetaan ambigu yang dipaksa.
FUZZY_MARGIN = 0.03
FUZZY_SUGGEST_MIN = 0.8

# Karakter yang dipertahankan di dalam kata: "c++", "c#", ".net", "node.js",
# "ci/cd", "react-native" tetap utuh setelah normalisasi.
_KEEP_INSIDE_WORD = set("+#./-")


def normalize_text(raw: str) -> str:
    """Normalisasi teks skill: lowercase, rapikan spasi, pertahankan +/#/./-/ di dalam kata.

    Titik di akhir kalimat dibuang ("Python." -> "python"), sedangkan titik
    yang menghubungkan bagian nama dipertahankan ("node.js", ".net").
    """
    text = (raw or "").strip().lower()
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch == " ":
            out.append(ch)
            continue
        if ch not in _KEEP_INSIDE_WORD:
            out.append(" ")
            continue
        prev_ok = bool(out) and (out[-1].isalnum() or out[-1] in _KEEP_INSIDE_WORD)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        nxt_alnumish = bool(nxt) and (nxt.isalnum() or nxt in _KEEP_INSIDE_WORD)
        if ch == ".":
            keep = (prev_ok or not out) and nxt_alnumish
        elif ch in "+#":
            keep = prev_ok and (nxt_alnumish or nxt == "")
        else:  # '/' atau '-'
            keep = prev_ok and nxt_alnumish
        out.append(ch if keep else " ")
    return " ".join("".join(out).split())


def seed_from_json(db: Session, json_path: Path | None = None) -> dict:
    """Seed/upsert taksonomi dari skills_v1.json secara idempoten.

    - Upsert skill kanon per slug (field di-update bila berubah).
    - Alias: tambah yang belum ada, lewati yang sudah terdaftar (alias unik
      global sehingga alias milik skill lain tidak pernah ditimpa).
    - taxonomy_version mengikuti versi berkas.
    """
    path = Path(json_path) if json_path else DEFAULT_TAXONOMY_JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    version = str(data["version"])

    existing_skills = {s.slug: s for s in db.scalars(select(CanonicalSkill))}
    existing_aliases: set[str] = set(db.scalars(select(SkillAlias.alias)))

    for item in data["skills"]:
        skill = existing_skills.get(item["slug"])
        if skill is None:
            skill = CanonicalSkill(slug=item["slug"])
            db.add(skill)
            existing_skills[item["slug"]] = skill
        skill.name = item["name"]
        skill.category = item["category"]
        skill.skill_type = item["skill_type"]
        skill.description = item.get("description")
        skill.related = item.get("related")
        skill.is_active = True
        skill.taxonomy_version = version
        db.flush()  # skill.id tersedia untuk alias baru

        for alias in item.get("aliases", []):
            alias_text = alias.strip().lower()
            if not alias_text or alias_text in existing_aliases:
                continue
            db.add(SkillAlias(canonical_skill_id=skill.id, alias=alias_text, source="seed"))
            existing_aliases.add(alias_text)

    db.commit()
    reset_alias_index_cache()  # taksonomi berubah → indeks cache tidak valid
    return {
        "canonical": int(db.scalar(select(func.count()).select_from(CanonicalSkill))),
        "aliases": int(db.scalar(select(func.count()).select_from(SkillAlias))),
        "version": version,
    }


def load_alias_index(db: Session) -> dict[str, CanonicalSkill]:
    """Indeks pencocokan: alias & nama kanon (lowercase) -> CanonicalSkill.

    Satu query (outerjoin); hanya skill aktif. Dipakai normalizer M3 dan
    autocomplete — nama kanon selalu menang atas alias bila bertabrakan.

    Di-cache per proses: incident 21 Nov 2026 — jalur rekomendasi memanggil ini
    sekali per lowongan (dan normalize_skill sekali per NAMA SKILL), sehingga
    satu request membayar ratusan query × 1-4 dtk melalui pooler (latensi
    menit-an per request). Taksonomi hanya berubah lewat seed_from_json
    (invalidasi di sana) dan reset_alias_index_cache() untuk test.
    """
    global _ALIAS_INDEX_CACHE
    if _ALIAS_INDEX_CACHE is not None:
        return _ALIAS_INDEX_CACHE
    rows = db.execute(
        select(CanonicalSkill, SkillAlias)
        .outerjoin(SkillAlias, SkillAlias.canonical_skill_id == CanonicalSkill.id)
        .where(CanonicalSkill.is_active.is_(True))
    ).all()
    index: dict[str, CanonicalSkill] = {}
    cached: set[uuid.UUID] = set()
    for skill, alias in rows:
        if skill.id not in cached:
            cached.add(skill.id)
            # Detach dari session peminta SEBELUM masuk cache: objek yang tetap
            # attached akan DI-EXPIRE oleh rollback session tsb. (mis. jalur
            # gagal embed di recommend_jobs), meracuni cache proses — request
            # berikutnya kena DetachedInstanceError saat membaca .slug/.name.
            db.expunge(skill)
        index[skill.name.strip().lower()] = skill
        if alias is not None:
            index.setdefault(alias.alias.strip().lower(), skill)
    _ALIAS_INDEX_CACHE = index
    return index


_ALIAS_INDEX_CACHE: dict[str, CanonicalSkill] | None = None

# Split lookup (names/aliases) untuk _ALIAS_INDEX_CACHE — hasil _split_lookup
# O(jumlah entri) yang dulu dijalankan ULANG pada SETIAP normalize_skill.
_SPLIT_CACHE: tuple[dict[str, CanonicalSkill], dict[str, CanonicalSkill]] | None = None

# Memo keputusan normalisasi per (query, fuzzy_threshold). NILAI MURNI DATA
# (str/float/bool/tuple) — tanpa objek ORM: keputusan fuzzy (SequenceMatcher
# ~954 entri per nama tanpa padanan exact/alias) dihitung SEKALI per nama unik;
# objek kanon direkonstruksi dari indeks alias yang sudah bebas-sesi.
# Incident 21 Nov 2026: tahap resolve skill rekomendasi 103 dtk CPU murni
# karena fuzzy matching berulang untuk nama skill yang sama antar lowongan.
_NORMALIZE_CACHE: dict[tuple[str, float], tuple] = {}


def reset_alias_index_cache() -> None:
    """Kosongkan seluruh cache taksonomi proses (test & CLI/scripts).

    Satu pintu invalidasi: indeks alias, split lookup, dan memo normalisasi —
    seed_from_json memanggil ini saat taksonomi berubah."""
    global _ALIAS_INDEX_CACHE, _SPLIT_CACHE, _NORMALIZE_CACHE
    _ALIAS_INDEX_CACHE = None
    _SPLIT_CACHE = None
    _NORMALIZE_CACHE.clear()


def _split_lookup(index: dict[str, CanonicalSkill]) -> tuple[dict, dict]:
    """Pisahkan indeks menjadi peta nama kanon dan peta alias (sudah dinormalisasi)."""
    names: dict[str, CanonicalSkill] = {}
    aliases: dict[str, CanonicalSkill] = {}
    for key, skill in index.items():
        nk = normalize_text(key)
        if nk == normalize_text(skill.name):
            names[nk] = skill
        else:
            aliases.setdefault(nk, skill)
    return names, aliases


def _get_split(index: dict[str, CanonicalSkill]) -> tuple[dict, dict]:
    """_split_lookup dengan cache proses — HANYA untuk indeks cache proses.

    _split_lookup O(entri) dan dulu dijalankan ulang per panggilan normalize_skill
    (954× normalize_text per nama skill — kontributor latensi tahap resolve).
    Indeks milik pemanggil (bukan cache proses) selalu dihitung segar: isinya
    bisa apa saja (test/CLI), memo tidak boleh mengasosiasinya dengan keputusan
    yang di-cache."""
    global _SPLIT_CACHE
    if index is _ALIAS_INDEX_CACHE and _SPLIT_CACHE is not None:
        return _SPLIT_CACHE
    names, aliases = _split_lookup(index)
    if index is _ALIAS_INDEX_CACHE:
        _SPLIT_CACHE = (names, aliases)
    return names, aliases


def _normalize_from_decision(
    decision: tuple, names: dict[str, CanonicalSkill], aliases: dict[str, CanonicalSkill]
) -> dict:
    """Bangun hasil normalize_skill dari keputusan ter-memo (nilai murni data).

    Objek kanon diambil dari indeks alias yang sudah bebas-sesi (expunge saat
    cache dibangun) — memo TIDAK menyimpan objek ORM agar rollback session
    mana pun tidak bisa meracuninya."""
    method, matched_nk, confidence, needs_review, suggestions = decision
    if method == "none":
        canonical = None
    elif method == "exact":
        canonical = names.get(matched_nk)
    elif method == "alias":
        canonical = aliases.get(matched_nk)
    else:  # fuzzy — pemenang adalah kunci dinormalisasi pada gabungan (kanon menang)
        canonical = {**aliases, **names}.get(matched_nk)
    return {
        "canonical": canonical,
        "confidence": confidence,
        "method": method,
        "needs_review": needs_review,
        # salinan segar per panggilan: pemanggil bebas memutasi hasilnya tanpa
        # mengubah keputusan yang di-cache.
        "suggestions": [dict(item) for item in suggestions],
    }


def normalize_skill(
    db: Session, raw: str, *, fuzzy_threshold: float = 0.92, index: dict | None = None
) -> dict:
    """Normalisasi satu nama skill mentah ke skill kanon (dictionary-first).

    Urutan: exact nama kanon (0.95) -> alias (0.95) -> fuzzy unik (0.8) ->
    gagal (needs_review + saran). Tidak pernah memaksa pemetaan ambigu.

    `index` opsional: peta alias yang sudah dimuat pemanggil (sekali untuk
    banyak nama — tanpa ini setiap nama memicu query indeks baru; incident
    21 Nov 2026). Bila None, indeks dimuat (cache per proses).

    Memo per proses: keputusan (exact/alias/fuzzy/none + saran) di-cache per
    (query, fuzzy_threshold) untuk indeks cache proses — fuzzy SequenceMatcher
    (~954 entri) dulu dijalankan ulang untuk SETIAP kemunculan nama yang sama
    antar lowongan (tahap resolve skill rekomendasi 103 dtk CPU). Hasil fungsi
    IDENTIK dengan tanpa memo: keputusan murni fungsi dari (query, indeks,
    threshold), objek kanon direkonstruksi dari indeks yang sama.

    Bentuk saran: [{"slug": ..., "name": ..., "score": ...}] (maks 3, score >= 0.8).
    """
    query = normalize_text(raw)
    if not query:
        return {
            "canonical": None,
            "confidence": 0.0,
            "method": "none",
            "needs_review": True,
            "suggestions": [],
        }

    resolved_index = index if index is not None else load_alias_index(db)
    names, aliases = _get_split(resolved_index)
    # Memo hanya untuk indeks cache proses (identitas sama) — keputusan adalah
    # fungsi murni dari (query, isi indeks, threshold).
    use_memo = resolved_index is _ALIAS_INDEX_CACHE
    memo_key = (query, fuzzy_threshold)
    if use_memo:
        cached_decision = _NORMALIZE_CACHE.get(memo_key)
        if cached_decision is not None:
            return _normalize_from_decision(cached_decision, names, aliases)

    hit = names.get(query)
    if hit is not None:
        if use_memo:
            _NORMALIZE_CACHE[memo_key] = ("exact", query, _CONFIDENCE_EXACT, False, ())
        return {
            "canonical": hit,
            "confidence": _CONFIDENCE_EXACT,
            "method": "exact",
            "needs_review": False,
            "suggestions": [],
        }
    hit = aliases.get(query)
    if hit is not None:
        if use_memo:
            _NORMALIZE_CACHE[memo_key] = ("alias", query, _CONFIDENCE_ALIAS, False, ())
        return {
            "canonical": hit,
            "confidence": _CONFIDENCE_ALIAS,
            "method": "alias",
            "needs_review": False,
            "suggestions": [],
        }

    # Fuzzy terkendali: nilai terbaik PER skill (satu skill bisa punya beberapa
    # alias dengan skor mirip — tidak dihitung sebagai kompetitor).
    per_skill: dict[object, tuple[float, CanonicalSkill]] = {}
    best_nk: dict[object, str] = {}
    for nk, skill in {**aliases, **names}.items():
        ratio = SequenceMatcher(None, query, nk).ratio()
        current = per_skill.get(skill.id)
        if current is None or ratio > current[0]:
            per_skill[skill.id] = (ratio, skill)
            best_nk[skill.id] = nk
    ranked = sorted(per_skill.values(), key=lambda item: item[0], reverse=True)

    best_ratio, best_skill = ranked[0]
    second_ratio = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_ratio >= fuzzy_threshold and second_ratio < best_ratio - FUZZY_MARGIN:
        if use_memo:
            decision = ("fuzzy", best_nk[best_skill.id], _CONFIDENCE_FUZZY, False, ())
            _NORMALIZE_CACHE[memo_key] = decision
        return {
            "canonical": best_skill,
            "confidence": _CONFIDENCE_FUZZY,
            "method": "fuzzy",
            "needs_review": False,
            "suggestions": [],
        }

    suggestions = [
        {"slug": skill.slug, "name": skill.name, "score": round(ratio, 4)}
        for ratio, skill in ranked[:3]
        if ratio >= FUZZY_SUGGEST_MIN
    ]
    if use_memo:
        _NORMALIZE_CACHE[memo_key] = ("none", None, 0.0, True, tuple(suggestions))
    return {
        "canonical": None,
        "confidence": 0.0,
        "method": "none",
        "needs_review": True,
        "suggestions": suggestions,
    }


def autocomplete(db: Session, q: str, limit: int = 10) -> list[CanonicalSkill]:
    """Saran skill untuk kolom pencarian (nama + alias, case-insensitive).

    q < 2 karakter setelah strip -> daftar kosong (dipilih: kosong, bukan 422).
    Peringkat: exact nama > prefix nama > exact alias > prefix alias > contains.
    """
    query = (q or "").strip().lower()
    if len(query) < 2:
        return []

    names, aliases = _get_split(load_alias_index(db))
    nq = normalize_text(query)

    ranked: dict[object, tuple[int, CanonicalSkill]] = {}

    def add(skill: CanonicalSkill, rank: int) -> None:
        current = ranked.get(skill.id)
        if current is None or rank < current[0]:
            ranked[skill.id] = (rank, skill)

    for nk, skill in names.items():
        if nk == nq:
            add(skill, 0)
        elif nk.startswith(nq):
            add(skill, 1)
        elif nq in nk:
            add(skill, 4)
    for nk, skill in aliases.items():
        if nk == nq:
            add(skill, 2)
        elif nk.startswith(nq):
            add(skill, 3)
        elif nq in nk:
            add(skill, 4)

    ordered = sorted(ranked.values(), key=lambda item: (item[0], len(item[1].name), item[1].name))
    return [skill for _, skill in ordered[:limit]]
