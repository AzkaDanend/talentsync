import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fetchPublicJobs, type Job, type Page } from "../../lib/api";
import { employmentTypeLabels, formatDate, formatSalary, workModeLabels } from "../../lib/format";
import { Badge, Button, Card, EmptyState, Input, Pagination, Select, Spinner } from "../../components/ui";

export function JobCard({ job }: { job: Job }) {
  return (
    <Card className="transition hover:ring-primary-300">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <Link
            to={`/app/jobs/${job.id}`}
            className="text-base font-bold text-slate-800 hover:text-primary-700"
          >
            {job.title}
          </Link>
          <p className="mt-0.5 text-sm text-slate-500">
            {job.company.name}
            {job.location_text ? ` • ${job.location_text}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5">
          <Badge value={job.employment_type} label={employmentTypeLabels[job.employment_type] ?? job.employment_type} />
          <Badge value={job.work_mode} label={workModeLabels[job.work_mode] ?? job.work_mode} />
        </div>
      </div>
      <p className="mt-2 line-clamp-2 text-sm text-slate-600">{job.description}</p>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span className="font-medium text-slate-700">{formatSalary(job.salary_min, job.salary_max, job.salary_currency)}</span>
        <span>Diposting {formatDate(job.posted_at)}</span>
        {job.closing_at && <span>Batas: {formatDate(job.closing_at)}</span>}
      </div>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {job.skills.slice(0, 6).map((s) => (
          <span
            key={s.name}
            className={
              "rounded-full px-2 py-0.5 text-xs font-medium " +
              (s.required ? "bg-primary-100 text-primary-700" : "bg-slate-100 text-slate-500")
            }
          >
            {s.name}
          </span>
        ))}
      </div>
      <div className="mt-3">
        <Link to={`/app/jobs/${job.id}`}>
          <Button variant="secondary">Lihat Detail</Button>
        </Link>
      </div>
    </Card>
  );
}

export default function JobDiscovery() {
  // Filter hidup di URL (?q=&location=&type=&mode=&page=) — navigasi pulang dari
  // detail lowongan / rekomendasi memulihkan hasil pencarian seperti semula
  // (incident 21 Nov 2026: navigasi job search "tidak mulus", state hilang).
  const [searchParams, setSearchParams] = useSearchParams();
  const [qInput, setQInput] = useState(searchParams.get("q") ?? "");
  const [locInput, setLocInput] = useState(searchParams.get("location") ?? "");
  const [q, setQ] = useState(searchParams.get("q") ?? "");
  const [location, setLocation] = useState(searchParams.get("location") ?? "");
  const [employmentType, setEmploymentType] = useState(searchParams.get("type") ?? "");
  const [workMode, setWorkMode] = useState(searchParams.get("mode") ?? "");
  const [page, setPage] = useState(Number(searchParams.get("page") ?? "1") || 1);
  const [result, setResult] = useState<Page<Job> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Guard race: respons dari request lama tidak boleh menimpa hasil request terbaru.
  const seqRef = useRef(0);

  // Tulis filter ke URL (replace — tidak menambah entri history per ketikan).
  // Via ref agar identitas stabil: setSearchParams berubah identitas setiap URL
  // berubah (react-router v7 deps [navigate, searchParams]) — tanpa ref, efek
  // debounce ikut berjalan ulang setiap navigasi dan MENIMPA page kembali ke 1
  // (terverifikasi di smoke test: klik halaman 2 ditimpa ±350 ms kemudian).
  const syncUrlRef = useRef<(next: {
    q: string;
    location: string;
    type: string;
    mode: string;
    page: number;
  }) => void>(() => {});
  syncUrlRef.current = (next: { q: string; location: string; type: string; mode: string; page: number }) => {
    const params = new URLSearchParams();
    if (next.q) params.set("q", next.q);
    if (next.location) params.set("location", next.location);
    if (next.type) params.set("type", next.type);
    if (next.mode) params.set("mode", next.mode);
    if (next.page > 1) params.set("page", String(next.page));
    setSearchParams(params, { replace: true });
  };

  // Mirror filter select agar timer debounce selalu menulis nilai TERKINI
  // (skenario: ketik → ubah select <350 ms → timer jalan dengan nilai basi).
  const filtersRef = useRef({ employmentType, workMode });
  filtersRef.current = { employmentType, workMode };

  useEffect(() => {
    // Efek ini HANYA untuk input teks (debounce): reset halaman + tulis URL.
    // Deps tidak memuat setSearchParams agar perubahan URL dari
    // select/pagination tidak memicu penulisan ulang yang menimpa state.
    const timer = setTimeout(() => {
      setQ(qInput);
      setLocation(locInput);
      setPage(1);
      syncUrlRef.current({
        q: qInput,
        location: locInput,
        type: filtersRef.current.employmentType,
        mode: filtersRef.current.workMode,
        page: 1,
      });
    }, 350);
    return () => clearTimeout(timer);
    // employmentType/workMode dibaca dari state terkini lewat ref di atas.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qInput, locInput]);

  const load = useCallback(async () => {
    const seq = ++seqRef.current;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPublicJobs({
        q: q || undefined,
        location: location || undefined,
        employment_type: employmentType || undefined,
        work_mode: workMode || undefined,
        page,
        size: 8,
      });
      if (seq !== seqRef.current) return; // sudah ada request lebih baru
      setResult(data);
    } catch (err) {
      if (seq !== seqRef.current) return;
      setError(err instanceof Error ? err.message : "Gagal memuat lowongan.");
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, [q, location, employmentType, workMode, page]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Cari Lowongan</h1>
        <p className="mt-1 text-sm text-slate-500">
          Lowongan aktif dari perusahaan di TalentSync AI.
        </p>
      </div>

      <Card className="grid gap-3 sm:grid-cols-5">
        <Input
          className="sm:col-span-2"
          placeholder="Cari judul pekerjaan…"
          value={qInput}
          onChange={(e) => setQInput(e.target.value)}
        />
        <Input
          placeholder="Lokasi…"
          value={locInput}
          onChange={(e) => setLocInput(e.target.value)}
        />
        <Select
          value={employmentType}
          onChange={(e) => {
            setEmploymentType(e.target.value);
            setPage(1);
            syncUrlRef.current({ q, location, type: e.target.value, mode: workMode, page: 1 });
          }}
        >
          <option value="">Semua Tipe</option>
          {Object.entries(employmentTypeLabels).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </Select>
        <Select
          value={workMode}
          onChange={(e) => {
            setWorkMode(e.target.value);
            setPage(1);
            syncUrlRef.current({ q, location, type: employmentType, mode: e.target.value, page: 1 });
          }}
        >
          <option value="">Semua Mode</option>
          {Object.entries(workModeLabels).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </Select>
      </Card>

      {loading ? (
        <Spinner />
      ) : error ? (
        <p className="text-sm text-red-600">{error}</p>
      ) : result && result.items.length > 0 ? (
        <>
          <p className="text-xs text-slate-400">{result.total} lowongan ditemukan</p>
          <div className="grid gap-4">
            {result.items.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </div>
          <Pagination
            page={result.page}
            pages={result.pages}
            onChange={(next) => {
              setPage(next);
              syncUrlRef.current({ q, location, type: employmentType, mode: workMode, page: next });
            }}
          />
        </>
      ) : (
        <EmptyState
          title="Tidak ada lowongan yang cocok"
          description="Coba ubah kata kunci atau filter pencarian Anda."
        />
      )}
    </div>
  );
}
