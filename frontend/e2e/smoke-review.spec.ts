/**
 * SEMENTARA — smoke review tahap-2 (dihapus setelah verifikasi):
 * 1. Filter/pagination bertahan di URL + pulih saat back-navigation.
 * 2. Empty state pencarian; state needCv rekomendasi; error state saat API mati.
 */

import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const HR = { name: "Rani Reviewer", email: `rani-smoke-${Date.now()}@e2e.dev`, password: "RahasiaKuat1!" };
const CAND = { name: "Sinta Smoke", email: `sinta-smoke-${Date.now()}@e2e.dev`, password: "RahasiaKuat1!" };
const CAND2 = { name: "Bima Smoke", email: `bima-smoke-${Date.now()}@e2e.dev`, password: "RahasiaKuat1!" };

async function apiRegister(request: APIRequestContext, role: "candidate" | "hr", p: { name: string; email: string; password: string }) {
  const res = await request.post("/api/v1/auth/register", {
    data: { role, full_name: p.name, email: p.email, password: p.password },
  });
  expect(res.ok(), `register ${role}: ${res.status()} ${await res.text()}`).toBeTruthy();
  return (await res.json()) as { access_token: string };
}

async function seedHrWithJobs(request: APIRequestContext) {
  const { access_token } = await apiRegister(request, "hr", HR);
  const headers = { Authorization: `Bearer ${access_token}` };
  const company = await request.post("/api/v1/me/company", {
    headers,
    data: { name: "PT Smoke Review", industry: "Teknologi", location: "Jakarta", website: "https://smoke.dev", description: "Perusahaan smoke." },
  });
  expect(company.ok(), await company.text()).toBeTruthy();
  const titles = ["Full-Stack Engineer", "Backend Engineer", "Frontend Engineer", "Data Engineer", "DevOps Engineer", "Mobile Engineer", "QA Engineer", "Platform Engineer", "Security Engineer", "ML Engineer"];
  for (const title of titles) {
    const job = await request.post("/api/v1/hr/jobs", {
      headers,
      data: {
        title,
        description: `Membangun sistem ${title} bersama tim produk.`,
        requirements: "Pengalaman 2+ tahun.",
        skills: [{ name: "Python", required: true }, { name: "React", required: false }],
        employment_type: "full_time",
        work_mode: "remote",
      },
    });
    expect(job.ok(), await job.text()).toBeTruthy();
    const pub = await request.post(`/api/v1/hr/jobs/${((await job.json()) as { id: string }).id}/publish`, { headers });
    expect(pub.ok(), await pub.text()).toBeTruthy();
  }
}

async function registerCandidateViaUi(page: Page, cand = CAND) {
  await page.goto("/register");
  await page.getByRole("button", { name: "Pencari Kerja" }).click();
  await page.getByLabel("Nama Lengkap").fill(cand.name);
  await page.getByLabel("Email").fill(cand.email);
  await page.locator('input[type="password"]').first().fill(cand.password);
  await page.locator('input[type="password"]').nth(1).fill(cand.password);
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Daftar Akun Sekarang" }).click();
  try {
    await expect(page).toHaveURL(/\/app/, { timeout: 15_000 });
  } catch {
    console.log(
      "DBG-REG-URL:",
      page.url(),
      "| ALERT:",
      (await page.locator("body").innerText()).replace(/\s+/g, " ").slice(0, 300),
    );
    throw new Error("registrasi gagal");
  }
}

test("Smoke: filter & pagination pulih via back-navigation", async ({ page, request }) => {
  await seedHrWithJobs(request);
  await registerCandidateViaUi(page);

  await page.getByRole("link", { name: "Cari Lowongan" }).click();
  await expect(page).toHaveURL(/\/app\/jobs$/);
  await page.getByPlaceholder("Cari judul pekerjaan…").fill("Engineer");
  await expect(page.getByText(/lowongan ditemukan/)).toBeVisible({ timeout: 15_000 });
  await expect(page).toHaveURL(/q=Engineer/);

  // Pagination komponen memakai tombol ‹ › (bukan angka)
  await page.getByRole("button", { name: "›", exact: true }).click();
  await expect(page).toHaveURL(/page=2/, { timeout: 15_000 });
  await expect(page.getByText(/lowongan ditemukan/)).toBeVisible();

  // Masuk detail lalu kembali (browser back) → filter & halaman pulih
  await page.getByRole("link", { name: "Backend Engineer" }).first().click();
  await expect(page).toHaveURL(/\/app\/jobs\/[^/]+$/);
  await page.goBack();
  await expect(page).toHaveURL(/q=Engineer/);
  await expect(page).toHaveURL(/page=2/);
  await expect(page.getByText(/lowongan ditemukan/)).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("Halaman 2 dari")).toBeVisible();
});

test("Smoke: empty state, needCv rekomendasi, dan error state API mati", async ({ page }) => {
  // Kandidat TERPISAH dari test pertama (email sama → 400 duplikat registrasi).
  await registerCandidateViaUi(page, CAND2);

  // Empty state pencarian
  await page.getByRole("link", { name: "Cari Lowongan" }).click();
  await page.getByPlaceholder("Cari judul pekerjaan…").fill("zzztidakadaxyz");
  await expect(page.getByText("Tidak ada lowongan yang cocok")).toBeVisible({ timeout: 15_000 });

  // Rekomendasi tanpa CV terkonfirmasi → state edukatif (bukan error mentah)
  await page.getByRole("link", { name: "Rekomendasi", exact: true }).click();
  try {
    await expect(page.getByText("Aktifkan CV utama yang sudah dikonfirmasi")).toBeVisible({ timeout: 15_000 });
  } catch {
    console.log("DBG-NEEDCV-URL:", page.url());
    console.log(
      "DBG-NEEDCV-BODY:",
      (await page.locator("body").innerText()).replace(/\s+/g, " ").slice(0, 500),
    );
    throw new Error("needCv state tidak muncul");
  }

  // Kembali ke halaman lowongan yang sudah termuat, lalu matikan API di tengah sesi
  await page.getByRole("link", { name: "Cari lowongan manual" }).click();
  await expect(page.getByText(/lowongan ditemukan|Tidak ada lowongan/)).toBeVisible({ timeout: 15_000 });
  await page.route("**/api/v1/**", (route) => route.abort());
  await page.getByPlaceholder("Cari judul pekerjaan…").fill("Engineer");
  // Pesan error ditampilkan (axios network error ditampilkan apa adanya —
  // catatan minor UX, bukan crash/spinner).
  await expect(
    page.getByText(/Gagal memuat lowongan|Network Error/i).first(),
  ).toBeVisible({ timeout: 15_000 });

  // Hard-reload dengan API mati → auth gagal pulih → diarahkan ke /login
  // (bukan spinner tanpa akhir) — perilaku degrade yang dapat dipahami.
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(`PAGEERROR: ${err.message}`));
  page.on("requestfailed", (req) =>
    errors.push(`REQFAILED: ${req.url().slice(0, 140)} :: ${req.failure()?.errorText}`),
  );
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`CONSOLE: ${msg.text().slice(0, 200)}`);
  });
  await page.goto("/app/jobs");
  await page.waitForTimeout(4000);
  for (const e of errors.slice(0, 8)) console.log("DBG-ERR:", e);
  await page.screenshot({ path: "D:/portfolio/AI/talentsync/frontend/dbg-apidown.png", fullPage: true });
  console.log("DBG-URL:", page.url());
  console.log("DBG-HTML:", (await page.content()).slice(0, 300).replace(/\s+/g, " "));
  console.log("DBG-BODY:", (await page.locator("body").innerText()).replace(/\s+/g, " ").slice(0, 400));
  await expect(page).toHaveURL(/login/, { timeout: 15_000 });
});
