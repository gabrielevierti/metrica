// Thin, typed client for the local METRICA API.
import type { CaseDetail, CaseSummary, Scene } from "./types";

const analyst = () => localStorage.getItem("metrica.analyst") || "analyst";

async function req<T>(method: string, url: string, body?: any): Promise<T> {
  const init: RequestInit = { method, headers: { "X-Metrica-Analyst": analyst() } };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as any)["Content-Type"] = "application/json";
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  const ct = r.headers.get("content-type") || "";
  return (ct.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export const api = {
  system: () => req<any>("GET", "/api/system"),
  models: () => req<any>("GET", "/api/models"),
  downloadModels: (opts: { keys?: string[]; include_optional?: boolean; accept_restricted?: boolean }) =>
    req<any>("POST", "/api/models/download", opts),
  cases: () => req<CaseSummary[]>("GET", "/api/cases"),
  createCase: (title: string, reference: string) => req<CaseDetail>("POST", "/api/cases", { title, reference }),
  case: (id: string) => req<CaseDetail>("GET", `/api/cases/${id}`),
  importPath: (id: string, path: string, copy = true) =>
    req<any>("POST", `/api/cases/${id}/videos/path`, { path, copy_into_case: copy }),
  analyze: (id: string, profile: string, overrides: Record<string, any>, force = false) =>
    req<any>("POST", `/api/cases/${id}/analyze`, { profile, overrides, force }),
  job: (jid: string) => req<any>("GET", `/api/jobs/${jid}`),
  cancel: (jid: string) => req<any>("POST", `/api/jobs/${jid}/cancel`),
  scene: (id: string) => req<Scene>("GET", `/api/cases/${id}/scene`),
  measure: (id: string, body: any) => req<any>("POST", `/api/cases/${id}/measurements`, body),
  measurements: (id: string) => req<any[]>("GET", `/api/cases/${id}/measurements`),
  withdraw: (id: string, mid: string) => req<any>("POST", `/api/cases/${id}/measurements/${mid}/withdraw`),
  calibrate: (id: string, body: any) => req<any>("POST", `/api/cases/${id}/calibration`, body),
  calibOff: (id: string, ref: string) => req<any>("POST", `/api/cases/${id}/calibration/${ref}/deactivate`),
  sync: (id: string, video: string, offset_s: number) => req<any>("POST", `/api/cases/${id}/sync`, { video, offset_s }),
  verify: (id: string, target: string, verdict: string, note: string) =>
    req<any>("POST", `/api/cases/${id}/verifications`, { target, verdict, note }),
  annotations: (id: string, kind: string) => req<any>("GET", `/api/cases/${id}/annotations/${kind}`),
  note: (id: string, text: string, target?: string) => req<any>("POST", `/api/cases/${id}/notes`, { text, target }),
  report: (id: string, analyst: string, purpose: string) =>
    req<any>("POST", `/api/cases/${id}/reports`, { analyst, purpose, pdf: true }),
  reportTask: (tid: string) => req<any>("GET", `/api/report-tasks/${tid}`),
  reports: (id: string) => req<any[]>("GET", `/api/cases/${id}/reports`),
  audit: (id: string) => req<any>("GET", `/api/cases/${id}/audit`),
  verifyEvidence: (id: string) => req<any>("GET", `/api/cases/${id}/verify`),
  snapshot: (id: string, blob: Blob, caption: string) => {
    const fd = new FormData();
    fd.append("file", blob, "snapshot.png");
    fd.append("caption", caption);
    return req<any>("POST", `/api/cases/${id}/snapshots`, fd);
  },
  upload: (id: string, file: File, onProgress?: (f: number) => void) =>
    new Promise<any>((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("POST", `/api/cases/${id}/videos`);
      x.setRequestHeader("X-Metrica-Analyst", analyst());
      x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      x.onload = () => {
        if (x.status >= 200 && x.status < 300) resolve(JSON.parse(x.responseText));
        else {
          try {
            reject(new Error(JSON.parse(x.responseText).detail));
          } catch {
            reject(new Error(x.statusText));
          }
        }
      };
      x.onerror = () => reject(new Error("upload failed"));
      const fd = new FormData();
      fd.append("file", file);
      x.send(fd);
    }),
};

export const fileUrl = (caseId: string, rel: string) => `/api/cases/${caseId}/files/${rel}`;
/** Files referenced from scene.json (layers, textures) are relative to derived/scene/. */
export const sceneFileUrl = (caseId: string, rel: string) => fileUrl(caseId, `scene/${rel}`);
export const frameUrl = (caseId: string, vid: string, frame: number) => `/api/cases/${caseId}/frame/${vid}/${frame}`;

export async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}
