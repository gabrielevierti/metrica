// Types for the METRICA scene graph (metrica-scene/1) and API payloads.
export type Vec3 = [number, number, number];

export interface ScaleRecord {
  status: "ARBITRARY" | "RELATIVE" | "ESTIMATED_METRIC" | "CALIBRATED_METRIC";
  unit: string;
  method: string;
  scale_sigma_rel: number | null;
  assumptions: string[];
  references: any[];
  explanation?: string;
}

export interface LayerMeta {
  id: string;
  unit: string;
  kind: "mesh" | "points" | "coverage";
  name?: string;
  files: Record<string, string>;
  stats: any;
  method?: string;
  corners_xz?: [number, number][];
  default_visible: boolean;
  legend?: Record<string, string>;
}

export interface Dimension {
  value: number;
  sigma?: number | null;
  unit?: string;
  basis?: string;
  note?: string;
}

export interface Entity {
  id: string;
  category: string;
  display?: string;
  unit?: string;
  geometry: any;
  dimensions?: Record<string, Dimension>;
  evidence?: { class: string; confidence?: number; views?: string[]; observations?: number; [k: string]: any };
  provenance?: { method?: string; algorithm?: string };
  source_observations?: any[];
  openings?: any[];
  inferred_patches?: any[];
  inferred_parts?: any[];
  texture?: { file: string; u_range: [number, number]; v_range: [number, number]; observed_fraction: number };
  relations?: any[];
  flags?: string[];
  regions?: Record<string, any>;
  tracks?: string[];
  shape_note?: string;
  note?: string;
}

export interface Sample {
  f: number;
  t: number;
  p: Vec3;
  s: number | null;
  conf: number;
  flags: string[];
  pose: string;
  b?: number[];
}

export interface Agent {
  id: string;
  video: string;
  category: string;
  group: string;
  samples: Sample[];
  smoothed?: { t: number; p: Vec3 }[];
  speed?: { t: number; v: number; s: number }[];
  summary?: any;
  stature?: { median: number; iqr: number; n: number };
  first_t: number;
  last_t: number;
  observed_count: number;
  gaps: any[];
  flags: any[];
  links: any[];
  crops: any[];
  faces: any[];
  color?: any;
  ocr_candidates?: any[];
  representation: any;
  stationary?: boolean;
  static_entity?: string;
}

export interface CameraMeta {
  id: string;
  video: string;
  kind: "static" | "moving" | "rotating";
  intrinsics: { fx: number; fy: number; cx: number; cy: number; width: number; height: number; hfov_deg: number };
  world_from_cam: number[][];
  poses?: { frames: number[]; t: number[]; world_from_cam: number[][][]; source: string[] };
  color: string;
  mode: string;
}

export interface VideoMeta {
  id: string;
  name: string;
  sha256: string;
  proxy: string;
  thumbnail: string;
  fps: number;
  frames: number;
  duration: number;
  width: number;
  height: number;
  offset: number;
  mode: string;
  unit: string;
  sync?: any;
  vfr: boolean;
  quality_flags: string[];
}

export interface SceneEvent {
  id: string;
  type: string;
  t: number;
  t_end?: number;
  video?: string;
  frame?: number | null;
  tracks?: string[];
  description: string;
  confidence: number;
  hypothesis?: boolean;
  requires_verification?: boolean;
  position?: Vec3 | null;
  status?: string;
}

export interface Scene {
  schema: string;
  case: { id: string; title: string; reference: string };
  coordinate_system: {
    unit: string;
    status: ScaleRecord["status"];
    status_text: string;
    scale: ScaleRecord;
    global_scale_factor: number;
    reference_video: string;
  };
  time: { reference_video: string; offsets: Record<string, number>; start: number; end: number };
  videos: VideoMeta[];
  cameras: CameraMeta[];
  layers: LayerMeta[];
  entities: Entity[];
  agents: Agent[];
  events: SceneEvent[];
  alignment: Record<string, any>;
  conflicts: any[];
  grouping?: { groups: string[][]; pairs: any[]; note: string } | null;
  warnings: { source: string; message: string }[];
  evidence_classes: Record<string, string>;
  categories: Record<string, { display: string; color: string; group: string }>;
  scene_labels: string[];
  stats: any;
}

export interface CaseSummary {
  path: string;
  id: string;
  title: string;
  created: string;
  reference: string;
  videos: number;
  status: string;
}

export interface StageStatus {
  name: string;
  title: string;
  status: string;
  duration_s?: number | null;
  warnings: string[];
  fallbacks: any[];
  error?: string | null;
  metrics?: any;
}

export interface CaseDetail {
  id: string;
  title: string;
  reference: string;
  created: string;
  path: string;
  videos: any[];
  status: any;
  stages: StageStatus[];
  job: any | null;
  has_scene: boolean;
  reports: string[];
}

export type Selection =
  | { kind: "entity"; id: string }
  | { kind: "agent"; id: string }
  | { kind: "event"; id: string }
  | { kind: "camera"; id: string }
  | null;

export const EVIDENCE = [
  "OBSERVED_MULTI_VIEW",
  "OBSERVED_SINGLE_VIEW",
  "FITTED_PRIMITIVE",
  "CLASS_PRIOR",
  "INFERRED",
  "UNKNOWN",
] as const;

export const EVIDENCE_COLORS: Record<string, string> = {
  OBSERVED_MULTI_VIEW: "#2ea05a",
  OBSERVED_SINGLE_VIEW: "#5a82dc",
  FITTED_PRIMITIVE: "#e6aa28",
  CLASS_PRIOR: "#be5ac8",
  INFERRED: "#dc5050",
  UNKNOWN: "#969696",
};

export const EVIDENCE_SHORT: Record<string, string> = {
  OBSERVED_MULTI_VIEW: "Observed (multi-view)",
  OBSERVED_SINGLE_VIEW: "Observed (single view)",
  FITTED_PRIMITIVE: "Fitted primitive",
  CLASS_PRIOR: "Class prior",
  INFERRED: "Inferred",
  UNKNOWN: "Unknown",
};
