/* =============================================================================
   cameraRegistry.js  — the single seam between DEMO and PRODUCTION.
   -----------------------------------------------------------------------------
   The dashboard NEVER owns the camera list. It asks this module. Today this is a
   client-side stand-in for the backend; it models the real Camera lifecycle from
   predictor.py:  GATHERING_CONTEXT (needs 60 frames) -> CALIBRATING (needs 15
   windows) -> LIVE. Cameras you add start cold and warm up honestly.

   TO GO LIVE: replace the bodies below.
     addCamera({zone, source})  ->  await fetch("/api/cameras", {POST, body:{source,zone}})
     removeCamera(id)           ->  await fetch(`/api/cameras/${id}`, {DELETE})
     advance/snapshot           ->  delete them; the WebSocket payload becomes the
                                    source of truth (useSentinelSocket supplies cameras).
   The dashboard code does not change — only this file does.
   ========================================================================== */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const ZONE_POOL = [
  "North Stand", "Gate A", "Concourse", "South Ramp", "East Wing", "West Gate",
  "Lower Bowl", "Tunnel 2", "Upper Tier", "Plaza", "Platform 1", "Exit 3",
];

// demo lifecycle timing (ticks ~1.5s). Real timing is governed by frame arrival.
const FRAMES_PER_TICK = 30;   // ~30fps over a 1.5s tick window
const CONTEXT_FRAMES  = 60;   // Camera.context_size
const CALIB_TICKS     = 3;    // compressed stand-in for 15 calibration windows

let cameras = [];   // internal descriptors (NOT what the UI sees)
let listeners = [];
let counter = 0;

const emit = () => listeners.forEach((l) => l());
export const subscribe = (fn) => { listeners.push(fn); return () => { listeners = listeners.filter((l) => l !== fn); }; };

// deterministic 0..1 from an id so each camera has a stable "personality"
function seedOf(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) % 100000;
  return h / 100000;
}

function makeGrid(cols, rows, bias) {
  const g = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) row.push(clamp(Math.sin(c / 3) * 0.2 + Math.random() * 0.5 + bias * 0.6, 0, 1));
    g.push(row);
  }
  return g;
}

function makeDescriptor({ zone, source, live = false }) {
  counter += 1;
  const id = `Camera_${String(counter).padStart(3, "0")}`;
  return {
    id,
    zone: zone || ZONE_POOL[(counter - 1) % ZONE_POOL.length],
    source: source || "",
    status: live ? "LIVE" : "GATHERING_CONTEXT",
    framesSeen: live ? 9999 : 0,
    calibTicks: 0,
    grid: makeGrid(12, 7, 0.2),
  };
}

/* ---- public API the dashboard uses ---- */
export const listCameras = () => cameras;

export function seedCameras(n) {
  if (cameras.length) return;
  for (let i = 0; i < n; i++) cameras.push(makeDescriptor({ live: true }));
  emit();
}

export function addCamera({ zone, source } = {}) {
  const d = makeDescriptor({ zone, source, live: false }); // starts cold, warms up
  cameras.push(d);
  emit();
  return d.id;
}

export function removeCamera(id) {
  cameras = cameras.filter((c) => c.id !== id);
  emit(); // in production this also frees the GPU thread server-side
}

// view object the UI renders (never leaks internal descriptor fields)
function view(c, bias) {
  if (c.status !== "LIVE") {
    return { id: c.id, zone: c.zone, status: c.status, grid: c.grid, riskScore: 0, load: 0, count: 0, density: 0, isAnomaly: false };
  }
  const personality = seedOf(c.id);
  const riskScore = clamp(bias * (0.5 + personality * 0.95) + (Math.random() - 0.5) * 0.05, 0.03, 0.99);
  const load = Math.round(clamp(riskScore * 100 + 8, 3, 99));
  const count = Math.round(70 + riskScore * 430);
  const density = +(0.5 + riskScore * 4 + (Math.random() - 0.5) * 0.15).toFixed(1);
  return { id: c.id, zone: c.zone, status: "LIVE", grid: c.grid, riskScore, load, count, density, isAnomaly: riskScore > 0.72 };
}

// snapshot WITHOUT advancing — used for instant UI updates on add/remove
export const snapshotCameras = (bias = 0.5) => cameras.map((c) => view(c, bias));

// advance the lifecycle one processing tick — used by the feed loop
export function advanceCameras(bias = 0.5) {
  cameras.forEach((c) => {
    if (c.status === "LIVE") return;
    c.framesSeen += FRAMES_PER_TICK;
    if (c.framesSeen < CONTEXT_FRAMES) {
      c.status = "GATHERING_CONTEXT";
    } else if (c.calibTicks < CALIB_TICKS) {
      c.status = "CALIBRATING";
      c.calibTicks += 1;
    } else {
      c.status = "LIVE";
      c.grid = makeGrid(12, 7, clamp(bias * (0.5 + seedOf(c.id) * 0.95), 0.05, 0.97));
    }
  });
  return snapshotCameras(bias);
}