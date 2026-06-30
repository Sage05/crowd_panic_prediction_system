import { useState, useEffect, useMemo, useRef } from "react";
import {
  seedCameras, addCamera, removeCamera,
  advanceCameras, snapshotCameras, subscribe, listCameras,
} from "./cameraRegistry";
import "./PanicPredictionDashboard.css";
import { useLiveSocket } from "./useLiveSocket"

/* =============================================================================
   PanicPredictionDashboard.jsx  (multi-camera, count-agnostic)
   -----------------------------------------------------------------------------
   The UI is a pure function of one `state` object — the contract between your
   ML pipeline -> FastAPI -> this component. Cameras are a DYNAMIC LIST: render
   whatever the backend sends (2, 12, N). A dropdown controls how many tiles are
   shown at once; pagination reaches the rest.

   Per-camera fields below mirror what predictor.py's Camera object yields, after
   build_payload() on the backend converts the raw result:
     camera_id          -> id
     (config lookup)     -> zone        (Camera has no zone; backend maps it)
     is_anomaly          -> isAnomaly   (sustained-breach panic flag)
     window_mse/thresh   -> riskScore   (0..1; backend normalizes the ratio)
     sum(current grid)   -> count        (CSRNet density integrates to a count)
     count / area        -> density      (ppl/m²; needs area calibration)
     current grid        -> grid         (live density heatmap)
     forecasted_grids    -> (feeds the +10/+20 forecast tabs, see DensityMap)
   ========================================================================== */

const C = {
  amber: "#e0a72c", amberSoft: "#3a2f17",
  green: "#3fae7e", greenSoft: "#16271f",
  red:   "#e0574f", redSoft:  "#2c1818",
  blue:  "#5b9bd5",
  text:  "#e9edf2", textDim: "#8a929e", textFaint: "#5a626d",
  borderSoft: "#161c25",
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const riskColor = (risk) =>
  risk === "Critical" || risk === "Severe" || risk === "High" ? C.red
  : risk === "Moderate" ? C.amber : C.green;
const loadColor = (v) => (v >= 70 ? C.red : v >= 50 ? C.amber : C.green);
const cellColor = (v) => (v > 0.66 ? C.red : v > 0.4 ? C.amber : C.green);

/* =============================================================================
   SCENARIO PRESETS — three model "moods". Each sets an overall risk bias; the
   per-camera values and all aggregates are DERIVED from it, so the dashboard
   stays coherent for any camera count.
   ========================================================================== */
const SCENARIOS = {
  Calm:     { riskClass: "Low",      panicBias: 0.16, confidence: 96, trend: -4 },
  Elevated: { riskClass: "Moderate", panicBias: 0.52, confidence: 94, trend: 16 },
  Critical: { riskClass: "Critical", panicBias: 0.82, confidence: 91, trend: 38 },
};

const CAPACITY = 8000;
const SEED_CAMERAS = 6; // how many cameras the "backend" starts with

function makeGrid(cols, rows, bias) {
  const g = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      row.push(clamp(Math.sin(c / 3) * 0.2 + Math.random() * 0.5 + bias * 0.6, 0, 1));
    }
    g.push(row);
  }
  return g;
}

/* =============================================================================
   SIMULATED FEED — REPLACE WITH useSentinelSocket() FOR PRODUCTION.
   Everything aggregate (people count, panic risk, zones, alerts) is DERIVED
   from the camera list, so it scales to any N.
   ========================================================================== */
function useSimulatedFeed(scenarioKey) {
  const base = SCENARIOS[scenarioKey];

  const densityGrid = useMemo(() => makeGrid(22, 6, base.panicBias), [scenarioKey]); // eslint-disable-line

  const historyRef = useRef({ key: null, data: [] });
  if (historyRef.current.key !== scenarioKey) {
    historyRef.current = {
      key: scenarioKey,
      data: Array.from({ length: 30 }, (_, i) =>
        clamp(base.panicBias * (0.45 + 0.55 * (i / 29)), 0.04, 0.99)
      ),
    };
  }

  // build the full payload from a camera list the registry hands us
  const buildFrom = (cameras) => {
    const now = new Date();
    const hh = now.toLocaleTimeString("en-GB");
    const confidence = Math.round(base.confidence + (Math.random() - 0.5) * 1.5);
    const live = cameras.filter((c) => c.status === "LIVE");

    const peopleCount = cameras.reduce((a, c) => a + c.count, 0);
    const maxRisk = live.length ? Math.max(...live.map((c) => c.riskScore)) : 0;
    const panicRisk = Math.round(maxRisk * 100);
    const riskClass = panicRisk > 75 ? "Critical" : panicRisk > 40 ? "Moderate" : "Low";
    const systemStatus = panicRisk > 75 ? "Critical" : panicRisk > 40 ? "Elevated" : "Calm";

    const zmap = {};
    live.forEach((c) => { if (!zmap[c.zone]) zmap[c.zone] = []; zmap[c.zone].push(c.load); });
    const zones = Object.entries(zmap)
      .map(([name, l]) => ({ name, load: Math.round(l.reduce((a, b) => a + b, 0) / l.length) }))
      .sort((a, b) => b.load - a.load)
      .slice(0, 6);

    const anom = live.filter((c) => c.isAnomaly);
    const alerts = [];
    anom.slice(0, 4).forEach((c) =>
      alerts.push({ time: hh, level: "WARN", zone: c.zone, msg: `Sustained density breach — ${c.density} ppl/m² on ${c.id}` })
    );
    const warming = cameras.filter((c) => c.status !== "LIVE");
    if (warming.length)
      alerts.push({ time: hh, level: "INFO", zone: "System", msg: `${warming.length} camera(s) warming up — predictions pending` });
    if (!anom.length && !warming.length)
      alerts.push({ time: hh, level: "INFO", zone: "System", msg: "Flow nominal across all monitored zones" });
    alerts.push({ time: hh, level: "OK", zone: "System", msg: `Model live — ${live.length}/${cameras.length} cameras · ${confidence}% confidence` });

    const next = clamp(base.panicBias * (0.85 + Math.random() * 0.3), 0.04, 0.99);
    historyRef.current.data = [...historyRef.current.data.slice(1), next];

    return {
      site: "Central Stadium — Sector 4", timestamp: now, fps: 30, confidence,
      systemStatus, riskClass, panicRisk, peopleCount, capacity: CAPACITY, trend: base.trend,
      cameras, zones, alerts,
      panicHistory: historyRef.current.data, densityGrid,
    };
  };

  const [payload, setPayload] = useState(() => buildFrom(snapshotCameras(base.panicBias)));

  useEffect(() => {
    if (listCameras().length === 0) seedCameras(SEED_CAMERAS);

    // instant re-render when a camera is added or removed (no waiting for a tick)
    const onChange = () => setPayload(buildFrom(snapshotCameras(base.panicBias)));
    const unsub = subscribe(onChange);
    onChange();

    // the "processing loop": advance each camera's lifecycle once per tick
    const id = setInterval(() => setPayload(buildFrom(advanceCameras(base.panicBias))), 1500);
    return () => { clearInterval(id); unsub(); };
  }, [scenarioKey]); // eslint-disable-line

  return payload;
}

/* ------------------------------ small atoms ------------------------------ */
function Dot({ color, pulse }) {
  return <span className={`ppd-dot${pulse ? " ppd-dot--pulse" : ""}`} style={{ background: color, boxShadow: `0 0 6px ${color}` }} />;
}
function Bar({ value, color, height = 4 }) {
  return (
    <div className="ppd-bar" style={{ height }}>
      <div className="ppd-bar__fill" style={{ width: `${value}%`, background: color }} />
    </div>
  );
}

/* ------------------------------- header ---------------------------------- */
function Header({ s, scenario, setScenario }) {
  const hh = s.timestamp.toLocaleTimeString("en-GB");
  const dd = s.timestamp.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  return (
    <header className="ppd-header">
      <div className="ppd-header__brand">
        <div className="ppd-logo"><div className="ppd-logo__ring" /></div>
        <div>
          <div className="ppd-title">Crowd Monitor</div>
          <div className="ppd-subtitle">{s.site}</div>
        </div>
      </div>

      <div className="ppd-scenario-wrap">
        <span className="ppd-scenario-label">Scenario</span>
        {Object.keys(SCENARIOS).map((k) => {
          const active = scenario === k;
          const col = riskColor(SCENARIOS[k].riskClass);
          return (
            <button key={k} className="ppd-scenario-btn" onClick={() => setScenario(k)}
              style={{ color: active ? C.text : C.textDim, background: active ? C.amberSoft : "transparent", borderColor: active ? col : "transparent" }}>
              <Dot color={active ? col : C.textFaint} /> {k}
            </button>
          );
        })}
      </div>

      <div className="ppd-header__right">
        <div className="ppd-model">
          <Dot color={C.green} pulse />
          <span className="ppd-dim">Model live · <span style={{ color: C.text }}>{s.confidence}% confidence</span></span>
        </div>
        <div className="ppd-clock">
          <div className="ppd-clock__time">{hh}</div>
          <div className="ppd-clock__date">{dd} · {s.fps} fps</div>
        </div>
      </div>
    </header>
  );
}

/* ---------------------------- camera wall -------------------------------- */
function CameraTile({ cam, onRemove }) {
  const col = cam.isAnomaly ? C.red : loadColor(cam.load);
  const badge = cam.isAnomaly
    ? { t: "ANOMALY", c: C.red, bg: C.redSoft }
    : cam.status === "CALIBRATING"
    ? { t: "CALIBRATING", c: C.amber, bg: C.amberSoft }
    : cam.status === "GATHERING_CONTEXT"
    ? { t: "WARMING UP", c: C.textDim, bg: C.borderSoft }
    : { t: "LIVE", c: C.green, bg: C.greenSoft };
  const cols = cam.grid[0].length;
  const warming = cam.status !== "LIVE";
  return (
    <div className={`ppd-cam-tile${cam.isAnomaly ? " ppd-cam-tile--alert" : ""}`}>
      <div className="ppd-cam-tile__head">
        <span className="ppd-cam-label"><Dot color={C.red} pulse /> {cam.id} · {cam.zone}</span>
        <div className="ppd-cam-tile__head-right">
          <span className="ppd-badge" style={{ color: badge.c, background: badge.bg }}>{badge.t}</span>
          <button className="ppd-cam-remove" title={`Remove ${cam.id}`} onClick={() => onRemove(cam.id)}>×</button>
        </div>
      </div>

      <div className="ppd-cam-tile__heat" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, opacity: warming ? 0.35 : 1 }}>
        {cam.grid.flatMap((row, r) =>
          row.map((v, c) => (
            <div key={`${r}-${c}`} className="ppd-cell" style={{ background: cellColor(v), opacity: 0.4 + v * 0.55 }} />
          ))
        )}
        {warming
          ? <span className="ppd-cam-tile__density ppd-mono">{badge.t}</span>
          : <span className="ppd-cam-tile__density ppd-mono">{cam.density} ppl/m²</span>}
      </div>

      <div className="ppd-cam-tile__foot">
        <div className="ppd-row-between" style={{ fontSize: 11.5, marginBottom: 6 }}>
          <span className="ppd-dim">Load</span>
          <b className="ppd-mono" style={{ color: warming ? C.textFaint : col }}>{warming ? "—" : `${cam.load}%`}</b>
        </div>
        <Bar value={warming ? 0 : cam.load} color={col} />
      </div>
    </div>
  );
}

function AddFeedForm({ onAdd, onClose }) {
  const [zone, setZone] = useState("");
  const [source, setSource] = useState("");
  const submit = () => {
    if (!source.trim()) return;          // a source is required — it feeds receive_frames()
    onAdd({ zone: zone.trim(), source: source.trim() });
    onClose();
  };
  return (
    <div className="ppd-panel ppd-addform">
      <div className="ppd-addform__row">
        <input className="ppd-input" placeholder="Zone / location (e.g. East Wing)" value={zone}
          onChange={(e) => setZone(e.target.value)} />
        <input className="ppd-input" placeholder="Stream source — RTSP URL, file path, or webcam index"
          value={source} onChange={(e) => setSource(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()} autoFocus />
      </div>
      <div className="ppd-addform__actions">
        <button className="ppd-btn ppd-btn--ghost" onClick={onClose}>Cancel</button>
        <button className="ppd-btn ppd-btn--primary" onClick={submit} disabled={!source.trim()}>Add feed</button>
      </div>
    </div>
  );
}

function CameraWall({ cameras }) {
  const total = cameras.length;
  const options = useMemo(() => {
    const o = [];
    for (let k = 2; k <= total; k += 2) o.push(k);
    if (total > 0 && o[o.length - 1] !== total) o.push(total); // ensure the real count is selectable
    return o.length ? o : [Math.max(1, total)];
  }, [total]);

  const [visible, setVisible] = useState(() => Math.min(6, total) || 1);
  const [page, setPage] = useState(0);
  const [showForm, setShowForm] = useState(false);

  // keep paging valid when the backend changes how many cameras it streams
  useEffect(() => { setVisible((v) => Math.min(v, total) || total); }, [total]);

  const pages = Math.max(1, Math.ceil(total / visible));
  const safePage = Math.min(page, pages - 1);
  const start = safePage * visible;
  const shown = cameras.slice(start, start + visible);

  const handleAdd = (feed) => {
    addCamera(feed);
    setPage(Math.ceil((total + 1) / visible) - 1); // jump to the page the new tile lands on
  };

  return (
    <>
      <div className="ppd-panel ppd-cam-bar">
        <div className="ppd-cam-controls__left">
          <span className="ppd-section-title">Camera feeds</span>
          <span className="ppd-count-badge">
            {total ? `${start + 1}–${Math.min(start + visible, total)} of ${total}` : "0 cameras"}
          </span>
        </div>

        <div className="ppd-cam-controls__right">
          <button className="ppd-btn ppd-btn--add" onClick={() => setShowForm((v) => !v)}>
            <span className="ppd-btn__plus">+</span> Add feed
          </button>

          {total > 0 && (
            <>
              <label className="ppd-dim" style={{ fontSize: 12 }}>Show</label>
              <select className="ppd-select" value={visible}
                onChange={(e) => { setVisible(Number(e.target.value)); setPage(0); }}>
                {options.map((o) => (
                  <option key={o} value={o}>{o === total ? `All (${o})` : o}</option>
                ))}
              </select>
            </>
          )}

          {pages > 1 && (
            <div className="ppd-pager">
              <button className="ppd-pager__btn" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={safePage === 0}>‹</button>
              <span className="ppd-mono" style={{ fontSize: 12 }}>{safePage + 1}/{pages}</span>
              <button className="ppd-pager__btn" onClick={() => setPage((p) => Math.min(pages - 1, p + 1))} disabled={safePage === pages - 1}>›</button>
            </div>
          )}
        </div>
      </div>

      {showForm && <AddFeedForm onAdd={handleAdd} onClose={() => setShowForm(false)} />}

      {total === 0 ? (
        <div className="ppd-panel ppd-cam-empty">No camera feeds yet. Use “+ Add feed” to register one.</div>
      ) : (
        <div className="ppd-cam-grid">
          {shown.map((c) => <CameraTile key={c.id} cam={c} onRemove={removeCamera} />)}
        </div>
      )}
    </>
  );
}

/* ---------------------------- sidebar cards ------------------------------ */
function StatusCard({ s }) {
  const col = riskColor(s.riskClass);
  return (
    <div className="ppd-panel ppd-status">
      <div><div className="ppd-kicker">System status</div><div className="ppd-status__big" style={{ color: col }}>{s.systemStatus}</div></div>
      <div style={{ textAlign: "right" }}><div className="ppd-kicker">Risk class</div><div className="ppd-status__big" style={{ color: col }}>{s.riskClass}</div></div>
    </div>
  );
}

function Gauge({ value, color }) {
  const r = 38, circ = 2 * Math.PI * r, off = circ * (1 - value / 100);
  return (
    <svg width="100" height="100" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r={r} fill="none" stroke={C.borderSoft} strokeWidth="9" />
      <circle cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round"
        strokeDasharray={circ} strokeDashoffset={off} transform="rotate(-90 50 50)"
        style={{ transition: "stroke-dashoffset .6s ease, stroke .35s ease" }} />
      <text x="50" y="50" textAnchor="middle" dominantBaseline="central" fill={C.text} fontSize="22" fontWeight="700">{value}%</text>
      <text x="50" y="66" textAnchor="middle" fill={C.textDim} fontSize="8.5" fontFamily="var(--ppd-mono)">panic risk</text>
    </svg>
  );
}

function CountCard({ s }) {
  const col = riskColor(s.riskClass);
  const pct = Math.round((s.peopleCount / s.capacity) * 100);
  const up = s.trend >= 0;
  return (
    <div className="ppd-panel ppd-count">
      <Gauge value={s.panicRisk} color={col} />
      <div style={{ flex: 1, minWidth: 150 }}>
        <div className="ppd-kicker">Live people count</div>
        <div className="ppd-count__num">{s.peopleCount.toLocaleString()}</div>
        <div className="ppd-count__trend">
          <span style={{ color: up ? C.amber : C.green }}>{up ? "▲" : "▼"} {Math.abs(s.trend)}%</span>
          <span className="ppd-dim">vs 5 min ago</span>
        </div>
        <Bar value={pct} color={col} />
        <div className="ppd-count__cap">{pct}% of {s.capacity.toLocaleString()} capacity</div>
      </div>
    </div>
  );
}

const LEVEL = { WARN: { c: C.amber, bg: C.amberSoft }, INFO: { c: C.blue, bg: "#16202c" }, OK: { c: C.green, bg: C.greenSoft } };
function Alerts({ s }) {
  const warn = s.alerts.filter((a) => a.level === "WARN").length;
  return (
    <div className="ppd-panel ppd-alerts">
      <div className="ppd-card-head ppd-alerts__head"><span>Active alerts</span><span className="ppd-count-badge">{warn}</span></div>
      <div className="ppd-alerts__list">
        {s.alerts.map((a, i) => {
          const lv = LEVEL[a.level] || LEVEL.INFO;
          return (
            <div key={i} className="ppd-alert-row" style={{ borderLeft: `2px solid ${lv.c}` }}>
              <div className="ppd-alert-row__meta">
                <span className="ppd-alert-row__time">{a.time}</span>
                <span className="ppd-level-tag" style={{ color: lv.c, background: lv.bg }}>{a.level}</span>
                <span className="ppd-alert-row__zone">{a.zone}</span>
              </div>
              <div className="ppd-alert-row__msg">{a.msg}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---------------------------- bottom panels ------------------------------ */
function DensityMap({ s }) {
  const [view, setView] = useState("Live");
  // NOTE: in production, +10/+20 should read camera.forecasted_grids[k] aggregated
  // across cameras. Here we just bias the live grid to fake the forecast.
  const tint = (v) => {
    const bias = view === "+20 min" ? 0.18 : view === "+10 min" ? 0.09 : 0;
    return cellColor(clamp(v + bias, 0, 1));
  };
  return (
    <div className="ppd-panel">
      <div className="ppd-section-head">
        <span className="ppd-section-title">Crowd density map</span>
        <div className="ppd-tabs">
          {["Live", "+10 min", "+20 min"].map((v) => (
            <button key={v} className="ppd-tab" onClick={() => setView(v)}
              style={{ background: view === v ? C.green : "transparent", color: view === v ? "#06120c" : C.textDim, fontWeight: view === v ? 600 : 400 }}>{v}</button>
          ))}
        </div>
      </div>
      <div className="ppd-grid-cells" style={{ gridTemplateColumns: `repeat(${s.densityGrid[0].length}, 1fr)` }}>
        {s.densityGrid.flatMap((row, r) => row.map((v, c) => (
          <div key={`${r}-${c}`} className="ppd-cell" style={{ background: tint(v), opacity: 0.55 + v * 0.45 }} />
        )))}
      </div>
      <div className="ppd-legend">
        <div className="ppd-legend__items">
          <span className="ppd-legend__item"><Dot color={C.green} /> Safe</span>
          <span className="ppd-legend__item"><Dot color={C.amber} /> Busy</span>
          <span className="ppd-legend__item"><Dot color={C.red} /> Critical</span>
        </div>
        <span className="ppd-mono">Live density · refreshed 47s</span>
      </div>
    </div>
  );
}

function ZoneLoad({ s }) {
  return (
    <div className="ppd-panel">
      <div className="ppd-section-title" style={{ marginBottom: 16 }}>Zone load</div>
      <div className="ppd-zones">
        {s.zones.map((z) => (
          <div key={z.name}>
            <div className="ppd-zone__row"><span style={{ color: C.text }}>{z.name}</span><b className="ppd-mono" style={{ color: loadColor(z.load) }}>{z.load}%</b></div>
            <Bar value={z.load} color={loadColor(z.load)} height={5} />
          </div>
        ))}
      </div>
    </div>
  );
}

function PanicChart({ s }) {
  const peak = Math.round(Math.max(...s.panicHistory) * 100);
  return (
    <div className="ppd-panel">
      <div className="ppd-section-head"><span className="ppd-section-title">Panic risk · 30 min</span><b className="ppd-mono" style={{ color: riskColor(s.riskClass) }}>{peak}%</b></div>
      <div className="ppd-chart-bars">
        {s.panicHistory.map((v, i) => (
          <div key={i} className="ppd-chart-bar" style={{ height: `${v * 100}%`, background: cellColor(v) }} />
        ))}
      </div>
      <div className="ppd-chart-axis"><span>-30m</span><span>-20m</span><span>-10m</span><span>now</span></div>
    </div>
  );
}

/* ---------------------------------- root --------------------------------- */
export default function PanicPredictionDashboard() {
  const [scenario, setScenario] = useState("Elevated");

  // ▼▼▼ SWAP for production:
  //     const s = useSentinelSocket("ws://localhost:8000/ws/live");
   const s = useLiveSocket("ws://localhost:8000/ws/live");

  // ▲▲▲

  if (!s) return <div className="ppd"><div className="ppd-connecting">Connecting</div></div>;

  return (
    <div className="ppd">
      <Header s={s} scenario={scenario} setScenario={setScenario} />

      <div className="ppd-grid">
        <div className="ppd-col">
          <CameraWall cameras={s.cameras} />
        </div>
        <div className="ppd-col">
          <StatusCard s={s} />
          <CountCard s={s} />
          <Alerts s={s} />
        </div>
      </div>

      <div className="ppd-bottom-row">
        <DensityMap s={s} />
        <ZoneLoad s={s} />
        <PanicChart s={s} />
      </div>
    </div>
  );
}