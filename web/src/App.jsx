import React, { useEffect, useState } from "react";

const W = 720, H = 280, M = { t: 16, r: 16, b: 28, l: 44 };
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const hourOf = (hhmm) => parseInt(hhmm.slice(0, 2), 10);

function useJson(url) {
  const [state, set] = useState({ data: null, error: null });
  useEffect(() => {
    fetch(url)
      .then((r) => (r.ok ? r.json() : r.json().then((j) => Promise.reject(new Error(j.detail || r.status)))))
      .then((data) => set({ data, error: null }))
      .catch((error) => set({ data: null, error }));
  }, [url]);
  return state;
}

function ScoreCurve({ curve, window: win }) {
  const [hover, setHover] = useState(null);
  const pts = curve.filter((c) => c.score != null);
  const x = (h) => M.l + (h / 23) * PW;
  const y = (s) => M.t + (1 - s / 100) * PH;
  const line = pts.map((c, i) => `${i ? "L" : "M"}${x(c.hour).toFixed(1)},${y(c.score).toFixed(1)}`).join("");
  const area = `${line}L${x(pts.at(-1).hour)},${y(0)}L${x(pts[0].hour)},${y(0)}Z`;
  const [bs, be] = [hourOf(win.start), hourOf(win.end)];
  const bx0 = x(bs) - PW / 46, bx1 = x(be - 1) + PW / 46;

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const h = Math.max(0, Math.min(23, Math.round(((px - M.l) / PW) * 23)));
    setHover(pts.find((c) => c.hour === h) || null);
  };

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Outdoor score by hour. Best window ${win.start} to ${win.end}.`}
           onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {[0, 25, 50, 75, 100].map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth="1" />
            <text x={M.l - 6} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        {[0, 3, 6, 9, 12, 15, 18, 21].map((h) => (
          <text key={h} x={x(h)} y={H - 8} textAnchor="middle">{String(h).padStart(2, "0")}</text>
        ))}
        <rect x={bx0} y={M.t} width={bx1 - bx0} height={PH} fill="var(--band)" />
        <text x={(bx0 + bx1) / 2} y={M.t + 12} textAnchor="middle" style={{ fill: "var(--ink)", fontWeight: 600 }}>
          best {win.start}–{win.end}
        </text>
        <path d={area} fill="var(--series)" opacity="0.1" />
        <path d={line} fill="none" stroke="var(--series)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {hover && (
          <g>
            <line x1={x(hover.hour)} x2={x(hover.hour)} y1={M.t} y2={y(0)} stroke="var(--ink-3)" strokeWidth="1" />
            <circle cx={x(hover.hour)} cy={y(hover.score)} r="5" fill="var(--series)" stroke="var(--surface)" strokeWidth="2" />
          </g>
        )}
      </svg>
      {hover && (
        <div className="tip" style={{ left: `${(x(hover.hour) / W) * 100}%`, top: `${(y(hover.score) / H) * 100 - 3}%` }}>
          <span className="t">{hover.time}</span>
          <b>{Math.round(hover.score)} / 100</b>
          feels {hover.feels_c}°C · AQI {hover.aqi} ({hover.air})<br />
          rain {hover.rain_pct}% · UV {hover.uv}{hover.is_day ? "" : " · night"}
        </div>
      )}
    </div>
  );
}

function HourTable({ curve, win }) {
  const [bs, be] = [hourOf(win.start), hourOf(win.end)];
  return (
    <table>
      <thead><tr><th>Hour</th><th>Score</th><th>Feels °C</th><th>AQI</th><th>Rain %</th><th>UV</th></tr></thead>
      <tbody>
        {curve.map((c) => (
          <tr key={c.hour} className={c.hour >= bs && c.hour < be ? "best" : ""}>
            <td>{c.time}</td><td>{c.score ?? "–"}</td><td>{c.feels_c ?? "–"}</td>
            <td>{c.aqi ?? "–"}</td><td>{c.rain_pct ?? "–"}</td><td>{c.uv ?? "–"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Accuracy({ ev }) {
  if (!ev || !ev.results) return null;
  const best = (r) => Math.min(r.final.mae, r.persistence.mae, r.yesterday.mae);
  return (
    <div className="card">
      <h2>How good is the PM2.5 forecast?</h2>
      <p className="sub">Mean absolute error in µg/m³ on the last {ev.test_days} days, scored once after choosing the model
        on the {ev.val_days} days before (lower is better).</p>
      <table>
        <thead><tr><th>Horizon</th><th>Window</th><th>TabPFN alone</th><th>Persistence</th><th>Yesterday</th></tr></thead>
        <tbody>
          {ev.results.map((r) => (
            <tr key={r.horizon_h}>
              <td>+{r.horizon_h}h</td>
              {[r.final, r.tabpfn, r.persistence, r.yesterday].map((m, i) => (
                <td key={i} style={m.mae === best(r) && i !== 1 ? { fontWeight: 600 } : undefined}>{m.mae}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">Window = TabPFN ({ev.selected?.name}) stacked with persistence and same-hour-yesterday, weights fit on
        validation. No CAMS-forecast baseline: Open-Meteo does not expose past forecasts. Air quality is CAMS model output,
        not station readings.</p>
    </div>
  );
}

export default function App() {
  const today = useJson("/today");
  const ev = useJson("/eval");
  const [table, setTable] = useState(false);

  if (today.error) return <main><p className="err">Could not load today's plan: {String(today.error.message)}</p></main>;
  if (!today.data) return <main><p className="meta">Planning…</p></main>;

  const { plan, window: win, curve, date, day, source, model } = today.data;
  const caution = { air: "Poor air", heat: "Very hot", uv: "High UV", rain: "Rain likely" };

  return (
    <main>
      <h1>Window</h1>
      <p className="meta">{day} · {date} · written by {source === "ollama" ? model : "a template (Ollama unavailable)"}</p>
      <p className="nudge">{plan.message}</p>
      <div className="chips">
        <span className="chip">{plan.start}–{plan.end}</span>
        <span className="chip">{plan.place}</span>
        <span className="chip">{plan.activity}</span>
        {(win.caution || []).map((c) => <span key={c} className="chip warn">⚠ {caution[c]}</span>)}
      </div>

      <div className="card">
        <h2>Outdoor score by hour</h2>
        <p className="sub">0–100, higher is better. Night hours are capped at 30.</p>
        <ScoreCurve curve={curve} window={win} />
        <button className="link" onClick={() => setTable(!table)}>{table ? "Hide" : "Show"} table</button>
        {table && <HourTable curve={curve} win={win} />}
      </div>

      <div className="card">
        <h2>Why</h2>
        <p className="why">{plan.reason}</p>
      </div>

      <Accuracy ev={ev.data} />
    </main>
  );
}
