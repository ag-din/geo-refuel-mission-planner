"""
animate_mission_html.py
------------------------
Generates a self-contained HTML page with an INTERACTIVE ANIMATION of the
mission solved by `rdvsq_planning_model.solve_optimal_rdv_seq`: the target
satellites move according to Keplerian motion and the service satellite
traces the real Lambert transfer arcs, with playback controls (play/pause,
speed, timeline) and a live propellant gauge (Tsiolkovsky, via
`mass_dynamics`).

"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbital_refuel.mission import generate_mission_case
from orbital_refuel.rdvsq_planning_model import solve_optimal_rdv_seq
from orbital_refuel.lambert import solve_lambert
from orbital_refuel.mass_dynamics import MassBudget, simulate_mission_mass
from orbital_refuel.keplerian import MU_EARTH

N_TARGETS = 6
SEED = 3
EVENLY = True
TF_PERIODS = 5.5          # total time budget, in orbital periods
TOF_MIN_FRAC = 0.03
TOF_MAX_FRAC = 1.3
M0_KG = 500.0
ISP_S = 300.0
CARGO_PER_LEG_KG = 40.0    # propellant delivered to each target upon arrival
DOCK_TIME_H = 2.0          # fixed docking/transfer dwell time, same at every stop
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "mission_visualization.html"


def build_payload(
    n_targets: int = N_TARGETS,
    seed: int = SEED,
    evenly: bool = EVENLY,
    tf_periods: float = TF_PERIODS,
    tof_min_frac: float = TOF_MIN_FRAC,
    tof_max_frac: float = TOF_MAX_FRAC,
    m0_kg: float = M0_KG,
    isp_s: float = ISP_S,
    cargo_per_leg_kg: float = CARGO_PER_LEG_KG,
    dock_time_h: float = DOCK_TIME_H,
):
    """
    Generates the mission scenario, solves the optimal sequence+time problem,
    computes the real Lambert arcs and the mass dynamics, and returns a
    dictionary with all the data needed for the HTML animation.
    """
    orbits, alphas = generate_mission_case(n_targets, seed=seed, evenly=evenly)
    T_ref = orbits[0].period_s
    tf = tf_periods * T_ref
    dock_time_s = dock_time_h * 3600.0

    best, _ = solve_optimal_rdv_seq(
        orbits, alphas, tf,
        tof_min=tof_min_frac * T_ref, tof_max=tof_max_frac * T_ref,
        n_samples=50, m0_kg=m0_kg, isp_s=isp_s, verbose=False,
        dock_time_s=dock_time_s,
    )

    seq_idx = best["sequence_idx"]  # [0, ...] indices in `orbits`
    tofs = best["tofs"]
    dv_per_leg = best["dv_per_leg"]
    n = len(tofs)

    # --- Rigorous mass dynamics (two tanks), for the live gauge ---
    cargo_per_leg = [cargo_per_leg_kg] * n
    cargo_total = sum(cargo_per_leg)
    dry_mass_kg = m0_kg - cargo_total

    # Maneuver propellant: compute the true minimum required (binary search,
    # same as in mass_dynamics_demo.py) and add a 25% margin — this makes the
    # animation gauge drop noticeably instead of staying almost full (which
    # would happen with an effectively "infinite" tank, and is self-
    # contradictory anyway: that extra propellant also weighs and increases
    # the cost of each burn).
    def _min_maneuver_propellant(dv_per_leg, cargo_per_leg, dry_mass_kg, cargo_total, isp_s):
        lo, hi = 0.0, 1e7
        for _ in range(60):
            mid = (lo + hi) / 2
            b = MassBudget(dry_mass_kg, mid, cargo_total, isp_s)
            if simulate_mission_mass(dv_per_leg, cargo_per_leg, b)["feasible"]:
                hi = mid
            else:
                lo = mid
        return hi

    required_kg = _min_maneuver_propellant(dv_per_leg, cargo_per_leg, dry_mass_kg,
                                            cargo_total, isp_s)
    maneuver_prop_kg = required_kg * 1.25

    budget = MassBudget(dry_mass_kg=dry_mass_kg, maneuver_prop_kg=maneuver_prop_kg,
                         cargo_prop_kg=cargo_total, isp_s=isp_s)
    mass_sim = simulate_mission_mass(dv_per_leg, cargo_per_leg, budget)
    assert mass_sim["feasible"], "the 25% margin was not enough — check it"

    # --- Real Lambert arcs for each leg, with real docking pause ---
    legs_data = []
    t = 0.0
    for i in range(n):
        a_idx, b_idx = seq_idx[i], seq_idx[i + 1]
        r1, _ = orbits[a_idx].state_at(t)
        r2, _ = orbits[b_idx].state_at(t + tofs[i])
        v1_transfer, _ = solve_lambert(r1, r2, tofs[i], MU_EARTH, prograde=True)

        leg_mass = mass_sim["legs"][i]
        legs_data.append({
            "from_idx": a_idx, "to_idx": b_idx,
            "from_name": orbits[a_idx].name, "to_name": orbits[b_idx].name,
            "t_depart": t, "tof": tofs[i], "dv": dv_per_leg[i],
            "r1": list(r1), "v1_transfer": list(v1_transfer),
            "cargo_delivered_kg": leg_mass.cargo_delivered_kg,
            "maneuver_prop_after_kg": leg_mass.maneuver_remaining_after_kg,
            "cargo_remaining_after_kg": leg_mass.cargo_remaining_after_kg,
            "dock_time": dock_time_s,
        })
        t += tofs[i] + dock_time_s  # <- the spacecraft stays docked before departing again

    orbits_data = [{
        "name": o.name, "r": o.radius_km, "inc": o.inclination_rad,
        "raan": o.raan_rad, "u0": o.u0_rad, "period": o.period_s,
    } for o in orbits]

    return {
        "mu": MU_EARTH,
        "r_earth": 6378.137,
        "orbits": orbits_data,
        "tour_labels": best["sequence_labels"],
        "tsa_deg": best["tsa_deg"],
        "legs": legs_data,
        "total_duration": t,
        "total_dv": best["total_dv"],
        "tf_budget": tf,
        "m0_kg": m0_kg,
        "maneuver_prop_initial_kg": budget.maneuver_prop_kg,
        "cargo_prop_initial_kg": budget.cargo_prop_kg,
        "dry_mass_kg": budget.dry_mass_kg,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Simulation — TSA + IP mission (Shen &amp; Tsiotras)</title>
<style>
  :root {
    --bg: #0b1220; --panel: #131c2f; --text: #e8edf7; --muted: #93a1bf;
    --accent: #ff9a3c; --line: #23304a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header {
    padding: 10px 18px; border-bottom: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 6px;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .sub { font-size: 12px; color: var(--muted); }
  #main { flex: 1; display: flex; min-height: 0; }
  #canvas-wrap { flex: 1; position: relative; min-width: 0; }
  canvas { display: block; width: 100%; height: 100%; }
  #side {
    width: 280px; border-left: 1px solid var(--line); background: var(--panel);
    padding: 14px; overflow-y: auto; font-size: 12.5px;
  }
  #side h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
             color: var(--muted); margin: 14px 0 8px; }
  #side h2:first-child { margin-top: 0; }
  .leg-row { padding: 7px 8px; border-radius: 6px; margin-bottom: 4px; border: 1px solid transparent; }
  .leg-row.active { background: rgba(255,154,60,0.15); border-color: var(--accent); }
  .leg-row.done { opacity: 0.45; }
  .leg-row .names { font-weight: 600; }
  .leg-row .stats { color: var(--muted); font-size: 11px; }
  #controls {
    border-top: 1px solid var(--line); background: var(--panel);
    padding: 10px 18px 14px; display: flex; flex-direction: column; gap: 8px;
  }
  #transport { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  button {
    background: var(--accent); border: none; color: #1a1206; font-weight: 700;
    padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 13px;
  }
  button.secondary { background: var(--line); color: var(--text); font-weight: 500; }
  input[type=range] { flex: 1; accent-color: var(--accent); }
  #scrub-row { display: flex; align-items: center; gap: 10px; }
  #readout { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; color: var(--muted); }
  #readout b { color: var(--text); }
  select { background: var(--line); color: var(--text); border: none; padding: 6px 8px; border-radius: 6px; font-size: 12.5px; }
  .dot-legend { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; vertical-align:middle;}
  .gauge-label { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); margin-bottom: 3px; }
  .gauge-track { background: #0d1524; border-radius: 5px; height: 10px; overflow: hidden; margin-bottom: 12px; border: 1px solid var(--line); }
  .gauge-fill { height: 100%; border-radius: 5px; transition: width 0.15s linear; }
  .gauge-fill.maneuver { background: linear-gradient(90deg,#ff9a3c,#ffd23c); }
  .gauge-fill.cargo { background: linear-gradient(90deg,#4da3ff,#4caf50); }
</style>
</head>
<body>

<header>
  <h1>&#128752; TSA + IP Mission — Shen &amp; Tsiotras (2002)</h1>
  <div class="sub">sequence = __TOUR_LABELS__ &middot; TSA = __TSA_DEG__&deg; &middot; &Delta;v total = __TOTAL_DV__ km/s &middot; duration = __TOTAL_DURATION_H__ h of tf=__TF_BUDGET_H__ h</div>
</header>

<div id="main">
  <div id="canvas-wrap"><canvas id="cv"></canvas></div>
  <div id="side">
    <h2>Fuel (Tsiolkovsky)</h2>
    <div class="gauge-label"><span>Maneuver propellant</span><span id="maneuver-pct">100%</span></div>
    <div class="gauge-track"><div class="gauge-fill maneuver" id="maneuver-gauge" style="width:100%"></div></div>
    <div class="gauge-label"><span>Payload to deliver</span><span id="cargo-pct">100%</span></div>
    <div class="gauge-track"><div class="gauge-fill cargo" id="cargo-gauge" style="width:100%"></div></div>

    <h2>Mission sequence</h2>
    <div id="legs-list"></div>

    <h2>Legend</h2>
    <div><span class="dot-legend" style="background:#ff5a5a"></span>Depot</div>
    <div><span class="dot-legend" style="background:#4da3ff"></span>Pending</div>
    <div><span class="dot-legend" style="background:#ffd23c"></span>Visiting</div>
    <div><span class="dot-legend" style="background:#4caf50"></span>Visited</div>
    <div><span class="dot-legend" style="background:#ff9a3c"></span>Service (in transit)</div>
  </div>
</div>

<div id="controls">
  <div id="scrub-row">
    <span style="width:70px; color:var(--muted); font-size:12px;">Time</span>
    <input type="range" id="scrub" min="0" max="1000" value="0">
    <span id="time-label" style="width:150px; color:var(--muted); font-size:12px;">0.0 h / 0.0 h</span>
  </div>
  <div id="transport">
    <button id="play">&#9654; Play</button>
    <button class="secondary" id="reset">&#8634; Reset</button>
    <label style="color:var(--muted); font-size:12.5px;">Speed
      <select id="speed">
        <option value="500">500x</option>
        <option value="1500">1500x</option>
        <option value="4000" selected>4000x</option>
        <option value="10000">10000x</option>
        <option value="30000">30000x</option>
      </select>
    </label>
    <div id="readout">
      <span>Current leg: <b id="leg-now">—</b></span>
      <span>Cumulative &Delta;v: <b id="dv-now">0.000</b> km/s</span>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD_JSON__;

// ---------- Física (port de Python) ----------
function circularState(orbit, t){
  const n = Math.sqrt(DATA.mu / Math.pow(orbit.r,3));
  const u = orbit.u0 + n*t;
  const cu=Math.cos(u), su=Math.sin(u);
  const cO=Math.cos(orbit.raan), sO=Math.sin(orbit.raan);
  const ci=Math.cos(orbit.inc), si=Math.sin(orbit.inc);
  const r = orbit.r;
  return [ r*(cO*cu - sO*su*ci), r*(sO*cu + cO*su*ci), r*(su*si) ];
}

function stumpffC(z){
  if (z > 1e-8){ const s=Math.sqrt(z); return (1-Math.cos(s))/z; }
  if (z < -1e-8){ const s=Math.sqrt(-z); return (Math.cosh(s)-1)/(-z); }
  return 0.5;
}
function stumpffS(z){
  if (z > 1e-8){ const s=Math.sqrt(z); return (s-Math.sin(s))/(s*s*s); }
  if (z < -1e-8){ const s=Math.sqrt(-z); return (Math.sinh(s)-s)/(s*s*s); }
  return 1/6;
}
function propagateUniversal(r0, v0, dt, mu){
  if (dt <= 0) return {r:r0.slice(), v:v0.slice()};
  const norm=(v)=>Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const r0n=norm(r0), v0n=norm(v0);
  const vr0=dot(r0,v0)/r0n;
  const alpha=2/r0n-(v0n*v0n)/mu;
  const sqrtMu=Math.sqrt(mu);
  let chi=sqrtMu*Math.abs(alpha)*dt;
  for (let k=0;k<100;k++){
    const z=alpha*chi*chi;
    const C=stumpffC(z), S=stumpffS(z);
    const F=(r0n*vr0/sqrtMu)*chi*chi*C+(1-alpha*r0n)*chi*chi*chi*S+r0n*chi-sqrtMu*dt;
    const dF=(r0n*vr0/sqrtMu)*chi*(1-alpha*chi*chi*S)+(1-alpha*r0n)*chi*chi*C+r0n;
    const dchi=F/dF; chi-=dchi;
    if (Math.abs(dchi)<1e-6) break;
  }
  const z=alpha*chi*chi;
  const C=stumpffC(z), S=stumpffS(z);
  const f=1-(chi*chi/r0n)*C;
  const g=dt-(1/sqrtMu)*chi*chi*chi*S;
  const r=[f*r0[0]+g*v0[0], f*r0[1]+g*v0[1], f*r0[2]+g*v0[2]];
  const rn=norm(r);
  const fdot=(sqrtMu/(rn*r0n))*(alpha*chi*chi*chi*S-chi);
  const gdot=1-(chi*chi/rn)*C;
  const v=[fdot*r0[0]+gdot*v0[0], fdot*r0[1]+gdot*v0[1], fdot*r0[2]+gdot*v0[2]];
  return {r, v};
}

function servicerState(t){
  const legs = DATA.legs;
  if (t <= legs[0].t_depart) return { pos: circularState(DATA.orbits[legs[0].from_idx], t), legIndex: -1, docked: false };
  for (let k=0;k<legs.length;k++){
    const L = legs[k];
    const arrival = L.t_depart + L.tof;
    const nextDepart = arrival + (L.dock_time || 0);
    if (t >= L.t_depart && t <= arrival){
      const {r} = propagateUniversal(L.r1, L.v1_transfer, t-L.t_depart, DATA.mu);
      return { pos: r, legIndex: k, docked: false };
    }
    if (t > arrival && t < nextDepart){
      // docked: moves WITH the target (same orbit) while docking lasts
      return { pos: circularState(DATA.orbits[L.to_idx], t), legIndex: k, docked: true };
    }
  }
  const last = legs[legs.length-1];
  return { pos: circularState(DATA.orbits[last.to_idx], t), legIndex: legs.length, docked: false };
}

function fuelState(t){
  const legs = DATA.legs;
  let maneuver = DATA.maneuver_prop_initial_kg;
  let cargo = DATA.cargo_prop_initial_kg;
  for (const L of legs){
    if (t >= L.t_depart + L.tof){
      maneuver = L.maneuver_prop_after_kg;
      cargo = L.cargo_remaining_after_kg;
    } else if (t > L.t_depart){
      // in transit: linearly interpolate the maneuver burn for this leg over time
      // (visual approximation; the real consumption is an impulsive burn)
      const frac = (t - L.t_depart) / L.tof;
      const before = maneuver;
      maneuver = before - (before - L.maneuver_prop_after_kg) * frac;
      break;
    } else break;
  }
  return { maneuver, cargo };
}

// ---------- Projection (planar problem: i=0, RAAN=0 for all) ----------
const ELEV = 18 * Math.PI/180, AZ = 10 * Math.PI/180; // slight aesthetic tilt only
function project(p){
  const x1 = p[0]*Math.cos(AZ) - p[1]*Math.sin(AZ);
  const y1 = p[0]*Math.sin(AZ) + p[1]*Math.cos(AZ);
  const z1 = p[2];
  const y2 = y1*Math.cos(ELEV) - z1*Math.sin(ELEV);
  return [x1, y2];
}

const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');
let W,H, scale, cx, cy;
function resize(){
  const wrap = document.getElementById('canvas-wrap');
  W = canvas.width = wrap.clientWidth * devicePixelRatio;
  H = canvas.height = wrap.clientHeight * devicePixelRatio;
  canvas.style.width = wrap.clientWidth+'px';
  canvas.style.height = wrap.clientHeight+'px';
  const maxR = Math.max(...DATA.orbits.map(o=>o.r));
  scale = Math.min(W,H) / (maxR*2.35);
  cx = W/2; cy = H/2;
}
window.addEventListener('resize', resize);
function toScreen(p3){ const [x,y]=project(p3); return [cx+x*scale, cy-y*scale]; }

function drawOrbitPath(orbit, color){
  ctx.beginPath();
  const N=128;
  for (let k=0;k<=N;k++){
    const u = orbit.u0 + 2*Math.PI*k/N;
    const fakeT = (u-orbit.u0) / Math.sqrt(DATA.mu/Math.pow(orbit.r,3));
    const [sx,sy] = toScreen(circularState(orbit, fakeT));
    if (k===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.globalAlpha = 0.35;
  ctx.stroke(); ctx.globalAlpha = 1;
}

let trail = [];
const COL_PENDING='#4da3ff', COL_VISITING='#ffd23c', COL_DONE='#4caf50', COL_DEPOT='#ff5a5a';

function statusColors(legIndex){
  const colors = {0: COL_DEPOT};
  DATA.orbits.forEach((o,idx)=>{ if(idx>0) colors[idx]=COL_PENDING; });
  DATA.legs.forEach((L,k)=>{
    if (legIndex > k) colors[L.to_idx] = COL_DONE;
    else if (legIndex === k) colors[L.to_idx] = COL_VISITING;
  });
  return colors;
}

function drawEarth(){
  const [sx,sy] = toScreen([0,0,0]);
  const rPix = DATA.r_earth*scale;
  const grad = ctx.createRadialGradient(sx-rPix*0.3, sy-rPix*0.3, rPix*0.1, sx, sy, rPix);
  grad.addColorStop(0,'#4d8fd6'); grad.addColorStop(1,'#0d2038');
  ctx.beginPath(); ctx.arc(sx,sy,Math.max(rPix,3),0,2*Math.PI);
  ctx.fillStyle = grad; ctx.fill();
}

function draw(t){
  const sv = servicerState(t);
  trail.push(sv.pos); if (trail.length>90) trail.shift();
  const colors = statusColors(sv.legIndex);

  ctx.clearRect(0,0,W,H); ctx.save();
  DATA.orbits.forEach((o,idx)=> drawOrbitPath(o, colors[idx]));
  drawEarth();

  DATA.orbits.forEach((o,idx)=>{
    const [sx,sy] = toScreen(circularState(o,t));
    const col = colors[idx];
    ctx.beginPath(); ctx.arc(sx,sy, idx===0?5:4, 0,2*Math.PI);
    ctx.fillStyle = col;
    if (col===COL_VISITING){ ctx.shadowColor=COL_VISITING; ctx.shadowBlur=10; }
    ctx.fill(); ctx.shadowBlur=0;
    ctx.fillStyle='#c8d3ea'; ctx.font='10px sans-serif';
    ctx.fillText(o.name, sx+7, sy-6);
  });

  ctx.beginPath();
  trail.forEach((p,k)=>{ const [sx,sy]=toScreen(p); if(k===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy); });
  ctx.strokeStyle='#ff9a3c'; ctx.lineWidth=2; ctx.globalAlpha=0.7; ctx.stroke(); ctx.globalAlpha=1;

  const [ssx,ssy] = toScreen(sv.pos);
  if (sv.docked){
    ctx.beginPath(); ctx.arc(ssx,ssy,11,0,2*Math.PI);
    ctx.strokeStyle='#ff9a3c'; ctx.lineWidth=1.5; ctx.globalAlpha=0.8; ctx.stroke(); ctx.globalAlpha=1;
  }
  ctx.beginPath(); ctx.arc(ssx,ssy,6,0,2*Math.PI);
  ctx.fillStyle='#ff9a3c'; ctx.shadowColor='#ff9a3c'; ctx.shadowBlur=12; ctx.fill(); ctx.shadowBlur=0;
  ctx.restore();
}

// ---------- UI ----------
const scrub=document.getElementById('scrub'), playBtn=document.getElementById('play'),
      resetBtn=document.getElementById('reset'), speedSel=document.getElementById('speed'),
      timeLabel=document.getElementById('time-label'), legNow=document.getElementById('leg-now'),
      dvNow=document.getElementById('dv-now'), legsList=document.getElementById('legs-list'),
      manGauge=document.getElementById('maneuver-gauge'), manPct=document.getElementById('maneuver-pct'),
      cargoGauge=document.getElementById('cargo-gauge'), cargoPct=document.getElementById('cargo-pct');

const SCRUB_MAX=1000;
let t=0, playing=false, lastFrame=null;

DATA.legs.forEach((L,k)=>{
  const div=document.createElement('div');
  div.className='leg-row'; div.id='leg-'+k;
  const dockTxt = L.dock_time>0 ? ` &middot; docking=${(L.dock_time/3600).toFixed(1)} h` : '';
  div.innerHTML = `<div class="names">${k+1}. ${L.from_name} &rarr; ${L.to_name}</div>
                    <div class="stats">&Delta;v=${L.dv.toFixed(3)} km/s &middot; tof=${(L.tof/3600).toFixed(1)} h &middot; payload=${L.cargo_delivered_kg.toFixed(0)} kg${dockTxt}</div>`;
  legsList.appendChild(div);
});

function updateSide(t){
  const sv = servicerState(t);
  let dvAcc = 0;
  DATA.legs.forEach((L,k)=>{
    const row = document.getElementById('leg-'+k);
    row.classList.remove('active','done');
    if (sv.legIndex > k){ row.classList.add('done'); dvAcc += L.dv; }
    else if (sv.legIndex === k) row.classList.add('active');
  });
  legNow.textContent = (sv.legIndex>=0 && sv.legIndex<DATA.legs.length)
      ? (sv.docked
          ? `docked at ${DATA.legs[sv.legIndex].to_name}`
          : `${DATA.legs[sv.legIndex].from_name} \u2192 ${DATA.legs[sv.legIndex].to_name}`)
      : (sv.legIndex<0 ? 'waiting for departure' : 'mission complete');
  dvNow.textContent = dvAcc.toFixed(3);
  timeLabel.textContent = `${(t/3600).toFixed(1)} h / ${(DATA.total_duration/3600).toFixed(1)} h`;

  const fuel = fuelState(t);
  const manFrac = 100*fuel.maneuver/DATA.maneuver_prop_initial_kg;
  const cargoFrac = 100*fuel.cargo/DATA.cargo_prop_initial_kg;
  manGauge.style.width = manFrac.toFixed(1)+'%';
  manPct.textContent = `${fuel.maneuver.toFixed(1)} kg (${manFrac.toFixed(0)}%)`;
  cargoGauge.style.width = cargoFrac.toFixed(1)+'%';
  cargoPct.textContent = `${fuel.cargo.toFixed(1)} kg (${cargoFrac.toFixed(0)}%)`;
}

function setT(newT){
  t = Math.max(0, Math.min(DATA.total_duration, newT));
  scrub.value = (t/DATA.total_duration*SCRUB_MAX).toFixed(0);
  draw(t); updateSide(t);
}

scrub.addEventListener('input', ()=>{
  playing=false; playBtn.textContent='\u25B6 Play'; trail=[];
  setT(scrub.value/SCRUB_MAX*DATA.total_duration);
});
playBtn.addEventListener('click', ()=>{
  playing=!playing; playBtn.textContent = playing?'\u23F8 Pause':'\u25B6 Play';
  lastFrame=null; if (playing) requestAnimationFrame(tick);
});
resetBtn.addEventListener('click', ()=>{
  playing=false; playBtn.textContent='\u25B6 Play'; trail=[]; setT(0);
});
function tick(now){
  if (!playing) return;
  if (lastFrame===null) lastFrame=now;
  const dtReal=(now-lastFrame)/1000; lastFrame=now;
  const rate=parseFloat(speedSel.value);
  setT(t+dtReal*rate);
  if (t>=DATA.total_duration){ playing=false; playBtn.textContent='\u25B6 Play'; }
  requestAnimationFrame(tick);
}

resize(); setT(0);
</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    """
    Renders the HTML template with the given payload data.

    Parameters
    ----------
    payload : dict
        A dictionary containing the mission data to be embedded in the HTML.

    Returns
    -------
    str
        The rendered HTML string with the payload data inserted.
    """
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    html = html.replace("__TOTAL_DV__", f"{payload['total_dv']:.4f}")
    html = html.replace("__TOTAL_DURATION_H__", f"{payload['total_duration']/3600:.1f}")
    html = html.replace("__TF_BUDGET_H__", f"{payload['tf_budget']/3600:.1f}")
    html = html.replace("__TOUR_LABELS__", str(payload["tour_labels"]))
    html = html.replace("__TSA_DEG__", f"{payload['tsa_deg']:.1f}")
    return html


def main(
    n_targets: int = N_TARGETS,
    seed: int = SEED,
    evenly: bool = EVENLY,
    tf_periods: float = TF_PERIODS,
    output_path: str | Path | None = None,
):
    payload = build_payload(
        n_targets=n_targets,
        seed=seed,
        evenly=evenly,
        tf_periods=tf_periods,
    )
    target_path = Path(output_path) if output_path is not None else OUTPUT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(payload)
    target_path.write_text(html)
    print(f"RDV Sequence: {payload['tour_labels']}  Total Δv: {payload['total_dv']:.4f} km/s")
    print(f"Visualization saved to: {target_path}")
    return target_path


if __name__ == "__main__":
    main()
