"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : (100 * x).toFixed(d) + "%");
const COL = { accepted: "#1b7f4c", uncertain: "#b7791f", rejected: "#b42318", unable: "#667085" };
const GROUP_NAME = { rerender: "Re-render support", ambiguity: "Ambiguity", continuity: "Continuity", missing: "Missing data", calibration: "Grid calibration", image: "Image quality" };
const GROUP_HELP = {
  rerender: "Does the image support the extracted trace when it is drawn back onto the picture?",
  ambiguity: "Were there competing ink strokes in the beat window?",
  continuity: "Is the extracted waveform physically smooth?",
  missing: "Were there columns with no readable trace (never filled in)?",
  calibration: "Is the paper grid consistent across the image (scale, tilt, residual)?",
  image: "Resolution, sharpness and ink-to-noise level around the beat.",
};
let overview = null, cur = null, selBeat = null, zoomMode = 0;

async function api(path, opts) {
  const r = await fetch(path, opts);
  let j = null;
  try { j = await r.json(); } catch (_) { /* non-JSON */ }
  if (!r.ok) throw new Error((j && j.error) || `${r.status} ${r.statusText}`);
  return j;
}

/* ---------------------------------------------------------------- tabs */
$$(".tabs button").forEach((b) => b.addEventListener("click", () => {
  $$(".tabs button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
  $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "evaluation") loadEval();
}));

/* ---------------------------------------------------------------- init */
async function init() {
  try {
    overview = await api("/api/overview");
    if (!overview.model_ready) {
      const w = $("#modelWarn"); w.hidden = false;
      w.textContent = "No trained trust model found (results/trust_model.json). Run: python scripts/run_evaluation.py --preset full";
    }
    const ex = await api("/api/examples");
    $("#exRecord").innerHTML = ex.records.map((r) => `<option value="${r.index}">Record ${r.index + 1} · PTB-XL ${esc(r.record)}</option>`).join("");
    $("#exCond").innerHTML = ex.conditions.map((c) => `<option value="${c.id}">${esc(c.label)}</option>`).join("");
  } catch (e) { setStatus("Could not reach the server: " + e.message, "bad"); }
  renderAbout();
}

function setStatus(msg, kind) {
  const s = $("#status");
  s.hidden = !msg; s.className = "status" + (kind ? " " + kind : ""); s.textContent = msg || "";
}

/* ---------------------------------------------------------------- analyze */
$("#btnExample").addEventListener("click", async () => {
  const i = $("#exRecord").value, c = $("#exCond").value;
  await runAnalysis(() => api(`/api/example/${i}/${c}/analyze`), "Digitizing the image and scoring each beat…");
});
$("#btnPick").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
function upload(file) {
  const fd = new FormData(); fd.append("image", file);
  runAnalysis(() => api("/api/analyze", { method: "POST", body: fd }), `Analysing ${file.name}…`);
}

async function runAnalysis(fn, busyMsg) {
  $("#btnExample").disabled = true;
  setStatus(busyMsg, "busy");
  try {
    cur = await fn();
    setStatus("");
    showResult();
  } catch (e) {
    $("#result").hidden = true;
    setStatus("Could not analyse this input: " + e.message, "bad");
  } finally { $("#btnExample").disabled = false; }
}

function showResult() {
  $("#result").hidden = false;
  selBeat = null;
  renderSummary();
  const img = $("#img");
  img.onload = () => { drawOverlay(); };
  img.src = `/api/image/${cur.token}/original.png`;
  renderLegend();
  renderStrip();
  $("#inspector").hidden = true;
  if (cur.beats.length) selectBeat(firstInteresting());
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function firstInteresting() {
  const order = { rejected: 0, uncertain: 1, accepted: 2 };
  return cur.beats.reduce((best, b, i) => (order[b.label] < order[cur.beats[best].label] ? i : best), 0);
}

function chip(label, text) { return `<span class="chip ${label}">${esc(text ?? label)}</span>`; }

function renderSummary() {
  const c = cur, el = $("#summary");
  if (c.status !== "ok" || !c.beats) {
    el.innerHTML = `<div class="src">${esc(c.source.label)}</div><div class="chips">${chip("unable", "unable")}</div>
      <p class="verdict"><b>No measurement is reported.</b> ${esc(c.message || c.reason || "The trace could not be digitized.")}</p>`;
    return;
  }
  const k = c.counts, n = c.beats.length, hr = c.heart_rate || {};
  const cal = c.calibration || {};
  const nAcc = k.accepted;
  el.innerHTML = `
    <div class="src">${esc(c.source.label)}</div>
    <div class="chips">${chip("accepted", `${k.accepted} accepted`)}${chip("uncertain", `${k.uncertain} uncertain`)}${chip("rejected", `${k.rejected} rejected`)}</div>
    <div class="stats">
      <div class="stat"><b>${n}</b><span>beats detected</span></div>
      <div class="stat"><b>${pct(c.valid_coverage, 0)}</b><span>of trace columns readable</span></div>
      <div class="stat"><b>${hr.bpm ? fmt(hr.bpm, 0) + " bpm" : "withheld"}</b><span>${hr.bpm ? `from ${hr.n_rr_intervals} accepted RR interval${hr.n_rr_intervals === 1 ? "" : "s"}` : "no RR interval with two accepted beats"}</span></div>
      <div class="stat"><b>${cal.px_per_mm_x ? fmt(cal.px_per_mm_x, 1) : "–"}</b><span>px per mm (read from the grid)</span></div>
    </div>
    <p class="verdict">${nAcc === 0 ? "<b>No beat is verifiable enough to measure.</b> All measurements are withheld." :
      `<b>${nAcc} of ${n} beats</b> can be measured from this image. Measurements for the other ${n - nAcc} are withheld, not estimated.`}
      <span class="hint"> Analysis took ${fmt(c.runtime_s, 1)} s.</span></p>`;
}

function renderLegend() {
  $("#legend").innerHTML = ["accepted", "uncertain", "rejected"].map((l) => `<span><i style="background:${COL[l]}"></i>${l}</span>`).join("") +
    `<span><i style="background:repeating-linear-gradient(45deg,#667085 0 2px,transparent 2px 5px)"></i>missing columns (not filled in)</span>`;
}

function drawOverlay() {
  const svg = $("#ov"), W = cur.image.width, H = cur.image.height;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (cur.status !== "ok") { svg.innerHTML = ""; return; }
  let h = `<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="#667085" stroke-width="3"/></pattern></defs>`;
  if ($("#tGaps").checked) cur.gaps.forEach(([a, b]) => { h += `<rect x="${a}" y="0" width="${b - a + 1}" height="${H}" fill="url(#hatch)" opacity=".28"/>`; });
  if ($("#tBeats").checked) cur.beats.forEach((b, i) => {
    const [a, z] = b.window_x, col = COL[b.label];
    h += `<g class="bw" data-i="${i}" style="cursor:pointer"><rect x="${a}" y="0" width="${z - a + 1}" height="${H}" fill="${col}" opacity="${i === selBeat ? ".24" : ".09"}"/>
      <rect x="${a}" y="${H - 10}" width="${z - a + 1}" height="10" fill="${col}"/><rect x="${a}" y="0" width="${z - a + 1}" height="${H}" fill="none" stroke="${col}" stroke-width="${i === selBeat ? 3 : 1}"/></g>`;
  });
  if ($("#tTrace").checked) {
    let d = "", pen = false;
    cur.trace.y.forEach((y, i) => { if (y === null) { pen = false; return; } d += `${pen ? "L" : "M"}${cur.trace.x0 + i} ${y} `; pen = true; });
    h += `<path d="${d}" fill="none" stroke="#0a7ea4" stroke-width="1.6" stroke-linejoin="round" opacity=".9"/>`;
  }
  svg.innerHTML = h;
  $$(".bw", svg).forEach((g) => g.addEventListener("click", () => selectBeat(+g.dataset.i)));
}
["tTrace", "tBeats", "tGaps"].forEach((id) => $("#" + id).addEventListener("change", () => cur && drawOverlay()));
$("#btnZoom").addEventListener("click", () => {
  zoomMode = (zoomMode + 1) % 3;
  const st = $("#stage"), names = ["fit", "100%", "200%"];
  const w = cur ? cur.image.width : 1000;
  st.style.width = zoomMode === 0 ? "100%" : `${w * (zoomMode === 1 ? 1 : 2)}px`;
  $("#btnZoom").textContent = "zoom: " + names[zoomMode];
});

function renderStrip() {
  $("#beatstrip").innerHTML = cur.beats.map((b, i) =>
    `<button class="bchip ${b.label}" data-i="${i}"><small>beat ${i + 1} · ${fmt(b.t_r_s, 2)} s</small><b>${fmt(b.trust, 0)}</b> <small>${b.label}</small></button>`).join("");
  $$(".bchip").forEach((b) => b.addEventListener("click", () => selectBeat(+b.dataset.i)));
}

function selectBeat(i) {
  selBeat = i;
  $$(".bchip").forEach((b) => b.classList.toggle("sel", +b.dataset.i === i));
  drawOverlay();
  renderInspector(cur.beats[i], i);
}

function gauge(b) {
  const ta = cur.thresholds.t_accept, tr = cur.thresholds.t_reject, p = Math.max(0, Math.min(100, b.trust));
  return `<div class="gauge"><i style="width:${Math.max(0, tr)}%;background:var(--rej-bg)"></i><i style="width:${Math.max(0, ta - tr)}%;background:var(--unc-bg)"></i><i style="width:${Math.max(0, 100 - ta)}%;background:var(--acc-bg)"></i>
    <span class="ptr" style="left:${p}%">${fmt(b.trust, 0)}</span></div>
    <div class="gscale"><span>0</span><span style="color:var(--rej)">reject &lt; ${fmt(tr, 0)}</span><span style="color:var(--acc)">accept ≥ ${fmt(ta, 0)}</span><span>100</span></div>`;
}

function evidenceHtml(b) {
  return b.evidence.map((g, k) => {
    const s = Math.max(0, Math.min(1, g.score)), col = s >= 0.75 ? COL.accepted : s >= 0.4 ? COL.uncertain : COL.rejected;
    return `<details class="eg" ${k === 0 ? "open" : ""}><summary><span title="${esc(GROUP_HELP[g.group] || "")}">${esc(GROUP_NAME[g.group] || g.group)}</span>
      <span class="meter"><i style="width:${(s * 100).toFixed(0)}%;background:${col}"></i></span><span class="val">${fmt(s, 2)}</span></summary>
      <div class="hint" style="margin-top:6px">${esc(GROUP_HELP[g.group] || "")}</div>
      <ul>${g.features.map((f) => `<li><b>${esc(f.name)}</b> = ${fmt(f.value, 3)}${f.subscore === null ? " (inactive)" : ` → sub-score ${fmt(f.subscore, 2)}`}<br>${esc(f.doc)}</li>`).join("")}</ul></details>`;
  }).join("");
}

function measHtml(b) {
  const m = b.measurements;
  if (b.label === "accepted") {
    const rows = [`<tr><td>R amplitude (above pre-QRS baseline)</td><td>${fmt(m.r_amplitude_mv, 2)} mV</td></tr>`];
    rows.push(m.rr_ms !== null ? `<tr><td>RR interval to previous beat</td><td>${fmt(m.rr_ms, 0)} ms</td></tr>`
      : `<tr><td>RR interval to previous beat</td><td>withheld · ${b.beat === 0 ? "first beat" : !b.rr_eligible ? "span not fully observed" : "previous beat not accepted"}</td></tr>`);
    return `<div class="meas ok"><b>Measurements reported</b> (this beat is verifiable against the image)<table>${rows.join("")}</table></div>`;
  }
  const why = b.label === "rejected" ? "The image does not support this beat well enough." : (!b.qrs_core_observed ? "The QRS core was not fully observed, so the beat can never be accepted." : "The evidence is not strong enough to accept it.");
  return `<div class="meas no"><b>Measurements withheld</b> · ${esc(m.status)}<br><span class="hint">${esc(why)} Nothing is estimated or filled in.</span></div>`;
}

function evalOnlyHtml(b) {
  const e = b.evaluation_only;
  if (!e) return "";
  const verdict = e.bad ? `<span class="verd no">bad beat</span>` : `<span class="verd yes">good beat</span>`;
  return `<div class="evalonly"><h3>Evaluation-only (reference signal) · not available for uploads</h3>
    <div class="hint">Computed after scoring from the original digital signal. It is never an input to the trust score.</div>
    ${verdict} waveform error ${e.nrmse === null ? "– (no matching reference beat)" : `NRMSE ${fmt(e.nrmse, 3)}`} · bad if unmatched or NRMSE &gt; 0.10
    ${e.r_amp_err_mv != null ? `· R-amplitude error ${fmt(e.r_amp_err_mv, 3)} mV` : ""}${e.rr_err_ms != null ? ` · RR error ${fmt(e.rr_err_ms, 0)} ms` : ""}
    ${e.curve ? curveSvg(e.curve) : ""}</div>`;
}

function curveSvg(c) {
  const xs = c.t, a = c.extracted_mv, r = c.reference_mv;
  const all = [...a, ...r].filter((v) => v !== null);
  if (!all.length) return "";
  const lo = Math.min(...all), hi = Math.max(...all), W = 420, H = 130, pad = 6;
  const sx = (t) => pad + (t / (xs[xs.length - 1] || 1)) * (W - 2 * pad), sy = (v) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
  const path = (arr) => { let d = "", pen = false; arr.forEach((v, i) => { if (v === null) { pen = false; return; } d += `${pen ? "L" : "M"}${sx(xs[i]).toFixed(1)} ${sy(v).toFixed(1)} `; pen = true; }); return d; };
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path d="${path(r)}" fill="none" stroke="#667085" stroke-width="2" stroke-dasharray="5 3"/><path d="${path(a)}" fill="none" stroke="#0a7ea4" stroke-width="1.8"/></svg>
    <div class="hint"><span style="color:#0a7ea4">━</span> extracted (offset removed) &nbsp; <span style="color:#667085">╍</span> reference · gaps are missing columns</div>`;
}

function renderInspector(b, i) {
  const el = $("#inspector"); el.hidden = false;
  el.innerHTML = `<h2>3 · Why this score? <span>Beat ${i + 1} · R peak at ${fmt(b.t_r_s, 2)} s</span> ${chip(b.label)}</h2>
    <div class="igrid">
      <div><img class="crop" src="/api/image/${cur.token}/beat/${i}.png" alt="Cropped image of beat ${i + 1} with the extracted trace drawn over it">
        ${gauge(b)}
        <div class="hint">Trust score ${fmt(b.trust, 1)} · a rank-ordered evidence index (0–100), not a probability of being correct.</div>
        ${measHtml(b)}</div>
      <div><div class="hint" style="margin-bottom:6px">Six evidence groups, worst first. 1.0 = fully consistent with the image, 0 = not. Open a group to see the numbers.</div>
        <div class="evid">${evidenceHtml(b)}</div>${evalOnlyHtml(b)}</div>
    </div>`;
}

/* ---------------------------------------------------------------- evaluation */
let evalLoaded = false;
async function loadEval() {
  if (evalLoaded) return;
  const body = $("#evalBody");
  let full = null;
  try { full = await api("/api/results/full"); } catch (e) { body.innerHTML = `<div class="panel"><b>Full results not available.</b> ${esc(e.message)}</div>`; return; }
  evalLoaded = true;
  body.innerHTML = renderEval(full);
}

const M_STYLE = {
  trust_operational: ["#0e5a64", 3.2, "", "Trust score"], accept_all: ["#444", 2, "6 4", "Accept every beat"], image_quality_snr: ["#4a7fb5", 1.8, "", "Image SNR"],
  continuity: ["#7a5c99", 1.8, "", "Continuity"], valid_fraction: ["#c27c0e", 1.8, "", "Valid fraction"], oracle_error_ranking: ["#9aa0a6", 1.6, "2 3", "Oracle (ideal)"],
};
const M_NAME = { trust_operational: "Trust score (operational)", trust_heuristic: "Trust · heuristic", trust_learned: "Trust · learned", accept_all: "Accept all beats", image_quality_snr: "Image SNR", continuity: "Continuity", valid_fraction: "Valid fraction", oracle_error_ranking: "Oracle ranking (upper bound)" };

function riskChart(curves) {
  const W = 760, H = 380, L = 52, R = 14, T = 12, B = 44;
  let ymax = 0;
  Object.entries(curves).forEach(([k, c]) => { if (M_STYLE[k]) c.mean_err.forEach((v) => { if (v > ymax) ymax = v; }); });
  ymax = Math.min(1, Math.ceil(ymax * 20) / 20 || 0.05);
  const sx = (v) => L + v * (W - L - R), sy = (v) => H - B - (v / ymax) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Risk-coverage curves: mean beat error against the fraction of beats accepted, for the trust score and baselines">`;
  for (let i = 0; i <= 5; i++) { const y = (ymax * i) / 5; s += `<line class="gl" x1="${L}" x2="${W - R}" y1="${sy(y)}" y2="${sy(y)}"/><text class="ax" x="${L - 6}" y="${sy(y) + 4}" text-anchor="end">${y.toFixed(2)}</text>`; }
  for (let i = 0; i <= 5; i++) { const x = i / 5; s += `<text class="ax" x="${sx(x)}" y="${H - B + 16}" text-anchor="middle">${(x * 100).toFixed(0)}%</text>`; }
  s += `<text class="ax" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">coverage: share of beats accepted (most trusted first)</text>`;
  s += `<text class="ax" transform="translate(13 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">mean beat error (NRMSE; unmatched = 1)</text>`;
  Object.entries(M_STYLE).forEach(([k, [col, w, dash]]) => {
    const c = curves[k]; if (!c) return;
    const d = c.coverage.map((cv, i) => `${i ? "L" : "M"}${sx(cv).toFixed(1)} ${sy(Math.min(ymax, c.mean_err[i])).toFixed(1)}`).join(" ");
    s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" stroke-dasharray="${dash}" stroke-linejoin="round"/>`;
  });
  return s + "</svg>" + `<div class="leg2">${Object.entries(M_STYLE).map(([k, [col, w, dash]]) => `<span><i style="border-color:${col};border-top-style:${dash ? "dashed" : "solid"}"></i>${esc(M_STYLE[k][3])}</span>`).join("")}</div>`;
}

function verdictPill(h, label) {
  return h.supported ? `<span class="verd yes">supported</span>` : `<span class="verd no">not supported</span>`;
}

function renderEval(r) {
  const t = r.test, h = t.hypotheses, op = t.operational, ds = r.dataset;
  const ci = (o) => `${fmt(o.point, 4)} <span class="s">[${fmt(o.ci95[0], 4)}, ${fmt(o.ci95[1], 4)}]</span>`;
  const H1 = h.H1_trust_beats_accept_all, H2 = h.H2_trust_beats_best_baseline;
  const cls = op.classes;
  const gate = r.feasibility_gate;
  let html = `<div class="panel"><h2>Held-out test result · ${t.n_records} records · ${t.n_beats} beats · ${ds.n_conditions} image conditions</h2>
    <p class="hint">Primary endpoint: AURC, the mean beat error averaged over coverage (lower is better), on the untouched PTB-XL test fold. 95% intervals come from a record-level bootstrap (${h.n_bootstrap} resamples). The hypotheses, the endpoint and the rule for choosing the ranking score were fixed before the run; the accept/reject thresholds below are post-hoc and labelled as such.</p>
    <div class="cards">
      <div class="card"><div class="k">AURC · accept every beat</div><div class="v">${ci(h.aurc_accept_all)}</div></div>
      <div class="card"><div class="k">AURC · trust score (${esc(op.score)})</div><div class="v">${ci(h.aurc_trust_operational)}</div></div>
      <div class="card"><div class="k">AURC · best simple baseline (${esc(h.best_simple_baseline)})</div><div class="v">${ci(h.aurc_best_baseline)}</div></div>
    </div>
    <div class="cards">
      <div class="card"><div class="k">H1 · trust beats accepting everything</div><div class="v">${verdictPill(H1)}</div><div class="s">AURC reduced by ${fmt(H1.delta_point, 4)} (${pct(H1.relative_reduction_point, 0)}); 95% CI of the improvement [${fmt(H1.delta_ci95[0], 4)}, ${fmt(H1.delta_ci95[1], 4)}]</div></div>
      <div class="card"><div class="k">H2 · trust beats the best simple baseline</div><div class="v">${verdictPill(H2)}</div><div class="s">improvement ${fmt(H2.delta_point, 4)}; 95% CI [${fmt(H2.delta_ci95[0], 4)}, ${fmt(H2.delta_ci95[1], 4)}]. “Supported” means the CI excludes 0.</div></div>
    </div></div>`;
  html += `<div class="panel chartwrap"><h2>Risk–coverage: accepting only the beats the score trusts most</h2>${riskChart(t.curves)}
    <p class="hint">Reading it: a useful score keeps its curve below the dashed “accept every beat” line. The gap to the dotted oracle line (which ranks beats by their true error, using the reference) shows how much headroom remains. “Valid fraction” coincides with accept-all because almost no beat has missing columns.</p></div>`;
  const clsTable = (c) => `<div class="tscroll"><table class="t"><tr><th>Decision</th><th>Beats</th><th>Share</th><th>Bad-beat rate</th><th>Mean error</th></tr>
    ${["accepted", "uncertain", "rejected", "all"].map((k) => `<tr><td>${chip(k === "all" ? "unable" : k, k)}</td><td>${c[k].n}</td><td>${pct(c[k].fraction)}</td><td>${pct(c[k].bad_rate)}</td><td>${fmt(c[k].mean_err, 3)}</td></tr>`).join("")}</table></div>`;
  html += `<div class="panel"><h2>Accept / uncertain / reject: the operating point the app uses <span class="verd no" style="margin-left:6px">post-hoc</span></h2>${clsTable(cls)}
    <p class="hint">Thresholds ${fmt(op.thresholds.t_reject, 1)} / ${fmt(op.thresholds.t_accept, 1)} were chosen on the <b>validation</b> split so that accepted beats are no worse than a clean render (accepted bad-beat rate ≤ 1%, the digitizer's own error floor of 0.8% on clean images). A “bad” beat is unmatched to a true beat or has NRMSE &gt; 0.10. This rule was added <b>after</b> seeing that the pre-registered one failed (below); the test split was used once, for reporting.</p></div>`;
  if (t.operational_prereg) {
    html += `<div class="panel"><h2>Pre-registered operating point: reported unchanged, and it does nothing <span class="verd no" style="margin-left:6px">negative result</span></h2>${clsTable(t.operational_prereg.classes)}
      <p class="hint">The pre-registered rule accepted every beat whose bad-rate target was ≤ 5%. The overall bad-beat rate (${pct(t.overall.bad_rate_all)}) is already below 5%, so the rule was vacuous: thresholds ${fmt(t.operational_prereg.thresholds.t_reject, 1)} / ${fmt(t.operational_prereg.thresholds.t_accept, 1)}, nothing rejected. The risk–coverage ranking above does not depend on thresholds and is unaffected.</p></div>`;
  }
  const m = t.measurements, hr = m.heart_rate_abs_error_bpm;
  const mrow = (name, a, b, unit, d) => `<tr><td>${name}</td><td>${a.n}</td><td>${fmt(a.median_abs, d)} ${unit}</td><td>${fmt(a.p90_abs, d)} ${unit}</td><td>${b.n}</td><td>${fmt(b.median_abs, d)} ${unit}</td><td>${fmt(b.p90_abs, d)} ${unit}</td></tr>`;
  html += `<div class="panel"><h2>Do the reported measurements get better when you only report accepted beats?</h2><div class="tscroll"><table class="t"><tr><th rowspan="2">Measurement</th><th colspan="3">all matched beats</th><th colspan="3">accepted beats only</th></tr>
    <tr><th>n</th><th>median |err|</th><th>p90 |err|</th><th>n</th><th>median |err|</th><th>p90 |err|</th></tr>
    ${mrow("R amplitude", m.r_amplitude_error_mv.all_matched_beats, m.r_amplitude_error_mv.accepted_beats, "mV", 3)}
    ${mrow("RR interval", m.rr_interval_error_ms.all_matched_pairs, m.rr_interval_error_ms.accepted_pairs, "ms", 1)}
    <tr><td>Heart rate (per image)</td><td>${hr.all_images.n}</td><td>${fmt(hr.all_images.median, 2)} bpm</td><td>${fmt(hr.all_images.p90, 2)} bpm</td><td>${hr.accepted_pairs_only.n}</td><td>${fmt(hr.accepted_pairs_only.median, 2)} bpm</td><td>${fmt(hr.accepted_pairs_only.p90, 2)} bpm</td></tr></table></div>
    <p class="hint">${hr.images_without_a_reportable_hr} image(s) had no reportable heart rate because no RR interval had two accepted bounding beats and a fully observed span. Withholding a measurement is a cost; it is reported here, not hidden.</p></div>`;
  const conds = Object.entries(t.by_condition);
  html += `<div class="panel"><h2>Where it works and where it does not (per image condition)</h2><div class="tscroll"><table class="t"><tr><th>Condition</th><th>images</th><th>unable</th><th>beats</th><th>bad-beat rate (all)</th><th>accepted share</th><th>bad-beat rate (accepted)</th><th>rejected share</th></tr>
    ${conds.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.n_images}</td><td>${v.n_unable}</td><td>${v.n_beats}</td><td>${pct(v.bad_rate_all)}</td><td>${pct(v.accepted_fraction, 0)}</td><td>${pct(v.accepted_bad_rate)}</td><td>${pct(v.rejected_fraction, 0)}</td></tr>`).join("")}</table></div></div>`;
  const ms = Object.entries(t.methods).filter(([k]) => !k.startsWith("ablation"));
  const abl = Object.entries(t.methods).filter(([k]) => k.startsWith("ablation"));
  html += `<div class="panel"><h2>Baselines and ablations (AURC, lower is better)</h2><div class="tscroll"><table class="t"><tr><th>Method</th><th>AURC</th><th>bad-beat rate at 70% coverage</th></tr>
    ${ms.map(([k, v]) => `<tr><td>${esc(M_NAME[k] || k)}</td><td>${fmt(v.aurc, 4)}</td><td>${pct((v.at_coverage["0.7"] || {}).bad_rate)}</td></tr>`).join("")}
    ${abl.map(([k, v]) => `<tr><td>${esc(k.replace("ablation_", "ablation · ").replace("_", " "))}</td><td>${fmt(v.aurc, 4)}</td><td>${pct((v.at_coverage["0.7"] || {}).bad_rate)}</td></tr>`).join("")}</table></div></div>`;
  html += `<div class="panel"><h2>Provenance</h2><p class="hint">Generated ${esc(r.generated_utc)} · ${fmt(r.runtime_s, 0)} s · Python ${esc(r.environment.python)}, OpenCV ${esc(r.environment.opencv)}, scikit-learn ${esc(r.environment.scikit_learn)}.
    Data: ${esc(ds.id)}, lead ${esc(ds.lead)}, CC BY 4.0. ${esc(ds.images)}.
    ${gate ? `Feasibility gate (fixed before evaluation): <b>${gate.passed ? "passed" : "FAILED"}</b>.` : ""}</p></div>`;
  return html;
}

/* ---------------------------------------------------------------- reproduce */
let poll = null;
async function startJob(mode) {
  $("#btnQuick").disabled = $("#btnFullCached").disabled = true;
  $("#job").hidden = false; $("#jobResult").innerHTML = ""; setBar(0, 1);
  $("#jobStage").textContent = "starting…"; $("#jobMsg").textContent = "";
  try { await api("/api/eval/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode }) }); }
  catch (e) { $("#jobStage").textContent = "could not start"; $("#jobMsg").textContent = e.message; $("#btnQuick").disabled = $("#btnFullCached").disabled = false; return; }
  poll = setInterval(pollJob, 1000);
}
function setBar(d, t) { $("#jobBar").style.width = (t ? Math.min(100, (100 * d) / t) : 0) + "%"; }
async function pollJob() {
  let s;
  try { s = await api("/api/eval/status"); } catch (e) { return; }
  $("#jobTime").textContent = fmt(s.elapsed_s, 0) + " s";
  if (s.state === "running") { $("#jobStage").textContent = `${s.stage} · ${s.done}/${s.total}`; $("#jobMsg").textContent = s.message || ""; setBar(s.done, s.total); return; }
  clearInterval(poll); $("#btnQuick").disabled = $("#btnFullCached").disabled = false;
  if (s.state === "error") { $("#jobStage").textContent = "failed"; $("#jobMsg").textContent = s.error; return; }
  $("#jobStage").textContent = "finished"; setBar(1, 1); $("#jobMsg").textContent = `outputs written to ${s.run_dir}`;
  const r = s.summary, c = s.comparison;
  $("#jobResult").innerHTML = `<div class="cards">
      <div class="card"><div class="k">AURC · accept all → trust</div><div class="v">${fmt(r.aurc_accept_all.point, 4)} → ${fmt(r.aurc_trust.point, 4)}</div><div class="s">${r.n_records} records · ${r.n_beats} beats · ${fmt(r.runtime_s, 0)} s</div></div>
      <div class="card"><div class="k">H1 · beats accepting everything</div><div class="v">${verdictPill(r.H1)}</div></div>
      <div class="card"><div class="k">H2 · beats best baseline (${esc(r.best_baseline)})</div><div class="v">${verdictPill(r.H2)}</div></div>
      <div class="card"><div class="k">Matches the shipped results file</div><div class="v">${c ? (c.identical ? '<span class="verd yes">identical</span>' : '<span class="verd no">differs</span>') : "–"}</div>
        <div class="s">${c ? `${c.n_values_compared} values compared · max |diff| ${c.max_abs_difference.toExponential(1)}` : "no shipped file to compare"}</div></div></div>
    ${r.mode === "quick" ? `<p class="hint">With only ${r.n_records} records the hypothesis tests are under-powered (their intervals can include 0). The point of this run is that the pipeline re-creates the shipped quick-preset numbers exactly; the confirmatory result is the full 80-record test above.</p>` : ""}`;
}
$("#btnQuick").addEventListener("click", () => startJob("quick"));
$("#btnFullCached").addEventListener("click", () => startJob("full_cached"));

/* ---------------------------------------------------------------- about */
function renderAbout() {
  $("#aboutBody").innerHTML = `
    <h2>What this is</h2>
    <p>Digitizers turn a picture of a paper ECG into a signal, and then every downstream number is silently trusted. ECG Trust Layer adds a missing step: for each heartbeat it asks <i>how well does the original image support this reconstruction?</i> It then reports measurements only for beats that are verifiable, and says plainly when it cannot.</p>
    <p>The score uses only the image and the extraction: the extracted trace is drawn back onto the picture, ambiguity, continuity, missing columns, grid calibration and image quality are checked. The reference signal is never an input.</p>
    <h2>Intended use</h2>
    <ul><li>Research on ECG digitization: deciding which reconstructed beats are good enough to include in an analysis.</li>
    <li>A transparent quality gate for RR interval, heart rate and R-amplitude measurements.</li></ul>
    <h2>Not intended for</h2>
    <ul><li>Diagnosis, triage or any clinical decision. This is not a medical device and has no clinical validation.</li>
    <li>Interval measurements such as QT or ST analysis, multi-lead layouts or rhythm classification. None of these is implemented.</li></ul>
    <h2>Limitations you should know about</h2>
    <ul>
      <li><b>Synthetic images.</b> Every image in the evaluation is a computer rendering of a PTB-XL signal with simulated degradations (blur, noise, tilt, occlusion, ink strokes…). No photographs of real paper ECGs were used, so real-world performance is unknown.</li>
      <li><b>Evidence, not probability.</b> The trust score ranks beats; 80 does not mean “80% correct”.</li>
      <li><b>Offset.</b> The absolute DC level of a printed trace is unobservable. Amplitudes are measured relative to the pre-QRS baseline.</li>
      <li><b>Visible beats only.</b> A beat that is hidden entirely (for example by occlusion) disappears from the list; it is shown as a gap, never invented.</li>
      <li><b>Single lead</b> (lead II), 10 s strips at standard speed and gain.</li>
      <li><b>Small evaluation.</b> 80 test records from one dataset; see the Evaluation tab for the intervals and for conditions where it does not help.</li>
    </ul>
    <h2>Data and licence</h2>
    <p>PTB-XL v1.0.3 (Wagner et al., PhysioNet), CC BY 4.0. Code under MIT; see NOTICE.md in the repository.</p>`;
}
init();
