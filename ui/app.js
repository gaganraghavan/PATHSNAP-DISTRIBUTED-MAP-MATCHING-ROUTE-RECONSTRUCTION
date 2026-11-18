/* PathSnap evaluator UI (Laptop 2).
 *
 * Talks only to the PathSnap API: GET /health, GET /trips, POST /predict,
 * GET /logs. It holds no model, no trip data and no reference routes. The
 * trip list, the predicted routes and all accuracy numbers come from the API.
 */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  };

  const css = getComputedStyle(document.documentElement);
  const C = (name) => css.getPropertyValue(name).trim();
  const COLORS = { hmm: C("--hmm"), clf: C("--clf"), gps: C("--gps"), net: C("--net") };

  // ------------------------------------------------------------------ API URL
  function defaultApi() {
    const q = new URLSearchParams(location.search).get("api");
    if (q) return q;
    const saved = localStorage.getItem("pathsnap.api");
    if (saved) return saved;
    if (window.PATHSNAP_API) return window.PATHSNAP_API;
    const host = location.protocol.startsWith("http") ? location.hostname : "localhost";
    return `http://${host}:8000`;
  }
  const normalise = (u) => u.trim().replace(/\/+$/, "");
  let API = normalise(defaultApi());

  async function api(path, opts = {}) {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), opts.timeout ?? 15000);
    try {
      const r = await fetch(API + path, {
        method: opts.method || "GET",
        headers: opts.body ? { "Content-Type": "application/json" } : undefined,
        body: opts.body ? JSON.stringify(opts.body) : undefined,
        signal: ctrl.signal,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      return data;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("request timed out");
      throw e;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ------------------------------------------------------------------ state
  const S = { trips: [], selected: null, running: false, results: {}, connected: false, logAfter: 0, logTimer: null };

  // ------------------------------------------------------------------ map
  const map = L.map("map", { zoomControl: true, preferCanvas: true, attributionControl: true }).setView([20, 0], 2);
  // Esri basemaps, as in the Folium maps: CARTO basemaps now need an API key, and
  // tile.openstreetmap.org blocks requests without a Referer (UI opened as a file).
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
  const BASES = {
    dark: L.tileLayer(`${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
      maxZoom: 20, maxNativeZoom: 16, attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    }),
    streets: L.tileLayer(`${ESRI}/World_Street_Map/MapServer/tile/{z}/{y}/{x}`, {
      maxZoom: 20, maxNativeZoom: 18, attribution: "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, USGS, &copy; OpenStreetMap contributors",
    }),
    none: null,
  };
  let base = BASES.dark.addTo(map);
  const renderer = L.canvas({ padding: 0.3 });
  const groups = {
    network: L.layerGroup().addTo(map),
    classifier: L.layerGroup().addTo(map),
    hmm: L.layerGroup().addTo(map),
    gps: L.layerGroup().addTo(map),
  };
  const ORDER = ["network", "classifier", "hmm", "gps"];

  function drawPrediction(p) {
    Object.values(groups).forEach((g) => g.clearLayers());
    // Road network near the track.
    if (p.network.length) {
      L.polyline(p.network, { renderer, color: COLORS.net, weight: 1, opacity: 0.6, interactive: false }).addTo(groups.network);
    }
    // Classifier-only route: dashed amber.
    const clf = p.routes.classifier;
    if (clf.lines.length) {
      L.polyline(clf.lines, { renderer, color: COLORS.clf, weight: 2.5, opacity: 0.95, dashArray: "6 6", lineCap: "butt" })
        .bindTooltip(`Classifier only · ${clf.n_arcs} arcs · ${clf.length_km.toFixed(2)} km`, { sticky: true })
        .addTo(groups.classifier);
    }
    // HMM route: dark casing under the line so it reads on any basemap.
    const hmm = p.routes.hmm;
    if (hmm.lines.length) {
      L.polyline(hmm.lines, { renderer, color: "#0b0d0f", weight: 7, opacity: 0.55, interactive: false }).addTo(groups.hmm);
      L.polyline(hmm.lines, { renderer, color: COLORS.hmm, weight: 4, opacity: 1 })
        .bindTooltip(`Classifier + HMM · ${hmm.n_arcs} arcs · ${hmm.length_km.toFixed(2)} km`, { sticky: true })
        .addTo(groups.hmm);
    }
    // Raw GPS.
    for (const [lat, lon, t] of p.gps) {
      L.circleMarker([lat, lon], { renderer, radius: 2, stroke: false, fillColor: COLORS.gps, fillOpacity: 0.8 })
        .bindTooltip(`t = ${Math.round(t)} s`)
        .addTo(groups.gps);
    }
    if (p.gps.length) {
      const first = p.gps[0], last = p.gps[p.gps.length - 1];
      const pin = (cls, title) => L.divIcon({ className: "", html: `<div class="pin ${cls}" title="${title}"></div>`, iconSize: [14, 14] });
      L.marker([first[0], first[1]], { icon: pin("pin-start", "Start"), keyboard: false }).bindTooltip("Start").addTo(groups.gps);
      L.marker([last[0], last[1]], { icon: pin("pin-end", "End"), keyboard: false }).bindTooltip("End").addTo(groups.gps);
    }
    syncLayers();
    map.fitBounds(p.bounds, { padding: [60, 60] });
  }

  function syncLayers() {
    for (const name of ORDER) {
      const on = document.querySelector(`.chip[data-layer="${name}"]`).classList.contains("on");
      if (on && !map.hasLayer(groups[name])) map.addLayer(groups[name]);
      if (!on && map.hasLayer(groups[name])) map.removeLayer(groups[name]);
    }
    // Keep draw order stable after toggling.
    for (const name of ORDER) if (map.hasLayer(groups[name])) groups[name].eachLayer((l) => l.bringToFront && l.bringToFront());
  }

  document.querySelectorAll(".chip[data-layer]").forEach((b) =>
    b.addEventListener("click", () => {
      b.classList.toggle("on");
      b.setAttribute("aria-pressed", b.classList.contains("on"));
      syncLayers();
    }));
  document.querySelectorAll(".seg[data-base]").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((s) => s.classList.toggle("on", s === b));
      if (base) map.removeLayer(base);
      base = BASES[b.dataset.base];
      if (base) base.addTo(map);
    }));

  // ------------------------------------------------------------------ formatting
  const pct = (x, d = 1) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
  const num = (x, d = 4) => (x == null ? "—" : Number(x).toFixed(d));
  const secs = (x) => (x == null ? "—" : x < 1 ? `${Math.round(x * 1000)} ms` : `${x.toFixed(2)} s`);
  const dur = (s) => {
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(r).padStart(2, "0")}s`;
  };

  // ------------------------------------------------------------------ connection
  function setStatus(state, text) {
    $("#api-status .dot").dataset.state = state;
    $("#api-status-text").textContent = text;
  }

  async function connect() {
    API = normalise($("#api-url").value || API);
    $("#api-url").value = API;
    localStorage.setItem("pathsnap.api", API);
    setStatus("busy", "Connecting…");
    S.connected = false;
    try {
      const h = await api("/health", { timeout: 6000 });
      const t = await api("/trips");
      S.trips = t.trips;
      S.connected = true;
      setStatus("ok", `Connected · ${h.trips} trips (${h.data_dir})`);
      renderTrips();
      startLogs(true);
    } catch (e) {
      setStatus("err", `Cannot reach API: ${e.message}`);
      S.trips = [];
      renderTrips();
      stopLogs();
      clientLog(`connection to ${API} failed: ${e.message}`, "error");
    }
  }
  $("#api-form").addEventListener("submit", (e) => { e.preventDefault(); connect(); });

  // ------------------------------------------------------------------ trip list
  function renderTrips() {
    const list = $("#trip-list");
    list.replaceChildren();
    $("#trip-count").textContent = S.trips.length ? `${S.trips.length} trips` : "–";
    S.trips.forEach((t, i) => {
      const r = S.results[t.trip_id];
      let score;
      if (S.running === t.trip_id) score = el("span", { class: "trip-score running" }, "running");
      else if (r && r.metrics.reference_available) score = el("span", { class: "trip-score done" }, pct(r.metrics.methods.hmm.top1_accuracy));
      else if (!t.evaluable) score = el("span", { class: "trip-score noref", title: "No reference route on the server" }, "no ref");
      else score = el("span", { class: "trip-score" }, "—");
      const btn = el("button", {
        class: "trip", style: `--i:${i}`, "aria-pressed": String(S.selected === t.trip_id),
        onclick: () => select(t.trip_id), ondblclick: () => { select(t.trip_id); run(); },
      },
        el("span", { class: "trip-n" }, String(i + 1).padStart(2, "0")),
        el("span", {}, el("span", { class: "trip-id" }, t.trip_id),
          el("span", { class: "trip-meta" }, `${t.n_obs.toLocaleString()} GPS pts · ${dur(t.duration_s)}`)),
        score);
      list.append(el("li", {}, btn));
    });
    updateRunButton();
  }

  function select(id) {
    if (S.running) return;
    S.selected = id;
    document.querySelectorAll(".trip").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.querySelector(".trip-id").textContent === id)));
    updateRunButton();
    if (S.results[id]) show(S.results[id]);
  }

  function updateRunButton() {
    const b = $("#run");
    const label = b.querySelector(".btn-run-label");
    b.disabled = !S.connected || !S.selected || !!S.running;
    label.textContent = S.running ? `Running trip ${S.running}…`
      : !S.connected ? "Not connected"
      : !S.selected ? "Select a trip"
      : S.results[S.selected] ? `Run trip ${S.selected} again` : `Run trip ${S.selected}`;
  }

  // ------------------------------------------------------------------ run
  let busyTimer = null;
  async function run() {
    if (!S.connected || !S.selected || S.running) return;
    const id = S.selected;
    S.running = id;
    renderTrips();
    $("#busy").hidden = false;
    $("#busy-title").textContent = `Reconstructing trip ${id}`;
    const t0 = performance.now();
    busyTimer = setInterval(() => { $("#busy-timer").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`; }, 100);
    clientLog(`POST ${API}/predict {"trip_id": "${id}"}`, "client");
    $("#log-dot").dataset.state = "busy";
    try {
      const p = await api("/predict", { method: "POST", body: { trip_id: id }, timeout: 600000 });
      p._roundtrip = (performance.now() - t0) / 1000;
      S.results[id] = p;
      clientLog(`response 200 for trip ${id} in ${p._roundtrip.toFixed(2)} s`, "client");
      if (S.selected === id) show(p);
    } catch (e) {
      clientLog(`predict ${id} failed: ${e.message}`, "error");
      setStatus("err", `Prediction failed: ${e.message}`);
    } finally {
      clearInterval(busyTimer);
      $("#busy").hidden = true;
      S.running = false;
      $("#log-dot").dataset.state = S.connected ? "ok" : "idle";
      renderTrips();
      pollLogs();
    }
  }
  $("#run").addEventListener("click", run);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && document.activeElement?.tagName !== "INPUT") { e.preventDefault(); run(); }
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && S.trips.length && document.activeElement?.tagName !== "INPUT") {
      e.preventDefault();
      const i = S.trips.findIndex((t) => t.trip_id === S.selected);
      const j = Math.max(0, Math.min(S.trips.length - 1, (i < 0 ? -1 : i) + (e.key === "ArrowDown" ? 1 : -1)));
      select(S.trips[j].trip_id);
      document.querySelectorAll(".trip")[j]?.focus();
    }
  });

  // ------------------------------------------------------------------ results
  function show(p) {
    $("#empty").hidden = true;
    $("#layers").hidden = false;
    $("#map-title").hidden = false;
    $("#map-trip").textContent = p.trip_id;
    $("#map-sub").textContent = `${p.n_obs.toLocaleString()} GPS points · ${dur(p.duration_s)}` +
      (p.gps_every > 1 ? ` · every ${p.gps_every}th point shown` : "");
    drawPrediction(p);
    renderKpis(p);
    renderTable(p);
    renderFacts(p);
    renderTiming(p);
  }

  function delta(h, c, lowerBetter = false) {
    if (h == null || c == null) return "";
    const d = (h - c) * 100;
    const good = lowerBetter ? d < 0 : d > 0;
    const cls = Math.abs(d) < 0.05 ? "" : good ? "up" : "down";
    const sign = d > 0 ? "+" : d < 0 ? "−" : "±";
    return `<span class="${cls}">${sign}${Math.abs(d).toFixed(1)} pts</span> vs classifier`;
  }

  function renderKpis(p) {
    const m = p.metrics;
    const cards = document.querySelectorAll(".kpi");
    const keys = [["top1_accuracy"], ["macro_f1"], ["route_f1"], ["route_continuity"]];
    cards.forEach((card, i) => {
      const k = keys[i][0];
      const v = card.querySelector(".kpi-value"), d = card.querySelector(".kpi-delta");
      if (!m.reference_available) {
        v.textContent = "n/a";
        d.textContent = "no reference route on server";
      } else {
        const h = m.methods.hmm[k], c = m.methods.classifier[k];
        v.textContent = pct(h);
        d.innerHTML = delta(h, c);
        d.title = c == null ? "" : `Classifier only: ${pct(c)}`;
      }
    });
  }

  const ROWS = [
    ["Observation level"],
    ["Top-1 accuracy", "top1_accuracy"],
    ["Macro precision", "macro_precision"],
    ["Macro recall", "macro_recall"],
    ["Macro F1", "macro_f1"],
    ["Log loss", "observation_log_loss", { lower: true, na: "HMM outputs a path, not probabilities" }],
    ["Sequence / route"],
    ["Route continuity", "route_continuity"],
    ["Route precision", "route_precision"],
    ["Route recall", "route_recall"],
    ["Route F1", "route_f1"],
    ["Route mismatch", "route_mismatch", { lower: true }],
    ["Route length (km)", "route_length_km", { plain: 2 }],
    ["Route breaks", "route_breaks", { int: true, lower: true }],
  ];

  function renderTable(p) {
    const body = $("#cmp tbody");
    body.replaceChildren();
    const m = p.metrics;
    if (!m.reference_available) {
      body.append(el("tr", {}, el("td", { colspan: 3, class: "muted" },
        "This trip has no reference route on the server, so accuracy cannot be computed. The predicted route is still shown on the map.")));
      return;
    }
    const c = m.methods.classifier, h = m.methods.hmm;
    for (const [label, key, o = {}] of ROWS) {
      if (!key) { body.append(el("tr", { class: "group" }, el("td", { colspan: 3 }, label))); continue; }
      const fmt = (x) => (x == null ? "—" : o.int ? String(x) : o.plain ? Number(x).toFixed(o.plain) : num(x));
      const cv = c[key], hv = h[key];
      let wc = "", wh = "";
      if (cv != null && hv != null && !o.plain && cv !== hv) {
        const hBetter = o.lower ? hv < cv : hv > cv;
        wh = hBetter ? "win" : ""; wc = hBetter ? "" : "win-clf";
      }
      const hCell = hv === undefined && o.na ? el("td", { class: "hint", title: o.na }, "n/a") : el("td", { class: wh }, fmt(hv));
      body.append(el("tr", {}, el("td", {}, label), el("td", { class: wc }, fmt(cv)), hCell));
    }
  }

  function renderFacts(p) {
    const m = p.metrics;
    const facts = [
      ["GPS points", p.n_obs.toLocaleString()],
      ["Matched", `${p.n_matched.toLocaleString()}`],
      ["Candidates", p.n_candidates.toLocaleString()],
      ["Evaluated pts", m.reference_available ? m.evaluated_observations.toLocaleString() : "n/a"],
      ["Candidate miss", m.reference_available ? pct(m.candidate_miss_rate, 2) : "n/a"],
      ["Decoder breaks", m.reference_available ? String(m.methods.hmm.decoder_breaks) : "—"],
    ];
    $("#facts").replaceChildren(...facts.map(([k, v]) => el("div", { class: "fact" }, el("span", {}, k), el("b", {}, v))));
  }

  const STAGES = [
    ["load_s", "Load trip", "#4b555c"],
    ["network_s", "Network", "#7c8890"],
    ["features_s", "Features", "#f0c27a"],
    ["classifier_s", "LambdaRank", "#e3a03f"],
    ["hmm_s", "HMM / Viterbi", "#2fbfae"],
    ["route_s", "Route rebuild", "#1f8a7e"],
    ["evaluation_s", "Scoring", "#c9786f"],
  ];
  const RTT = "_roundtrip";
  function renderTiming(p) {
    const t = p.timings;
    const total = STAGES.reduce((a, [k]) => a + (t[k] || 0), 0) || 1;
    $("#timing-total").textContent = secs(t.server_total_s);
    $("#timing-total").title = `server ${secs(t.server_total_s)}, browser round trip ${secs(p._roundtrip)}`;
    $("#timing-bar").replaceChildren(...STAGES.map(([k, label, color]) =>
      el("span", { title: `${label}: ${secs(t[k])}`, style: `flex-grow:${(t[k] || 0) / total};background:${color}` })));
    $("#timing-legend").replaceChildren(...STAGES.map(([k, label, color]) =>
      el("li", {}, el("i", { style: `background:${color}` }), label, el("b", {}, secs(t[k]))),
    ).concat(el("li", { class: "rtt" }, el("i", { style: "background:transparent;border:1px solid var(--muted)" }),
      "Round trip", el("b", {}, secs(p[RTT])))));
  }

  // ------------------------------------------------------------------ logs
  const logEl = $("#log");
  function addLog(time, msg, cls) {
    const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
    const ts = new Date(time * 1000).toLocaleTimeString([], { hour12: false });
    logEl.append(el("li", { class: cls }, el("time", {}, ts), msg));
    while (logEl.children.length > 400) logEl.firstChild.remove();
    if (stick) logEl.scrollTop = logEl.scrollHeight;
  }
  function clientLog(msg, cls = "client") { addLog(Date.now() / 1000, `[ui] ${msg}`, cls); }

  async function pollLogs() {
    if (!S.connected) return;
    try {
      const { lines } = await api(`/logs?after=${S.logAfter}`, { timeout: 5000 });
      for (const l of lines) {
        S.logAfter = Math.max(S.logAfter, l.id);
        const cls = l.level === "ERROR" ? "error" : l.level === "WARNING" ? "warn"
          : l.message.startsWith("EVAL") ? "eval" : l.message.startsWith("PREDICT") ? "predict" : "";
        addLog(l.time, l.message, cls);
      }
      if (!S.running) { $("#log-dot").dataset.state = "ok"; $("#log-state").textContent = "live"; }
    } catch {
      $("#log-dot").dataset.state = "err"; $("#log-state").textContent = "offline";
    }
  }
  function startLogs(reset) {
    stopLogs();
    if (reset) { S.logAfter = 0; logEl.replaceChildren(); }
    pollLogs();
    S.logTimer = setInterval(pollLogs, 1500);
  }
  function stopLogs() { clearInterval(S.logTimer); S.logTimer = null; $("#log-state").textContent = "idle"; $("#log-dot").dataset.state = "idle"; }

  // ------------------------------------------------------------------ boot
  $("#api-url").value = API;
  renderTrips();
  connect();
})();
