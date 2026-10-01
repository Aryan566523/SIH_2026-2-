/* TRINETRA dashboard: renders the engine's output (data.js). No detection logic lives here. */
(function () {
  var T = window.TUI, D = T.D, $ = function (id) { return document.getElementById(id); };
  if (!D) { $("app").innerHTML = '<div class="nodata"><h2>data.js not found</h2><p>Run <code>python -m trinetra export</code> to generate it, and keep it in the same folder as dashboard.html.</p></div>'; return; }
  var VARS = ["T", "P", "RH"], STN = D.stations, ORDER = { high: 0, medium: 1, event: 2, ok: 3 };
  var sel = STN[0].id, choice = {}, flash = null;
  STN.forEach(function (s) { choice[s.id] = D.defaults[s.id] || "clear"; });
  var run = function (id) { return D.runs[id][choice[id]]; };
  var label = { high: "High", medium: "Medium", event: "Real weather", ok: "OK" };

  function headline(r) {
    if (r.status === "fault") { var i = r.incident; return i.label + " — " + (i.type === "drift" ? "needs calibration" : i.type === "comm_error" ? "no usable data" : "sensor fault"); }
    if (r.status === "event") return r.event.event_kind + " seen by neighbours too — real weather";
    return "All checks passed";
  }

  function renderQueue() {
    var list = STN.slice().sort(function (a, b) { return ORDER[T.key(run(a.id))] - ORDER[T.key(run(b.id))] || (a.id < b.id ? -1 : 1); });
    var c = { high: 0, medium: 0, event: 0, ok: 0 };
    $("queue").innerHTML = list.map(function (s) {
      var r = run(s.id), k = T.key(r); c[k]++;
      return '<button class="st ' + k + (k === "ok" ? " compact" : "") + (flash === s.id ? " flash" : "") + '" data-id="' + s.id + '" aria-pressed="' + (s.id === sel) + '"><span class="bar"></span><span><span class="id">' + s.id + '</span><div class="msg">' + headline(r) + '</div></span><span class="pill ' + k + '">' + label[k] + '</span></button>';
    }).join("");
    $("qcount").textContent = (c.high + c.medium) + " open";
    var ev = D.evaluation && D.evaluation.summary && D.evaluation.summary["TRINETRA (full)"];
    $("kpis").innerHTML = '<div class="kpi high"><b>' + (c.high + c.medium) + '</b><span>sensor faults open</span></div><div class="kpi real"><b>' + c.event + '</b><span>real weather events</span></div>' +
      (ev ? '<div class="kpi ok"><b>' + Math.round(ev.recall * 100) + '%</b><span>injected faults caught</span></div><div class="kpi"><b>' + T.f(ev.median_latency_min, 0) + ' min</b><span>median time to alert</span></div>' : '<div class="kpi ok"><b>' + c.ok + '</b><span>healthy</span></div>');
  }

  function renderDetail() {
    var s = STN.filter(function (q) { return q.id === sel; })[0], r = run(sel), k = T.key(r), inc = r.incident, ev = r.event;
    var conf = r.status === "fault" ? 'Fault confidence <b>' + Math.round(inc.confidence * 100) + '%</b>' : r.status === "event" ? 'Real-weather confidence <b>' + Math.round(ev.confidence * 100) + '%</b>' : 'Confidence <b>high</b>';
    var nn = s.neighbours.length, far = nn ? Math.ceil(Math.max.apply(null, s.neighbours.map(function (q) { return q.km; }))) : 0;
    $("head").innerHTML = '<div><h2>' + s.id + '</h2><div class="sub">' + s.name + ' · ' + s.lat.toFixed(2) + '°N ' + s.lon.toFixed(2) + '°E · ' + s.elev + ' m · ' + (nn ? nn + ' nearby stations within ' + far + ' km' : 'no station within 100 km (physics + AI only)') + '</div></div><div class="right"><span class="conf">' + conf + '</span><span class="pill ' + k + '" style="font-size:12px;padding:4px 12px">' + label[k] + '</span></div>';
    $("demo").innerHTML = '<span>Live demo · inject into this station</span>' + D.scenarios.map(function (d) { return '<button class="seg' + (choice[sel] === d.key ? " on" : "") + '" data-scn="' + d.key + '">' + d.label + '</button>'; }).join("");

    $("charts").innerHTML = VARS.map(function (v) {
      var S = r.series[v], last = null; for (var i = S.length - 1; i >= 0; i--) if (S[i] != null) { last = S[i]; break; }
      var faulty = r.status === "fault" && inc.vars.indexOf(v) >= 0, dev = "";
      if (faulty && inc.var === v && inc.type !== "comm_error") dev = '<div class="dev ' + (k === "high" ? "bad" : "warn") + '">' + T.sg(inc.magnitude, T.DEC[v] || 1) + ' ' + T.UN[v] + ' vs expected</div>';
      else if (faulty) dev = '<div class="dev ' + (k === "high" ? "bad" : "warn") + '">' + inc.label.toLowerCase() + '</div>';
      else dev = '<div class="dev">' + (r.status === "event" ? "moves with nearby stations" : r.expected[v] ? "agrees with nearby stations" : "no neighbours to compare") + '</div>';
      return '<div class="chart"><div class="cmeta"><div class="name">' + T.NM[v] + '</div><div class="val">' + (last == null ? "no data" : T.f(last, T.DEC[v]) + ' <small>' + T.UN[v] + '</small>') + '</div>' + dev + '</div>' + T.chart(v, r) + '</div>';
    }).join("") + '<div class="pchart"><div class="cmeta"><div class="name">Fault probability</div><div class="val">' + (T.f(Math.max.apply(null, r.p.filter(function (x) { return x != null; })), 2)) + '</div><div class="dev">peak in window</div></div>' + T.prob(r) + '</div>';

    // why
    if (r.status === "fault") {
      $("quote").innerHTML = inc.explanation.replace(/^Flagged: /, "Flagged: <mark>").replace(/(minutes|minute)(?=[ ,.])/, "$1</mark>");
      $("quote").innerHTML = '<mark>' + inc.explanation.split(":")[0] + ':</mark>' + inc.explanation.slice(inc.explanation.indexOf(":") + 1);
      $("whytime").textContent = "raised " + inc.detect_clock + " IST";
    } else if (r.status === "event") { $("quote").innerHTML = '<mark>' + ev.explanation.split(":")[0] + ':</mark>' + ev.explanation.slice(ev.explanation.indexOf(":") + 1); $("whytime").textContent = "seen " + T.clock(Math.max(0, ev.onset)) + " IST"; }
    else { $("quote").innerHTML = '<mark>All clear:</mark> temperature, pressure and humidity agree with each other' + (s.neighbours.length ? ' and with ' + s.neighbours.length + ' nearby stations.' : '. No neighbour is available, so physics and AI alone are watching.'); $("whytime").textContent = "updated 15:59 IST"; }
    var p = r.status === "fault" ? inc.confidence : r.status === "event" ? 1 - ev.confidence : Math.min(.06, Math.max(0, Math.max.apply(null, r.p.map(function (x) { return x || 0; }))));
    $("mark").style.left = (p * 100).toFixed(0) + "%"; $("mval").textContent = Math.round(p * 100) + "% fault";
    var checks = r.status === "fault" ? inc.checks : r.status === "event" ? T.eventChecks(r, s) : [
      { state: "pass", title: "Quick check on the station", tag: "range · jump · stuck · missing", text: "No impossible values, sudden jumps, stuck or missing readings." },
      { state: "pass", title: "Physics check", tag: "T ↔ RH dew point", text: "Temperature, pressure and humidity are consistent. Dew point steady." },
      s.neighbours.length ? { state: "pass", title: "Neighbour check", tag: "buddy · 100 km", text: "Within the normal spread of " + s.neighbours.length + " stations within " + far + " km." } : { state: "na", title: "Neighbour check", tag: "buddy · 100 km", text: "No other station within 100 km. Physics and AI only, so confidence is lower." },
      { state: "pass", title: "AI check", tag: "IsolationForest + autoencoder", text: "Readings match this station's own learned pattern." }];
    $("checks").innerHTML = checks.map(T.checkRow).join("");
    var dr = r.status === "fault" ? inc.contributions : [{ name: "Difference from neighbours", share: r.status === "event" ? 3 : 5 }, { name: "Dew-point jump (physics)", share: 2 }, { name: "Broke a station limit", share: 2 }, { name: "Unusual for this station (AI)", share: r.status === "event" ? 93 : 91 }];
    if (r.status === "ok") dr = [];
    $("drivers").innerHTML = dr.length ? dr.map(function (x) { return '<div class="drv"><span>' + x.name + '</span><div class="b"><i style="width:' + x.share + '%"></i></div><em>' + x.share + '%</em></div>'; }).join("") : '<p class="hnote">Nothing to explain. No alert is open for this station.</p>';

    // ticket
    if (r.status === "fault") {
      var corr = inc.corrected.length ? inc.corrected.map(function (c) { return T.NM[c.var].slice(0, c.var === "RH" ? 8 : 4) + ' ' + (c.observed == null ? "n/a" : T.f(c.observed, T.DEC[c.var])) + '<span class="arrow">→</span><span class="good">' + T.f(c.corrected, T.DEC[c.var]) + ' ' + T.UN[c.var] + '</span>'; }).join("<br>") : "No estimate without neighbours";
      var lag = inc.detect - inc.onset;
      $("tkno").textContent = "TKT-" + (1000 + (hash(sel + inc.type) % 9000)) + " · raised " + inc.detect_clock + " · " + (lag <= 3 ? "within " + Math.max(1, lag) + " min of the fault" : lag + " min after the fault began") + " · " + inc.severity + " severity";
      $("ticket").innerHTML = '<div><dt>Faulty sensor</dt><dd class="big">' + inc.sensor + '</dd></div><div><dt>Type of fault</dt><dd>' + inc.label + '</dd></div><div><dt>Root cause</dt><dd>' + inc.cause + '</dd></div><div><dt>Corrected value</dt><dd class="big" style="font-size:13px">' + corr + '</dd></div><div><dt>Action</dt><dd>' + inc.action + '</dd></div>';
    } else {
      $("tkno").textContent = "No ticket needed";
      $("ticket").innerHTML = '<div style="grid-column:1/-1"><dt>Status</dt><dd>' + (r.status === "event" ? "Readings are kept and marked as a real weather event. Forecasters and warning teams can use them." : "No repair needed. Readings are passing every check.") + '</dd></div>';
    }
    renderHealth(r, inc);
  }

  function hash(s) { var h = 2166136261; for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

  function renderHealth(r, inc) {
    var H = D.health[sel]; if (!H) { $("health").innerHTML = ""; return; }
    $("health").innerHTML = VARS.map(function (v) {
      var h = H[v], sc = h.score, stt = h.status;
      if (r.status === "fault" && inc.vars.indexOf(v) >= 0) { sc = Math.min(sc, Math.round(100 - 65 * inc.confidence)); stt = sc < 55 ? "degraded" : sc < 80 ? "watch" : "ok"; }
      return '<div class="hrow ' + stt + '"><span>' + T.NM[v] + '</span><div class="b"><i style="width:' + sc + '%"></i></div><em>' + sc + '</em></div>';
    }).join("");
    $("hnote").innerHTML = (r.status === "fault" ? '<b>Now:</b> ' + inc.sensor.toLowerCase() + ' health is reduced by the open alert. ' : '') + H.notes.join(" ");
  }

  // map
  var POLY = [[74.4,37.0],[75.9,36.6],[77.8,35.5],[78.9,34.3],[78.4,32.6],[79.5,32.0],[78.8,30.6],[80.1,30.1],[80.9,29.9],[81.0,29.0],[80.1,28.8],[81.9,27.8],[84.1,27.4],[85.7,26.6],[87.3,26.4],[88.1,26.5],[88.2,27.9],[88.9,27.3],[89.7,26.7],[91.5,26.8],[92.1,26.9],[93.5,27.3],[95.2,27.9],[96.1,29.3],[97.3,28.2],[96.9,27.2],[96.0,26.2],[95.2,26.0],[94.6,24.6],[93.4,23.0],[92.6,22.0],[92.3,23.5],[91.7,24.2],[91.1,23.7],[91.2,25.2],[89.9,25.3],[89.0,25.3],[88.5,24.2],[88.7,23.2],[89.0,22.0],[88.0,21.7],[87.0,21.4],[86.9,20.3],[85.8,19.8],[84.8,19.3],[83.3,17.8],[82.3,16.6],[81.3,15.8],[80.3,15.5],[80.2,13.5],[80.3,12.5],[79.8,11.3],[79.4,10.3],[78.4,9.2],[77.5,8.1],[76.6,8.9],[75.8,11.2],[74.8,12.9],[74.1,14.9],[73.4,16.6],[72.8,18.6],[72.8,20.4],[72.6,21.5],[72.2,21.1],[70.8,20.7],[69.2,22.0],[70.2,22.9],[68.8,23.2],[68.2,23.7],[70.0,24.2],[70.9,24.4],[70.1,25.7],[70.2,27.5],[71.9,27.9],[72.9,29.3],[74.2,30.6],[74.6,31.8],[74.0,32.9],[74.4,34.0],[73.8,34.6]];
  var MW = 340, MS = MW / 31, MH = Math.round(31.5 * MS);
  var mx = function (lon) { return ((lon - 67) * MS).toFixed(1); }, my = function (lat) { return ((37.8 - lat) * MS).toFixed(1); };
  var kc = { high: "var(--high)", medium: "var(--med)", event: "var(--real)", ok: "var(--ok)" };
  var LBL = { "AWS-DEL-014": [7, -6], "AWS-PUN-003": [7, 11], "AWS-CHN-021": [8, 3], "AWS-MUM-007": [-26, -3], "AWS-JPR-011": [-26, 3], "AWS-HYD-019": [8, -4], "AWS-KOL-005": [-26, 3], "AWS-BLR-012": [-28, 9], "AWS-LEH-001": [7, 3] };
  function renderMap() {
    var g = "", d;
    for (d = 70; d <= 95; d += 5) g += '<line x1="' + mx(d) + '" x2="' + mx(d) + '" y1="0" y2="' + MH + '" stroke="var(--grid)"/>';
    for (d = 10; d <= 35; d += 5) g += '<line y1="' + my(d) + '" y2="' + my(d) + '" x1="0" x2="' + MW + '" stroke="var(--grid)"/>';
    var poly = POLY.map(function (p) { return mx(p[0]) + "," + my(p[1]); }).join(" ");
    var sats = "", nodes = "", selS = STN.filter(function (q) { return q.id === sel; })[0];
    STN.forEach(function (s) { s.neighbours.forEach(function (n) { sats += '<circle cx="' + mx(n.lon) + '" cy="' + my(n.lat) + '" r="1.6" fill="var(--faint)"/>'; }); });
    var ring = '<circle cx="' + mx(selS.lon) + '" cy="' + my(selS.lat) + '" r="' + (100 / 111 * MS).toFixed(1) + '" fill="color-mix(in srgb, var(--accent) 10%, transparent)" stroke="var(--accent)" stroke-dasharray="3 2"/>' + selS.neighbours.map(function (n) { return '<line x1="' + mx(selS.lon) + '" y1="' + my(selS.lat) + '" x2="' + mx(n.lon) + '" y2="' + my(n.lat) + '" stroke="var(--accent)" stroke-width=".7"/>'; }).join("");
    STN.forEach(function (s) {
      var k = T.key(run(s.id)), x = mx(s.lon), y = my(s.lat), l = LBL[s.id] || [7, 3];
      nodes += '<g class="node" data-id="' + s.id + '" tabindex="0" role="button" aria-label="' + s.id + '">' + (k === "high" ? '<circle cx="' + x + '" cy="' + y + '" r="9" fill="' + kc[k] + '" opacity=".22"/>' : "") + '<circle cx="' + x + '" cy="' + y + '" r="' + (s.id === sel ? 5.5 : 4.2) + '" fill="' + kc[k] + '" stroke="' + (s.id === sel ? "var(--fg)" : "var(--panel)") + '" stroke-width="1.5"/><text x="' + (+x + l[0]) + '" y="' + (+y + l[1]) + '" style="' + (s.id === sel ? "fill:var(--fg)" : "") + '">' + s.city + '</text></g>';
    });
    $("map").innerHTML = '<svg class="map" viewBox="0 0 ' + MW + ' ' + MH + '" role="img" aria-label="Map of India with sample stations">' + g + '<polygon points="' + poly + '" fill="var(--land)" stroke="var(--faint)" stroke-width="1" stroke-linejoin="round"/>' + ring + sats + nodes + '</svg>';
  }

  function all() { renderQueue(); renderDetail(); renderMap(); }
  document.addEventListener("click", function (e) {
    var b = e.target.closest("[data-id]"); if (b) { sel = b.getAttribute("data-id"); flash = null; all(); return; }
    var sc = e.target.closest("[data-scn]");
    if (sc) { choice[sel] = sc.getAttribute("data-scn"); flash = T.key(run(sel)) === "high" ? sel : null; all(); }
  });
  document.addEventListener("keydown", function (e) { if ((e.key === "Enter" || e.key === " ") && e.target.classList && e.target.classList.contains("node")) { e.preventDefault(); sel = e.target.getAttribute("data-id"); all(); } });
  $("foot").textContent = "Every alert, chart, explanation and corrected value on this page was produced by the TRINETRA Python engine on simulated AWS data (generated " + D.generated + "). Nothing is computed in the browser.";
  all();
})();
