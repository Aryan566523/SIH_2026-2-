/* Shared helpers for the TRINETRA pages. All data comes from data.js, which the Python engine generates. */
(function () {
  var D = window.TRINETRA_DATA || null;
  function f(x, d) { if (x == null || isNaN(x)) return "–"; d = d == null ? 1 : d; return (x < 0 ? "−" : "") + Math.abs(x).toFixed(d); }
  function sg(x, d) { d = d == null ? 1 : d; return (x < 0 ? "−" : "+") + Math.abs(x).toFixed(d); }
  function clock(i) { var m = 14 * 60 + i; return String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0"); }
  var UN = { T: "°C", P: "hPa", RH: "%" }, NM = { T: "Temperature", P: "Pressure", RH: "Humidity" }, DEC = { T: 1, P: 1, RH: 0 };
  var TONE = { high: "var(--high)", medium: "var(--med)", low: "var(--med)", event: "var(--real)", ok: "var(--ok)" };

  function key(run) { if (run.status === "fault") return run.incident.severity === "low" ? "medium" : run.incident.severity; return run.status === "event" ? "event" : "ok"; }

  function path(arr, X, Y) { var d = "", pen = false; arr.forEach(function (v, i) { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(i).toFixed(1) + "," + Y(v).toFixed(1); pen = true; }); return d; }

  function band(lo, hi, X, Y) {
    var out = "", i = 0, n = lo.length;
    while (i < n) {
      if (lo[i] == null || hi[i] == null) { i++; continue; }
      var j = i; while (j < n && lo[j] != null && hi[j] != null) j++;
      var up = "", dn = "";
      for (var k = i; k < j; k++) { up += (k === i ? "M" : "L") + X(k).toFixed(1) + "," + Y(hi[k]).toFixed(1); }
      for (k = j - 1; k >= i; k--) { dn += "L" + X(k).toFixed(1) + "," + Y(lo[k]).toFixed(1); }
      out += up + dn + "Z"; i = j;
    }
    return out;
  }

  /* One variable: station trace, what the neighbours say it should be, alert markers. */
  function chart(v, run, o) {
    o = o || {};
    var W = o.W || 640, H = o.H || 168, x0 = 50, x1 = W - 54, y0 = 12, y1 = H - 26, N = run.series[v].length;
    var S = run.series[v], E = run.expected[v];
    var vals = S.filter(function (q) { return q != null; });
    if (E) { vals = vals.concat(E.lo.filter(function (q) { return q != null; }), E.hi.filter(function (q) { return q != null; })); }
    var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    var pad = (mx - mn) * .1 || 1; mn -= pad; mx += pad;
    var X = function (i) { return x0 + (x1 - x0) * i / (N - 1); }, Y = function (q) { return y1 - (y1 - y0) * (q - mn) / (mx - mn); };
    var g = "", t, q;
    for (t = 0; t <= 4; t++) { q = mn + (mx - mn) * t / 4; g += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + Y(q) + '" y2="' + Y(q) + '" stroke="var(--grid)"/><text x="' + (x0 - 7) + '" y="' + (Y(q) + 4) + '" text-anchor="end">' + q.toFixed(DEC[v]) + '</text>'; }
    [0, 30, 60, 90, N - 1].forEach(function (i) { g += '<text x="' + X(i) + '" y="' + (H - 7) + '" text-anchor="' + (i === 0 ? "start" : i === N - 1 ? "end" : "middle") + '">' + clock(i) + '</text>'; });
    // missing data stretches
    var gaps = "", i = 0;
    while (i < N) { if (S[i] == null) { var j = i; while (j < N && S[j] == null) j++; if (j - i >= 2) { gaps += '<rect x="' + X(i) + '" y="' + y0 + '" width="' + (X(j - 1) - X(i)) + '" height="' + (y1 - y0) + '" fill="var(--muted)" opacity=".13"/><text x="' + ((X(i) + X(j - 1)) / 2) + '" y="' + ((y0 + y1) / 2) + '" text-anchor="middle">no data</text>'; } i = j; } else i++; }
    var st = run.status, inc = run.incident, ev = run.event, tone = "", mark = "";
    var faulty = st === "fault" && (inc.vars.indexOf(v) >= 0);
    if (faulty || st === "event") {
      var d = faulty ? inc : ev, from = Math.max(0, d.onset);
      tone = faulty ? (inc.severity === "medium" || inc.severity === "low" ? "var(--med)" : "var(--high)") : "var(--real)";
      var lbl = faulty ? "fault begins " + d.onset_clock : (ev.event_kind || "event").toLowerCase() + " " + clock(from);
      mark = '<rect x="' + X(from) + '" y="' + y0 + '" width="' + Math.max(2, X(Math.min(N - 1, d.end)) - X(from)) + '" height="' + (y1 - y0) + '" fill="' + tone + '" opacity=".12"/><line x1="' + X(from) + '" x2="' + X(from) + '" y1="' + y0 + '" y2="' + y1 + '" stroke="' + tone + '" stroke-dasharray="3 3"/><text x="' + (X(from) + 5) + '" y="' + (y0 + 11) + '" style="fill:' + tone + '">' + lbl + '</text>';
      if (faulty && inc.detect >= 0 && inc.detect < N) mark += '<line x1="' + X(inc.detect) + '" x2="' + X(inc.detect) + '" y1="' + (y1 - 14) + '" y2="' + y1 + '" stroke="' + tone + '" stroke-width="2"/>';
    }
    var exp = "";
    if (E) exp = '<path d="' + band(E.lo, E.hi, X, Y) + '" fill="var(--muted)" opacity=".2"/><path d="' + path(E.med, X, Y) + '" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>';
    var last = null, li = -1; for (i = N - 1; i >= 0; i--) if (S[i] != null) { last = S[i]; li = i; break; }
    var col = faulty ? tone : "var(--accent)";
    var dot = last == null ? "" : '<circle cx="' + X(li) + '" cy="' + Y(last) + '" r="5" fill="' + col + '" stroke="var(--panel)" stroke-width="2"/>';
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + NM[v] + ' over the last two hours">' + g + gaps + mark + exp + '<path d="' + path(S, X, Y) + '" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>' + dot + '</svg>';
  }

  /* fault probability from the fusion model, with the alert threshold */
  function prob(run, o) {
    o = o || {};
    var W = o.W || 640, H = o.H || 92, x0 = 50, x1 = W - 54, y0 = 8, y1 = H - 22, p = run.p, N = p.length;
    var X = function (i) { return x0 + (x1 - x0) * i / (N - 1); }, Y = function (q) { return y1 - (y1 - y0) * q; };
    var g = "";
    [0, .5, 1].forEach(function (q) { g += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + Y(q) + '" y2="' + Y(q) + '" stroke="var(--grid)"/><text x="' + (x0 - 7) + '" y="' + (Y(q) + 4) + '" text-anchor="end">' + q.toFixed(1) + '</text>'; });
    g += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + Y(.7) + '" y2="' + Y(.7) + '" stroke="var(--high)" stroke-dasharray="4 3" opacity=".7"/><text x="' + (x1 + 4) + '" y="' + (Y(.7) + 4) + '" style="fill:var(--high)">alert 0.7</text>';
    [0, 30, 60, 90, N - 1].forEach(function (i) { g += '<text x="' + X(i) + '" y="' + (H - 6) + '" text-anchor="' + (i === 0 ? "start" : i === N - 1 ? "end" : "middle") + '">' + clock(i) + '</text>'; });
    var d = path(p, X, Y), area = d ? d + "L" + X(N - 1) + "," + Y(0) + "L" + X(0) + "," + Y(0) + "Z" : "";
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Fault probability">' + g + '<path d="' + area + '" fill="var(--accent)" opacity=".15"/><path d="' + d + '" fill="none" stroke="var(--accent)" stroke-width="1.8"/></svg>';
  }

  function checkRow(c) {
    var ic = { pass: "✓", fail: "!", warn: "~", na: "–" }[c.state] || "✓";
    return '<div class="check ' + c.state + '"><span class="ic">' + ic + '</span><div><h3>' + c.title + '<span>' + c.tag + '</span></h3><p>' + c.text + '</p></div></div>';
  }

  function eventChecks(run, st) {
    var e = run.event, nn = e.n_neighbours;
    return [{ state: "pass", title: "Quick check on the station", tag: "range · jump · stuck · missing", text: "No impossible values, stuck or missing readings." },
      { state: "pass", title: "Physics check", tag: "T ↔ RH dew point", text: "Temperature, pressure and humidity changed together in a physically consistent way." },
      { state: "warn", title: "Neighbour check", tag: "buddy · 100 km", text: nn ? "Stations within " + Math.round(e.neighbour_km) + " km show the same change. That is weather, not a sensor." : "No neighbour to confirm." },
      { state: "warn", title: "AI check", tag: "IsolationForest + autoencoder", text: "Unusual for this station, but neighbours explain it, so it is kept as a real event." }];
  }

  window.TUI = { D: D, f: f, sg: sg, clock: clock, UN: UN, NM: NM, DEC: DEC, TONE: TONE, key: key, chart: chart, prob: prob, checkRow: checkRow, eventChecks: eventChecks };
})();
