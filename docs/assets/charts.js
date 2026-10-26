// Minimal SVG chart kit for the project site (no dependencies).
// Conventions: thin marks, hairline grid, legend for >= 2 series, hover tooltips with
// keyboard focus, and a table view for every chart. Text never uses series colours.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const SLOT = (i) => `var(--series-${i + 1})`;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, parent, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(min, max, n) {
    const span = max - min || 1;
    const step0 = span / (n || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const lo = Math.floor(min / step) * step;
    const hi = Math.ceil(max / step) * step;
    const out = [];
    for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? "–" :
    Number(v).toLocaleString(undefined, { maximumFractionDigits: d === undefined ? 2 : d,
                                          minimumFractionDigits: d === undefined ? 0 : d });

  // Tooltip shared by a figure.
  function tooltip(plot) {
    const tt = h("div", "tooltip", plot);
    return {
      show(x, y, title, rows) {
        tt.replaceChildren();
        if (title) h("div", "tt-title", tt, title);
        for (const r of rows) {
          const row = h("div", "tt-row", tt);
          if (r.color) {
            const k = h("span", "", row);
            k.style.cssText = `width:14px;height:2px;border-radius:1px;background:${r.color};display:inline-block`;
          }
          h("strong", "", row, r.value);
          if (r.name) h("span", "name", row, r.name);
        }
        tt.style.display = "block";
        const pw = plot.clientWidth;
        const tw = tt.offsetWidth;
        tt.style.left = Math.min(Math.max(x + 12, 0), pw - tw - 4) + "px";
        tt.style.top = Math.max(y - tt.offsetHeight - 8, 0) + "px";
      },
      hide() { tt.style.display = "none"; },
    };
  }

  // Figure shell: title, subtitle, table toggle, caption.
  function figure(container, opts) {
    const fig = h("figure", "chart", container);
    const head = h("div", "head", fig);
    const tbox = h("div", "", head);
    h("div", "title", tbox, opts.title || "");
    if (opts.subtitle) h("div", "subtitle", tbox, opts.subtitle);
    const btn = h("button", "", head, "Table");
    btn.setAttribute("aria-pressed", "false");
    const legend = h("div", "legend", fig);
    const plot = h("div", "plot", fig);
    const tw = h("div", "tablewrap", fig);
    tw.hidden = true;
    if (opts.caption) h("figcaption", "", fig, opts.caption);
    btn.addEventListener("click", () => {
      const on = tw.hidden;
      tw.hidden = !on;
      plot.hidden = on;
      btn.setAttribute("aria-pressed", String(on));
      btn.textContent = on ? "Chart" : "Table";
    });
    return { fig, plot, legend, tablewrap: tw };
  }

  function table(wrap, header, rows) {
    const t = h("table", "data", wrap);
    const tr = h("tr", "", h("thead", "", t));
    header.forEach((c) => h("th", "", tr, c));
    const tb = h("tbody", "", t);
    rows.forEach((r) => {
      const row = h("tr", "", tb);
      r.forEach((c) => h("td", "", row, typeof c === "number" ? fmt(c) : c));
    });
    return t;
  }

  function legendKeys(legend, series, kind) {
    if (series.length < 2) return;
    series.forEach((s, i) => {
      const k = h("span", "key", legend);
      const sw = h("span", kind === "line" ? "ln" : "sw", k);
      sw.style.background = s.color || SLOT(i);
      h("span", "", k, s.name);
    });
  }

  function frame(svg, W, H, m, x0, x1, y0, y1, xTitle, yTitle, xTicks, yTicks, xFmt) {
    const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    for (const t of yTicks) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "gridline" }, svg);
      el("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }, svg).textContent = fmt(t);
    }
    if (xTicks) for (const t of xTicks) {
      el("text", { x: sx(t), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent =
        xFmt ? xFmt(t) : fmt(t);
    }
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline" }, svg);
    if (xTitle) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" }, svg).textContent = xTitle;
    if (yTitle) el("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axis-title",
                             transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg).textContent = yTitle;
    return { sx, sy };
  }

  // Grouped columns: categories on x, one column per series.
  function columns(container, o) {
    const f = figure(container, o);
    const W = 640, H = o.height || 300, m = { l: 56, r: 12, t: 12, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title }, f.plot);
    const all = o.series.flatMap((s) => s.values).filter((v) => v !== null);
    const yTicks = niceTicks(Math.min(0, ...all), Math.max(...all), 5);
    const y0 = yTicks[0], y1 = yTicks[yTicks.length - 1];
    const { sy } = frame(svg, W, H, m, 0, 1, y0, y1, o.xTitle, o.yTitle, null, yTicks);
    const n = o.categories.length, k = o.series.length;
    const band = (W - m.l - m.r) / n;
    const bw = Math.min(24, (band * 0.7) / k);
    const tt = tooltip(f.plot);
    o.categories.forEach((c, i) => {
      const cx = m.l + band * (i + 0.5);
      el("text", { x: cx, y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = c;
      const g = el("g", { tabindex: 0, role: "listitem" }, svg);
      const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, class: "hit" }, g);
      o.series.forEach((s, j) => {
        const v = s.values[i];
        if (v === null || v === undefined) return;
        const x = cx - (k * bw + (k - 1) * 2) / 2 + j * (bw + 2);
        const yv = sy(v), yb = sy(Math.max(0, y0));
        const top = Math.min(yv, yb), hh = Math.abs(yb - yv);
        const r = Math.min(4, hh / 2, bw / 2);
        // rounded data-end, square at baseline
        const d = v >= 0
          ? `M${x},${yb} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${yb} Z`
          : `M${x},${yb} V${yb + hh - r} Q${x},${yb + hh} ${x + r},${yb + hh} H${x + bw - r} Q${x + bw},${yb + hh} ${x + bw},${yb + hh - r} V${yb} Z`;
        el("path", { d, fill: s.color || SLOT(j), class: "mark" }, g);
      });
      const showTip = (ev) => {
        const r = f.plot.getBoundingClientRect();
        const px = ev && ev.clientX !== undefined ? ev.clientX - r.left : (cx / W) * r.width;
        const py = ev && ev.clientY !== undefined ? ev.clientY - r.top : 40;
        tt.show(px, py, c, o.series.map((s, j) => ({ color: s.color || SLOT(j), value: fmt(s.values[i], o.digits), name: s.name })));
      };
      g.addEventListener("pointermove", showTip);
      g.addEventListener("focus", () => showTip());
      g.addEventListener("pointerleave", () => tt.hide());
      g.addEventListener("blur", () => tt.hide());
    });
    legendKeys(f.legend, o.series, "rect");
    table(f.tablewrap, [o.xTitle || "", ...o.series.map((s) => s.name)],
      o.categories.map((c, i) => [c, ...o.series.map((s) => s.values[i])]));
    return f;
  }

  // Multi-series line chart with crosshair; x numeric.
  function lines(container, o) {
    const f = figure(container, o);
    const W = 640, H = o.height || 300, m = { l: 56, r: 16, t: 12, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title }, f.plot);
    const ys = o.series.flatMap((s) => s.values).filter((v) => v !== null && !Number.isNaN(v));
    const yTicks = niceTicks(o.yMin !== undefined ? o.yMin : Math.min(...ys), o.yMax !== undefined ? o.yMax : Math.max(...ys), 5);
    const xTicks = o.xTicks || niceTicks(Math.min(...o.x), Math.max(...o.x), 6);
    const x0 = o.xMin !== undefined ? o.xMin : Math.min(...o.x, ...xTicks);
    const x1 = o.xMax !== undefined ? o.xMax : Math.max(...o.x, ...xTicks);
    const { sx, sy } = frame(svg, W, H, m, x0, x1, yTicks[0], yTicks[yTicks.length - 1], o.xTitle, o.yTitle, xTicks, yTicks, o.xFmt);
    if (o.diag) el("line", { x1: sx(Math.max(x0, yTicks[0])), y1: sy(Math.max(x0, yTicks[0])), x2: sx(Math.min(x1, yTicks[yTicks.length - 1])), y2: sy(Math.min(x1, yTicks[yTicks.length - 1])), stroke: "var(--axis)", "stroke-width": 1 }, svg);
    if (o.hline !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: sy(o.hline), y2: sy(o.hline), stroke: "var(--axis)", "stroke-width": 1 }, svg);
    o.series.forEach((s, j) => {
      const pts = o.x.map((x, i) => [x, s.values[i]]).filter((p) => p[1] !== null && !Number.isNaN(p[1]));
      el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0])},${sy(p[1])}`).join(" "), fill: "none",
                   stroke: s.color || SLOT(j), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      pts.forEach((p) => el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: s.color || SLOT(j), stroke: "var(--surface-1)", "stroke-width": 2 }, svg));
    });
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
    const tt = tooltip(f.plot);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit", tabindex: 0 }, svg);
    let idx = 0;
    const show = (i) => {
      idx = i;
      const X = sx(o.x[i]);
      cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.setAttribute("visibility", "visible");
      const r = f.plot.getBoundingClientRect();
      tt.show((X / W) * r.width, 30, (o.xFmt ? o.xFmt(o.x[i]) : fmt(o.x[i])) + (o.xUnit || ""),
        o.series.map((s, j) => ({ color: s.color || SLOT(j), value: fmt(s.values[i], o.digits), name: s.name })));
    };
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const X = ((ev.clientX - r.left) / r.width) * W;
      let best = 0, bd = Infinity;
      o.x.forEach((x, i) => { const d = Math.abs(sx(x) - X); if (d < bd) { bd = d; best = i; } });
      show(best);
    });
    hit.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight") show(Math.min(idx + 1, o.x.length - 1));
      if (ev.key === "ArrowLeft") show(Math.max(idx - 1, 0));
    });
    hit.addEventListener("focus", () => show(idx));
    const hide = () => { tt.hide(); cross.setAttribute("visibility", "hidden"); };
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("blur", hide);
    legendKeys(f.legend, o.series, "line");
    table(f.tablewrap, [o.xTitle || "x", ...o.series.map((s) => s.name)],
      o.x.map((x, i) => [o.xFmt ? o.xFmt(x) : x, ...o.series.map((s) => s.values[i])]));
    return f;
  }

  // 2-D density heatmap (sequential blue, log counts) with optional 1:1 line.
  function heatmap(container, o) {
    const f = figure(container, o);
    const W = 460, H = 420, m = { l: 56, r: 16, t: 12, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title }, f.plot);
    const xe = o.xEdges, ye = o.yEdges;
    const xTicks = niceTicks(xe[0], xe[xe.length - 1], 5);
    const yTicks = niceTicks(ye[0], ye[ye.length - 1], 5);
    const { sx, sy } = frame(svg, W, H, m, xe[0], xe[xe.length - 1], ye[0], ye[ye.length - 1], o.xTitle, o.yTitle, xTicks.filter((t) => t <= xe[xe.length - 1]), yTicks.filter((t) => t <= ye[ye.length - 1]));
    const maxc = Math.max(...o.z.flat());
    const steps = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
    const tt = tooltip(f.plot);
    const total = o.z.flat().reduce((a, b) => a + b, 0);
    o.z.forEach((row, yi) => row.forEach((c, xi) => {
      if (!c) return;
      const k = Math.min(steps.length - 1, Math.floor((Math.log(c) / Math.log(maxc + 1)) * steps.length));
      const r = el("rect", { x: sx(xe[xi]), y: sy(ye[yi + 1]), width: sx(xe[xi + 1]) - sx(xe[xi]) + 0.3,
                             height: sy(ye[yi]) - sy(ye[yi + 1]) + 0.3, fill: `var(${steps[k]})` }, svg);
      r.addEventListener("pointermove", (ev) => {
        const b = f.plot.getBoundingClientRect();
        tt.show(ev.clientX - b.left, ev.clientY - b.top,
          `${o.xTitle}: ${fmt(xe[xi], 1)}–${fmt(xe[xi + 1], 1)}; ${o.yTitle}: ${fmt(ye[yi], 1)}–${fmt(ye[yi + 1], 1)}`,
          [{ value: fmt(c, 0), name: `samples (${fmt((100 * c) / total, 2)} %)` }]);
      });
      r.addEventListener("pointerleave", () => tt.hide());
    }));
    if (o.diag) {
      const lo = Math.max(xe[0], ye[0]), hi = Math.min(xe[xe.length - 1], ye[ye.length - 1]);
      el("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), stroke: "var(--text-secondary)", "stroke-width": 1 }, svg);
    }
    const lg = f.legend;
    h("span", "", lg, "fewer");
    steps.forEach((s) => { const sw = h("span", "sw", lg); sw.style.background = `var(${s})`; sw.style.borderRadius = "0"; });
    h("span", "", lg, "more samples (log scale)");
    const rows = [];
    o.z.forEach((row, yi) => row.forEach((c, xi) => { if (c) rows.push([`${fmt(xe[xi], 1)}–${fmt(xe[xi + 1], 1)}`, `${fmt(ye[yi], 1)}–${fmt(ye[yi + 1], 1)}`, c]); }));
    table(f.tablewrap, [o.xTitle, o.yTitle, "samples"], rows);
    return f;
  }

  function themeToggle(btn) {
    const root = document.documentElement;
    const cur = () => root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const label = () => { btn.textContent = cur() === "dark" ? "Light mode" : "Dark mode"; };
    label();
    btn.addEventListener("click", () => { root.dataset.theme = cur() === "dark" ? "light" : "dark"; label();
      document.dispatchEvent(new CustomEvent("themechange")); });
  }

  window.Viz = { columns, lines, heatmap, table, figure, fmt, themeToggle, h };
})();
