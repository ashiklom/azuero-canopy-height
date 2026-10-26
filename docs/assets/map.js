// Interactive product viewer: Leaflet base map + per-year PNG overlays (Web Mercator).
(function () {
  "use strict";
  async function init() {
    const box = document.getElementById("map");
    if (!box || !window.L) return;
    const man = await (await fetch("data/map/manifest.json")).json();
    const map = L.map(box, { zoomControl: true, attributionControl: true });
    const imagery = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 17, attribution: "Imagery © Esri, Maxar, Earthstar Geographics" });
    const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 17, attribution: "© OpenStreetMap contributors" });
    imagery.addTo(map);
    L.control.layers({ "Satellite imagery": imagery, "OpenStreetMap": osm }, {}, { position: "topright" }).addTo(map);
    map.fitBounds(man.bounds);

    const yearInput = document.getElementById("map-year");
    const yearLabel = document.getElementById("map-year-label");
    const layerSel = document.getElementById("map-layer");
    const opacity = document.getElementById("map-opacity");
    const legend = document.getElementById("map-legend");
    yearInput.min = 0; yearInput.max = man.years.length - 1; yearInput.value = man.years.length - 1;
    for (const [k, v] of Object.entries(man.layers)) {
      const o = document.createElement("option"); o.value = k; o.textContent = v.label; layerSel.appendChild(o);
    }
    let overlay = null;
    function draw() {
      const y = man.years[+yearInput.value];
      const key = layerSel.value;
      const lay = man.layers[key];
      const yrs = lay.years || man.years;
      yearLabel.textContent = yrs.includes(y) ? String(y) : `${y} (not available for this layer)`;
      const url = `data/map/${key}_${y}.png`;
      if (overlay) map.removeLayer(overlay);
      if (!yrs.includes(y)) { overlay = null; return; }
      overlay = L.imageOverlay(url, man.bounds, { opacity: +opacity.value, interactive: false }).addTo(map);
      legend.replaceChildren();
      const lo = document.createElement("span"); lo.textContent = lay.min_label; legend.appendChild(lo);
      const bar = document.createElement("span"); bar.className = "bar";
      bar.style.background = `linear-gradient(to right, ${lay.colors.join(",")})`; legend.appendChild(bar);
      const hi = document.createElement("span"); hi.textContent = lay.max_label; legend.appendChild(hi);
      const u = document.createElement("span"); u.className = "muted"; u.textContent = lay.unit; legend.appendChild(u);
    }
    [yearInput, layerSel].forEach((e) => e.addEventListener("input", draw));
    opacity.addEventListener("input", () => overlay && overlay.setOpacity(+opacity.value));
    draw();

    // Click: read the pixel values from the product COGs (HTTP range requests via geotiff.js).
    map.on("click", async (ev) => {
      if (!window.GeoTIFF || !window.proj4 || !man.cog) return;
      const y = man.years[+yearInput.value];
      const [E, N] = proj4("EPSG:4326", man.cog.crs_proj4, [ev.latlng.lng, ev.latlng.lat]);
      const popup = L.popup().setLatLng(ev.latlng).setContent("Reading…").openOn(map);
      try {
        const v = {};
        for (const k of Object.keys(man.cog.files)) {
          if (k === "change" && y === man.years[0]) { v[k] = null; continue; }
          v[k] = await readPixel(man.cog.files[k].replace("{year}", y), E, N,
                                 man.cog.scale[k], man.cog.nodata[k]);
        }
        const div = document.createElement("div");
        const t = document.createElement("strong");
        t.textContent = `${y} · ${E.toFixed(0)} E, ${N.toFixed(0)} N (UTM 17N)`;
        div.appendChild(t);
        const f = (x, d) => (x === null || x === undefined ? "–" : x.toFixed(d));
        const lines = [];
        if (v.height === null) {
          lines.push(["", "No data (outside the AOI or water)"]);
        } else {
          const z = 1.645;
          lines.push(["Canopy height", `${f(v.height, 1)} m`]);
          if (v.sd !== null) lines.push(["90% interval", `${f(Math.max(v.height - z * v.sd, 0), 1)}–${f(v.height + z * v.sd, 1)} m`]);
          if (v.change !== null) {
            const csd = changeSD(v.height);
            lines.push(["Change vs. previous year", `${f(v.change, 1)} m (± ${f(csd, 1)} m, 1 SD)`]);
          }
          if (man.flags && man.flags[y] !== undefined) lines.push(["Quality flag", String(man.flags[y])]);
        }
        for (const [a, b] of lines) { const d = document.createElement("div"); d.textContent = a ? `${a}: ${b}` : b; div.appendChild(d); }
        popup.setContent(div);
      } catch (e) {
        popup.setContent("Could not read the product file.");
      }
    });

    function changeSD(h) {
      for (const b of man.change_sd || []) if (h >= b.lo && h < b.hi) return b.sd;
      return null;
    }
  }

  const tiffs = {};
  async function readPixel(url, E, N, scale, nodata) {
    if (!tiffs[url]) tiffs[url] = GeoTIFF.fromUrl(url).then((t) => t.getImage());
    const img = await tiffs[url];
    const [x0, y1] = img.getOrigin();
    const [rx, ry] = img.getResolution();
    const c = Math.floor((E - x0) / rx), r = Math.floor((N - y1) / ry);
    if (c < 0 || r < 0 || c >= img.getWidth() || r >= img.getHeight()) return null;
    const ras = await img.readRasters({ window: [c, r, c + 1, r + 1] });
    const val = ras[0][0];
    return val === nodata ? null : val * scale;
  }
  document.addEventListener("DOMContentLoaded", init);
})();
