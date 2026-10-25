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

    // Click: read the pixel from the product COG (HTTP range requests via geotiff.js).
    map.on("click", async (ev) => {
      if (!window.GeoTIFF || !window.proj4 || !man.cog) return;
      const y = man.years[+yearInput.value];
      const [E, N] = proj4("EPSG:4326", man.cog.crs_proj4, [ev.latlng.lng, ev.latlng.lat]);
      const popup = L.popup().setLatLng(ev.latlng).setContent("Reading…").openOn(map);
      try {
        const vals = Object.assign(
          await readPixel(man.cog.url.replace("{year}", y), E, N),
          await readPixel(man.cog.change_url.replace("{year}", y), E, N));
        const div = document.createElement("div");
        const t = document.createElement("strong");
        t.textContent = `${y} · ${E.toFixed(0)} E, ${N.toFixed(0)} N (UTM 17N)`;
        div.appendChild(t);
        if (vals.height === null) {
          const d = document.createElement("div"); d.textContent = "No data (outside AOI or water)"; div.appendChild(d);
        } else {
          const f = (v, d) => (v === null || v === undefined ? "–" : v.toFixed(d));
          for (const [a, v] of [
            ["Canopy height", `${f(vals.height, 1)} m`],
            ["90% interval", `${f(vals.lower90, 1)}–${f(vals.upper90, 1)} m`],
            ["Change vs. previous year", `${f(vals.change, 1)} ± ${f(vals.change_sd, 1)} m`],
            ["Quality flag", f(vals.flag, 0)],
          ]) { const d = document.createElement("div"); d.textContent = `${a}: ${v}`; div.appendChild(d); }
        }
        popup.setContent(div);
      } catch (e) {
        popup.setContent("Could not read the product file.");
      }
    });
  }

  const tiffs = {};
  async function readPixel(url, E, N) {
    if (!tiffs[url]) tiffs[url] = GeoTIFF.fromUrl(url).then((t) => t.getImage());
    const img = await tiffs[url];
    const [x0, y1] = img.getOrigin();
    const [rx, ry] = img.getResolution();
    const c = Math.floor((E - x0) / rx), r = Math.floor((N - y1) / ry);
    const out = {};
    const desc = man_desc[url.includes("_change") ? "change" : "main"];
    if (c < 0 || r < 0 || c >= img.getWidth() || r >= img.getHeight()) {
      desc.forEach((d) => { out[d.name] = null; });
      return out;
    }
    const ras = await img.readRasters({ window: [c, r, c + 1, r + 1] });
    desc.forEach((d, i) => {
      const v = ras[i][0];
      out[d.name] = v === d.nodata ? null : v * d.scale;
    });
    return out;
  }
  let man_desc = null;
  document.addEventListener("DOMContentLoaded", async () => {
    try { man_desc = (await (await fetch("data/map/manifest.json")).json()).cog.bands; } catch (e) { /* no map */ }
  });
  document.addEventListener("DOMContentLoaded", init);
})();
