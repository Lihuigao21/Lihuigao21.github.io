/* Actual-label visual explanations. No dynamics or orbital optimization is run. */
(function () {
  "use strict";
  function rotate(v, ax, ay) {
    const y = Math.cos(ax) * v[1] - Math.sin(ax) * v[2];
    const z = Math.sin(ax) * v[1] + Math.cos(ax) * v[2];
    return [Math.cos(ay) * v[0] + Math.sin(ay) * z, y,
      -Math.sin(ay) * v[0] + Math.cos(ay) * z];
  }
  function meanEnergy(energies, w0) {
    return w0 * energies[0] + (1 - w0) * energies[1];
  }
  function norm(v) { return Math.hypot(...v); }
  const math = { rotate, meanEnergy, norm };
  if (typeof module !== "undefined" && module.exports) module.exports = math;
  if (typeof document === "undefined") return;
  const colors = { 1: ["#ffffff", "#8a8e91", 6.5, ""],
    6: ["#555c60", "#40484c", 11, ""],
    7: ["#315f7d", "#244b65", 12, "N"],
    8: ["#a3654a", "#86503a", 12, "O"] };
  const cache = new Map();
  function load(url) {
    if (!cache.has(url)) cache.set(url, fetch(url).then(r => {
      if (!r.ok) throw new Error("Reference data could not be loaded.");
      return r.json();
    }).then(d => {
      if (d.schema !== "molecular-excited-states-example-v1" || d.positions.length !== 30 ||
        d.energies.length !== 2 || !d.energies.every(Number.isFinite)) {
        throw new Error("Reference data have an unexpected shape.");
      }
      return d;
    }));
    return cache.get(url);
  }
  function setupGeometry(el, d) {
    const canvas = el.querySelector("canvas"), ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas is unavailable; the static geometry remains below.");
    const yaw = el.querySelector("[data-yaw]"), pitch = el.querySelector("[data-pitch]");
    const state = el.querySelector("[data-force-state]"), status = el.querySelector("[data-geometry-status]");
    const center = [0, 1, 2].map(k => d.positions.reduce((sum, p) => sum + p[k], 0) / 30);
    const centered = d.positions.map(p => p.map((v, k) => v - center[k]));
    const bondPairs = [], radii = d.display.covalent_radii_angstrom;
    for (let i = 0; i < 30; i += 1) for (let j = 0; j < i; j += 1) {
      const distance = norm(d.positions[i].map((v, k) => v - d.positions[j][k]));
      const cutoff = d.display.bond_threshold_factor * (radii[d.atomic_numbers[i]] + radii[d.atomic_numbers[j]]);
      if (distance > 0.4 && distance < cutoff) bondPairs.push([i, j]);
    }
    const maximumForce = Math.max(...d.forces.flat().map(norm));
    function drawLine(a, b, color, width) {
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    }
    function draw() {
      const ax = Number(pitch.value) * Math.PI / 180, ay = Number(yaw.value) * Math.PI / 180;
      const xyz = centered.map(p => rotate(p, ax, ay));
      const radialExtent = Math.max(...centered.map(norm));
      const scale = Math.min(490, 310) / (2 * radialExtent);
      const points = xyz.map(p => [320 + p[0] * scale, 208 - p[1] * scale, p[2]]);
      ctx.clearRect(0, 0, 640, 430); ctx.fillStyle = "#fbfbfa"; ctx.fillRect(0, 0, 640, 430);
      ctx.font = "16px Arial, Helvetica, sans-serif";
      const objects = bondPairs.map(([i, j]) => ({ z: (points[i][2] + points[j][2]) / 2,
        draw: () => drawLine(points[i], points[j], "#a8aaa9", 4) }));
      if (state.value !== "none") {
        const which = Number(state.value), color = which === 0 ? "#315f7d" : "#a3654a";
        d.forces[which].forEach((f, i) => {
          const v = rotate(f, ax, ay), p = points[i], factor = 45 / maximumForce;
          const end = [p[0] + factor * v[0], p[1] - factor * v[1]];
          objects.push({ z: p[2] + 0.01, draw: () => {
            if (Math.hypot(end[0] - p[0], end[1] - p[1]) < 2) return;
            drawLine(p, end, color, 2);
            const angle = Math.atan2(end[1] - p[1], end[0] - p[0]);
            drawLine(end, [end[0] - 7 * Math.cos(angle - 0.4), end[1] - 7 * Math.sin(angle - 0.4)], color, 2);
            drawLine(end, [end[0] - 7 * Math.cos(angle + 0.4), end[1] - 7 * Math.sin(angle + 0.4)], color, 2);
          } });
        });
      }
      points.forEach((p, i) => objects.push({ z: p[2] + 0.02, draw: () => {
        const [fill, stroke, radius, label] = colors[d.atomic_numbers[i]];
        ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI);
        ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1.3; ctx.stroke();
        if (label) { ctx.fillStyle = "#ffffff"; ctx.textAlign = "center"; ctx.font = "bold 12px Arial"; ctx.fillText(label, p[0], p[1] + 4); }
      } }));
      objects.sort((a, b) => a.z - b.z).forEach(o => o.draw());
      ctx.textAlign = "left"; ctx.font = "16px Arial, Helvetica, sans-serif"; ctx.fillStyle = "#666666";
      ctx.fillText("C: gray    H: white    N: blue    O: brown", 24, 404);
      if (state.value === "none") status.textContent = "One fixed reference geometry. Rotate the view; the nuclei are not propagated.";
      else {
        const which = Number(state.value);
        status.textContent = `S${which} forces. Maximum 3D force magnitude: ${d.derived.maximum_force_norm_eV_per_angstrom[which].toFixed(4)} eV/Å. Both states use one arrow scale.`;
      }
      el.dataset.viewYaw = yaw.value; el.dataset.viewPitch = pitch.value; el.dataset.forceState = state.value;
      canvas.setAttribute("aria-label", status.textContent);
    }
    [yaw, pitch, state].forEach(input => input.addEventListener("input", draw));
    el.querySelector("[data-reset-view]").addEventListener("click", () => {
      yaw.value = "16"; pitch.value = "-24"; state.value = "none"; draw();
    });
    draw(); el.classList.add("is-ready");
  }
  function svgElement(name, attributes, text) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function setupAverage(el, d) {
    const slider = el.querySelector("[data-average-weight]"), svg = el.querySelector("svg");
    const output = el.querySelector("[data-average-value]"), weight = el.querySelector("[data-average-label]");
    const gap = d.derived.gap_eV, y = e => 265 - e / (gap * 1.15) * 210;
    svg.append(svgElement("title", {}, "Frozen-energy state-average illustration"));
    [[0, "S0", "#315f7d"], [gap, "S1", "#a3654a"]].forEach(([energy, label, color]) => {
      svg.append(svgElement("line", { x1: 100, x2: 345, y1: y(energy), y2: y(energy), stroke: color, "stroke-width": 4 }));
      svg.append(svgElement("text", { x: 115, y: y(energy) - 12, fill: color, "font-size": 20 }, label));
    });
    const meanLine = svgElement("line", { x1: 95, x2: 350, stroke: "#666666", "stroke-width": 2, "stroke-dasharray": "7 5" });
    const meanText = svgElement("text", { x: 373, "font-size": 17, fill: "#666666" });
    svg.append(meanLine, meanText);
    function update() {
      const w0 = Number(slider.value), energy = meanEnergy(d.energies, w0);
      const relative = (1 - w0) * gap;
      meanLine.setAttribute("y1", y(relative)); meanLine.setAttribute("y2", y(relative));
      meanText.setAttribute("y", y(relative) + 6); meanText.textContent = `${relative.toFixed(4)} eV`;
      weight.textContent = `w0 = ${w0.toFixed(2)}, w1 = ${(1 - w0).toFixed(2)}`;
      output.textContent = `${energy.toFixed(7)} Ha`;
      el.dataset.meanHa = String(energy); el.dataset.weightS0 = String(w0);
    }
    slider.addEventListener("input", update);
    el.querySelector("[data-reset-weight]").addEventListener("click", () => { slider.value = "0.5"; update(); });
    update(); el.classList.add("is-ready");
  }
  document.querySelectorAll("[data-molecular-example]").forEach(el => {
    load(el.dataset.exampleSrc).then(data => {
      if (el.hasAttribute("data-molecular-geometry")) setupGeometry(el, data);
      if (el.hasAttribute("data-molecular-average")) setupAverage(el, data);
    }).catch(error => {
      const status = el.querySelector("[data-widget-error]");
      if (status) status.textContent = `${error.message} The static figure and reference data remain available.`;
    });
  });
}());
