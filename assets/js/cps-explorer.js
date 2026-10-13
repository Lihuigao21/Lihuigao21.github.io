/* Interactive analytic examples for MQC VI and VII. No network requests. */
(function () {
  "use strict";
  const W = (Math.sqrt(3) - 1) / 2;
  const BLUE = "#316583", ORANGE = "#b46c35", GRAY = "#aaaaaa", INK = "#222222";
  const dot = (a, b) => a.reduce((v, x, i) => v + x * b[i], 0);
  const direction = (theta, phi, length = 1) => [length * Math.sin(theta) * Math.cos(phi), length * Math.sin(theta) * Math.sin(phi), length * Math.cos(theta)];
  const symbol = (gamma, r, n) => (1 + (1 + 2 * gamma) * dot(r, n)) / 2;
  const components = r => [(1 + r[2]) / 2, (1 - r[2]) / 2, r[0] / 2, -r[1] / 2];

  function reconstruct(gamma, r, matched = true) {
    const s = 1 + 2 * gamma;
    if (!(s > 0)) throw new RangeError("The CPS radius must be positive.");
    // Four-point Gauss-Legendre nz and eight equispaced azimuths.
    // This product rule integrates the degree-two reconstruction exactly.
    const nz = [-.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526];
    const weights = [.3478548451374538, .6521451548625461, .6521451548625461, .3478548451374538];
    const scale = matched ? 3 / s : s;
    let trace = 0;
    const rebuilt = [0, 0, 0];
    nz.forEach((z, i) => {
      for (let j = 0; j < 8; j++) {
        const phi = 2 * Math.PI * j / 8;
        const n = [Math.sqrt(1 - z * z) * Math.cos(phi), Math.sqrt(1 - z * z) * Math.sin(phi), z];
        const factor = 2 * weights[i] / 16 * symbol(gamma, r, n);
        trace += factor;
        n.forEach((value, k) => { rebuilt[k] += factor * scale * value; });
      }
    });
    const target = components(r), values = components(rebuilt);
    const delta = rebuilt.map((v, i) => v - r[i]);
    const error = Math.max(Math.abs(delta[2]) / 2, Math.hypot(delta[0], delta[1]) / 2, Math.abs(trace - 1) / 2);
    return { trace, bloch: rebuilt, target, values, error };
  }

  function rotation(angle) {
    return { spinor: [Math.cos(angle / 2), Math.sin(angle / 2)], bloch: [Math.sin(angle), 0, Math.cos(angle)] };
  }
  function generator(gamma, nz) { return { matrix: -1, scalar: -(((1 + 2 * gamma) * nz) ** 2) }; }
  // Node can verify the same arithmetic that the browser uses.
  if (typeof module !== "undefined" && module.exports) module.exports = { W, direction, symbol, reconstruct, rotation, generator };
  if (typeof document === "undefined") return;

  const fmt = x => Math.abs(x) < 5e-7 ? "0.000000" : x.toFixed(6);
  function context(canvas, height) {
    const width = Math.max(240, canvas.clientWidth);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.height = height + "px";
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    const c = canvas.getContext("2d");
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, width, height);
    c.font = "13px system-ui, sans-serif";
    c.fillStyle = INK; c.lineWidth = 1;
    return { c, width, height };
  }
  function line(c, a, b, color = GRAY, width = 1) {
    c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.strokeStyle = color; c.lineWidth = width; c.stroke();
  }
  function point(c, x, y, color) { c.beginPath(); c.arc(x, y, 5, 0, 2 * Math.PI); c.fillStyle = color; c.fill(); }
  function arrow(c, a, b, color) {
    line(c, a, b, color, 2);
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    line(c, b, [b[0] - 9 * Math.cos(angle - .4), b[1] - 9 * Math.sin(angle - .4)], color, 2);
    line(c, b, [b[0] - 9 * Math.cos(angle + .4), b[1] - 9 * Math.sin(angle + .4)], color, 2);
  }
  function circlePlot(canvas, v, labels, title) {
    const { c, width, height } = context(canvas, 265);
    const cx = width / 2, cy = 140, radius = Math.min((width - 76) / 2, 88);
    c.fillStyle = INK; c.textAlign = "center"; c.fillText(title, cx, 24);
    c.beginPath(); c.arc(cx, cy, radius, 0, 2 * Math.PI); c.strokeStyle = GRAY; c.lineWidth = 1; c.stroke();
    line(c, [cx - radius - 12, cy], [cx + radius + 12, cy]);
    line(c, [cx, cy - radius - 12], [cx, cy + radius + 12]);
    c.fillStyle = INK; c.fillText(labels[0], cx + radius + 17, cy - 10);
    c.fillText(labels[1], cx + 16, cy - radius - 16);
    arrow(c, [cx, cy], [cx + radius * v[0], cy - radius * v[1]], BLUE);
    point(c, cx + radius * v[0], cy - radius * v[1], BLUE);
    c.fillStyle = INK; c.fillText("(" + fmt(v[0]) + ", " + fmt(v[1]) + ")", cx, height - 10);
  }

  function heatColor(value) {
    // One fixed color scale throughout the explorer: -1 (blue), .5 (white), 2 (red).
    const t = Math.max(0, Math.min(1, (value + 1) / 3));
    const low = [41, 99, 155], high = [159, 34, 53], neutral = [247, 247, 247];
    const a = t <= .5 ? low : neutral, b = t <= .5 ? neutral : high, f = t <= .5 ? t * 2 : (t - .5) * 2;
    return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * f)).join(",") + ")";
  }
  function heatmap(canvas, gamma, r) {
    const { c, width } = context(canvas, 310);
    const left = 38, right = width - 10, top = 25, bottom = 208;
    const cols = 160, rows = 90, dx = (right - left) / cols, dy = (bottom - top) / rows;
    for (let y = 0; y < rows; y++) {
      const nz = 1 - 2 * (y + .5) / rows;
      for (let x = 0; x < cols; x++) {
        const phi = -Math.PI + 2 * Math.PI * (x + .5) / cols;
        const n = [Math.sqrt(1 - nz * nz) * Math.cos(phi), Math.sqrt(1 - nz * nz) * Math.sin(phi), nz];
        c.fillStyle = heatColor(symbol(gamma, r, n));
        c.fillRect(left + x * dx, top + y * dy, dx + .5, dy + .5);
      }
    }
    c.fillStyle = INK; c.textAlign = "right";
    c.fillText("1", left - 8, top + 5); c.fillText("0", left - 8, (top + bottom) / 2 + 4); c.fillText("−1", left - 8, bottom);
    c.textAlign = "left"; c.fillText("n_z", 3, 15);
    c.textAlign = "center"; c.fillText("−π", left, bottom + 19); c.fillText("0", (left + right) / 2, bottom + 19); c.fillText("π", right - 5, bottom + 19);
    c.fillText("Azimuth φ", width / 2, bottom + 40);
    for (let i = 0; i < 120; i++) {
      c.fillStyle = heatColor(-1 + 3 * i / 119); c.fillRect(left + i * (right - left) / 120, 290, (right - left) / 120 + .5, 8);
    }
    c.fillStyle = INK; c.textAlign = "left"; c.fillText("−1", left, 285);
    c.textAlign = "center"; c.fillText("fρ: 0.5", (left + right) / 2, 285);
    c.textAlign = "right"; c.fillText("2", right, 285);
  }

  function generatorPlot(canvas, gamma, nz) {
    const { c, width, height } = context(canvas, 280);
    const l = 42, right = width - 14, top = 25, bottom = 223;
    const yMin = -9, X = n => l + (n + 1) / 2 * (right - l), Y = y => top + y / yMin * (bottom - top);
    c.fillStyle = INK; c.textAlign = "right";
    for (const y of [0, -1, -4, -9]) { line(c, [l, Y(y)], [right, Y(y)], "#dddddd"); c.fillText(String(y), l - 8, Y(y) + 4); }
    c.setLineDash([5, 4]); line(c, [l, Y(-1)], [right, Y(-1)], INK, 2); c.setLineDash([]);
    c.beginPath();
    for (let i = 0; i <= 300; i++) {
      const n = -1 + 2 * i / 300, y = generator(gamma, n).scalar;
      if (i === 0) c.moveTo(X(n), Y(y)); else c.lineTo(X(n), Y(y));
    }
    c.strokeStyle = BLUE; c.lineWidth = 2; c.stroke();
    point(c, X(nz), Y(generator(gamma, nz).scalar), ORANGE);
    c.fillStyle = INK; c.textAlign = "center";
    for (const n of [-1, 0, 1]) c.fillText(String(n), X(n), bottom + 19);
    c.fillText("Bloch direction n_z", width / 2, height - 8);
  }

  function watch(root, draw) {
    root.addEventListener("input", draw);
    root.addEventListener("change", draw);
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(draw).observe(root);
    else window.addEventListener("resize", draw);
    draw();
  }
  function gammaButtons(root, input, draw) {
    root.querySelectorAll("[data-gamma]").forEach(button => button.addEventListener("click", () => {
      input.value = button.dataset.gamma === "W" ? W : Number(button.dataset.gamma);
      // A range slider quantizes W; retain its exact value until the next drag.
      input.dataset.exact = button.dataset.gamma === "W" ? String(W) : "";
      draw();
    }));
    input.addEventListener("input", () => { delete input.dataset.exact; });
  }
  const gammaValue = input => Number(input.dataset.exact || input.value);

  const orbit = document.getElementById("cps-orbit-explorer");
  if (orbit) {
    const slider = orbit.querySelector("input");
    const draw = () => {
      const degrees = Number(slider.value), data = rotation(degrees * Math.PI / 180);
      orbit.querySelector("output").textContent = degrees + "°";
      circlePlot(orbit.querySelector("[data-spinor]"), data.spinor, ["c₀", "c₁"], "Real spinor slice");
      circlePlot(orbit.querySelector("[data-bloch]"), [data.bloch[0], data.bloch[2]], ["nₓ", "n_z"], "Bloch sphere: n_y = 0 section");
      orbit.querySelector(".cps-result").textContent = "c₀ = " + fmt(data.spinor[0]) + ", c₁ = " + fmt(data.spinor[1]) + "; nₓ = " + fmt(data.bloch[0]) + ", n_z = " + fmt(data.bloch[2]) + ".";
    };
    orbit.querySelectorAll("[data-angle]").forEach(button => button.addEventListener("click", () => { slider.value = button.dataset.angle; draw(); }));
    watch(orbit, draw);
  }

  const mapping = document.getElementById("cps-mapping-explorer");
  if (mapping) {
    const gamma = mapping.querySelector('[name="gamma"]'), theta = mapping.querySelector('[name="theta"]');
    const phi = mapping.querySelector('[name="phi"]'), length = mapping.querySelector('[name="length"]');
    const mode = mapping.querySelector("select");
    const draw = () => {
      const g = gammaValue(gamma), t = Number(theta.value), p = Number(phi.value), len = Number(length.value);
      const r = direction(t * Math.PI / 180, p * Math.PI / 180, len), result = reconstruct(g, r, mode.value === "dual");
      const values = { gamma: g.toFixed(6), theta: t.toFixed(1) + "°", phi: p.toFixed(1) + "°", length: len.toFixed(2) };
      mapping.querySelectorAll("output[data-value]").forEach(el => { el.textContent = values[el.dataset.value]; });
      heatmap(mapping.querySelector("canvas"), g, r);
      mapping.querySelectorAll("tbody tr").forEach((row, i) => {
        row.cells[1].textContent = fmt(result.target[i]); row.cells[2].textContent = fmt(result.values[i]);
      });
      mapping.querySelector(".cps-result").textContent = "Maximum matrix-entry error: " + result.error.toExponential(2) + ". Symbol range: [" + fmt((1 - (1 + 2 * g) * len) / 2) + ", " + fmt((1 + (1 + 2 * g) * len) / 2) + "].";
    };
    gammaButtons(mapping, gamma, draw);
    watch(mapping, draw);
  }

  const coupled = document.getElementById("cps-generator-explorer");
  if (coupled) {
    const gamma = coupled.querySelector('[name="gamma"]'), nz = coupled.querySelector('[name="nz"]');
    const draw = () => {
      const g = gammaValue(gamma), z = Number(nz.value), result = generator(g, z);
      coupled.querySelector('[data-value="gamma"]').textContent = g.toFixed(6);
      coupled.querySelector('[data-value="nz"]').textContent = z.toFixed(3);
      generatorPlot(coupled.querySelector("canvas"), g, z);
      coupled.querySelector(".cps-result").textContent = "Matrix generator = −1; scalar generator = " + fmt(result.scalar) + "; pointwise difference = " + fmt(result.scalar - result.matrix) + ".";
    };
    gammaButtons(coupled, gamma, draw);
    watch(coupled, draw);
  }
}());
