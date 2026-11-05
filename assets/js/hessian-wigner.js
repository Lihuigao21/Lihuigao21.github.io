/* Independent-oscillator thermal Gaussians; no nuclear/electronic propagation. */
(function () {
  "use strict";

  const HBAR = 1.054571817e-34;
  const AMU = 1.66053906660e-27;
  const C = 29979245800;
  const KB = 1.380649e-23;
  const EV = 1.602176634e-19;
  const REF = 2 * Math.PI * C * 1000;
  const BASE_SEED = 2026110501;
  const BLUE = "#315f85";
  const BROWN = "#925e35";
  const INK = "#292d30";
  const GRAY = "#90979c";
  const TEACHING = "Strict harmonic Gaussian teaching figure; not a replay of SHARC 4.1 finite-temperature output.";

  function finite(value, label) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(label + " must be finite.");
    }
    return value;
  }

  function thermal(freq, temperature, hbar, speedOfLight) {
    finite(freq, "Frequency");
    finite(temperature, "Temperature");
    if (freq <= 0 || temperature < 0) throw new Error("Frequency must be positive and temperature nonnegative.");
    const omega = 2 * Math.PI * speedOfLight * freq;
    const x = temperature === 0 ? Infinity : hbar * omega / (2 * KB * temperature);
    const coth = x > 350 ? 1 : 1 + 2 / Math.expm1(2 * x);
    return { omega: omega, coth: coth, sigmaQ2: hbar * coth / (2 * omega), sigmaP2: hbar * omega * coth / 2 };
  }

  function variance(freq, temperature) {
    const result = thermal(freq, temperature, HBAR, C);
    const classicalFactor = 2 * KB * temperature / (HBAR * result.omega);
    function scaled(factor) {
      const uVariance = REF * factor / result.omega;
      const vVariance = result.omega * factor / REF;
      return {
        uVariance: uVariance, vVariance: vVariance,
        uRms: Math.sqrt(uVariance), vRms: Math.sqrt(vVariance),
        energy_eV: HBAR * result.omega * factor / (2 * EV)
      };
    }
    result.quantum = scaled(result.coth);
    result.classical = scaled(classicalFactor);
    result.classical.energy_eV = KB * temperature / EV;
    return result;
  }

  function validateData(data) {
    if (!data || !Array.isArray(data.symbols) || data.symbols.length < 1) throw new Error("Missing atom symbols.");
    const n = data.symbols.length;
    const modes = data.modes_mass_weighted;
    const frequencies = data.frequencies_cm1;
    if (!Array.isArray(frequencies) || !frequencies.length || !Array.isArray(modes) || modes.length !== frequencies.length) {
      throw new Error("Normal-mode frequencies and vectors do not match.");
    }
    function xyz(rows, label) {
      if (!Array.isArray(rows) || rows.length !== n) throw new Error(label + " has the wrong atom count.");
      rows.forEach(function (row) {
        if (!Array.isArray(row) || row.length !== 3) throw new Error(label + " needs three Cartesian components.");
        row.forEach(function (v) { finite(v, label); });
      });
    }
    xyz(data.coordinates_angstrom, "Equilibrium geometry");
    if (!Array.isArray(data.masses_amu) || data.masses_amu.length !== n) throw new Error("Missing atomic masses.");
    data.masses_amu.forEach(function (m) { if (finite(m, "Mass") <= 0) throw new Error("Atomic masses must be positive."); });
    modes.forEach(function (mode, i) {
      xyz(mode, "Mode " + (i + 1));
      if (finite(frequencies[i], "Frequency") <= 0) throw new Error("Only positive vibrational modes can be displayed.");
      const norm = mode.reduce(function (a, row) { return a + row.reduce(function (b, v) { return b + v * v; }, 0); }, 0);
      if (Math.abs(norm - 1) > 5e-4) throw new Error("Mass-weighted mode " + (i + 1) + " is not normalized.");
    });
    ["hbar_Js", "amu_kg", "c_cm_s"].forEach(function (key) {
      if (!data.constants || finite(data.constants[key], key) <= 0) throw new Error("Missing physical constant " + key + ".");
    });
    if (data.bonds !== undefined) {
      if (!Array.isArray(data.bonds)) throw new Error("Invalid bond list.");
      data.bonds.forEach(function (bond) {
        if (!Array.isArray(bond) || bond.length !== 2 || bond.some(function (i) { return !Number.isInteger(i) || i < 0 || i >= n; })) {
          throw new Error("Bond indices must be zero-based atom indices.");
        }
      });
    }
    return data;
  }

  function reconstruct(data, xis, etas, temperature) {
    validateData(data);
    const count = data.frequencies_cm1.length;
    if (!Array.isArray(xis) || !Array.isArray(etas) || xis.length !== count || etas.length !== count) {
      throw new Error("Position and momentum draws must each cover every mode.");
    }
    const displacements = data.symbols.map(function () { return [0, 0, 0]; });
    const velocities = data.symbols.map(function () { return [0, 0, 0]; });
    const qs = [], ps = [], sigmaQ = [], sigmaP = [];
    data.frequencies_cm1.forEach(function (freq, k) {
      const t = thermal(freq, temperature, data.constants.hbar_Js, data.constants.c_cm_s);
      sigmaQ[k] = Math.sqrt(t.sigmaQ2);
      sigmaP[k] = Math.sqrt(t.sigmaP2);
      qs[k] = sigmaQ[k] * finite(xis[k], "Position draw");
      ps[k] = sigmaP[k] * finite(etas[k], "Momentum draw");
      data.symbols.forEach(function (_, i) {
        const inverseSqrtMass = 1 / Math.sqrt(data.masses_amu[i] * data.constants.amu_kg);
        for (let a = 0; a < 3; a += 1) {
          const e = data.modes_mass_weighted[k][i][a];
          displacements[i][a] += e * qs[k] * inverseSqrtMass / 1e-10;
          velocities[i][a] += e * ps[k] * inverseSqrtMass;
        }
      });
    });
    function rms(rows) {
      return Math.sqrt(rows.reduce(function (sum, row) { return sum + row.reduce(function (a, b) { return a + b * b; }, 0); }, 0) / rows.length);
    }
    return {
      positions_angstrom: data.coordinates_angstrom.map(function (row, i) { return row.map(function (v, a) { return v + displacements[i][a]; }); }),
      displacements_angstrom: displacements,
      velocities_m_s: velocities,
      normal_coordinates_SI: qs,
      normal_momenta_SI: ps,
      sigmaQ_SI: sigmaQ,
      sigmaP_SI: sigmaP,
      rms_displacement_angstrom: rms(displacements),
      rms_velocity_m_s: rms(velocities)
    };
  }

  /* Mulberry32 + Box–Muller. Every draw uses an explicit unsigned 32-bit seed. */
  function gaussianDraws(seed, count) {
    let state = seed >>> 0;
    function uniform() {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    const xi = [], eta = [];
    for (let k = 0; k < count; k += 1) {
      const radius = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1 / 4294967296)));
      const angle = 2 * Math.PI * uniform();
      xi.push(radius * Math.cos(angle));
      eta.push(radius * Math.sin(angle));
    }
    return { xi: xi, eta: eta };
  }

  window.HessianWignerMath = Object.freeze({ variance: variance, reconstruct: reconstruct, gaussianDraws: gaussianDraws });

  function element(root, selector) {
    const found = root.querySelector(selector);
    if (!found) throw new Error("A required figure control is missing: " + selector);
    return found;
  }
  function field(root, name) { return element(root, '[data-hw-field="' + name + '"]'); }
  function output(root, name, value) { element(root, '[data-hw-value="' + name + '"]').textContent = value; }
  function readout(root, parts) {
    const box = element(root, "[data-hw-readout]");
    box.replaceChildren();
    parts.forEach(function (part) {
      const span = document.createElement("span");
      span.textContent = part;
      box.appendChild(span);
    });
  }
  function number(value, digits) { return value.toFixed(digits === undefined ? 3 : digits); }
  function sci(value) { return value.toExponential(3); }
  function showError(root, error) {
    root.classList.remove("is-ready");
    root.classList.add("has-error");
    const status = root.querySelector("[data-hw-status]");
    if (status) status.textContent = "Interactive figure unavailable: " + error.message + " The static figure remains available. Reload the page to retry.";
    console.error("Harmonic figure:", error);
  }

  function canvasContext(canvas) {
    const width = Math.max(180, Math.round(canvas.getBoundingClientRect().width));
    const height = Math.round(canvas.getBoundingClientRect().height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("This browser does not provide a 2D canvas.");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "12px " + window.getComputedStyle(canvas).fontFamily;
    ctx.fillStyle = INK;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1;
    return { ctx: ctx, width: width, height: height };
  }
  function line(ctx, a, b) {
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  function arrow(ctx, a, b) {
    line(ctx, a, b);
    const theta = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const head = 5;
    ctx.beginPath(); ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - head * Math.cos(theta - 0.45), b[1] - head * Math.sin(theta - 0.45));
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - head * Math.cos(theta + 0.45), b[1] - head * Math.sin(theta + 0.45));
    ctx.stroke();
  }

  function drawMolecule(canvas, data, result, yaw, showVelocity) {
    const surface = canvasContext(canvas), ctx = surface.ctx;
    const width = surface.width, height = surface.height;
    const angle = yaw * Math.PI / 180, tilt = 0.32;
    const center = [0, 1, 2].map(function (a) { return data.coordinates_angstrom.reduce(function (s, p) { return s + p[a]; }, 0) / data.symbols.length; });
    function project(point, isVector) {
      const p = point.map(function (v, a) { return v - (isVector ? 0 : center[a]); });
      const x = Math.cos(angle) * p[0] + Math.sin(angle) * p[2];
      const depth = -Math.sin(angle) * p[0] + Math.cos(angle) * p[2];
      return [x, Math.cos(tilt) * p[1] - Math.sin(tilt) * depth, Math.sin(tilt) * p[1] + Math.cos(tilt) * depth];
    }
    const reference = data.coordinates_angstrom.map(function (p) { return project(p, false); });
    const positions = result.positions_angstrom.map(function (p) { return project(p, false); });
    /* Fixed orientation-independent molecular scale: thermal changes remain visible. */
    const radius = Math.max.apply(null, data.coordinates_angstrom.map(function (p) { return Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]); }));
    const scale = Math.min((width - 52) / (2 * radius + 3), (height - 76) / (2 * radius + 3));
    function pixel(p) { return [width / 2 + p[0] * scale, height / 2 - p[1] * scale]; }
    const refPixels = reference.map(pixel), pixels = positions.map(pixel);
    ctx.save();
    ctx.beginPath(); ctx.rect(4, 25, width - 8, height - 54); ctx.clip();
    ctx.strokeStyle = "#b7bcc0"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    (data.bonds || []).forEach(function (b) { line(ctx, refPixels[b[0]], refPixels[b[1]]); });
    reference.forEach(function (_, i) {
      ctx.beginPath(); ctx.arc(refPixels[i][0], refPixels[i][1], data.symbols[i] === "H" ? 3.2 : 5, 0, 2 * Math.PI); ctx.stroke();
    });
    ctx.setLineDash([]); ctx.strokeStyle = "#5c6267"; ctx.lineWidth = 1.6;
    (data.bonds || []).forEach(function (b) { line(ctx, pixels[b[0]], pixels[b[1]]); });
    positions.map(function (p, i) { return { i: i, z: p[2] }; }).sort(function (a, b) { return a.z - b.z; }).forEach(function (item) {
      const i = item.i, symbol = data.symbols[i];
      ctx.fillStyle = symbol === "H" ? "#f1f0ec" : (symbol === "N" ? BLUE : (symbol === "O" ? "#9c4d42" : "#555e64"));
      ctx.strokeStyle = symbol === "H" ? "#727a80" : "#42494e"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(pixels[i][0], pixels[i][1], symbol === "H" ? 3.2 : 5.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    });
    const projectedVelocities = result.velocities_m_s.map(function (v) { return project(v, true); });
    let arrowScale = 0.035;
    if (showVelocity) {
      projectedVelocities.forEach(function (v, i) {
        const vx = v[0], vy = -v[1];
        const xSpace = vx > 0 ? width - 13 - pixels[i][0] : pixels[i][0] - 13;
        const ySpace = vy > 0 ? height - 35 - pixels[i][1] : pixels[i][1] - 31;
        if (Math.abs(vx) > 1e-8 && xSpace > 0) arrowScale = Math.min(arrowScale, xSpace / Math.abs(vx));
        if (Math.abs(vy) > 1e-8 && ySpace > 0) arrowScale = Math.min(arrowScale, ySpace / Math.abs(vy));
      });
    }
    if (showVelocity) {
      ctx.strokeStyle = BLUE; ctx.lineWidth = 1.2;
      projectedVelocities.forEach(function (projected, i) {
        const tip = [pixels[i][0] + arrowScale * projected[0], pixels[i][1] - arrowScale * projected[1]];
        if (Math.hypot(tip[0] - pixels[i][0], tip[1] - pixels[i][1]) > 1) arrow(ctx, pixels[i], tip);
      });
    }
    ctx.restore();
    ctx.fillStyle = INK; ctx.textAlign = "left";
    ctx.fillText("Equilibrium outline / displaced geometry", 5, 15);
    const clipped = pixels.some(function (p) { return p[0] < 8 || p[0] > width - 8 || p[1] < 29 || p[1] > height - 33; });
    if (showVelocity) {
      const legendSpeed = arrowScale * 1000 < 15 ? 5000 : 1000;
      ctx.strokeStyle = BLUE; ctx.lineWidth = 1.2;
      arrow(ctx, [8, height - 13], [8 + legendSpeed * arrowScale, height - 13]);
      ctx.fillStyle = BLUE; ctx.fillText(legendSpeed + " m/s (projected)", 14 + legendSpeed * arrowScale, height - 9);
    } else {
      ctx.strokeStyle = INK; line(ctx, [8, height - 15], [8 + scale, height - 15]);
      ctx.fillText("1 Å", 13 + scale, height - 11);
    }
    return { clipped: clipped, arrowScale: arrowScale };
  }

  async function initMolecule(root) {
    const canvas = element(root, "[data-hw-canvas]");
    const source = root.getAttribute("data-source");
    if (!source) throw new Error("The normal-mode data URL is missing.");
    element(root, "[data-hw-status]").textContent = "Loading the molecular normal modes. The static figure remains available.";
    const response = await fetch(source);
    if (!response.ok) throw new Error("Normal-mode data could not be loaded (HTTP " + response.status + ").");
    const data = validateData(await response.json());
    if (data.symbols.length !== 30 || data.frequencies_cm1.length !== 84) throw new Error("The molecular figure requires the 30-atom, 84-mode data set.");
    const mode = field(root, "mode"), position = field(root, "position"), temperature = field(root, "temperature");
    const scope = field(root, "scope"), velocity = field(root, "velocity"), yaw = field(root, "yaw");
    mode.replaceChildren();
    data.frequencies_cm1.forEach(function (freq, i) {
      const option = document.createElement("option"); option.value = String(i); option.textContent = "Mode " + (i + 1) + " · " + number(freq, 1) + " cm⁻¹"; mode.appendChild(option);
    });
    let seed = BASE_SEED;
    let allDraws = gaussianDraws(seed, data.frequencies_cm1.length);
    const stableMomentum = gaussianDraws(3141592653, data.frequencies_cm1.length).eta;
    function update() {
      const k = Number(mode.value), t = Number(temperature.value), all = scope.value === "all";
      const xi = all ? allDraws.xi.slice() : data.frequencies_cm1.map(function () { return 0; });
      const eta = all ? allDraws.eta.slice() : data.frequencies_cm1.map(function () { return 0; });
      if (!all) { xi[k] = Number(position.value); eta[k] = stableMomentum[k]; }
      position.disabled = all;
      output(root, "position", number(Number(position.value), 2)); output(root, "temperature", String(t)); output(root, "yaw", yaw.value);
      const result = reconstruct(data, xi, eta, t);
      const drawing = drawMolecule(canvas, data, result, Number(yaw.value), velocity.checked);
      const selectedSigma = result.sigmaQ_SI[k];
      const cartesianSigma = data.symbols.map(function (_, i) {
        const norm = Math.hypot.apply(null, data.modes_mass_weighted[k][i]);
        return selectedSigma * norm / Math.sqrt(data.masses_amu[i] * data.constants.amu_kg) / 1e-10;
      });
      readout(root, [
        "Mode " + (k + 1) + ": " + number(data.frequencies_cm1[k], 2) + " cm⁻¹; T = " + t + " K",
        "Selected-mode σQ = " + sci(selectedSigma) + " kg½ m; σP = " + sci(result.sigmaP_SI[k]) + " kg½ m/s",
        "Selected-mode largest atomic RMS displacement = " + number(Math.max.apply(null, cartesianSigma)) + " Å",
        "Draw RMS displacement = " + number(result.rms_displacement_angstrom) + " Å; RMS speed = " + number(result.rms_velocity_m_s, 1) + " m/s",
        all ? "All 84 modes; Gaussian seed = " + seed : "One mode; ξ = " + number(xi[k], 2) + "; independent η = " + number(eta[k], 3) + " (fixed seed 3141592653)"
      ]);
      element(root, "[data-hw-status]").textContent = TEACHING + " σQ is the position standard deviation of a mass-weighted normal coordinate. Gray outlines are the reference. " + (all ? "Independent Gaussian position and momentum draws are retained when temperature changes. " : "The slider changes only Q; its independent P draw stays fixed. ") + (velocity.checked ? "Arrow scale: 1000 m/s = " + number(drawing.arrowScale * 1000, 1) + " CSS pixels; this scale adapts to the viewport. Only projected components are shown. " : "") + (drawing.clipped ? "The displaced geometry extends beyond the fixed demonstration viewport. " : "") + "The slider range and viewport are for demonstration, not a production-sampling restriction.";
      root.dataset.hwSeed = String(all ? seed : 3141592653);
    }
    function safeUpdate() { try { update(); } catch (error) { showError(root, error); } }
    [mode, position, temperature, scope, velocity, yaw].forEach(function (control) { control.addEventListener("input", safeUpdate); control.addEventListener("change", safeUpdate); });
    element(root, '[data-hw-action="resample"]').addEventListener("click", function () {
      seed = (seed + 1) >>> 0; allDraws = gaussianDraws(seed, data.frequencies_cm1.length); scope.value = "all"; safeUpdate();
    });
    root.classList.add("is-ready"); update(); observe(root, safeUpdate);
  }

  function axes(ctx, box, xRange, yRange, xLabel, yLabel) {
    function map(x, y) { return [box.x + (x + xRange) * box.w / (2 * xRange), box.y + box.h - (y + yRange) * box.h / (2 * yRange)]; }
    ctx.strokeStyle = "#d7dadd"; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.strokeStyle = "#b7bcc0"; line(ctx, map(-xRange, 0), map(xRange, 0)); line(ctx, map(0, -yRange), map(0, yRange));
    ctx.fillStyle = INK;
    [-xRange, 0, xRange].forEach(function (v) { const p = map(v, -yRange); ctx.textAlign = "center"; ctx.fillText(String(v), p[0], p[1] + 16); });
    [-yRange, yRange].forEach(function (v) { const p = map(-xRange, v); ctx.textAlign = "right"; ctx.fillText(String(v), p[0] - 6, p[1] + 4); });
    ctx.textAlign = "center"; ctx.fillText(xLabel, box.x + box.w / 2, box.y + box.h + 34);
    ctx.textAlign = "left"; ctx.fillText(yLabel, box.x + 2, box.y - 9);
    return map;
  }

  function drawPhase(canvas, values) {
    canvas.style.height = canvas.getBoundingClientRect().width < 520 ? "560px" : "390px";
    const surface = canvasContext(canvas), ctx = surface.ctx;
    const w = surface.width, h = surface.height, stack = w < 520;
    const leftSize = stack ? Math.min(w - 52, 242) : Math.min((w - 90) * 0.46, 272);
    const left = { x: stack ? (w - leftSize) / 2 + 10 : 34, y: 40, w: leftSize, h: leftSize };
    const right = stack ? { x: 38, y: left.y + left.h + 94, w: w - 52, h: h - left.y - left.h - 143 } : { x: left.x + left.w + 57, y: 40, w: w - left.x - left.w - 73, h: leftSize };
    ctx.fillStyle = BLUE; ctx.fillRect(8, 3, 16, 3); ctx.fillText("Wigner", 30, 9);
    ctx.strokeStyle = BROWN; ctx.setLineDash([5, 4]); line(ctx, [106, 5], [126, 5]); ctx.setLineDash([]);
    ctx.fillStyle = BROWN; ctx.fillText("Classical", 132, 9);
    const map = axes(ctx, left, 8, 8, "u", "v · one-σ ellipses");
    ctx.save(); ctx.beginPath(); ctx.rect(left.x, left.y, left.w, left.h); ctx.clip();
    function ellipse(stats, color, dashed) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash(dashed ? [5, 4] : []);
      const origin = map(0, 0);
      if (stats.uRms === 0) { ctx.beginPath(); ctx.arc(origin[0], origin[1], 3, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.beginPath(); ctx.ellipse(origin[0], origin[1], stats.uRms * left.w / 16, stats.vRms * left.h / 16, 0, 0, 2 * Math.PI); ctx.stroke(); }
    }
    ellipse(values.quantum, BLUE, false); ellipse(values.classical, BROWN, true); ctx.restore();
    const maxDensity = 1 / (Math.sqrt(2 * Math.PI) * Math.min(values.quantum.uRms, values.classical.uRms || Infinity));
    const densityMax = Math.max(0.1, maxDensity * 1.12);
    function densityPoint(u, density) { return [right.x + (u + 8) * right.w / 16, right.y + right.h - density * right.h / densityMax]; }
    ctx.strokeStyle = "#d7dadd"; ctx.setLineDash([]); ctx.lineWidth = 1; ctx.strokeRect(right.x, right.y, right.w, right.h);
    ctx.strokeStyle = "#b7bcc0"; line(ctx, densityPoint(0, 0), densityPoint(0, densityMax));
    ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.fillText("Position density p(u)", right.x, right.y - 9);
    ctx.textAlign = "center";
    [-8, 0, 8].forEach(function (u) { ctx.fillText(String(u), densityPoint(u, 0)[0], right.y + right.h + 16); });
    ctx.fillText("u", right.x + right.w / 2, right.y + right.h + 34);
    ctx.textAlign = "left"; ctx.fillText(number(densityMax, 2), right.x + 4, right.y + 14);
    ctx.save(); ctx.beginPath(); ctx.rect(right.x, right.y, right.w, right.h); ctx.clip();
    function gaussian(stats, color, dashed) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash(dashed ? [5, 4] : []);
      if (stats.uRms === 0) { arrow(ctx, densityPoint(0, 0), densityPoint(0, densityMax * 0.86)); return; }
      /* Include the exact origin and at least 240 intervals per visible Gaussian σ. */
      const intervals = Math.min(12000, Math.max(800, Math.ceil(16 * 240 / stats.uRms)));
      const evenIntervals = intervals + intervals % 2;
      ctx.beginPath();
      for (let j = 0; j <= evenIntervals; j += 1) {
        const u = -8 + 16 * j / evenIntervals;
        const density = Math.exp(-0.5 * u * u / stats.uVariance) / (Math.sqrt(2 * Math.PI) * stats.uRms);
        const p = densityPoint(u, density);
        if (j === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }
    gaussian(values.quantum, BLUE, false); gaussian(values.classical, BROWN, true); ctx.restore(); ctx.setLineDash([]);
    ctx.fillStyle = INK; ctx.textAlign = "left";
    const clipped = Math.max(values.quantum.uRms, values.quantum.vRms, values.classical.uRms, values.classical.vRms) > 8;
    return clipped;
  }

  function initPhase(root) {
    const canvas = element(root, "[data-hw-canvas]"), frequency = field(root, "frequency"), temperature = field(root, "temperature");
    function update() {
      const freq = Number(frequency.value), t = Number(temperature.value);
      const result = variance(freq, t);
      output(root, "frequency", String(freq)); output(root, "temperature", String(t));
      const clipped = drawPhase(canvas, result);
      readout(root, [
        "Wigner: RMS u = " + number(result.quantum.uRms) + "; RMS v = " + number(result.quantum.vRms),
        "Classical: RMS u = " + number(result.classical.uRms) + "; RMS v = " + number(result.classical.vRms),
        "Mean oscillator energy: Wigner = " + number(result.quantum.energy_eV, 5) + " eV; classical = " + number(result.classical.energy_eV, 5) + " eV"
      ]);
      element(root, "[data-hw-status]").textContent = TEACHING + " u = Q / √(ℏ / 2ωref), v = P / √(ℏωref / 2), with ωref = 2πc × 1000 cm⁻¹. Fixed phase-space axes span −8 to 8; the density panel rescales only its vertical density axis. " + (clipped ? "The one-σ ellipse exceeds this viewport; the RMS readout gives its full extent. " : "Gaussian tails outside the viewport are not shown. ") + (t === 0 ? "At T = 0, the classical point and delta marginal have zero width; the arrow denotes the delta distribution, not a finite peak height." : "Ellipses have one-σ semiaxes, not 68% joint coverage.");
    }
    function safeUpdate() { try { update(); } catch (error) { showError(root, error); } }
    [frequency, temperature].forEach(function (control) { control.addEventListener("input", safeUpdate); control.addEventListener("change", safeUpdate); });
    element(root, '[data-hw-action="reset"]').addEventListener("click", function () { frequency.value = "1000"; temperature.value = "0"; safeUpdate(); });
    root.classList.add("is-ready"); update(); observe(root, safeUpdate);
  }

  function observe(root, update) {
    let pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(function () { pending = false; if (root.classList.contains("is-ready")) update(); });
    }
    if (typeof window.ResizeObserver === "function") {
      let lastWidth = -1;
      const observer = new window.ResizeObserver(function (entries) {
        const width = Math.round(entries[0].contentRect.width);
        if (width !== lastWidth) { lastWidth = width; schedule(); }
      });
      observer.observe(root);
    } else { window.addEventListener("resize", schedule, { passive: true }); }
  }
  function start() {
    document.querySelectorAll('[data-hw-widget="molecule"], [data-hw-widget="phase"]').forEach(function (root) {
      if (root.dataset.hwInitialized) return;
      root.dataset.hwInitialized = "true";
      Promise.resolve().then(function () { return root.dataset.hwWidget === "molecule" ? initMolecule(root) : initPhase(root); }).catch(function (error) { showError(root, error); });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
}());
