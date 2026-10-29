(() => {
  "use strict";

  document.querySelectorAll("[data-music-player]").forEach((root) => {
    const audio = root.querySelector("audio");
    const spectrum = root.querySelector(".music-spectrum");
    const canvas = spectrum?.querySelector("canvas");
    const toggle = root.querySelector("[data-spectrum-toggle]");
    const status = root.querySelector(".music-spectrum-status");
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const Context = window.AudioContext || window.webkitAudioContext;
    const paint = canvas?.getContext("2d");
    let context;
    let analyser;
    let bins;
    let source;
    let starting;
    let frame = 0;
    let lastFrame = 0;
    let width = 0;
    let height = 0;

    if (!audio || !toggle) return;
    toggle.checked = !motion.matches;
    const transport = root.querySelector(".music-transport");
    const playButtons = [...root.querySelectorAll("[data-play-toggle]")];
    const seek = root.querySelector("[data-music-seek]");
    const elapsed = root.querySelector("[data-music-elapsed]");
    const duration = root.querySelector("[data-music-duration]");
    const mute = root.querySelector("[data-music-mute]");
    const formatTime = (time) => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, "0")}`;

    function wakeAudio() {
      return context && context.state !== "running" ? context.resume() : Promise.resolve();
    }

    function refreshMotion() {
      root.classList.toggle("is-still", !toggle.checked || motion.matches || document.hidden);
    }

    function refreshTransport() {
      const playing = !audio.paused && !audio.ended;
      root.classList.toggle("is-playing", playing);
      playButtons.forEach((button) => button.setAttribute("aria-label", `${playing ? "暂停" : "播放"} Photograph`));
      const hint = root.querySelector(".music-record-hint");
      if (hint) hint.textContent = playing ? "点击唱片暂停" : "点击唱片播放";
      if (seek && elapsed && duration) {
        const length = Number.isFinite(audio.duration) ? audio.duration : 0;
        seek.disabled = !length;
        seek.max = length || 282.378;
        seek.value = audio.currentTime;
        seek.style.setProperty("--progress", `${length ? audio.currentTime / length * 100 : 0}%`);
        seek.setAttribute("aria-valuetext", `${formatTime(audio.currentTime)} / ${formatTime(length)}`);
        elapsed.textContent = formatTime(audio.currentTime);
        duration.textContent = length ? formatTime(length) : "4:42";
      }
      if (mute) {
        mute.setAttribute("aria-pressed", String(audio.muted));
        mute.setAttribute("aria-label", audio.muted ? "取消静音" : "静音");
      }
      refreshMotion();
    }

    if (transport && seek && elapsed && duration && playButtons.length) {
      playButtons.forEach((button) => {
        button.hidden = false;
        button.addEventListener("click", () => {
          if (!audio.paused) return audio.pause();
          // Audio output must resume even when visual effects are switched off.
          wakeAudio().catch(() => {});
          audio.play().catch(() => {
            root.classList.remove("has-transport");
            transport.hidden = true;
          });
        });
      });
      seek.addEventListener("input", () => {
        if (Number.isFinite(audio.duration)) audio.currentTime = Math.min(Number(seek.value), audio.duration);
        refreshTransport();
      });
      mute?.addEventListener("click", () => { audio.muted = !audio.muted; });
      audio.addEventListener("error", () => {
        root.classList.remove("has-transport");
        transport.hidden = true;
      });
      ["play", "pause", "ended", "timeupdate", "loadedmetadata", "durationchange", "seeked", "volumechange"].forEach((event) => audio.addEventListener(event, refreshTransport));
      transport.hidden = false;
      root.classList.add("has-transport");
      const hint = root.querySelector(".music-record-hint");
      if (hint) hint.hidden = false;
      refreshTransport();
    }
    toggle.addEventListener("change", refreshMotion);
    document.addEventListener("visibilitychange", refreshMotion);
    motion.addEventListener("change", () => { toggle.checked = !motion.matches; refreshMotion(); });
    refreshMotion();

    function stop() {
      cancelAnimationFrame(frame);
      frame = 0;
    }

    function draw(active = false) {
      if (!paint) return;
      paint.clearRect(0, 0, width, height);
      if (active) analyser.getByteFrequencyData(bins);
      const horizon = (x) => height * (0.64 + 0.07 * Math.sin(x / width * Math.PI * 1.6 - 0.5));
      const count = width < 600 ? 64 : 144;
      const step = width / count;
      const binHz = analyser ? context.sampleRate / analyser.fftSize : 1;
      const maxHz = context ? Math.min(14000, context.sampleRate / 2) : 14000;
      const floor = paint.createLinearGradient(0, height * 0.55, 0, height);
      floor.addColorStop(0, "#bbaa8620");
      floor.addColorStop(1, "#bbaa8600");
      paint.beginPath();
      paint.moveTo(0, horizon(0));
      for (let x = 0; x <= width; x += 8) paint.lineTo(x, horizon(x));
      paint.lineTo(width, height);
      paint.lineTo(0, height);
      paint.closePath();
      paint.fillStyle = floor;
      paint.fill();
      paint.beginPath();
      for (let x = 0; x <= width; x += 8) x ? paint.lineTo(x, horizon(x)) : paint.moveTo(x, horizon(x));
      paint.strokeStyle = "#8b806852";
      paint.lineWidth = 1;
      paint.stroke();
      paint.strokeStyle = "#837658";
      paint.lineWidth = width < 600 ? 1.5 : 2;
      paint.lineCap = "round";
      for (let i = 0; i < count; i += 1) {
        let strength = 0;
        if (active) {
          // Log-spaced bands preserve the quieter high-frequency detail.
          const low = Math.max(1, Math.floor(60 * (maxHz / 60) ** (i / count) / binHz));
          const high = Math.min(bins.length, Math.max(low + 1, Math.ceil(60 * (maxHz / 60) ** ((i + 1) / count) / binHz)));
          for (let bin = low; bin < high; bin += 1) strength += bins[bin];
          strength /= (high - low) * 255;
        }
        const x = i * step + step / 2;
        const base = horizon(x);
        const bar = Math.max(0.8, strength ** 1.8 * height * 0.76);
        paint.beginPath();
        paint.moveTo(x, base - bar);
        paint.lineTo(x, base + bar * 0.22);
        paint.stroke();
      }
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      paint.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(Boolean(analyser && toggle.checked && !audio.paused && !document.hidden));
    }

    function animate(now) {
      frame = 0;
      if (!toggle.checked || audio.paused || audio.ended || document.hidden || !analyser) return;
      if (now - lastFrame >= 32) {
        draw(true);
        lastFrame = now;
      }
      frame = requestAnimationFrame(animate);
    }

    function update() {
      stop();
      if (!toggle.checked) {
        status.textContent = "动效已关闭";
        draw();
      } else if (audio.paused || audio.ended) {
        status.textContent = audio.currentTime ? "已暂停" : "播放后显示实时频谱";
        draw();
      } else if (analyser && context.state === "running") {
        status.textContent = "实时频谱";
        if (!document.hidden) frame = requestAnimationFrame(animate);
      }
    }

    async function start() {
      if (source && context.state !== "running") await wakeAudio().catch(() => {});
      if (!toggle.checked) return update();
      if (!starting) {
        starting = (async () => {
          context ||= new Context();
          if (context.state !== "running") await context.resume();
          // Keep native playback intact until the context can actually run.
          if (context.state !== "running") return;
          if (!source) {
            analyser = context.createAnalyser();
            analyser.fftSize = 2048;
            analyser.smoothingTimeConstant = 0.82;
            analyser.minDecibels = -90;
            analyser.maxDecibels = -20;
            bins = new Uint8Array(analyser.frequencyBinCount);
            source = context.createMediaElementSource(audio);
            source.connect(context.destination);
            source.connect(analyser);
          }
        })().catch(() => {
          toggle.checked = false;
          status.textContent = "当前浏览器未启用频谱，可继续听歌";
        }).finally(() => { starting = null; });
      }
      await starting;
      update();
    }

    if (audio && Context && paint && toggle && status) {
      spectrum.hidden = false;
      toggle.checked = !motion.matches;
      resize();
      if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
      else window.addEventListener("resize", resize);
      audio.addEventListener("play", start);
      audio.addEventListener("pause", update);
      audio.addEventListener("ended", update);
      // Resuming on a gesture also covers mobile browsers' audio policy.
      audio.addEventListener("pointerdown", () => { wakeAudio().catch(() => {}); });
      audio.addEventListener("keydown", () => { wakeAudio().catch(() => {}); });
      toggle.addEventListener("change", () => audio.paused ? update() : start());
      motion.addEventListener("change", () => { toggle.checked = !motion.matches; audio.paused ? update() : start(); });
      document.addEventListener("visibilitychange", update);
      window.addEventListener("pagehide", stop);
      window.addEventListener("pageshow", update);
      update();
    }

    async function prepareLyrics() {
      const section = root.querySelector(".song-lyrics");
      const toolbar = root.querySelector(".music-lyrics-toolbar");
      const follow = root.querySelector("[data-lyrics-follow]");
      const timing = root.querySelector(".music-lyrics-timing");
      const offsetLabel = root.querySelector("[data-lyrics-offset]");
      if (!audio || !section || !toolbar || !follow || !root.dataset.timings) return;
      const response = await fetch(root.dataset.timings);
      if (!response.ok) return;
      const { lineStarts } = await response.json();
      const paragraphs = [...section.querySelectorAll("p")];
      const lines = paragraphs.slice(1).flatMap((p, stanza) => {
        const parts = p.innerHTML.split(/<br\s*\/?\s*>/i);
        return parts.map((html, index) => {
          const text = document.createElement("span");
          text.innerHTML = html;
          return { text: text.textContent, stanzaStart: index === 0 && stanza > 0 };
        });
      });
      if (!Array.isArray(lineStarts) || lineStarts.length !== lines.length ||
          lineStarts.some((t, i) => !Number.isFinite(t) || t < 0 || (i > 0 && t < lineStarts[i - 1]))) return;
      const viewport = document.createElement("div");
      viewport.className = "music-lyrics-window";
      viewport.tabIndex = 0;
      viewport.setAttribute("role", "region");
      viewport.setAttribute("aria-label", "Photograph 歌词");
      const title = paragraphs[0].cloneNode(true);
      title.className = "music-lyrics-title";
      viewport.append(title);
      let lastButton;
      const buttons = lines.map((line, i) => {
        // Shared LRC cues stay together instead of skipping half a phrase.
        if (i > 0 && lineStarts[i] === lineStarts[i - 1]) {
          lastButton.append(document.createElement("br"), document.createTextNode(line.text));
          return lastButton;
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "music-lyric-line";
        button.textContent = line.text;
        button.dataset.lyricStart = lineStarts[i];
        if (line.stanzaStart) button.dataset.stanzaStart = "";
        const seconds = lineStarts[i];
        button.title = `跳转至 ${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
        button.addEventListener("click", () => {
          if (Number.isFinite(audio.duration)) audio.currentTime = Math.max(0, Math.min(seconds + offset, audio.duration));
          follow.checked = true;
          sync(true);
        });
        viewport.append(button);
        lastButton = button;
        return button;
      });
      section.replaceChildren(viewport);
      toolbar.hidden = false;
      const help = root.querySelector(".music-lyrics-help");
      if (help) help.hidden = false;
      let current = -1;
      let offset = 0;

      function center(instant = false) {
        if (!follow.checked || current < 0) return;
        const button = buttons[current];
        // Scroll only this frame, never the surrounding article.
        const top = button.offsetTop - viewport.clientHeight / 2 + button.offsetHeight / 2;
        viewport.scrollTo({ top, behavior: instant || motion.matches ? "instant" : "smooth" });
      }

      function sync(instant = false) {
        let next = -1;
        for (let i = 0; i < lineStarts.length && lineStarts[i] + offset <= audio.currentTime; i += 1) next = i;
        if (next !== current) {
          if (current >= 0) buttons[current].removeAttribute("aria-current");
          current = next;
          if (current >= 0) buttons[current].setAttribute("aria-current", "true");
          center(instant);
        } else if (instant) center(true);
      }

      const release = () => { follow.checked = false; };
      viewport.addEventListener("wheel", release, { passive: true });
      viewport.addEventListener("touchstart", release, { passive: true });
      viewport.addEventListener("pointerdown", (event) => { if (event.target === viewport) release(); });
      viewport.addEventListener("keydown", (event) => {
        if (["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " "].includes(event.key)) release();
      });
      follow.addEventListener("change", () => center(true));
      if (timing && offsetLabel) {
        timing.hidden = false;
        timing.querySelectorAll("[data-lyrics-adjust]").forEach((button) => {
          button.addEventListener("click", () => {
            offset = button.dataset.lyricsAdjust === "reset" ? 0 : Math.max(-10, Math.min(10, offset + Number(button.dataset.lyricsAdjust)));
            offsetLabel.textContent = offset === 0 ? "原始时间" : `${offset < 0 ? "提前" : "延后"} ${Math.abs(offset).toFixed(1)} 秒`;
            sync(true);
          });
        });
      }
      audio.addEventListener("timeupdate", () => sync());
      audio.addEventListener("seeked", () => sync(true));
      audio.addEventListener("loadedmetadata", () => sync(true));
      if (window.ResizeObserver) new ResizeObserver(() => center(true)).observe(viewport);
      sync(true);
    }

    // The full static lyrics and native controls remain usable if enhancement fails.
    prepareLyrics().catch(() => {});
  });
})();
