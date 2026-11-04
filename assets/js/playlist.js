(() => {
  "use strict";
  const root = document.querySelector("[data-playlist]");
  if (!root) return;
  const audio = root.querySelector("audio");
  const rows = [...root.querySelectorAll("[data-playlist-track]")];
  const heading = root.querySelector("[data-playlist-title]");
  const artist = root.querySelector("[data-playlist-artist]");
  const duration = root.querySelector("[data-playlist-duration]");
  const details = root.querySelector("[data-playlist-details]");
  const status = root.querySelector("[data-playlist-status]");
  const previous = root.querySelector("[data-playlist-previous]");
  const next = root.querySelector("[data-playlist-next]");
  if (!audio || !rows.length || !heading || !artist || !duration || !details || !status || !previous || !next) return;
  let index = 0;
  let selection = 0;
  const time = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

  function refresh() {
    const playing = !audio.paused && !audio.ended;
    rows.forEach((row, i) => {
      row.toggleAttribute("data-selected", i === index);
      if (i === index) row.setAttribute("aria-current", "true");
      else row.removeAttribute("aria-current");
      const button = row.querySelector("[data-playlist-play]");
      button.textContent = i === index && playing ? "暂停" : "播放";
      button.setAttribute("aria-label", `${i === index && playing ? "暂停" : "播放"} ${row.dataset.title}`);
      button.setAttribute("aria-pressed", String(i === index && playing));
    });
    previous.disabled = index === 0;
    next.disabled = index === rows.length - 1;
  }

  function play() {
    const current = selection;
    audio.play().catch((error) => {
      if (current !== selection || error.name === "AbortError") return;
      status.textContent = "请点击播放器继续播放，或打开歌曲页面。";
      refresh();
    });
  }

  function select(i, shouldPlay = true) {
    if (i < 0 || i >= rows.length) return;
    selection += 1;
    audio.pause();
    index = i;
    const row = rows[index];
    heading.textContent = row.dataset.title;
    artist.textContent = row.dataset.artist;
    duration.textContent = time(Number(row.dataset.duration));
    details.href = row.querySelector(".playlist-track-title").getAttribute("href");
    details.setAttribute("aria-label", `打开 ${row.dataset.title} 的歌词与随记`);
    audio.setAttribute("aria-label", `${row.dataset.title}，${row.dataset.artist}`);
    audio.src = row.dataset.media;
    audio.load();
    status.textContent = `已选择：${row.dataset.title}`;
    refresh();
    if (shouldPlay) play();
  }

  rows.forEach((row, i) => {
    const button = row.querySelector("[data-playlist-play]");
    button.hidden = false;
    button.addEventListener("click", () => {
      if (i !== index) select(i);
      else if (audio.paused || audio.ended) play();
      else audio.pause();
    });
  });
  previous.addEventListener("click", () => select(index - 1));
  next.addEventListener("click", () => select(index + 1));
  audio.addEventListener("play", () => { status.textContent = `正在播放：${rows[index].dataset.title}`; refresh(); });
  audio.addEventListener("pause", () => { if (!audio.ended) status.textContent = `已暂停：${rows[index].dataset.title}`; refresh(); });
  audio.addEventListener("loadedmetadata", () => {
    if (Number.isFinite(audio.duration)) duration.textContent = time(audio.duration);
    refresh();
  });
  audio.addEventListener("ended", () => {
    if (index < rows.length - 1) select(index + 1);
    else { status.textContent = "歌单已播放完。"; refresh(); }
  });
  audio.addEventListener("error", () => {
    status.textContent = "这首歌暂时无法播放，可以切换歌曲或打开歌曲页面。";
    refresh();
  });
  root.querySelector(".playlist-controls").hidden = false;
  refresh();
})();
