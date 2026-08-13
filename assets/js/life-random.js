(() => {
  const button = document.querySelector("[data-random-life]");
  const status = document.querySelector("[data-random-life-status]");
  if (!button) return;

  const defaultStatus = "从已经公开的生活记录里，随机翻开一页。";

  const resetButton = () => {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    if (status) status.textContent = defaultStatus;
  };

  window.addEventListener("pageshow", resetButton);

  const modulePages = new Set([
    "coffee-talk.html", "daily.html", "dreams-creation.html", "editorial.html",
    "food.html", "gpt-deep-chat.html", "life-experience.html", "memories.html",
    "people.html", "positive-feedback.html", "reflection.html", "relationship.html",
    "skill.html", "travel.html",
  ]);

  const latestFallback = () =>
    [...document.querySelectorAll("#latest-life-notes .archive-list a")].map(
      (link) => new URL(link.href, window.location.href).href,
    );

  const loadLifeNotes = async () => {
    const response = await fetch("sitemap.xml", { cache: "no-cache" });
    if (!response.ok) throw new Error(`Sitemap request failed: ${response.status}`);
    const xml = new DOMParser().parseFromString(await response.text(), "application/xml");
    if (xml.querySelector("parsererror")) throw new Error("Sitemap could not be parsed");

    return [...xml.querySelectorAll("loc")]
      .map((node) => node.textContent.trim())
      .filter((url) => {
        const parsed = new URL(url, window.location.origin);
        const file = parsed.pathname.split("/").pop();
        return parsed.pathname.includes("/life/") && file.endsWith(".html") && !modulePages.has(file);
      });
  };

  button.addEventListener("click", async () => {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    if (status) status.textContent = "正在翻开一页过去……";

    try {
      let notes;
      try { notes = await loadLifeNotes(); } catch { notes = latestFallback(); }
      if (!notes.length) throw new Error("No life notes are available");

      const previous = sessionStorage.getItem("lastRandomLifeNote");
      const choices = notes.length > 1 ? notes.filter((url) => url !== previous) : notes;
      const target = choices[Math.floor(Math.random() * choices.length)];
      sessionStorage.setItem("lastRandomLifeNote", target);
      window.location.assign(target);
    } catch {
      resetButton();
      if (status) status.textContent = "暂时没有找到过去的记录，请稍后再试。";
    }
  });
})();
