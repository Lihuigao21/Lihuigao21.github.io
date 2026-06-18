(function () {
  const articles = window.SITE_ARTICLES || [];
  const form = document.querySelector("[data-search-form]");
  const input = document.querySelector("[data-search-input]");
  const summary = document.querySelector("[data-search-summary]");
  const results = document.querySelector("[data-search-results]");
  const clearLink = document.querySelector("[data-search-clear]");

  if (!form || !input || !summary || !results) return;

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (char) => {
      const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      };
      return entities[char];
    });

  const slugifyTag = (value) =>
    String(value)
      .trim()
      .toLowerCase()
      .replace(/&/g, "and")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");

  const normalize = (value) => String(value).trim().toLowerCase();
  const tokenize = (query) => normalize(query).split(/\s+/).filter(Boolean);
  const byDateDesc = (left, right) =>
    right.date.localeCompare(left.date) || left.title.localeCompare(right.title);

  function searchableText(article) {
    return normalize([article.title, ...(article.tags || [])].join(" "));
  }

  function matchesArticle(article, tokens) {
    if (!tokens.length) return true;
    const haystack = searchableText(article);
    return tokens.every((token) => haystack.includes(token));
  }

  function articleList(items) {
    if (!items.length) {
      return '<p class="muted-note">No matching articles were found.</p>';
    }

    return `
      <div class="taxonomy-list">
        ${items
          .map(
            (article) => `
              <article class="taxonomy-item">
                <time datetime="${escapeHtml(article.date)}">${escapeHtml(article.dateText)}</time>
                <div>
                  <h3><a href="${escapeHtml(article.path)}">${escapeHtml(article.title)}</a></h3>
                  <p>${escapeHtml(article.description)}</p>
                  <div class="tags" aria-label="Article tags">
                    ${(article.tags || [])
                      .map(
                        (tag) =>
                          `<a href="tags.html?tag=${encodeURIComponent(slugifyTag(tag))}">${escapeHtml(tag)}</a>`
                      )
                      .join("")}
                  </div>
                </div>
              </article>
            `
          )
          .join("")}
      </div>
    `;
  }

  function updateUrl(query) {
    const next = query ? `search.html?q=${encodeURIComponent(query)}` : "search.html";
    window.history.replaceState({}, "", next);
  }

  function render(query) {
    const trimmed = query.trim();
    const tokens = tokenize(trimmed);
    const matches = articles.filter((article) => matchesArticle(article, tokens)).sort(byDateDesc);

    input.value = query;
    clearLink.hidden = !trimmed;
    summary.textContent = trimmed
      ? `${matches.length} result${matches.length === 1 ? "" : "s"} for "${trimmed}".`
      : `Showing all ${matches.length} article${matches.length === 1 ? "" : "s"}.`;
    results.innerHTML = articleList(matches);
  }

  const params = new URLSearchParams(window.location.search);
  const initialQuery = params.get("q") || "";

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = input.value.trim();
    updateUrl(query);
    render(query);
  });

  input.addEventListener("input", () => {
    const query = input.value.trim();
    updateUrl(query);
    render(query);
  });

  render(initialQuery);
})();
