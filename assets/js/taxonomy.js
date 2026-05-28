(function () {
  const articles = window.SITE_ARTICLES || [];
  const series = window.SITE_SERIES || [];
  const seriesById = new Map(series.map((item) => [item.id, item]));

  const slugify = (value) =>
    String(value)
      .trim()
      .toLowerCase()
      .replace(/&/g, "and")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");

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

  const byDateDesc = (left, right) =>
    right.date.localeCompare(left.date) || left.title.localeCompare(right.title);

  const bySeriesPart = (left, right) =>
    (left.part ?? 999) - (right.part ?? 999) || left.title.localeCompare(right.title);

  const params = new URLSearchParams(window.location.search);

  function articleList(items) {
    if (!items.length) {
      return '<p class="muted-note">No articles are listed here yet.</p>';
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
                    ${article.tags
                      .map(
                        (tag) =>
                          `<a href="tags.html?tag=${encodeURIComponent(slugify(tag))}">${escapeHtml(tag)}</a>`
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

  function buildTagMap() {
    const tagMap = new Map();
    articles.forEach((article) => {
      article.tags.forEach((tag) => {
        const slug = slugify(tag);
        if (!tagMap.has(slug)) {
          tagMap.set(slug, { name: tag, slug, articles: [] });
        }
        tagMap.get(slug).articles.push(article);
      });
    });
    [...tagMap.values()].forEach((tag) => tag.articles.sort(byDateDesc));
    return tagMap;
  }

  function renderTagsPage() {
    const cloud = document.querySelector("[data-tag-cloud]");
    const results = document.querySelector("[data-tag-results]");
    if (!cloud || !results) return;

    const tagMap = buildTagMap();
    const tags = [...tagMap.values()].sort((left, right) => left.name.localeCompare(right.name));
    const selectedSlug = params.get("tag") || "";
    const selected = tagMap.get(selectedSlug);

    cloud.innerHTML = tags
      .map(
        (tag) => `
          <a class="taxonomy-chip${selectedSlug === tag.slug ? " is-active" : ""}"
             href="tags.html?tag=${encodeURIComponent(tag.slug)}"
             ${selectedSlug === tag.slug ? 'aria-current="page"' : ""}>
            <span>${escapeHtml(tag.name)}</span>
            <span>${tag.articles.length}</span>
          </a>
        `
      )
      .join("");

    if (selectedSlug && !selected) {
      results.innerHTML = `
        <div class="taxonomy-block">
          <h2>Tag Not Found</h2>
          <p class="muted-note">The requested tag does not exist in the current article index.</p>
        </div>
      `;
      return;
    }

    if (selected) {
      document.title = `${selected.name} Articles | Lihui Gao`;
      results.innerHTML = `
        <div class="taxonomy-block">
          <p class="section-label">Selected Tag</p>
          <h2>${escapeHtml(selected.name)}</h2>
          <p class="muted-note">${selected.articles.length} article${selected.articles.length === 1 ? "" : "s"} tagged with ${escapeHtml(selected.name)}.</p>
          ${articleList(selected.articles)}
        </div>
      `;
      return;
    }

    results.innerHTML = tags
      .map(
        (tag) => `
          <section class="taxonomy-block">
            <h2 id="tag-${escapeHtml(tag.slug)}">${escapeHtml(tag.name)}</h2>
            ${articleList(tag.articles)}
          </section>
        `
      )
      .join("");
  }

  function collectionArticles(seriesId) {
    return articles
      .filter((article) => article.series === seriesId)
      .sort(bySeriesPart);
  }

  function renderSeriesPage() {
    const overview = document.querySelector("[data-series-overview]");
    const results = document.querySelector("[data-series-results]");
    if (!overview || !results) return;

    const selectedId = params.get("series") || "";
    const selectedSeries = seriesById.get(selectedId);

    overview.innerHTML = series
      .map((item) => {
        const count = collectionArticles(item.id).length;
        return `
          <a class="taxonomy-card${selectedId === item.id ? " is-active" : ""}"
             href="series.html?series=${encodeURIComponent(item.id)}"
             ${selectedId === item.id ? 'aria-current="page"' : ""}>
            <span class="section-label">${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.title)}</strong>
            <span>${escapeHtml(item.description)}</span>
            <span class="taxonomy-count">${count} article${count === 1 ? "" : "s"}</span>
          </a>
        `;
      })
      .join("");

    if (selectedId && !selectedSeries) {
      results.innerHTML = `
        <div class="taxonomy-block">
          <h2>Series Not Found</h2>
          <p class="muted-note">The requested series does not exist in the current article index.</p>
        </div>
      `;
      return;
    }

    const collections = selectedSeries ? [selectedSeries] : series;
    if (selectedSeries) {
      document.title = `${selectedSeries.title} | Lihui Gao`;
    }

    results.innerHTML = collections
      .map((item) => {
        const items = collectionArticles(item.id);
        return `
          <section class="taxonomy-block" id="${escapeHtml(item.id)}">
            <p class="section-label">${escapeHtml(item.label)}</p>
            <h2>${escapeHtml(item.title)}</h2>
            <p class="muted-note">${escapeHtml(item.description)}</p>
            ${articleList(items)}
          </section>
        `;
      })
      .join("");
  }

  renderTagsPage();
  renderSeriesPage();
})();
