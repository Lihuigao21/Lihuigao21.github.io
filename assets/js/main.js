const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelectorAll(".site-nav a");
const year = document.querySelector("#year");
const tagLabels = document.querySelectorAll(".tags span");

const articlePath = window.location.pathname;
const lifeModulePages = new Set([
  "coffee-talk.html",
  "daily.html",
  "dreams-creation.html",
  "editorial.html",
  "food.html",
  "gpt-deep-chat.html",
  "life-experience.html",
  "memories.html",
  "people.html",
  "positive-feedback.html",
  "reflection.html",
  "relationship.html",
  "skill.html",
  "travel.html",
]);
const articleFile = articlePath.split("/").pop();
const isTechnicalArticle = /\/posts\/[^/]+\.html$/.test(articlePath);
const isLifeArticle = /\/life\/[^/]+\.html$/.test(articlePath) && !lifeModulePages.has(articleFile);

const taxonomyRoot = window.location.pathname.includes("/posts/") ? "../" : "";
const slugifyTag = (value) =>
  String(value)
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

if (year) {
  year.textContent = new Date().getFullYear();
}

tagLabels.forEach((tag) => {
  const label = tag.textContent.trim();
  const link = document.createElement("a");
  link.href = `${taxonomyRoot}tags.html?tag=${encodeURIComponent(slugifyTag(label))}`;
  link.textContent = label;
  link.setAttribute("aria-label", `View all articles tagged ${label}`);
  tag.replaceWith(link);
});

if (navToggle) {
  navToggle.addEventListener("click", () => {
    const isOpen = document.body.classList.toggle("nav-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
    navToggle.setAttribute("aria-label", isOpen ? "关闭导航" : "打开导航");
  });
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
    navToggle?.setAttribute("aria-label", "打开导航");
  });
});

if (isTechnicalArticle || isLifeArticle) {
  const article = document.querySelector("article.post");
  const postNavigation = article?.querySelector(".post-nav");

  if (article) {
    const comments = document.createElement("section");
    comments.className = "article-comments";
    comments.setAttribute("aria-labelledby", "article-comments-title");
    comments.innerHTML = `
      <div class="section-heading">
        <p class="section-label">COMMENTS</p>
        <h2 id="article-comments-title">留言</h2>
      </div>
      <p class="article-comments-note">留言会公开展示，并需要登录 GitHub。请不要填写不适合公开的信息。</p>
    `;

    const giscus = document.createElement("script");
    giscus.src = "https://giscus.app/client.js";
    giscus.async = true;
    giscus.crossOrigin = "anonymous";
    Object.entries({
      repo: "Lihuigao21/Lihuigao21.github.io",
      "repo-id": "R_kgDOObs2TQ",
      category: "General",
      "category-id": "DIC_kwDOObs2Tc4DC-gE",
      mapping: "pathname",
      strict: "1",
      "reactions-enabled": "1",
      "emit-metadata": "0",
      "input-position": "top",
      theme: "preferred_color_scheme",
      lang: "zh-CN",
      loading: "lazy",
    }).forEach(([key, value]) => giscus.setAttribute(`data-${key}`, value));
    comments.append(giscus);

    if (postNavigation) {
      article.insertBefore(comments, postNavigation);
    } else {
      article.append(comments);
    }
  }
}
