const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelectorAll(".site-nav a");
const year = document.querySelector("#year");
const tagLabels = document.querySelectorAll(".tags span");

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
    navToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
  });
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
    navToggle?.setAttribute("aria-label", "Open navigation");
  });
});
