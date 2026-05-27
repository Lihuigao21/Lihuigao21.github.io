# Website Agent Guidelines

This file defines how future agents should write, revise, and publish articles for this GitHub Pages site.

## Core Style

- All published site content must be in English.
- Keep a consistent academic and technical style across articles: restrained, clear, reproducible, and not promotional.
- Keep titles compact and consistent with the existing pattern:
  - Series articles: `DVR VII: Excited-Ground Kubo Population Correlation`
  - Standalone articles: short technical title without decorative wording.
- Keep typography and layout consistent with the current site. Do not introduce article-specific fonts, oversized headings, bright palettes, or decorative layouts unless the whole site style is intentionally revised.
- Use the existing HTML article structure: header metadata, `series-nav` when applicable, lead paragraph, clear `h2` sections, figures with captions, code links, and footer navigation.

## Article Shape

Each article should be concise and useful. Prefer a compact technical note over a long transcript of raw source material.

Every article should normally contain:

1. Background: what problem the note addresses and why it matters.
2. Principle: the central mathematical, physical, or computational idea.
3. Workflow: how the calculation or method is actually carried out.
4. Result: at least one staged result, benchmark, diagnostic, or test.
5. Code: links to the executable or source files used in the note.

Avoid purely theoretical articles with no result. If the source material is theoretical, add a minimal benchmark, numerical check, derivation check, or reproducible toy example.

## Equations And Figures

- Do not publish equations or explanatory text as screenshots.
- Convert equations to MathJax/LaTeX text inside the HTML article.
- Figures should show actual numerical, conceptual, or workflow results, not decorative filler.
- Every figure must have a precise caption explaining:
  - what the axes mean,
  - what each curve/color/line style represents,
  - what units or scaling factors are used,
  - what conclusion the reader should draw.
- If a supplied figure contains UI overlays, cropping artifacts, or irrelevant marks, clean it before publishing when this can be done without changing the data.

## Code Links

- All code used by an article must be linked inside the article.
- Prefer compact, executable scripts under `assets/code/` for published examples.
- Keep raw notebooks, PDFs, drafts, and large binary files in `articles/`; this folder is ignored by git and should not be published wholesale.
- Do not publish huge dumps such as `.npz`, cache folders, or notebook outputs unless they are intentionally cleaned, documented, and small enough for GitHub Pages.
- When adapting code from a notebook, make a clean script that can run from the repository root or from its own path.

## Series Logic

- Articles in the same series must progress step by step. Avoid logical jumps.
- Each series article should briefly say how it depends on previous parts and what new idea it adds.
- Update every article in the series when adding a new part:
  - top `series-nav`,
  - bottom `DVR series map`,
  - any text that incorrectly calls an older part the endpoint.
- At the end of each series article, include links to all parts in the same series.

## Editing Existing Articles

- Preserve the existing site tone and structure unless the user explicitly asks for a redesign.
- Do not simply paste raw source notes into an article. Rewrite, compress, and organize them.
- Remove duplicated derivations, unclear commentary, and irrelevant implementation details.
- Keep technical claims grounded in the supplied source files, papers, code, or generated results.
- If formulas are OCRed from screenshots or PDFs, verify them against source LaTeX or code when possible.

## Publishing Workflow

Before committing or pushing article changes:

1. Confirm that the article is English-only.
2. Check that all local links and image paths resolve.
3. Run any compact scripts linked from the article when feasible.
4. Verify that each article has at least one result, benchmark, or diagnostic.
5. Update `index.html`, `sitemap.xml`, and `README.md` when adding a new published page.
6. Check desktop and mobile layout for long titles, formulas, figures, and code blocks.
7. Commit with a clear message and push to `origin/main` only when the user asks to publish or the task explicitly includes upload/deploy.

## Site Boundaries

- `posts/` contains published HTML articles.
- `assets/img/` contains published images.
- `assets/code/` contains published code attachments.
- `articles/` contains local raw materials and should stay unpublished.
- Keep the site static and GitHub Pages friendly; do not add a build system unless the user asks for one.
