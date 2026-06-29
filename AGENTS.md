# Website Agent Guidelines

This file defines how future agents should write, revise, and publish articles for this GitHub Pages site.

## Core Style

- All published technical article content must be in English by default.
- Exception: the `life.html`, `mind.html`, and `life/` pages form an intentionally Chinese-language “生活记录” section for daily life and personal reflection.
- Outside academic/technical article content, default all public-facing site text to Chinese, including lifestyle pages, personal notes, captions, section labels, and newly added interface copy.
- Keep a consistent academic and technical style across articles: restrained, clear, reproducible, and not promotional.
- Keep titles compact and consistent with the existing pattern:
  - Series articles: `DVR VII: Excited-Ground Kubo Population Correlation`
  - Standalone articles: short technical title without decorative wording.
- Keep typography and layout consistent with the current site. Do not introduce article-specific fonts, oversized headings, bright palettes, or decorative layouts unless the whole site style is intentionally revised.
- Use the existing HTML article structure: header metadata, `series-nav` when applicable, lead paragraph, clear `h2` sections, figures with captions, code links, and footer navigation.

## 生活记录 Section

- The “生活记录” section is a Chinese-language public section for recording daily life. Its modules include “精神世界”, “日常”, “感情”, “出游”, “美食”, “学习”, and “交友”.
- Treat `life.html` as the section landing page. Treat `mind.html` as the “精神世界” module, not as a separate top-level site category. Put additional lifestyle module pages under `life/`.
- Do not force this section into the technical-article requirements for benchmark results, code links, or reproducible computational workflows.
- Keep the tone restrained, sincere, precise, and non-performative. Avoid decorative prose, motivational slogans, raw diary dumps, or sensational private confession.
- Future notes in this section should stay in Chinese unless the user explicitly asks for bilingual or English text.
- Daily-life notes should normally preserve: date, module, one concrete public-safe fact, one real feeling or observation, and one judgment or question that can be revisited later.
- Preserve public-site safety especially carefully here: do not publish private names, raw personal incidents, addresses, real-time locations, exact travel routes, receipts, account details, contact details beyond the site chrome, therapy/medical details, or identifying information about other people unless the user explicitly asks and confirms it is public-safe.
- For “感情”, “交友”, and other relationship notes, anonymize or generalize other people by default and do not publish private conversations verbatim without explicit user confirmation.

## Public-Site Safety

- Treat the website as fully public. Do not publish private local filesystem paths, absolute machine paths, usernames, email addresses beyond the intentional public contact already in the site chrome, account names, tokens, API keys, internal URLs, or other personal information.
- Article text, captions, code snippets, and linked example scripts must avoid machine-specific paths such as local workspace directories. Use repository-relative paths, published URLs, or generic placeholders when a path is necessary.
- Before publishing, scan new article text and code blocks for private paths, personal identifiers, raw terminal prompts, environment dumps, and accidental credentials.

## Article Shape

Each article should be useful and self-contained. It may be moderately long when needed, but the length should clarify the science or workflow rather than reproduce raw source material.

Every article should normally contain:

1. Background: what problem the note addresses and why it matters.
2. Principle: the central mathematical, physical, or computational idea.
3. Workflow: how the calculation or method is actually carried out.
4. Result: at least one staged result, benchmark, diagnostic, or test.
5. Code: links to the executable or source files used in the note.

The background, principle, and workflow sections should be explicit enough that a reader can understand the motivation, the governing idea, and the successful computational path without needing private notes or trial logs.

Articles are not lab notebooks or experiment reports. Do not include tuning history, failed attempts, debugging records, environment mishaps, or chronological error logs unless they are essential to a scientific conclusion. Present a cleaned, complete, successful path with the final choices and enough rationale to reproduce it.

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

1. Confirm that technical articles are English-only; the `life.html`, `mind.html`, and `life/` pages are the explicit Chinese-language exceptions.
2. Confirm that no absolute local paths, private personal information, credentials, or raw private environment details appear in published article text, captions, or code snippets.
3. Check that all local links and image paths resolve.
4. Run any compact scripts linked from the article when feasible.
5. Verify that each article has at least one result, benchmark, or diagnostic.
6. Update `index.html`, `sitemap.xml`, and `README.md` when adding a new published page.
7. Check desktop and mobile layout for long titles, formulas, figures, and code blocks.
8. Commit with a clear message and push to `origin/main` after validation by default, unless the user explicitly asks to keep the change local or not to publish.

## Site Boundaries

- `posts/` contains published HTML articles.
- `life.html` is the public Chinese landing page for the “生活记录” section.
- `mind.html` is the public Chinese “精神世界” module within “生活记录”.
- `life/` contains public Chinese lifestyle module pages such as daily life, relationship, travel, food, study, and friends.
- `assets/img/` contains published images.
- `assets/code/` contains published code attachments.
- `articles/` contains local raw materials and should stay unpublished.
- Keep the site static and GitHub Pages friendly; do not add a build system unless the user asks for one.
