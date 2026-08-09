# Technical Notes Website

This is a lightweight static website for technical articles, designed for GitHub Pages. It does not require a build step: open `index.html` locally or push the repository to `Lihuigao21.github.io`.

## Add a New Article

1. Copy `posts/technical-note-template.html` and rename it, for example `posts/my-first-note.html`.
2. Update the title, date, tags, description, and body in the new file.
3. Add the article metadata to `assets/js/article-data.js`, including tags and series membership.
4. Add the new link to the "Latest Articles" and "Archive" sections in `index.html`.
5. Update `sitemap.xml`.
6. Commit and push to GitHub. GitHub Pages will update automatically.

The homepage search plus the tag and series browsers are static GitHub Pages features powered by `assets/js/article-data.js`, `assets/js/search.js`, and `assets/js/taxonomy.js`. Search matches article titles and canonical tags only. Homepage tag labels are converted into links by `assets/js/main.js`, so tag names should match the canonical names in `article-data.js`.

The `life.html`, `mind.html`, and `life/` pages form a separate Chinese-language life-notes section. It is linked from the top navigation and homepage, but it is not part of the technical article search, tag index, or benchmark/code-link article workflow.

Published technical articles and individual life-note articles receive a public, per-page Giscus discussion thread through `assets/js/main.js`. Module landing pages do not create discussion threads.

Keep raw source materials, notebooks, PDFs, and drafts in the local `articles/` folder. That folder is ignored by Git so working materials do not get published accidentally.

For published computational notes, prefer compact executable scripts under `assets/code/`. Large notebooks and binary dumps should stay local unless they are deliberately cleaned and documented.

## Deploy to GitHub Pages

1. Use the repository `Lihuigao21.github.io`.
2. Commit this directory and push it to the default branch.
3. In `Settings -> Pages`, make sure the source is the default branch root.
4. Visit `https://lihuigao21.github.io` after GitHub Pages finishes deploying.
