# Technical Notes Website

This is a lightweight static website for technical articles, designed for GitHub Pages. It does not require a build step: open `index.html` locally or push the repository to `Lihuigao21.github.io`.

## Structure

```text
.
|-- index.html
|-- robots.txt
|-- sitemap.xml
|-- posts
|   |-- cayley-transform-ring-polymer.html
|   |-- dvr-wavepacket-methods.html
|   |-- dvr-operator-matrices.html
|   |-- dvr-wavepacket-dynamics.html
|   |-- dvr-ensemble-evolution.html
|   |-- dvr-kubo-correlation.html
|   |-- dvr-flux-side-correlation.html
|   `-- technical-note-template.html
|-- assets
|   |-- code
|   |   |-- cayley_dt_scan.py
|   |   `-- dvr
|   |       |-- dvr_fd_benchmark.py
|   |       |-- dvr_ensemble_demo.py
|   |       |-- dvr_kubo_minimal.py
|   |       `-- source
|   |-- css
|   |   `-- styles.css
|   |-- img
|   |   |-- cayley
|   |   |-- dvr-series
|   |   `-- favicon.svg
|   `-- js
|       `-- main.js
`-- README.md
```

## Published Notes

- `posts/cayley-transform-ring-polymer.html`
- `posts/dvr-wavepacket-methods.html`
- `posts/dvr-operator-matrices.html`
- `posts/dvr-wavepacket-dynamics.html`
- `posts/dvr-ensemble-evolution.html`
- `posts/dvr-kubo-correlation.html`
- `posts/dvr-flux-side-correlation.html`

## Add a New Article

1. Copy `posts/technical-note-template.html` and rename it, for example `posts/my-first-note.html`.
2. Update the title, date, tags, description, and body in the new file.
3. Add the new link to the "Latest Articles" and "Archive" sections in `index.html`.
4. Commit and push to GitHub. GitHub Pages will update automatically.

Keep raw source materials, notebooks, PDFs, and drafts in the local `articles/` folder. That folder is ignored by Git so working materials do not get published accidentally.

For published computational notes, prefer compact executable scripts under `assets/code/`. Large notebooks and binary dumps should stay local unless they are deliberately cleaned and documented.

## Deploy to GitHub Pages

1. Use the repository `Lihuigao21.github.io`.
2. Commit this directory and push it to the default branch.
3. In `Settings -> Pages`, make sure the source is the default branch root.
4. Visit `https://lihuigao21.github.io` after GitHub Pages finishes deploying.
