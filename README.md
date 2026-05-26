# Technical Notes Website

This is a lightweight static website for technical articles, designed for GitHub Pages. It does not require a build step: open `index.html` locally or push the repository to `username.github.io`.

## Structure

```text
.
|-- index.html
|-- posts
|   |-- cayley-transform-ring-polymer.html
|   `-- technical-note-template.html
|-- assets
|   |-- css
|   |   `-- styles.css
|   |-- img
|   |   |-- favicon.svg
|   |   `-- cayley/
|   `-- js
|       `-- main.js
`-- README.md
```

## Add a New Article

1. Copy `posts/technical-note-template.html` and rename it, for example `posts/my-first-note.html`.
2. Update the title, date, tags, description, and body in the new file.
3. Add the new link to the "Latest Articles" and "Archive" sections in `index.html`.
4. Commit and push to GitHub. GitHub Pages will update automatically.

Keep raw source materials, notebooks, PDFs, and drafts in the local `articles/` folder. That folder is ignored by Git so working materials do not get published accidentally.

## Deploy to GitHub Pages

1. Use a repository named `your-github-username.github.io`.
2. Commit this directory and push it to the default branch.
3. In `Settings -> Pages`, make sure the source is the default branch root.
4. Visit `https://your-github-username.github.io` after GitHub Pages finishes deploying.
