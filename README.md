# Technical Notes Website

This is a lightweight static website for technical articles, designed for GitHub Pages. It does not require a build step: open `index.html` locally or push the repository to `Lihuigao21.github.io`.

## Structure

```text
.
|-- index.html
|-- robots.txt
|-- sitemap.xml
|-- posts
|   |-- trpmd-qtip4pf-water-ir-spectrum.html
|   |-- normal-mode-free-ring-polymer.html
|   |-- rpmd-sho-correlation.html
|   |-- pimc-sho-metropolis.html
|   |-- pimd-ring-polymer-basics.html
|   |-- pimd-nvt-gle-piglet.html
|   |-- matsubara-lscivr-phase-space.html
|   |-- matsubara-modes-and-phase.html
|   |-- matsubara-quartic-benchmark.html
|   |-- spcf-rpmd-water-ir-spectrum.html
|   |-- allegro-mlpes-mapbi3-workflow.html
|   |-- cayley-transform-ring-polymer.html
|   |-- dvr-wavepacket-methods.html
|   |-- dvr-operator-matrices.html
|   |-- dvr-wavepacket-dynamics.html
|   |-- dvr-ensemble-evolution.html
|   |-- dvr-kubo-correlation.html
|   |-- dvr-flux-side-correlation.html
|   |-- dvr-excited-ground-kubo.html
|   `-- technical-note-template.html
|-- assets
|   |-- code
|   |   |-- pimd
|   |   |   |-- pimd_sho_benchmark.py
|   |   |   |-- pimd_gle_piglet_toy.py
|   |   |   |-- pimc_sho_metropolis.py
|   |   |   |-- rpmd_sho_correlation.py
|   |   |   `-- normal_mode_vs_bead.py
|   |   |-- matsubara
|   |   |   |-- matsubara_lscivr_benchmark.py
|   |   |   |-- matsubara_mode_filter.py
|   |   |   |-- matsubara_quartic_benchmark.py
|   |   |   |-- willatt_fig39_partial_repro.csv
|   |   |   |-- willatt_fig39_partial_repro_meta.csv
|   |   |   `-- willatt_fig39_partial_repro.png
|   |   |-- rpmd-ir
|   |   |   |-- compute_ir_spectrum_mudot.py
|   |   |   `-- run_mudot_analysis.sh
|   |   |-- trpmd-ir
|   |   |   |-- analyze_method.py
|   |   |   |-- compute_ir_spectrum_mudot.py
|   |   |   |-- experiment_nalpha_bertie_lan_1996.csv
|   |   |   |-- input-rpmd-lambda-0p001.template.xml
|   |   |   |-- input-trpmd-lambda-0p5.template.xml
|   |   |   |-- parse_qtip4pf_dipoles.py
|   |   |   |-- render_input.py
|   |   |   `-- run-qtip4pf-trpmd.slurm
|   |   |-- allegro
|   |   |   |-- collect_nve_samples.py
|   |   |   |-- formal_nve970.yaml
|   |   |   |-- prepare_nve_branches.py
|   |   |   |-- run_mlpes_md.py
|   |   |   |-- run_mlpes_nve.py
|   |   |   `-- run_train_then_md.slurm
|   |   |-- cayley_dt_scan.py
|   |   `-- dvr
|   |       |-- dvr_fd_benchmark.py
|   |       |-- dvr_ensemble_demo.py
|   |       |-- dvr_excited_ground_kubo.py
|   |       |-- dvr_kubo_minimal.py
|   |       `-- source
|   |-- css
|   |   `-- styles.css
|   |-- img
|   |   |-- pimd-series
|   |   |-- matsubara-series
|   |   |-- rpmd-ir
|   |   |-- trpmd-ir
|   |   |-- allegro
|   |   |-- cayley
|   |   |-- dvr-series
|   |   `-- favicon.svg
|   `-- js
|       `-- main.js
`-- README.md
```

## Published Notes

- `posts/trpmd-qtip4pf-water-ir-spectrum.html`
- `posts/normal-mode-free-ring-polymer.html`
- `posts/rpmd-sho-correlation.html`
- `posts/pimc-sho-metropolis.html`
- `posts/pimd-ring-polymer-basics.html`
- `posts/pimd-nvt-gle-piglet.html`
- `posts/cayley-transform-ring-polymer.html`
- `posts/matsubara-lscivr-phase-space.html`
- `posts/matsubara-modes-and-phase.html`
- `posts/matsubara-quartic-benchmark.html`
- `posts/spcf-rpmd-water-ir-spectrum.html`
- `posts/allegro-mlpes-mapbi3-workflow.html`
- `posts/dvr-wavepacket-methods.html`
- `posts/dvr-operator-matrices.html`
- `posts/dvr-wavepacket-dynamics.html`
- `posts/dvr-ensemble-evolution.html`
- `posts/dvr-kubo-correlation.html`
- `posts/dvr-flux-side-correlation.html`
- `posts/dvr-excited-ground-kubo.html`

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
