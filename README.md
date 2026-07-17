# Technical Notes Website

This is a lightweight static website for technical articles, designed for GitHub Pages. It does not require a build step: open `index.html` locally or push the repository to `Lihuigao21.github.io`.

## Structure

```text
.
|-- index.html
|-- life.html
|-- mind.html
|-- tags.html
|-- series.html
|-- robots.txt
|-- sitemap.xml
|-- posts
|   |-- hefeinamd-training-hands-on-workflow.html
|   |-- mes-pimd-geometric-phase-thermodynamics.html
|   |-- geometric-phase-key-theory-reproductions.html
|   |-- geometric-phase-jahn-teller-thermodynamics.html
|   |-- jahn-teller-li3-na3.html
|   |-- cmd-curvature-redshift-champagne-bottle.html
|   |-- h2o-abacus-rttddft-electronic-absorption.html
|   |-- cmd-effective-surfaces-tully-sac.html
|   |-- perovskite-softness-controlled-nvt.html
|   |-- spin-mint-three-state-morse-reproduction.html
|   |-- hamgnn-openmx-tio2-hamiltonian-workflow.html
|   |-- spin-mapping-math-foundations.html
|   |-- mmst-zpe-leakage.html
|   |-- pmatrix-decoherence-balance.html
|   |-- ehrenfest-tully-benchmark.html
|   |-- fssh-tully-benchmark.html
|   |-- mqc-background.html
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
|-- life
|   |-- leave-comfort-zone-plan-2026-07-13.html
|   |-- daily.html
|   |-- reflection.html
|   |-- reflection-family-meal-planning-2026-07-17.html
|   |-- daily-record-2026-07-17.html
|   |-- daily-record-2026-07-16.html
|   |-- daily-record-2026-07-15.html
|   |-- daily-record-2026-07-14.html
|   |-- daily-record-2026-07-13.html
|   |-- daily-record-2026-07-12.html
|   |-- daily-record-2026-07-11.html
|   |-- daily-record-2026-07-10.html
|   |-- daily-record-2026-07-09.html
|   |-- daily-record-2026-07-08.html
|   |-- daily-record-2026-07-07.html
|   |-- daily-record-2026-07-06.html
|   |-- daily-record-2026-07-05.html
|   |-- daily-record-2026-07-04.html
|   |-- daily-record-2026-07-03.html
|   |-- daily-record-2026-07-02.html
|   |-- daily-record-2026-07-01.html
|   |-- daily-record-2026-06-30.html
|   |-- daily-summary-2026-06-29.html
|   |-- mind-state-from-blame-to-repair-2026-07-17.html
|   |-- mind-state-responsibility-as-ability-2026-07-17.html
|   |-- mind-state-family-personality-2026-07-16.html
|   |-- mind-state-calm-heart-2026-07-13.html
|   |-- mind-state-language-action-2026-07-12.html
|   |-- mind-state-2026-07-11.html
|   |-- mind-state-worldview-consistency-2026-07-10.html
|   |-- mind-state-worldview-nodes-2026-07-10.html
|   |-- mind-state-2026-07-09.html
|   |-- mind-state-2026-07-07.html
|   |-- mind-state-2026-07-06.html
|   |-- mind-state-2026-07-05.html
|   |-- mind-state-2026-07-04.html
|   |-- mind-state-2026-07-03.html
|   |-- mind-state-2026-07-02.html
|   |-- mind-state-2026-07-01.html
|   |-- mind-state-2026-06-30.html
|   |-- p-to-j-plan-2026-06-29.html
|   |-- relationship.html
|   |-- relationship-words-once-believed-2026-07-17.html
|   |-- relationship-what-we-wanted-2026-07-16.html
|   |-- relationship-qiu-love-lesson-2026-07-16.html
|   |-- relationship-marriage-view-2026-07-15.html
|   |-- relationship-leave-small-pond-2026-07-14.html
|   |-- relationship-self-release-process-2026-07-13.html
|   |-- relationship-admiring-strength-self-frame-2026-07-13.html
|   |-- relationship-rational-emotional-cycle-2026-07-12.html
|   |-- relationship-obsession-2026-07-11.html
|   |-- relationship-destination-scenery-2026-07-11.html
|   |-- relationship-leave-vortex-2026-07-11.html
|   |-- relationship-burial-2026-07-09.html
|   |-- relationship-not-an-option-2026-07-09.html
|   |-- relationship-competition-self-polishing-2026-07-06.html
|   |-- relationship-pain-loop-2026-07-05.html
|   |-- relationship-journey-scenery-2026-07-04.html
|   |-- relationship-cost-of-love-2026-07-03.html
|   |-- relationship-hunger-cohabitation-2026-07-03.html
|   |-- relationship-softness-2026-07-02.html
|   |-- relationship-tragedy-roots-2026-07-01.html
|   |-- relationship-summary-2026-06-29.html
|   |-- travel.html
|   |-- food.html
|   |-- food-xinjiang-first-day-2026-07-14.html
|   |-- life-experience.html
|   |-- skill.html
|   |-- escape-comfort-zone.html
|   |-- escape-dream-song-2026-07-16.html
|   |-- escape-wake-up-to-2026-07-15.html
|   |-- escape-english-task-2026-07-14.html
|   |-- memories.html
|   |-- editorial.html
|   |-- editorial-upward-comfort-zone-2026-07-12.html
|   |-- dreams-creation.html
|   |-- dream-heart-knot-released-2026-07-17.html
|   |-- dream-journey-end-2026-07-16.html
|   |-- dream-chasing-phantom-2026-07-15.html
|   |-- dream-forgetting-is-hard-2026-07-14.html
|   |-- creation-home-2026-07-14.html
|   |-- dream-wangchuan-forgetting-river-2026-07-13.html
|   |-- gpt-deep-chat.html
|   |-- gpt-chat-what-is-true-love-2026-07-13.html
|   |-- gpt-chat-imagined-future-her-2026-07-12.html
|   |-- gpt-chat-dreams-of-reunion-2026-07-12.html
|   |-- gpt-chat-deprived-participation-2026-07-12.html
|   |-- dream-lingering-obsession-2026-07-12.html
|   |-- creation-angel-2026-07-11.html
|   |-- dream-mortal-dust-2026-07-10.html
|   |-- dream-judgment-2026-07-09.html
|   |-- dream-discrete-moments-2026-07-06.html
|   |-- dream-two-weeks-memorial-2026-07-05.html
|   `-- dream-meet-again-2026-07-05.html
|-- assets
|   |-- code
|   |   |-- jt
|   |   |   |-- li3_na3_jt_model.py
|   |   |   |-- mes_pi_gp_logic_demo.py
|   |   |   |-- gp_pseudorotor_thermo.py
|   |   |   |-- gp_fig1d_heat_capacity_reproduction.py
|   |   |   `-- gp_key_theory_reproductions.py
|   |   |-- rt-tddft-h2o
|   |   |-- perovskite-softness
|   |   |-- hamgnn-tio2
|   |   |-- cmd
|   |   |   |-- cmd_curvature_validation.py
|   |   |   |-- cmd_curvature_potential_surface.py
|   |   |   `-- run_cmd_curvature_smoke.py
|   |   |-- mqc
|   |   |   |-- cmd_sac_benchmark.py
|   |   |   |-- cmd_sac_dvr_convergence.py
|   |   |   |-- cmd_sac_extend_fssh.py
|   |   |   |-- tully_common.py
|   |   |   |-- dvr_tully_sac_reference.py
|   |   |   |-- fssh_tully_sac.py
|   |   |   |-- ehrenfest_tully_sac.py
|   |   |   |-- p_matrix.py
|   |   |   |-- mmst.py
|   |   |   |-- mmst_sac_mqc_comparison.py
|   |   |   |-- mmst_gamma_zpe_scan.py
|   |   |   |-- mmst_correction_comparison.py
|   |   |   |-- pmatrix_sac_comparison.py
|   |   |   |-- pmatrix_decoherence_balance_demo.py
|   |   |   |-- spin_mapping_sac_comparison.py
|   |   |   |-- three_state_morse_pes_nac.py
|   |   |   |-- three_state_morse_method_comparison.py
|   |   |   |-- reproduce_spin_mint_three_state_morse.py
|   |   |   `-- su_n_spin_mapping_checks.py
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
|   |   |-- jt
|   |   |-- life
|   |   |-- rt-tddft-h2o
|   |   |-- perovskite-softness
|   |   |-- hamgnn-tio2
|   |   |-- cmd-series
|   |   |-- mqc-series
|   |   |-- pimd-series
|   |   |-- matsubara-series
|   |   |-- rpmd-ir
|   |   |-- trpmd-ir
|   |   |-- allegro
|   |   |-- cayley
|   |   |-- dvr-series
|   |   `-- favicon.svg
|   `-- js
|       |-- article-data.js
|       |-- search.js
|       |-- taxonomy.js
|       `-- main.js
`-- README.md
```

## Published Notes

- `life.html` - Chinese life-notes landing page.
- `life/leave-comfort-zone-plan-2026-07-13.html` - Chinese pinned life-plan note.
- `mind.html` - Chinese inner-world module under life notes.
- `life/daily.html` - Chinese daily-life module.
- `life/reflection.html` - Chinese reflection module.
- `life/reflection-family-meal-planning-2026-07-17.html` - Chinese reflection note.
- `life/daily-record-2026-07-17.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-16.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-15.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-14.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-13.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-12.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-11.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-10.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-09.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-08.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-07.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-06.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-05.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-04.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-03.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-02.html` - Chinese daily-life record note.
- `life/daily-record-2026-07-01.html` - Chinese daily-life record note.
- `life/daily-record-2026-06-30.html` - Chinese daily-life record note.
- `life/daily-summary-2026-06-29.html` - Chinese daily-life summary note.
- `life/mind-state-from-blame-to-repair-2026-07-17.html` - Chinese mental-state reflection note.
- `life/mind-state-responsibility-as-ability-2026-07-17.html` - Chinese mental-state reflection note.
- `life/mind-state-family-personality-2026-07-16.html` - Chinese mental-state reflection note.
- `life/mind-state-calm-heart-2026-07-13.html` - Chinese mental-state reflection note.
- `life/mind-state-language-action-2026-07-12.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-11.html` - Chinese mental-state reflection note.
- `life/mind-state-worldview-consistency-2026-07-10.html` - Chinese mental-state reflection note.
- `life/mind-state-worldview-nodes-2026-07-10.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-09.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-07.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-06.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-05.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-04.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-03.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-02.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-07-01.html` - Chinese mental-state reflection note.
- `life/mind-state-2026-06-30.html` - Chinese mental-state reflection note.
- `life/p-to-j-plan-2026-06-29.html` - Chinese daily-life pinned P-to-J plan.
- `life/relationship.html` - Chinese relationship module.
- `life/relationship-words-once-believed-2026-07-17.html` - Chinese relationship reflection note.
- `life/relationship-what-we-wanted-2026-07-16.html` - Chinese relationship reflection note.
- `life/relationship-qiu-love-lesson-2026-07-16.html` - Chinese relationship reflection note.
- `life/relationship-marriage-view-2026-07-15.html` - Chinese relationship reflection note.
- `life/relationship-leave-small-pond-2026-07-14.html` - Chinese relationship reflection note.
- `life/relationship-self-release-process-2026-07-13.html` - Chinese relationship reflection note.
- `life/relationship-admiring-strength-self-frame-2026-07-13.html` - Chinese relationship reflection note.
- `life/relationship-rational-emotional-cycle-2026-07-12.html` - Chinese relationship reflection note.
- `life/relationship-obsession-2026-07-11.html` - Chinese relationship reflection note.
- `life/relationship-destination-scenery-2026-07-11.html` - Chinese relationship reflection note.
- `life/relationship-leave-vortex-2026-07-11.html` - Chinese relationship reflection note.
- `life/relationship-burial-2026-07-09.html` - Chinese relationship reflection note.
- `life/relationship-not-an-option-2026-07-09.html` - Chinese relationship reflection note.
- `life/relationship-competition-self-polishing-2026-07-06.html` - Chinese relationship reflection note.
- `life/relationship-pain-loop-2026-07-05.html` - Chinese relationship reflection note.
- `life/relationship-journey-scenery-2026-07-04.html` - Chinese relationship reflection note.
- `life/relationship-cost-of-love-2026-07-03.html` - Chinese relationship reflection note.
- `life/relationship-hunger-cohabitation-2026-07-03.html` - Chinese relationship reflection note.
- `life/relationship-softness-2026-07-02.html` - Chinese relationship reflection note.
- `life/relationship-tragedy-roots-2026-07-01.html` - Chinese relationship reflection note.
- `life/relationship-summary-2026-06-29.html` - Chinese relationship reflection note.
- `life/travel.html` - Chinese travel module.
- `life/food.html` - Chinese food module.
- `life/food-xinjiang-first-day-2026-07-14.html` - Chinese food note.
- `life/life-experience.html` - Chinese life-experience module.
- `life/skill.html` - Chinese skill-learning module.
- `life/escape-comfort-zone.html` - Chinese escape-comfort-zone module.
- `life/escape-dream-song-2026-07-16.html` - Chinese escape-comfort-zone note.
- `life/escape-wake-up-to-2026-07-15.html` - Chinese escape-comfort-zone note.
- `life/escape-english-task-2026-07-14.html` - Chinese escape-comfort-zone note.
- `life/memories.html` - Chinese memories module.
- `life/editorial.html` - Chinese editorial module.
- `life/editorial-upward-comfort-zone-2026-07-12.html` - Chinese editorial note.
- `life/dreams-creation.html` - Chinese dreams and creation module.
- `life/dream-heart-knot-released-2026-07-17.html` - Chinese dreams and creation note.
- `life/dream-journey-end-2026-07-16.html` - Chinese dreams and creation note.
- `life/dream-chasing-phantom-2026-07-15.html` - Chinese dreams and creation note.
- `life/dream-forgetting-is-hard-2026-07-14.html` - Chinese dreams and creation note.
- `life/creation-home-2026-07-14.html` - Chinese dreams and creation note.
- `life/dream-wangchuan-forgetting-river-2026-07-13.html` - Chinese dreams and creation note.
- `life/gpt-deep-chat.html` - Chinese deep GPT chat module.
- `life/gpt-chat-what-is-true-love-2026-07-13.html` - Chinese deep GPT chat note.
- `life/gpt-chat-imagined-future-her-2026-07-12.html` - Chinese deep GPT chat note.
- `life/gpt-chat-dreams-of-reunion-2026-07-12.html` - Chinese deep GPT chat note.
- `life/gpt-chat-deprived-participation-2026-07-12.html` - Chinese deep GPT chat note.
- `life/dream-lingering-obsession-2026-07-12.html` - Chinese dreams and creation note.
- `life/creation-angel-2026-07-11.html` - Chinese dreams and creation note.
- `life/dream-mortal-dust-2026-07-10.html` - Chinese dreams and creation special note.
- `life/dream-judgment-2026-07-09.html` - Chinese dreams and creation note.
- `life/dream-discrete-moments-2026-07-06.html` - Chinese dreams and creation note.
- `life/dream-two-weeks-memorial-2026-07-05.html` - Chinese dreams and creation special note.
- `life/dream-meet-again-2026-07-05.html` - Chinese dreams and creation note.
- `posts/geometric-phase-key-theory-reproductions.html`
- `posts/geometric-phase-jahn-teller-thermodynamics.html`
- `posts/hefeinamd-training-hands-on-workflow.html`
- `posts/jahn-teller-li3-na3.html`
- `posts/cmd-curvature-redshift-champagne-bottle.html`
- `posts/h2o-abacus-rttddft-electronic-absorption.html`
- `posts/cmd-effective-surfaces-tully-sac.html`
- `posts/perovskite-softness-controlled-nvt.html`
- `posts/hamgnn-openmx-tio2-hamiltonian-workflow.html`
- `posts/spin-mint-three-state-morse-reproduction.html`
- `posts/spin-mapping-math-foundations.html`
- `posts/mmst-zpe-leakage.html`
- `posts/pmatrix-decoherence-balance.html`
- `posts/ehrenfest-tully-benchmark.html`
- `posts/fssh-tully-benchmark.html`
- `posts/mqc-background.html`
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
3. Add the article metadata to `assets/js/article-data.js`, including tags and series membership.
4. Add the new link to the "Latest Articles" and "Archive" sections in `index.html`.
5. Update `sitemap.xml`.
6. Commit and push to GitHub. GitHub Pages will update automatically.

The homepage search plus the tag and series browsers are static GitHub Pages features powered by `assets/js/article-data.js`, `assets/js/search.js`, and `assets/js/taxonomy.js`. Search matches article titles and canonical tags only. Homepage tag labels are converted into links by `assets/js/main.js`, so tag names should match the canonical names in `article-data.js`.

The `life.html`, `mind.html`, and `life/` pages form a separate Chinese-language life-notes section. It is linked from the top navigation and homepage, but it is not part of the technical article search, tag index, or benchmark/code-link article workflow.

Keep raw source materials, notebooks, PDFs, and drafts in the local `articles/` folder. That folder is ignored by Git so working materials do not get published accidentally.

For published computational notes, prefer compact executable scripts under `assets/code/`. Large notebooks and binary dumps should stay local unless they are deliberately cleaned and documented.

## Deploy to GitHub Pages

1. Use the repository `Lihuigao21.github.io`.
2. Commit this directory and push it to the default branch.
3. In `Settings -> Pages`, make sure the source is the default branch root.
4. Visit `https://lihuigao21.github.io` after GitHub Pages finishes deploying.
