# Thesis-extension analysis (`analysis_thesis/`)

Follow-up experiments for the master's thesis, built on the workshop paper
"Physical Safety Queries Reveal What Video World Models Forget". The scripts reuse the paper's
modules under `src/safety_probing/`.

Run scripts from this directory unless stated otherwise:

```bash
PYTHONPATH=. ../.venv/bin/python <script>.py                       # analysis (torch, numpy)
PYTHONPATH=. ../external/LTX-2/.venv/bin/python make_figures_*.py  # figures (needs matplotlib)
```

`common.py` reuses the paper's `scripts/build_statistics.py`, so metrics match Table 1 definitions.
`check_repro.py` reproduces Table 1 from the saved probes.

## Notes on the data

1. **Replicas.** The 810 conditions of ForkSafe-CF v1 are 270 physical configurations run three
   times; the seed is not used by the physics. The paper split spreads replicas across
   train/val/test (`replica_leak.py`). Use the physically disjoint protocols in `psq_cv.py`:
   `config5` (5-fold over configurations) and `logo9` (leave one manoeuvre group out).
2. **Stability labels.** The dataset logger pairs PhysX link masses with link poses taken in USD
   traversal order, so the payload mass is placed at the rear-wheel swivel and the chassis mass
   at the mast (`sim_probe/probe_mass_properties.py`,
   `com_identify.py`). `truecom.py` rebuilds the CoM from the true link masses and computes a
   multi-body ZMP; against the wheel-load centre of pressure it is within 1.5 cm (longitudinal)
   and 3 cm (lateral), where the logged ZMP is off by 52 cm and 8 cm (`mechanics_true.py`).
   All results here use targets built with `truecom_build_targets.py` (`*_truecom.pt`). Mass,
   configured CoM offset, roll and tip-over labels are unaffected.
3. **Configured vs physical CoM offset.** The payload CoM offset is set in the scaled local frame
   of the load cube, so a configured 0.25 m is 0.225 m physically.
4. **Lateral CoM offset is weakly observable.** On v1 the same manoeuvre and mass repeat on a grid
   and the simulator is deterministic, which lets a readout separate the three offset levels. On
   the continuous v2 set neither the video readout nor a readout on ground-truth kinematics
   recovers the offset; payload mass is recovered in both.

## Mechanics

| Script | What it does |
|---|---|
| `sim_probe/probe_mass_properties.py` | One-off Isaac Sim query: link masses, link CoMs, PhysX link order vs USD traversal order |
| `com_identify.py` | System CoM from wheel loads (centre of pressure), independent of the logged CoM |
| `truecom.py` | True rigid-body CoM, multi-body ZMP, outward margins; `corrected_rows()` feeds the target builder |
| `mechanics_true.py` | Stability arm, static rollover threshold, lateral demand, ZMP-vs-CoP validation per run |
| `physics_pipeline.py` | Identified payload plugged into the static stability model, compared with the learned readout |
| `rollover_physics.py`, `analytic_*.py`, `comfix*.py`, `zmp_*.py`, `cop_check.py` | Earlier analyses based on the logged CoM |

## Prediction experiments (ForkSafe-CF v1)

| Script | What it does |
|---|---|
| `psq_cv.py` | Readout with physically disjoint folds; `--with-command` appends the planned drive/steering command, `--variant command_only` uses no video |
| `run_v1_truecom_jobs.sh` | Job runner: unseen manoeuvre, unseen payload-manoeuvre combination, observation lengths 1.0-2.71 s |
| `run_truecom_paper_split.sh`, `truecom_eval.py` | Paper split (observed video, real-future oracle, generated future) on the true-CoM targets |
| `rollover_criterion.py` | Roll-over about the outer fork-roller line: turn severity against g (b - y_c) / h with no fitted constant, compared with the support-triangle threshold |
| `pair_outcome_flip.py` | Matched pairs with the same command and different outcomes: does the tipped member get the higher score? |
| `psq_heldout.py`, `kinematics_*.py`, `psq_ablation.py`, `baseline_physical.py` | Held-out levels, no-video kinematics ceiling, readout ablations, simple baselines |
| `lead_time.py`, `pair_effect_size.py`, `replica_leak.py`, `dense_table.py` | Lead time before support loss, realised motion difference within pairs, replica structure |

## ForkSafe-CF v2 (`v2/`)

Continuous payload mass, lateral CoM offset, drive command and steering command; one mass or CoM
intervention per pair; no replicas. Parts `forksafe_v2_part_01..05` cover drive commands 15-25,
parts `forksafe_v2_band_01..05` cover 18-22, where the payload decides the outcome.

| Script | What it does |
|---|---|
| `build_v2_config.py`, `run_v2_generation*.sh` | Sample a dataset part and run the simulator |
| `build_v2_manifest.py` | Manifest and pair list for any set of parts (labels from the paper's analysis code) |
| `psq_v2.py`, `psq_v2_cv.py` | Readout on a fixed split / pair-grouped K-fold (all runs out of fold) |
| `kinematics_v2.py` | No-video ceiling on ground-truth kinematics |
| `hypothesis_v2.py` | Forward model conditioned on a payload hypothesis; identification by matching; what-if prediction |
| `run_v2_final.sh`, `run_v2_band.sh`, `run_v2_obslen.sh` | Full pipelines (features, true-CoM targets, readouts, learning curve, observation length) |
| `run_v2_obslen_cv.sh` | Observation-length sweep with pair-grouped CV (all runs out of fold) |
| `payload_value_v2.py` | Value of payload knowledge without video: command, + true mass, + true lateral CoM offset |
| `physics_stage2_v2.py` | Physics-based second stage: saturating severity law over the roller-line threshold, compared with the learned stage |
| `build_v2_sweep_config.py`, `run_v2_generation_sweep.sh`, `run_v2_sweep.sh` | Boundary sweep (regular grid over payload mass / lateral CoM offset and drive command): configs, generation, analysis pipeline |
| `merge_sets.py`, `sweep_eval_v2.py` | Evaluate readouts trained on one set on a held-out set; boundary map, replicates, critical drive command per payload |
| `refresh_v2cv_eval.py` | Re-evaluate saved CV predictions to add new metrics to the result files |
| `early_direction_check.py` | Shows that the signed roll is unpredictable before the turn because the turn direction is not yet visible |
| `ltx_v2.py`, `run_v2_ltx.sh` | Generated-future audit on a v2 set (real-future readout applied to LTX rollouts of held-out pairs) |

## Figures

`make_figures_mechanics.py`, `make_fig_roller_line.py`, `make_figures_cv.py`, `make_figures_early.py`, `make_figures_v1.py`,
`make_fig_forecast_gallery.py`, `v2/make_figures_v2.py`, `v2/make_fig_v2_dataset.py`,
`v2/make_fig_v2_information.py`, `v2/make_fig_v2_gallery.py`, `v2/make_fig_v2_early.py`,
`v2/make_fig_v2_early_cv.py`, `v2/make_fig_v2_sweep.py` write PDF and
PNG files to `figures/` using `thesis_style.py`. Result summaries are JSON files under `out/`;
prediction arrays (`*.npz`) and checkpoints (`*.pt`) are not tracked.
