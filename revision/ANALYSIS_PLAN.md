# MEI Manuscript Revision — Analysis Plan
**Date:** 2026-03-11
**Manuscript:** Meta-awareness, mind-wandering, and the control of 'default' external and internal orientations of attention
**Authors:** Isaac R. Christian, Samuel A. Nastase, Lauren K. Kim, Michael S. A. Graziano

---

## Data Overview

| Source | Location | Structure |
|---|---|---|
| ISC (200 ROIs) | `data/work/isc_dat/isc_ALL_roi_values.csv` | 181,600 rows; Subject, cond(ext/int), mov, Roi(0–199), run(1–4), isc_val, salience |
| ISC with sub IDs | `data/work/isc_dat/n39_sub_tracked_stacked_data_MERGED.csv` | 187,200 rows; same + sub_id (sub-002 etc.) |
| ISC voxel data | `data/work/isc_dat/n39_ext_isc.npz` / `n39_int_isc.npz` | Per movie: (TRs, 112179 voxels, 4 runs, ~19 subs) |
| Behavioral | `data/behavioral/sub-XXX_behav.npy` | Dict{Internal/External → movie → run → {start_tr, end_tr, bpress[timestamps]}} |
| Button counts | `data/work/button_press_counts/` | Per-subject total counts by condition |
| Engagement | `data/behavioral/high-low-sal_data.csv` | 228 rows; Subject ID, movie, engagement(1–10), salience |
| WSFC (pre-computed) | `data/work/wsfc/` | 34-network connectivity matrices; pre-computed rep1→rep4 |
| Schaefer atlas | `data/work/schaefer_atlas/` | 200 parcels, 17 networks (34 with hemispheres) |

**Salience assignments:**
- High salience: shrek, office, sherlock
- Low salience: brushing, origami (oragami in files), cake

**N = 39 total subjects** (counterbalanced: ~19–20 per condition per movie)

**Environment:** `mei_` conda env — statsmodels 0.13.5, scipy 1.7.3, pingouin 0.1.5

---

## Analysis 1 — Region-by-Region ISC

**Script:** `revision/code/analysis1_region_isc.py`
**Data:** `n39_sub_tracked_stacked_data_MERGED.csv`
**Outputs:** `revision/results/region_isc/`

**Model:** Linear mixed-effects for each of 200 ROIs independently:
```
isc_val ~ run * salience * cond + (1 | sub_id)
```
- `run` coded as continuous (1–4; linear habituation)
- `salience` reference level: high
- `cond` reference level: ext (external)
- Random intercept per subject
- Fit with `statsmodels.formula.api.mixedlm`, REML=False, optimizer=L-BFGS-B

**Multiple comparisons:** Benjamini-Hochberg FDR correction applied separately per model term, across 200 ROIs.

**Descriptive:** Δ ISC (rep4 − rep1) computed per ROI × condition × salience for visualization.

**Outputs:**
- `roi_lme_results.csv` — full results (ROI, term, coef, SE, t, p, p_fdr, sig_fdr)
- `sig_roi_summary.csv` — FDR-significant ROIs only
- `delta_isc_run4minus1.csv` — descriptive delta ISC
- `region_isc_delta_heatmap.png` — bar plot of Δ ISC per ROI
- `summary.txt` — results narrative

---

## Analysis 2 — By-Repetition Connectivity Change

**Script:** `revision/code/analysis2_connectivity_transition.py`
**SLURM:** `revision/code/analysis2_slurm.sh` (3 parallel jobs, array 1–3)
**Data:** `n39_ext_isc.npz`, `n39_int_isc.npz`, Schaefer atlas
**Outputs:** `revision/results/connectivity/`, `revision/plots/connectivity_transitions.png`

**Methodology** (follows `wsfc_3.ipynb` exactly):
1. Load raw voxel-level fMRI data (TRs × voxels × 4 runs × ~19 subs per movie)
2. Parcellate to 34-network level (Schaefer 200-parcel 17-network atlas, hemisphere-specific)
3. For each of 3 transitions (rep1→rep2, rep2→rep3, rep3→rep4):
   - Compute WSFC matrices for run `n` and `n+1`, for each condition (external, internal)
   - Compute int − ext difference matrix at each run
   - Compute transition effect: Δ[int−ext]_{n+1} − Δ[int−ext]_n
   - Average across 6 movies
   - 10,000 sign-flip permutations for null distribution
   - FDR correction (Benjamini-Hochberg)
4. Plot 3 × 34×34 connectivity matrices side by side

**SLURM setup:**
- 3 array jobs (transition=1, 2, 3) in parallel
- 4 CPUs, 32 GB RAM, 6 hr time limit
- Intermediate results saved after each job
- Combined figure generated when all 3 complete

**Expected finding per R&R:** Increases emerge primarily between reps 1 and 2, then stabilize.

---

## Analysis 3 — Temporal Distribution of Button Presses

**Script:** `revision/code/analysis3_button_press_temporal.py`
**Data:** `data/behavioral/sub-XXX_behav.npy` (button press timestamps in seconds)
**Outputs:** `revision/plots/button_press_temporal_distribution.png`, `revision/results/button_press_temporal/press_timestamps_by_movie.csv`

**Figure layout:** 2 rows × 3 columns
- Top row: high salience (Shrek, The Office, Sherlock)
- Bottom row: low salience (Brushing, Origami, Cake)

**Per subplot:** All button presses from all subjects × all repetitions × all conditions (collapsed)
- Each press = one vertical line
- Line opacity ∝ local press density (KDE with Scott's bandwidth rule)
- Density-shaded background fill for visual reference
- Duration inferred from max observed timestamp per movie

**Note:** bpress=-1 treated as no event (0 presses). Timestamps are in seconds from movie onset.

---

## Analysis 4 — Participant-Specific Engagement Ratings (Supplementary)

**Script:** `revision/code/analysis4_engagement_ratings.py`
**Data:** `high-low-sal_data.csv` + `sub-XXX_behav.npy`
**Outputs:** `revision/results/engagement_ratings/`, `revision/plots/engagement_scatter.png`

**Steps:**
1. **Data inspection:** Print column names, shapes, unique values for both sources
2. **Validation:** Point-biserial correlation (engagement [continuous] vs salience [binary high/low])
   - Expected: r ≈ 0.74, p < 0.001 (per R&R)
3. **External condition regression:**
   - `total_bpress ~ engagement + (1 | subject)` — statsmodels.mixedlm
   - Report: β, SE, t, p, marginal R²
4. **Internal condition regression:** Same model
5. **Visualization:** Two-panel scatter (external | internal), x=engagement rating, y=total button presses, OLS regression line with 95% CI bootstrap, individual points colored by participant

**Merge logic:** Button press counts summed across all 4 repetitions per subject × movie. Merged on (sub_id, movie_norm) where 'oragami' → 'origami'.

**Output summary.txt:** APA-format write-up ready for manuscript.

---

## Analysis 5 — Post-Hoc Power Analysis & Effect Sizes

**Script:** `revision/code/analysis5_power_effect_sizes.py`
**Data:** `high-low-sal_data.csv` + `sub-XXX_behav.npy`
**Outputs:** `revision/results/power_and_effect_sizes/`

**Models analyzed (all non-neural, non-GLM):**

| Analysis | Model | Effect metric |
|---|---|---|
| Main behavioral | `bpress ~ run × cond × salience + (1\|sub)` | Marginal R², Conditional R², partial η² per term (via LRT) |
| External vs Internal | Paired t-test | Cohen's d |
| High vs Low salience (External) | Paired t-test | Cohen's d |
| High vs Low salience (Internal) | Paired t-test | Cohen's d |
| Run 1 vs Run 4 (per condition) | Paired t-test | Cohen's d |
| Engagement × bpress (External) | `bpress ~ engagement + (1\|sub)` | partial η² via LRT, Marginal R² |
| Engagement × bpress (Internal) | Same | Same |

**R² computation (Nakagawa & Schielzeth 2013):**
- Marginal R² = Var(fixed) / (Var(fixed) + Var(random) + Var(residual))
- Conditional R² = (Var(fixed) + Var(random)) / same

**Power analysis:**
- Cohen's d → power via non-central t distribution (scipy.stats.nct)
- Cohen's f² → power via non-central F distribution (scipy.stats.ncf)
- Parameters: N=39, α=0.05, two-tailed

**Note on neural analyses:** Fisher z-transformed correlations serve as effect size analogs for ISC and connectivity analyses — no separate effect sizes computed for those.

**Outputs:**
- `behavioral_effect_sizes.csv` — columns: analysis, test_type, term, statistic, effect_size_type, effect_size_value, cohens_f2, N, power, p
- `summary.txt` — APA-format, flags underpowered analyses

---

## Analysis 6 — Updated Figures with Individual Data Points

**Script:** `revision/code/analysis6_updated_figures.py`
**Data:** `sub-XXX_behav.npy`, `n39_sub_tracked_stacked_data_MERGED.csv`
**Outputs:** `revision/plots/fig2_violin_bpress.png`, `revision/plots/fig2_violin_sal_cond.png`, `revision/plots/fig3_isc_violin.png`

**Figure 2 remake (button presses):**
- 2×3 layout (same as original): 2 rows (high/low salience) × 3 columns (movies)
- Per subplot per repetition: violin plot + strip plot (jittered individual points)
- Mean trajectory line connecting repetition means
- External = steelblue, Internal = sandybrown

**Figure 2 salience summary:**
- 1×2 layout: External | Internal
- Violin + strip by repetition, separated by salience (high/low)

**ISC figure remake:**
- 1×2 layout: External | Internal
- ISC values averaged across 200 ROIs per subject per run per salience
- Violin + strip by repetition, separated by salience

**Important:** All saved as new files — original figures not modified.

---

## Analysis 7 — Source Data Assembly

**Script:** `revision/code/analysis7_source_data.py`
**Outputs:** `revision/data/source_data/`

**Files produced:**
- `fig2_button_presses.xlsx` — raw trial data + summary by condition/salience/repetition
- `fig3_isc_values.xlsx` — per-subject mean ISC + per-ROI values
- `fig6_connectivity_transitions.xlsx` — per-transition connectivity matrices (when Analysis 2 complete)
- `figS_engagement.xlsx` — engagement ratings + button press counts merged

**Note:** Requires `openpyxl` (installed via pip if not present). Falls back to CSV if unavailable.

---

## Execution Order

| Phase | Analyses | Method | Est. time |
|---|---|---|---|
| 1 (parallel, now) | 1, 3, 4, 5, 6 | Local Python | 5–30 min |
| 2 (SLURM) | 2 | SLURM array job (3 tasks) | 2–6 hr |
| 3 (after outputs) | 7 | Local Python | 5 min |

**SLURM submission:**
```bash
cd /jukebox/graziano/coolCatIsaac/mei/revision/code
sbatch analysis2_slurm.sh
```

**Resumability:** Each analysis saves intermediate outputs (per-ROI results, per-transition connectivity npy files). If interrupted, re-running will regenerate from those checkpoints.

---

## Key Assumptions

1. `run` coded continuous (1–4) — tests linear change across repetitions
2. Figure 3 layout: 2×3 (6 movies total; "3×3" in original brief was a typo)
3. bpress=-1 → 0 events; timestamps in seconds from movie onset
4. Engagement merge: 'oragami' (behavioral files) → 'origami' (engagement CSV) normalization
5. ISC subject index: uses `sub_id` from MERGED CSV (N=39 mapped back)
6. WSFC atlas resampling: requires a reference fMRI nifti from `data/work/glm_data/`
7. Fisher z-transformed correlations serve as effect size analogs for all neural analyses
8. N=39 for all power analyses unless a specific test used a different sample size
