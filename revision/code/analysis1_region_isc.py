#!/usr/bin/env python
"""
Analysis 1 — Region-by-Region ISC
Fits LME: isc_val ~ run * salience * cond + (1|sub_id)
for each of 200 brain regions. Applies FDR correction.
Run with: conda run -n mei_ python analysis1_region_isc.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/jukebox/graziano/coolCatIsaac/mei'
ISC_CSV = f'{BASE}/data/work/isc_dat/n39_sub_tracked_stacked_data_MERGED.csv'
OUT_DIR = '/jukebox/graziano/coolCatIsaac/mei/revision/results/region_isc'
PLOT_DIR = '/jukebox/graziano/coolCatIsaac/mei/revision/plots'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("Analysis 1 — Region-by-Region ISC")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading ISC data...")
df = pd.read_csv(ISC_CSV)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Conditions: {df['cond'].unique()}")
print(f"  Runs: {sorted(df['run'].unique())}")
print(f"  Movies: {df['mov'].unique()}")
print(f"  Num ROIs: {df['Roi'].nunique()}")
print(f"  Num unique sub_ids: {df['sub_id'].nunique()}")
print(f"  Sample rows:\n{df.head(3)}")

# Standardize condition labels
df['cond'] = df['cond'].str.lower().str.strip()
df['cond'] = df['cond'].replace({'external': 'ext', 'internal': 'int'})

# Add salience from movie name (MERGED CSV lacks salience column)
HIGH_SAL_MOVS = {'shrek', 'office', 'sherlock'}
LOW_SAL_MOVS  = {'brushing', 'oragami', 'cake'}
df['salience'] = df['mov'].str.lower().map(
    lambda m: 'high' if m in HIGH_SAL_MOVS else ('low' if m in LOW_SAL_MOVS else 'unknown')
)
print(f"  Salience (derived from movie): {df['salience'].value_counts().to_dict()}")

# Ensure run is numeric
df['run'] = df['run'].astype(float)

# Check for NaN
nan_count = df['isc_val'].isna().sum()
print(f"\n  NaN in isc_val: {nan_count}")
df = df.dropna(subset=['isc_val', 'sub_id', 'run', 'salience', 'cond'])

# ── Fit LME per ROI ───────────────────────────────────────────────────────────
print("\n[2] Fitting LME for each of 200 ROIs...")
print("    Model: isc_val ~ run * salience * cond + (1|sub_id)")

rois = sorted(df['Roi'].unique())
print(f"    ROIs: {len(rois)} (range {min(rois)}–{max(rois)})")

terms_of_interest = ['run', 'salience[T.low]', 'cond[T.int]',
                     'run:salience[T.low]', 'run:cond[T.int]',
                     'salience[T.low]:cond[T.int]',
                     'run:salience[T.low]:cond[T.int]']

results_rows = []

for i, roi in enumerate(rois):
    if i % 50 == 0:
        print(f"    ROI {i}/{len(rois)}...")

    roi_df = df[df['Roi'] == roi].copy()

    try:
        # Fit model — run as continuous, salience and cond as categorical
        model = mixedlm(
            'isc_val ~ run * salience * cond',
            data=roi_df,
            groups=roi_df['sub_id']
        )
        result = model.fit(reml=False, method='lbfgs', maxiter=500)

        for term in result.params.index:
            if term == 'Group Var':
                continue
            row = {
                'Roi': roi,
                'term': term,
                'coef': result.params[term],
                'se': result.bse.get(term, np.nan),
                't': result.tvalues.get(term, np.nan),
                'p': result.pvalues.get(term, np.nan),
                'converged': result.converged
            }
            results_rows.append(row)

    except Exception as e:
        # Record failure for all terms
        for term in terms_of_interest + ['Intercept']:
            results_rows.append({
                'Roi': roi, 'term': term,
                'coef': np.nan, 'se': np.nan, 't': np.nan, 'p': np.nan,
                'converged': False
            })

results_df = pd.DataFrame(results_rows)
print(f"\n  Total result rows: {len(results_df)}")

# ── FDR correction ────────────────────────────────────────────────────────────
print("\n[3] Applying FDR correction (Benjamini-Hochberg) per term...")

fdr_rows = []
for term, term_df in results_df.groupby('term'):
    term_df = term_df.copy()
    valid = term_df['p'].notna()
    p_vals = term_df.loc[valid, 'p'].values

    if len(p_vals) > 0:
        _, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
        term_df.loc[valid, 'p_fdr'] = p_fdr
    else:
        term_df['p_fdr'] = np.nan

    term_df['sig_fdr'] = term_df['p_fdr'] < 0.05
    fdr_rows.append(term_df)

results_df = pd.concat(fdr_rows, ignore_index=True)

# Summary of significant regions
print("\n  Significant ROIs per term (FDR p < 0.05):")
for term, grp in results_df.groupby('term'):
    nsig = grp['sig_fdr'].sum()
    if nsig > 0:
        print(f"    {term}: {int(nsig)} significant ROIs")

# ── Compute delta ISC (rep4 – rep1) ──────────────────────────────────────────
print("\n[4] Computing delta ISC (run4 – run1) per ROI/condition/salience...")

run1 = df[df['run'] == 1].groupby(['Roi', 'cond', 'salience'])['isc_val'].mean()
run4 = df[df['run'] == 4].groupby(['Roi', 'cond', 'salience'])['isc_val'].mean()
delta_isc = (run4 - run1).reset_index()
delta_isc.columns = ['Roi', 'cond', 'salience', 'delta_isc']

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\n[5] Saving outputs...")

# Full results
out_path = f'{OUT_DIR}/roi_lme_results.csv'
results_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

# Significant ROIs only
sig_df = results_df[results_df['sig_fdr'] == True]
sig_path = f'{OUT_DIR}/sig_roi_summary.csv'
sig_df.to_csv(sig_path, index=False)
print(f"  Saved: {sig_path} ({len(sig_df)} rows)")

# Delta ISC
delta_path = f'{OUT_DIR}/delta_isc_run4minus1.csv'
delta_isc.to_csv(delta_path, index=False)
print(f"  Saved: {delta_path}")

# ── Plot: heatmap of delta ISC ─────────────────────────────────────────────────
print("\n[6] Plotting delta ISC heatmap...")

fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
combos = [('ext', 'high'), ('ext', 'low'), ('int', 'high'), ('int', 'low')]
titles = ['External\nHigh Salience', 'External\nLow Salience',
          'Internal\nHigh Salience', 'Internal\nLow Salience']
colors = ['steelblue', 'lightsteelblue', 'sandybrown', 'wheat']

for ax, (cond, sal), title, col in zip(axes, combos, titles, colors):
    sub = delta_isc[(delta_isc['cond'] == cond) & (delta_isc['salience'] == sal)]
    if len(sub) == 0:
        ax.set_title(title + '\n(no data)')
        continue
    sub = sub.set_index('Roi')['delta_isc'].reindex(range(200)).fillna(0)
    ax.barh(range(200), sub.values, color=col, alpha=0.8, height=1.0)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Δ ISC (rep4 − rep1)', fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel('ROI index', fontsize=9)

plt.suptitle('Change in ISC from Repetition 1 to 4\nby Condition and Salience', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/region_isc_delta_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/region_isc_delta_heatmap.png")

# ── Summary text ─────────────────────────────────────────────────────────────
print("\n[7] Writing summary...")

sig_terms = {}
for term, grp in results_df.groupby('term'):
    nsig = int(grp['sig_fdr'].sum())
    if nsig > 0:
        sig_terms[term] = nsig

with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write("Analysis 1 — Region-by-Region ISC\n")
    f.write("=" * 50 + "\n\n")
    f.write("Model: isc_val ~ run * salience * cond + (1|sub_id)\n")
    f.write("Multiple comparison correction: Benjamini-Hochberg FDR (p < 0.05)\n")
    f.write("Applied separately per model term across 200 ROIs\n\n")
    f.write(f"Total ROIs tested: {len(rois)}\n\n")
    f.write("Significant ROIs per term (FDR-corrected p < 0.05):\n")
    for term, nsig in sig_terms.items():
        f.write(f"  {term}: {nsig} ROIs\n")
    if not sig_terms:
        f.write("  No terms survived FDR correction.\n")
    f.write("\nNote: run coded as continuous (1–4).\n")
    f.write("salience reference level: high; cond reference level: ext\n")

print(f"  Saved: {OUT_DIR}/summary.txt")
print("\nAnalysis 1 complete.")
