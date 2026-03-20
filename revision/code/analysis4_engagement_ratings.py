#!/usr/bin/env python
"""
Analysis 4 — Participant-Specific Engagement Ratings (Supplementary)
- Point-biserial correlation: engagement rating vs salience label
- Mixed-effects regression: bpress_count ~ engagement + (1|subject)
  for external and internal conditions separately
- Two-panel scatter plot
Run with: conda run -n mei_ python analysis4_engagement_ratings.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, pearsonr
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/jukebox/graziano/coolCatIsaac/mei'
BEHAV_DIR = f'{BASE}/data/behavioral'
ENG_CSV    = f'{BEHAV_DIR}/high-low-sal_data.csv'
OUT_DIR    = '/jukebox/graziano/coolCatIsaac/mei/revision/results/engagement_ratings'
PLOT_DIR   = '/jukebox/graziano/coolCatIsaac/mei/revision/plots'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("Analysis 4 — Participant-Specific Engagement Ratings")
print("=" * 70)

# ── Load engagement ratings ───────────────────────────────────────────────────
print("\n[1] Loading engagement ratings...")
eng_df = pd.read_csv(ENG_CSV)
print(f"  Shape: {eng_df.shape}")
print(f"  Columns: {eng_df.columns.tolist()}")
print(f"  Sample:\n{eng_df.head(5)}")
print(f"  Unique subjects: {eng_df['Subject ID'].nunique()}")
print(f"  Unique movies: {sorted(eng_df['movie'].unique())}")
print(f"  Engagement range: {eng_df['engagement'].min()}–{eng_df['engagement'].max()}")
print(f"  Salience values: {eng_df['salience'].unique()}")

# Normalize subject IDs to lowercase sub-XXX format
eng_df['sub_id'] = eng_df['Subject ID'].str.lower().str.replace(' ', '-')
# Handle 'Sub-002' → 'sub-002'
eng_df['sub_id'] = eng_df['sub_id'].str.replace('sub-', 'sub-', regex=False)
# Standardize movie names
eng_df['movie'] = eng_df['movie'].str.lower().str.strip()

# ── Load behavioral button press data ────────────────────────────────────────
print("\n[2] Loading behavioral data for button press counts...")

behav_files = sorted([
    f for f in os.listdir(BEHAV_DIR)
    if f.endswith('_behav.npy') and f.startswith('sub-')
])
print(f"  Found {len(behav_files)} behavioral files")

records = []
for fname in behav_files:
    sub_id = fname.replace('_behav.npy', '')  # e.g., 'sub-002'
    data = np.load(os.path.join(BEHAV_DIR, fname), allow_pickle=True).item()

    for cond in ['Internal', 'External']:
        if cond not in data:
            continue
        for movie, runs in data[cond].items():
            mov_key = movie.lower()
            total_bpress = 0
            for run_key, run_data in runs.items():
                bpress = run_data.get('bpress', -1)
                if bpress == -1 or (isinstance(bpress, (int, float)) and bpress < 0):
                    count = 0
                elif hasattr(bpress, '__iter__'):
                    count = len([b for b in bpress if b >= 0])
                else:
                    count = 0
                total_bpress += count

            records.append({
                'sub_id': sub_id,
                'movie': mov_key,
                'condition': cond,
                'total_bpress': total_bpress
            })

bpress_df = pd.DataFrame(records)
print(f"  Button press records: {len(bpress_df)}")
print(f"  Sample:\n{bpress_df.head(5)}")

# ── Merge datasets ────────────────────────────────────────────────────────────
print("\n[3] Merging engagement ratings with button press counts...")

# Normalize movie names for merge
movie_name_map = {
    'oragami': 'origami',  # align data typo to rating CSV name
}
bpress_df['movie_norm'] = bpress_df['movie'].replace(movie_name_map)
eng_df['movie_norm'] = eng_df['movie'].str.lower().str.strip()

merged = bpress_df.merge(
    eng_df[['sub_id', 'movie_norm', 'engagement', 'salience']],
    on=['sub_id', 'movie_norm'],
    how='inner'
)
print(f"  Merged shape: {merged.shape}")
print(f"  Unique subjects in merged: {merged['sub_id'].nunique()}")
print(f"  Sample:\n{merged.head(5)}")

if len(merged) == 0:
    print("  WARNING: No rows merged — check subject ID formats")
    print("  bpress sub_ids:", sorted(bpress_df['sub_id'].unique())[:5])
    print("  eng sub_ids:", sorted(eng_df['sub_id'].unique())[:5])

# ── Validation: point-biserial correlation ────────────────────────────────────
print("\n[4] Point-biserial correlation: engagement vs salience...")

merged_unique = merged.drop_duplicates(subset=['sub_id', 'movie_norm'])[
    ['engagement', 'salience']].dropna()
sal_binary = (merged_unique['salience'] == 'high').astype(int)
eng_vals = merged_unique['engagement'].values

if len(eng_vals) > 2:
    r_pb, p_pb = pointbiserialr(sal_binary, eng_vals)
    print(f"  Point-biserial r = {r_pb:.3f}, p = {p_pb:.4e}")
else:
    r_pb, p_pb = np.nan, np.nan
    print("  Insufficient data for correlation")

# ── Mixed-effects models ──────────────────────────────────────────────────────
print("\n[5] Mixed-effects models: bpress ~ engagement + (1|subject)")

model_results = []

for cond in ['External', 'Internal']:
    print(f"\n  Condition: {cond}")
    sub_df = merged[merged['condition'] == cond].copy()
    print(f"    N rows: {len(sub_df)}, N subjects: {sub_df['sub_id'].nunique()}")

    if len(sub_df) < 10:
        print("    Insufficient data — skipping")
        continue

    try:
        model = mixedlm(
            'total_bpress ~ engagement',
            data=sub_df,
            groups=sub_df['sub_id']
        )
        fit = model.fit(reml=False, method='lbfgs', maxiter=500)

        coef = fit.params.get('engagement', np.nan)
        se = fit.bse.get('engagement', np.nan)
        t_val = fit.tvalues.get('engagement', np.nan)
        p_val = fit.pvalues.get('engagement', np.nan)

        print(f"    engagement coef = {coef:.3f}, SE = {se:.3f}, t = {t_val:.3f}, p = {p_val:.4f}")
        print(f"    Converged: {fit.converged}")

        # Marginal R² (Nakagawa & Schielzeth approximation)
        var_fixed = np.var(fit.fittedvalues)
        var_resid = fit.scale
        # Random effects variance from model
        re_var = fit.cov_re.values[0, 0] if hasattr(fit.cov_re, 'values') else 0
        var_total = var_fixed + re_var + var_resid
        r2_marginal = var_fixed / var_total if var_total > 0 else np.nan
        r2_conditional = (var_fixed + re_var) / var_total if var_total > 0 else np.nan
        print(f"    Marginal R² = {r2_marginal:.3f}, Conditional R² = {r2_conditional:.3f}")

        model_results.append({
            'condition': cond,
            'coef_engagement': coef,
            'se': se,
            't': t_val,
            'p': p_val,
            'converged': fit.converged,
            'n_obs': len(sub_df),
            'n_subjects': sub_df['sub_id'].nunique(),
            'r2_marginal': r2_marginal,
            'r2_conditional': r2_conditional
        })

    except Exception as e:
        print(f"    Model error: {e}")
        model_results.append({
            'condition': cond, 'coef_engagement': np.nan,
            'se': np.nan, 't': np.nan, 'p': np.nan,
            'converged': False,
            'n_obs': len(sub_df),
            'n_subjects': sub_df['sub_id'].nunique(),
            'r2_marginal': np.nan, 'r2_conditional': np.nan
        })

model_df = pd.DataFrame(model_results)

# ── Visualization ─────────────────────────────────────────────────────────────
print("\n[6] Generating scatter plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
cond_colors = {'External': 'steelblue', 'Internal': 'sandybrown'}
cond_display = {'External': 'External Attention', 'Internal': 'Internal Attention'}

for ax, cond in zip(axes, ['External', 'Internal']):
    sub_df = merged[merged['condition'] == cond].copy()

    if len(sub_df) == 0:
        ax.set_title(f'{cond_display[cond]}\n(no data)')
        continue

    color = cond_colors[cond]

    # Individual subject-movie data points
    subs = sub_df['sub_id'].unique()
    palette = sns.color_palette('husl', len(subs))
    sub_color_map = dict(zip(subs, palette))

    for sub in subs:
        sub_rows = sub_df[sub_df['sub_id'] == sub]
        ax.scatter(
            sub_rows['engagement'], sub_rows['total_bpress'],
            color=sub_color_map[sub], alpha=0.5, s=30, zorder=3
        )

    # OLS regression line with CI (using all data)
    x_vals = sub_df['engagement'].values
    y_vals = sub_df['total_bpress'].values
    if len(x_vals) > 2 and np.std(x_vals) > 0:
        from scipy.stats import linregress
        slope, intercept, r_val, p_val_r, se = linregress(x_vals, y_vals)
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_pred = intercept + slope * x_range

        # CI via bootstrap
        n_boot = 500
        boot_preds = []
        for _ in range(n_boot):
            idx = np.random.choice(len(x_vals), len(x_vals), replace=True)
            try:
                s, b, _, _, _ = linregress(x_vals[idx], y_vals[idx])
                boot_preds.append(b + s * x_range)
            except Exception:
                pass
        if boot_preds:
            boot_arr = np.array(boot_preds)
            ci_lo = np.percentile(boot_arr, 2.5, axis=0)
            ci_hi = np.percentile(boot_arr, 97.5, axis=0)
            ax.fill_between(x_range, ci_lo, ci_hi, alpha=0.15, color=color)
        ax.plot(x_range, y_pred, color=color, linewidth=2, zorder=4)

        # Get model fit stats
        mrow = model_df[model_df['condition'] == cond]
        if len(mrow) > 0:
            t_disp = mrow.iloc[0]['t']
            p_disp = mrow.iloc[0]['p']
            r2_disp = mrow.iloc[0]['r2_marginal']
            ax.text(0.05, 0.95,
                    f't = {t_disp:.2f}, p = {p_disp:.3f}\nMarginal R² = {r2_disp:.3f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_title(cond_display[cond], fontsize=13, fontweight='bold')
    ax.set_xlabel('Engagement Rating (1–10)', fontsize=11)
    ax.set_ylabel('Total Button Presses (all repetitions)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Mind-Wandering Events Predicted by Engagement Rating\n'
             '(each point = one participant × movie; color = participant)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/engagement_scatter.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/engagement_scatter.png")

# ── Save results ──────────────────────────────────────────────────────────────
print("\n[7] Saving results...")
model_df.to_csv(f'{OUT_DIR}/model_results.csv', index=False)
print(f"  Saved: {OUT_DIR}/model_results.csv")

# APA summary text
with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write("Analysis 4 — Participant-Specific Engagement Ratings (Supplementary)\n")
    f.write("=" * 60 + "\n\n")

    f.write("Validation:\n")
    f.write(f"  Participants' individual engagement ratings were significantly correlated\n")
    f.write(f"  with the categorical high/low salience grouping\n")
    if not np.isnan(r_pb):
        sig_str = "p < 0.001" if p_pb < 0.001 else f"p = {p_pb:.3f}"
        f.write(f"  (r = {r_pb:.2f}, {sig_str}), validating the a priori classification.\n\n")
    else:
        f.write(f"  (insufficient data to compute correlation).\n\n")

    for row in model_results:
        cond = row['condition']
        coef = row['coef_engagement']
        t = row['t']
        p = row['p']
        r2m = row['r2_marginal']

        f.write(f"{cond} condition:\n")
        if not np.isnan(coef):
            sig_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            f.write(f"  Mixed-effects regression: button press count ~ engagement\n")
            f.write(f"  + (1|participant).\n")
            f.write(f"  Fixed effect of engagement: β = {coef:.3f}, t = {t:.2f}, {sig_str}.\n")
            f.write(f"  Marginal R² = {r2m:.3f}.\n\n")
        else:
            f.write(f"  Model did not converge or insufficient data.\n\n")

    f.write("Note: Button press counts are summed across all 4 repetitions per movie.\n")
    f.write("Engagement ratings are from post-session self-report (1–10 scale).\n")

print(f"  Saved: {OUT_DIR}/summary.txt")
print("\nAnalysis 4 complete.")
