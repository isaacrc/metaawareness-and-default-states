#!/usr/bin/env python
"""
Analysis 6 — Updated Figures with Individual Data Points
Regenerates Figure 2 (button presses) and relevant ISC figures
with violin/boxplot + overlaid individual points.
New files only — not replacements.
Run with: conda run -n mei_ python analysis6_updated_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/jukebox/graziano/coolCatIsaac/mei'
BEHAV_DIR = f'{BASE}/data/behavioral'
ISC_CSV   = f'{BASE}/data/work/isc_dat/n39_sub_tracked_stacked_data_MERGED.csv'
PLOT_DIR  = '/jukebox/graziano/coolCatIsaac/mei/revision/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set(style='white', context='talk', font_scale=1.0, rc={"lines.linewidth": 2})

print("=" * 70)
print("Analysis 6 — Updated Figures with Individual Data Points")
print("=" * 70)

# ── Load behavioral data ──────────────────────────────────────────────────────
print("\n[1] Loading behavioral data...")

behav_files = sorted([
    f for f in os.listdir(BEHAV_DIR)
    if f.endswith('_behav.npy') and f.startswith('sub-')
])
print(f"  Found {len(behav_files)} behavioral files")

HIGH_SAL = {'shrek', 'office', 'sherlock'}
LOW_SAL  = {'brushing', 'oragami', 'cake'}
MOVIES = ['shrek', 'sherlock', 'office', 'brushing', 'oragami', 'cake']
DISPLAY_NAMES = {
    'shrek': 'Shrek', 'office': 'The Office', 'sherlock': 'Sherlock',
    'brushing': 'Brushing', 'oragami': 'Origami', 'cake': 'Cake'
}
NUM_RUNS = 4

records = []
for fname in behav_files:
    sub_id = fname.replace('_behav.npy', '')
    data = np.load(os.path.join(BEHAV_DIR, fname), allow_pickle=True).item()

    for cond in ['Internal', 'External']:
        if cond not in data:
            continue
        for movie, runs in data[cond].items():
            mov_key = movie.lower()
            if mov_key not in set(MOVIES):
                continue
            for run_key, run_data in runs.items():
                run_num = int(run_key.split('-')[1])
                bpress = run_data.get('bpress', -1)
                if bpress == -1 or (isinstance(bpress, (int, float)) and bpress < 0):
                    count = 0
                elif hasattr(bpress, '__iter__'):
                    count = len([b for b in bpress if b >= 0])
                else:
                    count = 0
                records.append({
                    'sub_id': sub_id, 'condition': cond,
                    'movie': mov_key, 'run': run_num,
                    'bpress': count,
                    'salience': 'high' if mov_key in HIGH_SAL else 'low'
                })

df_bpress = pd.DataFrame(records)
print(f"  Total records: {len(df_bpress)}")

# ── Figure 2 Remake: Violin + strip per movie ─────────────────────────────────
print("\n[2] Generating Figure 2 violin remake (per movie)...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
run_nums = np.array([1, 2, 3, 4])
run_labels = ['Rep 1', 'Rep 2', 'Rep 3', 'Rep 4']

ext_color = 'steelblue'
int_color = 'sandybrown'

legend_elements = [
    Line2D([0], [0], color=ext_color, linewidth=2.5, label='External'),
    Line2D([0], [0], color=int_color, linewidth=2.5, label='Internal'),
]

for idx, movie in enumerate(MOVIES):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    movie_df = df_bpress[df_bpress['movie'] == movie]

    # Build long dataframe with run and condition columns
    plot_df = movie_df[['sub_id', 'condition', 'run', 'bpress']].copy()
    plot_df['run_label'] = plot_df['run'].map({1:'Rep 1',2:'Rep 2',3:'Rep 3',4:'Rep 4'})

    ext_df = plot_df[plot_df['condition'] == 'External']
    int_df = plot_df[plot_df['condition'] == 'Internal']

    # Violin plots per run
    positions_ext = run_nums - 0.18
    positions_int = run_nums + 0.18
    width = 0.32

    for r, pos_e, pos_i in zip(run_nums, positions_ext, positions_int):
        ext_vals = ext_df[ext_df['run'] == r]['bpress'].values
        int_vals = int_df[int_df['run'] == r]['bpress'].values

        for vals, pos, color in [(ext_vals, pos_e, ext_color), (int_vals, pos_i, int_color)]:
            if len(vals) > 2:
                parts = ax.violinplot(vals, positions=[pos], widths=width,
                                      showmedians=True, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.5)
                parts['cmedians'].set_color(color)
                parts['cmedians'].set_linewidth(2)

            # Strip plot (individual points)
            jitter = np.random.uniform(-0.08, 0.08, len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.6,
                       s=20, zorder=4, edgecolors='none')

    # Mean line connecting runs
    for data_df, color in [(ext_df, ext_color), (int_df, int_color)]:
        run_means = data_df.groupby('run')['bpress'].mean()
        offset = -0.18 if color == ext_color else 0.18
        ax.plot(run_nums + offset, run_means.reindex(run_nums).values,
                color=color, linewidth=1.5, marker='o', markersize=4,
                zorder=5, alpha=0.8)

    ax.set_title(DISPLAY_NAMES[movie], fontsize=12, fontweight='bold')
    ax.set_xticks(run_nums)
    ax.set_xticklabels(run_labels, fontsize=9)
    ax.set_xlabel('Repetition', fontsize=9)
    if col == 0:
        ax.set_ylabel('Button Presses', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if idx == 2:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Row label
    if col == 0:
        sal_label = 'High Salience' if row == 0 else 'Low Salience'
        ax.annotate(sal_label, xy=(-0.25, 0.5), xycoords='axes fraction',
                    fontsize=11, fontweight='bold', rotation=90, va='center')

plt.suptitle('Button Presses by Movie, Condition, and Repetition\n'
             '(Violin + individual data points)',
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(f'{PLOT_DIR}/fig2_violin_bpress.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/fig2_violin_bpress.png")

# ── Figure 2 Remake: High/Low salience panels, Internal vs External lines ──────
# Matches style of bpress_analysis-6-22-23.ipynb exactly:
# Left panel = High Salience, Right panel = Low Salience
# Each panel: External (steelblue) and Internal (sandybrown) lines across repetitions
print("\n[3] Generating Figure 2 style — High vs Low salience, Internal vs External...")

HIGH_SAL = {'shrek', 'office', 'sherlock'}
LOW_SAL  = {'brushing', 'oragami', 'cake'}

# Build result dict: result[salience][run_num]['External'] = [bpress per sub]
result = {}
for sal_key, sal_movies in [('high', HIGH_SAL), ('low', LOW_SAL)]:
    run_dic = {}
    for run_num in run_nums:
        ext_vals, int_vals = [], []
        for sub_id in df_bpress['sub_id'].unique():
            sub_df = df_bpress[(df_bpress['sub_id'] == sub_id) &
                               (df_bpress['movie'].isin(sal_movies)) &
                               (df_bpress['run'] == run_num)]
            for cond_label, target_list in [('External', ext_vals), ('Internal', int_vals)]:
                cond_sub = sub_df[sub_df['condition'] == cond_label]
                if len(cond_sub) > 0:
                    target_list.append(cond_sub['bpress'].sum())
        run_dic[run_num] = {'External': ext_vals, 'Internal': int_vals}
    result[sal_key] = run_dic

legend_elements_sal = [
    Line2D([0], [0], marker='o', color='sandybrown', label='Internal',
           linewidth=7.5, markersize=15),
    Line2D([0], [0], marker='o', color='steelblue', label='External',
           linewidth=7.5, markersize=15),
]

sns.set(style='white', context='talk', font_scale=1.0, rc={"lines.linewidth": 2})
fig, axs = plt.subplots(1, 2, figsize=(11, 7), sharey=True, dpi=300)

vln_offset = 0.18
vln_width  = 0.30
pt_size    = 30

for ind, sal_key in enumerate(['high', 'low']):
    ax = axs[ind]

    ext_means, int_means = [], []

    for r in run_nums:
        ext_vals = np.array(result[sal_key][r]['External'], dtype=float)
        int_vals = np.array(result[sal_key][r]['Internal'], dtype=float)
        pos_e = r - vln_offset
        pos_i = r + vln_offset

        for vals, pos, color in [(ext_vals, pos_e, 'steelblue'),
                                  (int_vals, pos_i, 'sandybrown')]:
            if len(vals) > 2:
                parts = ax.violinplot(vals, positions=[pos], widths=vln_width,
                                      showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
            jitter = np.random.uniform(-0.07, 0.07, len(vals))
            ax.scatter(pos + jitter, vals, color=color, alpha=0.7,
                       s=pt_size, zorder=4, edgecolors='none')

        ext_means.append(np.mean(ext_vals) if len(ext_vals) > 0 else np.nan)
        int_means.append(np.mean(int_vals) if len(int_vals) > 0 else np.nan)

    # Mean lines
    ax.plot(run_nums - vln_offset, ext_means, color='steelblue',
            linewidth=3.5, marker='o', markersize=9, zorder=5)
    ax.plot(run_nums + vln_offset, int_means, color='sandybrown',
            linewidth=3.5, marker='o', markersize=9, zorder=5)

    if ind == 1:
        ax.legend(handles=legend_elements_sal, loc='upper right', fontsize=25)

    ax.set_title(sal_key.capitalize() + ' Salience', fontsize=35)
    ax.set_ylabel('# Button Presses', fontsize=35)
    ax.set_xticks(run_nums)
    ax.set_xticklabels(run_nums, fontsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlabel('Repetition', fontsize=35)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
sns.despine()
fig.savefig(f'{PLOT_DIR}/fig2_violin_sal_cond.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOT_DIR}/fig2_violin_sal_cond.png")

# ── ISC Figure Remake ─────────────────────────────────────────────────────────
print("\n[4] Loading ISC data for violin remake...")

try:
    isc_df = pd.read_csv(ISC_CSV)
    isc_df['cond'] = isc_df['cond'].str.lower()
    # MERGED CSV lacks salience — derive from movie name
    _HIGH = {'shrek', 'office', 'sherlock'}
    _LOW  = {'brushing', 'oragami', 'cake'}
    isc_df['salience'] = isc_df['mov'].str.lower().map(
        lambda m: 'high' if m in _HIGH else ('low' if m in _LOW else 'unknown')
    )
    print(f"  ISC shape: {isc_df.shape}")
    print(f"  Columns: {isc_df.columns.tolist()}")

    # Average ISC across ROIs per subject per run per condition per salience
    isc_avg = isc_df.groupby(['sub_id', 'cond', 'salience', 'run'])['isc_val'].mean().reset_index()
    print(f"  Averaged ISC rows: {len(isc_avg)}")

    # ── ISC by salience × run — matching bpress figure style ──
    # Left panel = High Salience, Right panel = Low Salience
    # Each panel: External (steelblue) and Internal (sandybrown) ISC lines
    print("\n[5] Generating ISC figure — High/Low salience panels, Int vs Ext lines...")

    # Build result_isc dict: result_isc[salience][run]['External'|'Internal'] = [ISC per sub]
    isc_run_nums = np.array([1, 2, 3, 4])
    _HIGH_M = {'shrek', 'office', 'sherlock'}
    _LOW_M  = {'brushing', 'oragami', 'cake'}

    result_isc = {}
    for sal_key, sal_movies in [('high', _HIGH_M), ('low', _LOW_M)]:
        run_dic_isc = {}
        for run_num in isc_run_nums:
            ext_isc_vals, int_isc_vals = [], []
            cond_map = {'external': ext_isc_vals, 'internal': int_isc_vals,
                        'ext': ext_isc_vals, 'int': int_isc_vals}
            for sub_id in isc_avg['sub_id'].unique():
                sub_df = isc_avg[(isc_avg['sub_id'] == sub_id) &
                                 (isc_avg['run'] == run_num)]
                for cond_label in ['external', 'ext']:
                    cond_sub = sub_df[(sub_df['cond'] == cond_label) &
                                      (sub_df['salience'] == sal_key)]
                    if len(cond_sub) > 0:
                        ext_isc_vals.append(cond_sub['isc_val'].mean())
                        break
                for cond_label in ['internal', 'int']:
                    cond_sub = sub_df[(sub_df['cond'] == cond_label) &
                                      (sub_df['salience'] == sal_key)]
                    if len(cond_sub) > 0:
                        int_isc_vals.append(cond_sub['isc_val'].mean())
                        break
            run_dic_isc[run_num] = {'External': ext_isc_vals, 'Internal': int_isc_vals}
        result_isc[sal_key] = run_dic_isc

    legend_elements_isc = [
        mpatches.Patch(facecolor='lightsalmon', edgecolor='black', linewidth=1.2, label='Internal'),
        mpatches.Patch(facecolor='steelblue',   edgecolor='black', linewidth=1.2, label='External'),
    ]

    sns.set(style='white', context='talk', font_scale=1.0, rc={"lines.linewidth": 2})
    s1, s2 = 25, 35
    fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True, dpi=300)

    bar_width  = 0.375
    bar_offset = bar_width / 2   # bars touch with no gap
    pt_size    = 30
    np.random.seed(42)

    for ind, sal_key in enumerate(['high', 'low']):
        ax = axs[ind]

        for r in isc_run_nums:
            ext_vals = np.array(result_isc[sal_key][r]['External'], dtype=float)
            int_vals = np.array(result_isc[sal_key][r]['Internal'], dtype=float)
            pos_e = r - bar_offset
            pos_i = r + bar_offset

            for vals, pos, color in [(ext_vals, pos_e, 'steelblue'),
                                      (int_vals, pos_i, 'lightsalmon')]:
                mean = np.nanmean(vals)
                sem  = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))
                ax.bar(pos, mean, width=bar_width, color=color,
                       zorder=2, edgecolor='black', linewidth=1.2)
                ax.errorbar(pos, mean, yerr=sem, fmt='none', color='black',
                            capsize=4, linewidth=1.5, zorder=3)
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals, color=color, alpha=0.6,
                           s=pt_size, zorder=4, edgecolors='black', linewidths=0.3)

        if ind == 0:
            ax.set_ylabel('Mean ISC (Fisher z)', fontsize=s2)
        if ind == 1:
            ax.legend(handles=legend_elements_isc, loc='upper right', fontsize=s1)

        ax.set_title(sal_key.capitalize() + ' Salience', fontsize=s2)
        ax.set_xticks(isc_run_nums)
        ax.set_xticklabels(isc_run_nums, fontsize=s1)
        ax.tick_params(axis='y', labelsize=s1)
        ax.set_xlabel('Repetition', fontsize=s2)
        ax.set_xlim(0.5, 4.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine()
    fig.savefig(f'{PLOT_DIR}/fig3_isc_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/fig3_isc_bar.png")

    # ── Version without y-label on second panel ───────────────────────────────
    s1, s2 = 25, 35
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 7), sharey=True, dpi=300)
    np.random.seed(42)

    for ind, sal_key in enumerate(['high', 'low']):
        ax = axs2[ind]
        for r in isc_run_nums:
            ext_vals = np.array(result_isc[sal_key][r]['External'], dtype=float)
            int_vals = np.array(result_isc[sal_key][r]['Internal'], dtype=float)
            pos_e = r - bar_offset
            pos_i = r + bar_offset
            for vals, pos, color in [(ext_vals, pos_e, 'steelblue'),
                                      (int_vals, pos_i, 'lightsalmon')]:
                mean = np.nanmean(vals)
                sem  = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))
                ax.bar(pos, mean, width=bar_width, color=color,
                       zorder=2, edgecolor='black', linewidth=1.2)
                ax.errorbar(pos, mean, yerr=sem, fmt='none', color='black',
                            capsize=4, linewidth=1.5, zorder=3)
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals, color=color, alpha=0.6,
                           s=pt_size, zorder=4, edgecolors='black', linewidths=0.3)

        ax.set_ylabel('Mean ISC (Fisher z)' if ind == 0 else '', fontsize=s2)
        if ind == 1:
            ax.legend(handles=legend_elements_isc, loc='upper right', fontsize=s1)
        ax.set_title(sal_key.capitalize() + ' Salience', fontsize=s2)
        ax.set_xticks(isc_run_nums)
        ax.set_xticklabels(isc_run_nums, fontsize=s1)
        ax.tick_params(axis='y', labelsize=s1)
        ax.set_xlabel('Repetition', fontsize=s2)
        ax.set_xlim(0.5, 4.5)

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    sns.despine()
    fig2.savefig(f'{PLOT_DIR}/fig3_isc_bar_nolabel2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/fig3_isc_bar_nolabel2.png")

except Exception as e:
    print(f"  ISC violin error: {e}")

# ── Significant Voxel Count Bar Plot ──────────────────────────────────────────
print("\n[6] Generating Significant Voxel Count bar plot...")

import nibabel as nib

SIGVOX_PATH = f'/jukebox/graziano/coolCatIsaac/mei/data/work/isc_dat/n39_isc_sigvox_nii_dic.npy'
try:
    nt_vox = np.load(SIGVOX_PATH, allow_pickle=True).item()

    high_movies = ['shrek', 'office', 'sherlock']
    low_movies  = ['brushing', 'oragami', 'cake']
    run_list    = [1, 2, 3, 4]
    cond_list   = ['external', 'internal']
    thresh      = 0

    # Build sal dict: sal[salience][cond] = list of [run1, run2, run3, run4] per movie
    high_dict = {'internal': [], 'external': []}
    low_dict  = {'internal': [], 'external': []}

    for mov in high_movies + low_movies:
        for cond in cond_list:
            run_count = []
            for run in run_list:
                x_3d = nt_vox[cond][mov][run].get_fdata()
                run_count.append(x_3d[x_3d > thresh].shape[0])
            if mov in high_movies:
                high_dict[cond].append(run_count)
            else:
                low_dict[cond].append(run_count)

    sal_sigvox = {'high': high_dict, 'low': low_dict}

    # ── Plot ──────────────────────────────────────────────────────────────────
    s1, s2 = 25, 35
    x_labels = [1, 2, 3, 4]
    n = len(x_labels)
    r = np.arange(n)
    width = 0.375

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7),
                             sharex=True, sharey=True, dpi=300)

    legend_handles = []
    legend_labels  = []

    for idx, sal_cond in enumerate(['high', 'low']):
        ax = axes[idx]
        for cond in cond_list:
            color    = 'lightsalmon' if cond == 'internal' else 'steelblue'
            run_count = np.mean(np.array(sal_sigvox[sal_cond][cond]), axis=0)

            if cond == 'external':
                bars = ax.bar(r, run_count, color=color, width=width,
                              edgecolor='black', label='External')
            else:
                bars = ax.bar(r + width, run_count, color=color, width=width,
                              edgecolor='black', label='Internal')

            if idx == 0:
                legend_handles.append(bars)
                legend_labels.append('External' if cond == 'external' else 'Internal')

        ax.set_title(sal_cond.capitalize() + ' Salience', fontsize=s2)
        ax.set_ylim(0, 60000)
        ax.set_ylabel('Sig. Voxel Count' if idx == 0 else '', fontsize=s2)
        ax.set_xlabel('Repetition', fontsize=s2)

        bar_positions = [x + width / 2 for x in r]
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(x_labels, fontsize=s1)

        formatter = plt.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='y', labelsize=s1)

    fig.legend(legend_handles, legend_labels, loc='upper center',
               fontsize=s1, bbox_to_anchor=(0.84, 0.9))

    plt.tight_layout()
    sns.despine()
    fig.savefig(f'{PLOT_DIR}/fig_sigvox_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/fig_sigvox_bar.png")

except Exception as e:
    print(f"  Sig voxel error: {e}")

print("\nAnalysis 6 complete.")
