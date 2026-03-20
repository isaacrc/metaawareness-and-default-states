#!/usr/bin/env python
"""
Analysis 3 — Temporal Distribution of Button Presses
Shows when participants mind-wandered during each movie.
Layout: 2 rows × 3 columns (top=high salience, bottom=low salience)
Each subplot: vertical lines per press, opacity ∝ local density.
Run with: conda run -n mei_ python analysis3_button_press_temporal.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/jukebox/graziano/coolCatIsaac/mei'
BEHAV_DIR = f'{BASE}/data/behavioral'
OUT_DIR = '/jukebox/graziano/coolCatIsaac/mei/revision/results/button_press_temporal'
PLOT_DIR = '/jukebox/graziano/coolCatIsaac/mei/revision/plots'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("Analysis 3 — Temporal Distribution of Button Presses")
print("=" * 70)

# ── Movie salience assignments ─────────────────────────────────────────────────
HIGH_SAL_MOVIES = ['shrek', 'office', 'sherlock']
LOW_SAL_MOVIES  = ['brushing', 'oragami', 'cake']   # note: spelled 'oragami' in data
ALL_MOVIES = HIGH_SAL_MOVIES + LOW_SAL_MOVIES

# ── Load behavioral data ───────────────────────────────────────────────────────
print("\n[1] Loading behavioral data from all subjects...")

behav_files = sorted([
    f for f in os.listdir(BEHAV_DIR)
    if f.endswith('_behav.npy') and f.startswith('sub-')
])
print(f"  Found {len(behav_files)} behavioral files")

# Collect all timestamps per movie (across all subjects, conditions, repetitions)
press_by_movie = {m: [] for m in ALL_MOVIES}
press_details = []  # for saving CSV

total_subs = 0
for fname in behav_files:
    sub_id = fname.replace('_behav.npy', '')
    data = np.load(os.path.join(BEHAV_DIR, fname), allow_pickle=True).item()
    total_subs += 1

    for cond in ['Internal', 'External']:
        if cond not in data:
            continue
        for movie, runs in data[cond].items():
            # normalize movie name
            mov_key = movie.lower()
            if mov_key not in press_by_movie:
                continue

            for run_key, run_data in runs.items():
                bpress = run_data.get('bpress', -1)
                run_num = int(run_key.split('-')[1])

                if bpress == -1 or (isinstance(bpress, (int, float)) and bpress < 0):
                    continue

                if not hasattr(bpress, '__iter__'):
                    bpress = [bpress]

                for ts in bpress:
                    if ts >= 0:
                        press_by_movie[mov_key].append(float(ts))
                        press_details.append({
                            'sub_id': sub_id, 'movie': mov_key,
                            'condition': cond, 'run': run_num,
                            'timestamp_s': float(ts)
                        })

print(f"\n  Subjects processed: {total_subs}")
print("\n  Button presses per movie:")
for m in ALL_MOVIES:
    presses = press_by_movie[m]
    dur = max(presses) if presses else 0
    print(f"    {m:12s}: {len(presses):4d} presses  (max time: {dur:.1f}s)")

# Save CSV
details_df = pd.DataFrame(press_details)
csv_path = f'{OUT_DIR}/press_timestamps_by_movie.csv'
details_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

# ── Determine movie durations ─────────────────────────────────────────────────
# Infer from max observed timestamp, rounded up to nearest 10s
movie_durations = {}
for m in ALL_MOVIES:
    presses = press_by_movie[m]
    if presses:
        movie_durations[m] = np.ceil(max(presses) / 10) * 10
    else:
        movie_durations[m] = 120.0  # fallback ~2 min
    print(f"  {m}: inferred duration = {movie_durations[m]:.0f}s")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n[2] Generating temporal distribution figure...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
movie_grid = [HIGH_SAL_MOVIES, LOW_SAL_MOVIES]
row_labels = ['High Salience', 'Low Salience']
# title mapping for display (normalize oragami → origami for display)
display_names = {
    'shrek': 'Shrek', 'office': 'The Office', 'sherlock': 'Sherlock',
    'brushing': 'Brushing', 'oragami': 'Origami', 'cake': 'Cake'
}

for row_idx, (row_movies, row_label) in enumerate(zip(movie_grid, row_labels)):
    for col_idx, movie in enumerate(row_movies):
        ax = axes[row_idx, col_idx]
        presses = np.array(press_by_movie[movie])
        dur = movie_durations[movie]

        if len(presses) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_title(display_names[movie], fontsize=12, fontweight='bold')
            continue

        # Histogram with bin size = 1.5 seconds
        bin_size = 1.5
        bins = np.arange(0, dur + bin_size, bin_size)
        bar_color = '#c0392b' if row_idx == 0 else '#2c5f8a'
        ax.hist(presses, bins=bins, color=bar_color, alpha=0.75, edgecolor='white',
                linewidth=0.4)
        ax.set_xlim(0, dur)

        # Press count annotation
        ax.text(0.02, 0.96, f'n={len(presses)} presses',
                transform=ax.transAxes, fontsize=8, va='top',
                color='gray')

        ax.set_title(display_names[movie], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Count', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Row label on left
    axes[row_idx, 0].annotate(
        row_label, xy=(-0.18, 0.5), xycoords='axes fraction',
        fontsize=12, fontweight='bold', rotation=90, va='center',
        color='#c0392b' if row_idx == 0 else '#2c5f8a'
    )

plt.suptitle(
    'Temporal Distribution of Mind-Wandering Events\n'
    '(All participants, all repetitions and conditions collapsed)',
    fontsize=12, y=1.01
)
plt.tight_layout()
fig.savefig(
    f'{PLOT_DIR}/button_press_temporal_distribution.png',
    dpi=200, bbox_inches='tight'
)
plt.close()
print(f"  Saved: {PLOT_DIR}/button_press_temporal_distribution.png")

print("\nAnalysis 3 complete.")
