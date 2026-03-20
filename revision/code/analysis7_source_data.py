#!/usr/bin/env python
"""
Analysis 7 — Source Data Assembly
Assembles numerical source data for all figures into .xlsx files.
One file per figure, one sheet per panel.
Run with: conda run -n mei_ python analysis7_source_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Check for Excel library ────────────────────────────────────────────────────
try:
    import openpyxl
    EXCEL_ENGINE = 'openpyxl'
    print("Using openpyxl for Excel output")
except ImportError:
    try:
        import xlwt
        EXCEL_ENGINE = 'xlwt'
        print("Using xlwt for Excel output")
    except ImportError:
        try:
            import xlsxwriter
            EXCEL_ENGINE = 'xlsxwriter'
            print("Using xlsxwriter for Excel output")
        except ImportError:
            print("WARNING: No Excel library found. Falling back to CSV.")
            EXCEL_ENGINE = None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/jukebox/graziano/coolCatIsaac/mei'
BEHAV_DIR = f'{BASE}/data/behavioral'
ISC_CSV   = f'{BASE}/data/work/isc_dat/n39_sub_tracked_stacked_data_MERGED.csv'
ENG_CSV   = f'{BEHAV_DIR}/high-low-sal_data.csv'
WSFC_DIR  = f'{BASE}/data/work/wsfc'
CONN_DIR  = '/jukebox/graziano/coolCatIsaac/mei/revision/results/connectivity'
OUT_DIR   = '/jukebox/graziano/coolCatIsaac/mei/revision/data/source_data'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("Analysis 7 — Source Data Assembly")
print("=" * 70)


def save_excel(sheets_dict, filename):
    """Save dict of {sheet_name: dataframe} to Excel or CSV fallback."""
    filepath = os.path.join(OUT_DIR, filename)
    if EXCEL_ENGINE == 'openpyxl':
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet, df in sheets_dict.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"  Saved: {filepath}")
    elif EXCEL_ENGINE == 'xlsxwriter':
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            for sheet, df in sheets_dict.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"  Saved: {filepath}")
    else:
        # CSV fallback — save each sheet as separate file
        base = filename.replace('.xlsx', '')
        for sheet, df in sheets_dict.items():
            csv_path = os.path.join(OUT_DIR, f'{base}_{sheet}.csv')
            df.to_csv(csv_path, index=False)
            print(f"  Saved (CSV): {csv_path}")


# ── Figure 2: Button Presses ──────────────────────────────────────────────────
print("\n[1] Figure 2 — Button Press Source Data")

behav_files = sorted([
    f for f in os.listdir(BEHAV_DIR)
    if f.endswith('_behav.npy') and f.startswith('sub-')
])

HIGH_SAL = {'shrek', 'office', 'sherlock'}
LOW_SAL  = {'brushing', 'oragami', 'cake'}

records_bpress = []
for fname in behav_files:
    sub_id = fname.replace('_behav.npy', '')
    data = np.load(os.path.join(BEHAV_DIR, fname), allow_pickle=True).item()

    for cond in ['Internal', 'External']:
        if cond not in data:
            continue
        for movie, runs in data[cond].items():
            mov_key = movie.lower()
            salience = 'high' if mov_key in HIGH_SAL else ('low' if mov_key in LOW_SAL else 'unknown')
            for run_key, run_data in runs.items():
                run_num = int(run_key.split('-')[1])
                bpress = run_data.get('bpress', -1)
                count = 0
                if bpress != -1 and not (isinstance(bpress, (int, float)) and bpress < 0):
                    if hasattr(bpress, '__iter__'):
                        count = len([b for b in bpress if b >= 0])
                records_bpress.append({
                    'subject_id': sub_id, 'condition': cond,
                    'movie': mov_key, 'salience': salience,
                    'repetition': run_num, 'button_press_count': count
                })

df_bpress_src = pd.DataFrame(records_bpress)
print(f"  Records: {len(df_bpress_src)}")

# Summary version (mean per condition × salience × run for Figure 2 main)
fig2_summary = df_bpress_src.groupby(
    ['condition', 'salience', 'repetition']
)['button_press_count'].agg(['mean', 'std', 'count']).reset_index()
fig2_summary.columns = ['condition', 'salience', 'repetition', 'mean_bpress', 'sd_bpress', 'n']

save_excel({
    'raw_trial_data': df_bpress_src,
    'summary_by_cond_sal_rep': fig2_summary
}, 'fig2_button_presses.xlsx')

# ── Figure 3: ISC Values ──────────────────────────────────────────────────────
print("\n[2] Figure 3 — ISC Source Data")

try:
    isc_df = pd.read_csv(ISC_CSV)
    isc_df['cond'] = isc_df['cond'].str.lower()
    # MERGED CSV lacks 'salience' column — derive from movie name
    _H = {'shrek', 'office', 'sherlock'}
    _L = {'brushing', 'oragami', 'cake'}
    isc_df['salience'] = isc_df['mov'].str.lower().map(
        lambda m: 'high' if m in _H else ('low' if m in _L else 'unknown')
    )

    # Summary ISC by subject × condition × salience × run (averaged across ROIs)
    isc_avg_sub = isc_df.groupby(
        ['sub_id', 'cond', 'salience', 'run']
    )['isc_val'].mean().reset_index()
    isc_avg_sub.columns = ['subject_id', 'condition', 'salience', 'repetition', 'mean_isc']

    # Summary: mean/SD across subjects
    isc_summary = isc_avg_sub.groupby(
        ['condition', 'salience', 'repetition']
    )['mean_isc'].agg(['mean', 'std', 'count']).reset_index()
    isc_summary.columns = ['condition', 'salience', 'repetition', 'mean_isc', 'sd_isc', 'n']

    # Per-ROI values (for supplementary)
    isc_roi = isc_df[['sub_id', 'cond', 'salience', 'mov', 'Roi', 'run', 'isc_val']].copy()
    isc_roi.columns = ['subject_id', 'condition', 'salience', 'movie', 'roi', 'repetition', 'isc_value']

    save_excel({
        'subject_mean_isc': isc_avg_sub,
        'summary_by_cond_sal_rep': isc_summary,
        'per_roi_values': isc_roi
    }, 'fig3_isc_values.xlsx')

except Exception as e:
    print(f"  ISC data error: {e}")

# ── Figure 6: Connectivity ────────────────────────────────────────────────────
print("\n[3] Figure 6 — Connectivity Source Data")

try:
    # Load pre-existing rep1→rep4 connectivity for both conditions
    for cond_key, fname in [('internal', '6vid_wsfc_run4-1_int_34-Net_fish.npy'),
                             ('external', '6vid_wsfc_run4-1_ext_34-Net_fish.npy')]:
        fp = os.path.join(WSFC_DIR, fname)
        if not os.path.exists(fp):
            print(f"  Missing: {fp}")
            continue

        dat = np.load(fp, allow_pickle=True).item()
        actual = dat.get('actual_cors_across_movs', [])
        if isinstance(actual, list) and len(actual) > 0:
            # Stack movies and average
            ac_stacked = np.stack(actual)  # (n_movies, n_connections)
            ac_mean = np.mean(ac_stacked, axis=0)

            conn_df = pd.DataFrame({
                'connection_idx': range(len(ac_mean)),
                'mean_connectivity_z': ac_mean,
                'condition': cond_key
            })
            conn_df.to_csv(
                os.path.join(OUT_DIR, f'fig6_connectivity_{cond_key}_rep1to4.csv'),
                index=False
            )
            print(f"  Saved: fig6_connectivity_{cond_key}_rep1to4.csv")

    # Also check for per-transition saved files
    transition_sheets = {}
    for key in ['1to2', '2to3', '3to4']:
        fp = os.path.join(CONN_DIR, f'transition_{key}_wsfc.npy')
        if os.path.exists(fp):
            res = np.load(fp, allow_pickle=True).item()
            mat = res['actual_mat']
            sig = res['sig_mat']
            n_nets = mat.shape[0]
            rows = []
            for i in range(n_nets):
                for j in range(n_nets):
                    rows.append({
                        'network_i': i, 'network_j': j,
                        'connectivity_z': mat[i, j],
                        'significant_fdr': int(sig[i, j])
                    })
            transition_sheets[f'rep{key}'] = pd.DataFrame(rows)

    if transition_sheets:
        save_excel(transition_sheets, 'fig6_connectivity_transitions.xlsx')

except Exception as e:
    print(f"  Connectivity source data error: {e}")

# ── Supplementary: Engagement ─────────────────────────────────────────────────
print("\n[4] Supplementary — Engagement Source Data")

try:
    eng_df = pd.read_csv(ENG_CSV)
    eng_df['sub_id'] = eng_df['Subject ID'].str.lower().str.strip()
    eng_df['movie'] = eng_df['movie'].str.lower().str.strip()

    # Merge with button press counts
    movie_name_map = {'oragami': 'origami'}
    df_bpress_copy = df_bpress_src.copy()
    df_bpress_copy['movie_norm'] = df_bpress_copy['movie'].replace(movie_name_map)
    eng_df['movie_norm'] = eng_df['movie']

    df_total = df_bpress_copy.groupby(
        ['subject_id', 'condition', 'movie', 'movie_norm', 'salience']
    )['button_press_count'].sum().reset_index()
    df_total.rename(columns={'button_press_count': 'total_bpress'}, inplace=True)

    merged_eng = df_total.merge(
        eng_df[['sub_id', 'movie_norm', 'engagement', 'salience']].rename(
            columns={'sub_id': 'subject_id', 'salience': 'salience_label'}
        ),
        on=['subject_id', 'movie_norm'], how='left'
    )

    save_excel({
        'engagement_ratings_raw': eng_df[['sub_id', 'movie', 'engagement', 'salience']],
        'bpress_vs_engagement': merged_eng
    }, 'figS_engagement.xlsx')

except Exception as e:
    print(f"  Engagement source data error: {e}")

print("\nAnalysis 7 complete.")
