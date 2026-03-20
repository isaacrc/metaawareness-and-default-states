#!/usr/bin/env python
"""
Analysis 5 — Post-Hoc Power Analysis & Effect Sizes
Computes effect sizes for all non-neural behavioral mixed-effects models and
pairwise t-tests. Runs post-hoc power analysis for each.
Run with: conda run -n mei_ python analysis5_power_effect_sizes.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.formula.api import mixedlm
from statsmodels.stats.power import TTestIndPower, TTestPower
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = '/jukebox/graziano/coolCatIsaac/mei'
BEHAV_DIR = f'{BASE}/data/behavioral'
ENG_CSV    = f'{BEHAV_DIR}/high-low-sal_data.csv'
OUT_DIR    = '/jukebox/graziano/coolCatIsaac/mei/revision/results/power_and_effect_sizes'
os.makedirs(OUT_DIR, exist_ok=True)

N = 39   # total sample size per paper
ALPHA = 0.05

print("=" * 70)
print("Analysis 5 — Post-Hoc Power Analysis & Effect Sizes")
print("=" * 70)

# ── Helper functions ──────────────────────────────────────────────────────────

def nakagawa_r2(fit):
    """Compute marginal and conditional R² (Nakagawa & Schielzeth 2013)."""
    var_fixed = np.var(fit.fittedvalues)
    var_resid = fit.scale
    try:
        re_var = float(fit.cov_re.values.flat[0])
    except Exception:
        re_var = 0.0
    var_total = var_fixed + re_var + var_resid
    if var_total == 0:
        return np.nan, np.nan
    r2_m = var_fixed / var_total
    r2_c = (var_fixed + re_var) / var_total
    return r2_m, r2_c

def cohens_d_paired(a, b):
    """Cohen's d for paired comparison."""
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / np.std(diff, ddof=1)

def cohens_d_ind(a, b):
    """Cohen's d for independent samples."""
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan

def lrt_partial_eta2(full_fit, reduced_fit):
    """Partial η² via likelihood ratio test (LRT)."""
    try:
        ll_full = full_fit.llf
        ll_red = reduced_fit.llf
        lr_stat = 2 * (ll_full - ll_red)
        return lr_stat / (lr_stat + full_fit.nobs) if lr_stat >= 0 else 0.0
    except Exception:
        return np.nan

def compute_power_d(d, n, alpha=0.05):
    """Post-hoc power for paired t-test given Cohen's d."""
    from scipy.stats import t, nct
    df = n - 1
    nc = d * np.sqrt(n)  # non-centrality parameter
    t_crit = t.ppf(1 - alpha/2, df)  # two-tailed
    power = 1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc)
    return float(power)

def compute_power_f2(f2, n_groups, n_obs, alpha=0.05):
    """Post-hoc power for F-test given Cohen's f² and sample size."""
    from scipy.stats import f as fdist
    df1 = n_groups - 1
    df2 = n_obs - n_groups
    if df1 <= 0 or df2 <= 0:
        return np.nan
    lambda_nc = f2 * n_obs
    f_crit = fdist.ppf(1 - alpha, df1, df2)
    from scipy.stats import ncf
    power = 1 - ncf.cdf(f_crit, df1, df2, nc=lambda_nc)
    return float(power)

# ── Load behavioral data ──────────────────────────────────────────────────────
print("\n[1] Loading behavioral data...")

# Load engagement ratings
eng_df = pd.read_csv(ENG_CSV)
eng_df['sub_id'] = eng_df['Subject ID'].str.lower().str.strip()
eng_df['movie'] = eng_df['movie'].str.lower().str.strip()
print(f"  Engagement ratings: {eng_df.shape}")
print(f"  Columns: {eng_df.columns.tolist()}")

# Load button press counts from behavioral npy files
behav_files = sorted([
    f for f in os.listdir(BEHAV_DIR)
    if f.endswith('_behav.npy') and f.startswith('sub-')
])
print(f"  Behavioral files: {len(behav_files)}")

records = []
for fname in behav_files:
    sub_id = fname.replace('_behav.npy', '')
    data = np.load(os.path.join(BEHAV_DIR, fname), allow_pickle=True).item()

    # High/low salience mapping
    HIGH_SAL = {'shrek', 'office', 'sherlock'}
    LOW_SAL  = {'brushing', 'oragami', 'cake'}

    for cond in ['Internal', 'External']:
        if cond not in data:
            continue
        for movie, runs in data[cond].items():
            mov_key = movie.lower()
            salience = 'high' if mov_key in HIGH_SAL else ('low' if mov_key in LOW_SAL else 'unknown')

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
                    'movie': mov_key, 'salience': salience,
                    'run': run_num, 'bpress_count': count
                })

df = pd.DataFrame(records)
print(f"  Button press records: {len(df)}")
print(f"  Unique subjects: {df['sub_id'].nunique()}")
print(f"  Conditions: {df['condition'].unique()}")
print(f"  Salience: {df['salience'].unique()}")

# Aggregate total bpress per subject × condition × movie (across runs)
df_total = df.groupby(['sub_id', 'condition', 'movie', 'salience'])['bpress_count'].sum().reset_index()
df_total.rename(columns={'bpress_count': 'total_bpress'}, inplace=True)
print(f"\n  Aggregated: {df_total.shape} (subject × movie × condition total presses)")

# Merge with engagement ratings
movie_name_map = {'oragami': 'origami'}
df_total['movie_norm'] = df_total['movie'].replace(movie_name_map)
eng_df['movie_norm'] = eng_df['movie']
merged = df_total.merge(
    eng_df[['sub_id', 'movie_norm', 'engagement', 'salience']],
    on=['sub_id', 'movie_norm'], suffixes=('', '_eng'), how='left'
)
print(f"  Merged with engagement: {merged.shape}")

# ── Analysis A: Main behavioral model ────────────────────────────────────────
print("\n[2] Main behavioral LME: bpress ~ condition × salience × run + (1|subject)")

# Use per-run data for this model
df_model = df.copy()
df_model['run_cont'] = df_model['run'].astype(float)
df_model['cond_int'] = (df_model['condition'] == 'Internal').astype(int)
df_model['sal_low'] = (df_model['salience'] == 'low').astype(int)

effect_rows = []

try:
    full_model = mixedlm(
        'bpress_count ~ run_cont * cond_int * sal_low',
        data=df_model,
        groups=df_model['sub_id']
    )
    full_fit = full_model.fit(reml=False, method='lbfgs', maxiter=1000)
    r2_m, r2_c = nakagawa_r2(full_fit)
    print(f"  Full model: R²m = {r2_m:.3f}, R²c = {r2_c:.3f}, converged = {full_fit.converged}")

    # LRT partial η² for each term
    terms = ['run_cont', 'cond_int', 'sal_low', 'run_cont:cond_int',
             'run_cont:sal_low', 'cond_int:sal_low', 'run_cont:cond_int:sal_low']

    for term in terms:
        # Reduced model: remove this term
        other_terms = [t for t in terms if t != term and ':' not in t or
                       all(p in term for p in t.split(':'))]
        # Simplified: just drop from formula
        formula_main = ['run_cont', 'cond_int', 'sal_low',
                        'run_cont:cond_int', 'run_cont:sal_low',
                        'cond_int:sal_low', 'run_cont:cond_int:sal_low']
        reduced_terms = [t for t in formula_main if t != term]
        if not reduced_terms:
            continue
        reduced_formula = 'bpress_count ~ ' + ' + '.join(reduced_terms)
        try:
            red_fit = mixedlm(reduced_formula, data=df_model,
                              groups=df_model['sub_id']).fit(
                                  reml=False, method='lbfgs', maxiter=1000)
            eta2 = lrt_partial_eta2(full_fit, red_fit)
        except Exception:
            eta2 = np.nan

        coef = full_fit.params.get(term, np.nan)
        t_val = full_fit.tvalues.get(term, np.nan)
        p_val = full_fit.pvalues.get(term, np.nan)
        f2 = eta2 / (1 - eta2) if (eta2 is not None and not np.isnan(eta2) and eta2 < 1) else np.nan
        power = compute_power_f2(f2, 2, N) if not np.isnan(f2) else np.nan

        effect_rows.append({
            'analysis': 'Main behavioral LME',
            'test_type': 'LME term',
            'statistic': f't={t_val:.3f}, p={p_val:.4f}',
            'effect_size_type': 'partial_eta2',
            'effect_size_value': eta2,
            'cohens_f2': f2,
            'N': N,
            'power': power,
            'term': term,
            'coef': coef,
            'p': p_val
        })
        print(f"    {term}: η² = {eta2:.3f}, f² = {f2:.3f} → power = {power:.3f}"
              if not np.isnan(eta2) else f"    {term}: η² = NaN")

    # Also add model-level R²
    effect_rows.append({
        'analysis': 'Main behavioral LME',
        'test_type': 'LME model',
        'statistic': 'Nakagawa R²',
        'effect_size_type': 'R2_marginal',
        'effect_size_value': r2_m,
        'cohens_f2': np.nan,
        'N': N, 'power': np.nan, 'term': 'model', 'coef': np.nan, 'p': np.nan
    })
    effect_rows.append({
        'analysis': 'Main behavioral LME',
        'test_type': 'LME model',
        'statistic': 'Nakagawa R²',
        'effect_size_type': 'R2_conditional',
        'effect_size_value': r2_c,
        'cohens_f2': np.nan,
        'N': N, 'power': np.nan, 'term': 'model', 'coef': np.nan, 'p': np.nan
    })

except Exception as e:
    print(f"  Main LME error: {e}")

# ── Analysis B: Pairwise t-tests ──────────────────────────────────────────────
print("\n[3] Pairwise t-tests with Cohen's d...")

def paired_t_test(group_a, group_b, label_a, label_b, analysis_name):
    """Align by subject, run paired t-test, compute Cohen's d."""
    subs_a = set(group_a['sub_id'])
    subs_b = set(group_b['sub_id'])
    common = sorted(subs_a & subs_b)
    if len(common) < 5:
        print(f"  {analysis_name}: insufficient paired subs ({len(common)})")
        return None

    a_vals = group_a.set_index('sub_id').reindex(common)['total_bpress'].values
    b_vals = group_b.set_index('sub_id').reindex(common)['total_bpress'].values

    # Drop NaN pairs
    valid = ~(np.isnan(a_vals) | np.isnan(b_vals))
    a_vals, b_vals = a_vals[valid], b_vals[valid]
    n_pairs = len(a_vals)

    if n_pairs < 5:
        print(f"  {analysis_name}: insufficient valid pairs ({n_pairs})")
        return None

    t_stat, p_val = ttest_rel(a_vals, b_vals)
    d = cohens_d_paired(a_vals, b_vals)
    power = compute_power_d(abs(d), n_pairs)

    print(f"  {analysis_name}: t({n_pairs-1}) = {t_stat:.3f}, p = {p_val:.4f}, d = {d:.3f}, power = {power:.3f}")
    return {
        'analysis': analysis_name,
        'test_type': 'paired t-test',
        'statistic': f't({n_pairs-1})={t_stat:.3f}, p={p_val:.4f}',
        'effect_size_type': 'cohens_d',
        'effect_size_value': abs(d),
        'cohens_f2': np.nan,
        'N': n_pairs,
        'power': power,
        'term': f'{label_a} vs {label_b}',
        'coef': t_stat,
        'p': p_val
    }

# Per-subject averages across runs
df_sub_cond = df_total.groupby(['sub_id', 'condition'])['total_bpress'].mean().reset_index()

# B1: Internal vs External (overall)
int_df = df_sub_cond[df_sub_cond['condition'] == 'Internal']
ext_df = df_sub_cond[df_sub_cond['condition'] == 'External']
row = paired_t_test(ext_df, int_df, 'External', 'Internal', 'External vs Internal bpress')
if row: effect_rows.append(row)

# B2: High vs Low salience within External
ext_total = df_total[df_total['condition'] == 'External']
ext_high = ext_total[ext_total['salience'] == 'high'].groupby('sub_id')['total_bpress'].mean().reset_index()
ext_low  = ext_total[ext_total['salience'] == 'low'].groupby('sub_id')['total_bpress'].mean().reset_index()
row = paired_t_test(ext_high, ext_low, 'HighSal', 'LowSal', 'External: High vs Low salience bpress')
if row: effect_rows.append(row)

# B3: High vs Low salience within Internal
int_total = df_total[df_total['condition'] == 'Internal']
int_high = int_total[int_total['salience'] == 'high'].groupby('sub_id')['total_bpress'].mean().reset_index()
int_low  = int_total[int_total['salience'] == 'low'].groupby('sub_id')['total_bpress'].mean().reset_index()
row = paired_t_test(int_high, int_low, 'HighSal', 'LowSal', 'Internal: High vs Low salience bpress')
if row: effect_rows.append(row)

# B4: Run 1 vs Run 4 (repetition effect) within each condition
for cond in ['External', 'Internal']:
    r1 = df[(df['condition'] == cond) & (df['run'] == 1)].groupby('sub_id')['bpress_count'].mean().reset_index()
    r4 = df[(df['condition'] == cond) & (df['run'] == 4)].groupby('sub_id')['bpress_count'].mean().reset_index()
    # rename bpress_count to total_bpress so paired_t_test can find it
    r1 = r1.rename(columns={'bpress_count': 'total_bpress'})
    r4 = r4.rename(columns={'bpress_count': 'total_bpress'})
    row = paired_t_test(r1, r4, 'Run1', 'Run4', f'{cond}: Run1 vs Run4 bpress')
    if row: effect_rows.append(row)

# ── Analysis C: Engagement models ─────────────────────────────────────────────
print("\n[4] Engagement regression models...")

movie_name_map = {'oragami': 'origami'}
df_total2 = df_total.copy()
df_total2['movie_norm'] = df_total2['movie'].replace(movie_name_map)
eng_df2 = eng_df.copy()
eng_df2['movie_norm'] = eng_df2['movie']
merged2 = df_total2.merge(
    eng_df2[['sub_id', 'movie_norm', 'engagement']],
    on=['sub_id', 'movie_norm'], how='left'
)

for cond in ['External', 'Internal']:
    sub_df = merged2[merged2['condition'] == cond].dropna(subset=['engagement', 'total_bpress'])
    if len(sub_df) < 10:
        print(f"  {cond} engagement model: insufficient data")
        continue
    try:
        full_fit = mixedlm('total_bpress ~ engagement', data=sub_df,
                           groups=sub_df['sub_id']).fit(
                               reml=False, method='lbfgs', maxiter=500)
        r2_m, r2_c = nakagawa_r2(full_fit)

        # LRT for engagement term
        red_fit = mixedlm('total_bpress ~ 1', data=sub_df,
                          groups=sub_df['sub_id']).fit(
                              reml=False, method='lbfgs', maxiter=500)
        eta2 = lrt_partial_eta2(full_fit, red_fit)
        f2 = eta2 / (1 - eta2) if not np.isnan(eta2) and eta2 < 1 else np.nan
        power = compute_power_f2(f2, 2, len(sub_df)) if not np.isnan(f2) else np.nan

        coef = full_fit.params.get('engagement', np.nan)
        t_val = full_fit.tvalues.get('engagement', np.nan)
        p_val = full_fit.pvalues.get('engagement', np.nan)

        print(f"  {cond}: β={coef:.3f}, t={t_val:.3f}, p={p_val:.4f}, η²={eta2:.3f}, power={power:.3f}")

        effect_rows.append({
            'analysis': f'Engagement ~ bpress ({cond})',
            'test_type': 'LME term',
            'statistic': f't={t_val:.3f}, p={p_val:.4f}',
            'effect_size_type': 'partial_eta2',
            'effect_size_value': eta2,
            'cohens_f2': f2,
            'N': sub_df['sub_id'].nunique(),
            'power': power,
            'term': 'engagement',
            'coef': coef,
            'p': p_val
        })
        effect_rows.append({
            'analysis': f'Engagement ~ bpress ({cond})',
            'test_type': 'LME model',
            'statistic': 'R2_marginal',
            'effect_size_type': 'R2_marginal',
            'effect_size_value': r2_m,
            'cohens_f2': np.nan,
            'N': sub_df['sub_id'].nunique(),
            'power': np.nan,
            'term': 'model', 'coef': np.nan, 'p': np.nan
        })

    except Exception as e:
        print(f"  {cond} engagement model error: {e}")

# ── Analysis D: ISC unthresholded effect sizes ────────────────────────────────
print("\n[5] ISC unthresholded effect sizes (Cohen's d)...")

ISC_CSV = f'{BASE}/data/work/isc_dat/n39_sub_tracked_stacked_data_MERGED.csv'
try:
    isc_df = pd.read_csv(ISC_CSV)
    isc_df['cond'] = isc_df['cond'].str.lower().str.strip()
    isc_df['cond'] = isc_df['cond'].replace({'external': 'ext', 'internal': 'int'})
    HIGH_SAL = {'shrek', 'office', 'sherlock'}
    LOW_SAL  = {'brushing', 'oragami', 'cake'}
    isc_df['salience'] = isc_df['mov'].str.lower().map(
        lambda m: 'high' if m in HIGH_SAL else ('low' if m in LOW_SAL else 'unknown')
    )
    isc_df = isc_df.dropna(subset=['isc_val'])

    # Average ISC across ROIs and movies per subject × condition × run × salience
    isc_sub = isc_df.groupby(['sub_id', 'cond', 'run', 'salience'])['isc_val'].mean().reset_index()

    def isc_paired_d(group_a, group_b, label, analysis_name):
        common = sorted(set(group_a['sub_id']) & set(group_b['sub_id']))
        a = group_a.set_index('sub_id').reindex(common)['isc_val'].values
        b = group_b.set_index('sub_id').reindex(common)['isc_val'].values
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        d = cohens_d_paired(a, b)
        power = compute_power_d(abs(d), len(a))
        t_stat, p_val = stats.ttest_rel(a, b)
        print(f"  {analysis_name}: d = {abs(d):.3f}, t({len(a)-1}) = {t_stat:.3f}, p = {p_val:.4f}, power = {power:.3f}")
        return {
            'analysis': analysis_name,
            'test_type': 'paired t-test',
            'statistic': f't({len(a)-1})={t_stat:.3f}, p={p_val:.4f}',
            'effect_size_type': 'cohens_d',
            'effect_size_value': abs(d),
            'cohens_f2': np.nan,
            'N': len(a),
            'power': power,
            'term': label,
            'coef': t_stat,
            'p': p_val
        }

    # D1: External vs Internal (averaged across all runs and salience)
    ext_isc = isc_sub[isc_sub['cond'] == 'ext'].groupby('sub_id')['isc_val'].mean().reset_index()
    int_isc = isc_sub[isc_sub['cond'] == 'int'].groupby('sub_id')['isc_val'].mean().reset_index()
    row = isc_paired_d(ext_isc, int_isc, 'External vs Internal', 'ISC: External vs Internal')
    if row: effect_rows.append(row)

    # D2: Run 1 vs Run 4 per condition
    for cond, label in [('ext', 'External'), ('int', 'Internal')]:
        r1 = isc_sub[(isc_sub['cond'] == cond) & (isc_sub['run'] == 1)].groupby('sub_id')['isc_val'].mean().reset_index()
        r4 = isc_sub[(isc_sub['cond'] == cond) & (isc_sub['run'] == 4)].groupby('sub_id')['isc_val'].mean().reset_index()
        row = isc_paired_d(r1, r4, 'Run1 vs Run4', f'ISC {label}: Run 1 vs Run 4')
        if row: effect_rows.append(row)

    # D3: High vs Low salience (averaged across conditions and runs)
    hi_isc = isc_sub[isc_sub['salience'] == 'high'].groupby('sub_id')['isc_val'].mean().reset_index()
    lo_isc = isc_sub[isc_sub['salience'] == 'low'].groupby('sub_id')['isc_val'].mean().reset_index()
    row = isc_paired_d(hi_isc, lo_isc, 'High vs Low salience', 'ISC: High vs Low Salience')
    if row: effect_rows.append(row)

    # D4: ISC LME partial η² for run, condition, run×condition
    print("\n  ISC LME partial η² (LRT)...")
    isc_model_df = isc_df.dropna(subset=['sub_id']).copy()
    isc_model_df['sub_id'] = isc_model_df['sub_id'].astype(str)
    isc_model_df['run_cont'] = isc_model_df['run'].astype(float)
    isc_model_df['cond_int'] = (isc_model_df['cond'] == 'int').astype(float)

    isc_reduced_formulas = {
        'run_cont':         'isc_val ~ cond_int',
        'cond_int':         'isc_val ~ run_cont',
        'run_cont:cond_int':'isc_val ~ run_cont + cond_int',
    }

    try:
        isc_full_fit = mixedlm('isc_val ~ run_cont * cond_int', data=isc_model_df,
                               groups=isc_model_df['sub_id']).fit(
                                   reml=False, method='lbfgs', maxiter=500)
        print(f"  ISC full model converged: {isc_full_fit.converged}")

        def safe_r2m(fit, y):
            """Marginal R²: variance of fixed-effect predictions / total variance."""
            try:
                r2m, r2c = nakagawa_r2(fit)
                if not np.isnan(r2m):
                    return r2m, r2c
            except Exception:
                pass
            # Fallback: R² = corr(fixed predictions, observed)^2
            try:
                pred = fit.predict()
                r2m = float(np.corrcoef(pred, y)[0, 1] ** 2)
                return r2m, np.nan
            except Exception:
                return np.nan, np.nan

        y_obs = isc_model_df['isc_val'].values
        r2m_full, r2c_full = safe_r2m(isc_full_fit, y_obs)
        print(f"  ISC full model: R²m = {r2m_full:.3f}, R²c = {r2c_full}")

        for term, red_formula in isc_reduced_formulas.items():
            try:
                red_fit = mixedlm(red_formula, data=isc_model_df,
                                  groups=isc_model_df['sub_id']).fit(
                                      reml=False, method='lbfgs', maxiter=500)
                r2m_red, _ = safe_r2m(red_fit, y_obs)
                # Unique marginal R² contribution of this term
                delta_r2m = r2m_full - r2m_red
                delta_r2m = max(delta_r2m, 0.0)
                t_val = isc_full_fit.tvalues.get(term, np.nan)
                p_val = isc_full_fit.pvalues.get(term, np.nan)
                print(f"    {term}: ΔR²m = {delta_r2m:.3f}, t = {t_val:.3f}")
                effect_rows.append({
                    'analysis': 'ISC LME',
                    'test_type': 'LME term',
                    'statistic': f't={t_val:.3f}, p={p_val:.4f}' if not np.isnan(t_val) else 'N/A',
                    'effect_size_type': 'delta_R2_marginal',
                    'effect_size_value': delta_r2m,
                    'cohens_f2': np.nan,
                    'N': N,
                    'power': np.nan,
                    'term': term,
                    'coef': t_val,
                    'p': p_val
                })
            except Exception as e:
                print(f"    {term}: error — {e}")

        # Also save full model R²
        for r2_type, r2_val in [('R2_marginal', r2m_full), ('R2_conditional', r2c_full)]:
            effect_rows.append({
                'analysis': 'ISC LME', 'test_type': 'LME model',
                'statistic': 'Nakagawa R²', 'effect_size_type': r2_type,
                'effect_size_value': r2_val, 'cohens_f2': np.nan,
                'N': N, 'power': np.nan, 'term': 'model', 'coef': np.nan, 'p': np.nan
            })
    except Exception as e:
        import traceback
        print(f"  ISC LME error: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"  ISC effect size error: {e}")

# ── Compile and save ──────────────────────────────────────────────────────────
print("\n[6] Saving results...")

effect_df = pd.DataFrame(effect_rows)
# Select meaningful columns for the output CSV
out_cols = ['analysis', 'test_type', 'term', 'statistic',
            'effect_size_type', 'effect_size_value', 'cohens_f2', 'N', 'power', 'p']
effect_df_out = effect_df[[c for c in out_cols if c in effect_df.columns]]
effect_df_out.to_csv(f'{OUT_DIR}/behavioral_effect_sizes.csv', index=False)
print(f"  Saved: {OUT_DIR}/behavioral_effect_sizes.csv")

# ── APA summary text ──────────────────────────────────────────────────────────
underpowered = effect_df[effect_df['power'].notna() & (effect_df['power'] < 0.80)]
well_powered = effect_df[effect_df['power'].notna() & (effect_df['power'] >= 0.80)]

with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write("Analysis 5 — Post-Hoc Power Analysis & Effect Sizes\n")
    f.write("=" * 60 + "\n\n")
    f.write("Sample size: N = 39, α = 0.05 (two-tailed)\n")
    f.write("ISC isc_val are Fisher z-transformed correlations.\n\n")

    f.write("Mixed-Effects Models (Nakagawa & Schielzeth R²):\n")
    lme_rows = effect_df[(effect_df['test_type'].isin(['LME term', 'LME model']))]
    for _, row in lme_rows.iterrows():
        es_val = row['effect_size_value']
        pwr = row['power']
        es_str = f"{es_val:.3f}" if not np.isnan(float(es_val)) else "N/A"
        pwr_str = f"{pwr:.3f}" if not np.isnan(float(pwr)) else "N/A"
        f.write(f"  [{row['analysis']} — {row['term']}]\n")
        f.write(f"    {row['effect_size_type']} = {es_str}")
        if pwr_str != "N/A":
            f.write(f", post-hoc power = {pwr_str}")
        f.write(f"\n")

    f.write("\nPairwise t-Tests (Cohen's d) — Behavioral:\n")
    t_rows = effect_df[(effect_df['test_type'] == 'paired t-test') & (~effect_df['analysis'].str.startswith('ISC'))]
    for _, row in t_rows.iterrows():
        f.write(f"  {row['analysis']}: d = {row['effect_size_value']:.3f}, post-hoc power = {row['power']:.3f}\n")

    f.write("\nPairwise t-Tests (Cohen's d) — ISC Unthresholded:\n")
    isc_t_rows = effect_df[(effect_df['test_type'] == 'paired t-test') & (effect_df['analysis'].str.startswith('ISC'))]
    for _, row in isc_t_rows.iterrows():
        f.write(f"  {row['analysis']}: d = {row['effect_size_value']:.3f}, {row['statistic']}, post-hoc power = {row['power']:.3f}\n")

    f.write("\nISC LME — unique marginal R² contribution per term (ΔR²m = R²m_full − R²m_reduced):\n")
    isc_lme_rows = effect_df[(effect_df['analysis'] == 'ISC LME') & (effect_df['test_type'] == 'LME term')]
    for _, row in isc_lme_rows.iterrows():
        f.write(f"  {row['term']}: ΔR²m = {row['effect_size_value']:.3f}, {row['statistic']}\n")
    isc_model_rows = effect_df[(effect_df['analysis'] == 'ISC LME') & (effect_df['test_type'] == 'LME model')]
    for _, row in isc_model_rows.iterrows():
        f.write(f"  model {row['effect_size_type']} = {row['effect_size_value']:.3f}\n")

    f.write("\nPower Summary:\n")
    f.write(f"  Well-powered analyses (power ≥ 0.80): {len(well_powered)}\n")
    f.write(f"  Potentially underpowered (power < 0.80): {len(underpowered)}\n")

    if len(underpowered) > 0:
        f.write("\nUnderpowered analyses:\n")
        for _, row in underpowered.iterrows():
            f.write(f"  [{row['analysis']} — {row['term']}]: power = {row['power']:.3f}\n")
        f.write(
            "\nWe acknowledge that smaller effects, such as those in "
            + ", ".join(underpowered['analysis'].unique().tolist())
            + ", may be underpowered. These results should be interpreted "
            "with caution, and future work with larger samples is warranted.\n"
        )

print(f"  Saved: {OUT_DIR}/summary.txt")
print("\nAnalysis 5 complete.")
