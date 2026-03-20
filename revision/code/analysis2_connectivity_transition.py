#!/usr/bin/env python
"""
Analysis 2 — By-Repetition Connectivity Change
Computes WSFC at 34-network level for consecutive run transitions:
  rep1→rep2, rep2→rep3, rep3→rep4
Each transition: internal−external difference matrix, averaged across movies,
significance tested via 10,000 sign-flip permutations.
Produces a single figure with 3 side-by-side 34×34 matrices.

Usage:
    python analysis2_connectivity_transition.py --transition 1  # rep1→rep2
    python analysis2_connectivity_transition.py --transition 2  # rep2→rep3
    python analysis2_connectivity_transition.py --transition 3  # rep3→rep4
    python analysis2_connectivity_transition.py --transition all  # all (slow)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--transition', type=str, default='all',
                    help='Which transition to compute: 1, 2, 3, or all')
parser.add_argument('--n_perms', type=int, default=10000,
                    help='Number of permutations (default 10000)')
args = parser.parse_args()

TRANSITION = args.transition
N_PERMS = args.n_perms
print(f"Running transition={TRANSITION}, n_perms={N_PERMS}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/jukebox/graziano/coolCatIsaac/mei'
DATA_DIR  = f'{BASE}/data/work'
ISC_DIR   = f'{DATA_DIR}/isc_dat'
SCAF_DIR  = f'{DATA_DIR}/schaefer_atlas'
GLM_DIR   = f'{DATA_DIR}/glm_data'
MASK_DIR  = '/jukebox/graziano/coolCatIsaac/MEI/data/work/masks'  # uppercase MEI path
CODE_DIR  = f'{BASE}/code/analysis'

OUT_DIR   = '/jukebox/graziano/coolCatIsaac/mei/revision/results/connectivity'
PLOT_DIR  = '/jukebox/graziano/coolCatIsaac/mei/revision/plots'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Add the analysis code directory to path for utils_anal
sys.path.insert(0, CODE_DIR)

# ── Import dependencies ───────────────────────────────────────────────────────
import nibabel as nib
import nilearn as nil
import brainiak.utils.fmrisim as sim
from nilearn.image import resample_to_img
from nilearn import datasets
from nilearn.masking import unmask
from brainiak.isc import squareform_isfc
from statsmodels.stats.multitest import multipletests

try:
    from utils_anal import get_network_labels
    print("Loaded get_network_labels from utils_anal")
except ImportError:
    print("WARNING: utils_anal not found, defining get_network_labels locally")
    def get_network_labels(num_parc, num_net):
        label_fn = f'{SCAF_DIR}/Schaefer2018_{num_parc}Parcels_{num_net}Networks_order.txt'
        with open(label_fn) as f:
            networks = [' '.join((label.split('_')[1][0], label.split('_')[2]))
                        for label in f.readlines()]
        idxs = np.unique(networks, return_index=True)[1]
        network_labels = [networks[idx] for idx in sorted(idxs)]
        network_idxs = [int(np.median([i for i, n in enumerate(networks) if n == network]))
                        for network in network_labels]
        return networks, network_labels, network_idxs

# ── Atlas setup ───────────────────────────────────────────────────────────────
NUM_PARC = 200
NUM_NET  = 17

print(f"\n[1] Loading Schaefer atlas ({NUM_PARC} parcels, {NUM_NET} networks)...")
dataset = datasets.fetch_atlas_schaefer_2018(n_rois=NUM_PARC, yeo_networks=NUM_NET)
atlas_filename = dataset.maps
labels = dataset.labels
networks, network_labels, network_idxs = get_network_labels(NUM_PARC, NUM_NET)
N_NETS = len(network_labels)
print(f"  Number of networks: {N_NETS}")

# Resample atlas to match fMRI data
print("  Resampling atlas to fMRI space...")

# Load a reference EPI for resampling
dimensions = np.asarray([78, 93, 65])
mask_brain, _ = sim.mask_brain(dimensions, mask_self=False)

# Load a reference nifti from GLM data: structure is {cond → movie → {'f': [nifti, ...]}}
ref_img = None
try:
    glm_files = sorted([f for f in os.listdir(GLM_DIR) if f.endswith('_bpress_fmri_data.npy')])
    if glm_files:
        s_dat = np.load(os.path.join(GLM_DIR, glm_files[0]), allow_pickle=True).item()
        for cond_val in s_dat.values():
            if not isinstance(cond_val, dict):
                continue
            for mov_val in cond_val.values():
                if isinstance(mov_val, dict) and 'f' in mov_val:
                    f_list = mov_val['f']
                    if f_list and hasattr(f_list[0], 'affine'):
                        ref_img = f_list[0]  # 4D nifti — use as reference for affine
                        print(f"  Reference nifti loaded: shape={ref_img.shape}, affine={ref_img.affine[0]}")
                        break
            if ref_img is not None:
                break
except Exception as e:
    print(f"  Could not load GLM reference: {e}")

if ref_img is None:
    print("  WARNING: No reference nifti found. Using synthetic mask.")
    # Create synthetic affine based on typical fMRI
    affine = np.eye(4) * 2.0
    affine[3, 3] = 1.0
    ref_img = nib.Nifti1Image(mask_brain.astype(np.float32), affine)

# Load brain mask — 112179 non-zero voxels matching the ISC npz data
BRAIN_MASK_PATH = f'{MASK_DIR}/whole_b_bnk.nii.gz'
brain_mask_nii = nib.load(BRAIN_MASK_PATH)
brain_mask_data = brain_mask_nii.get_fdata()
print(f"  Brain mask: {BRAIN_MASK_PATH}")
print(f"  Brain mask shape: {brain_mask_data.shape}, n_voxels={int(np.sum(brain_mask_data > 0))}")

# Resample atlas to same space as brain mask
atlas_nii_raw = nib.load(atlas_filename)
atlas_nii = resample_to_img(atlas_nii_raw, brain_mask_nii, interpolation='nearest')
atlas_img = atlas_nii.get_fdata()
print(f"  Atlas shape: {atlas_img.shape}")
unique_parcels = np.unique(atlas_img)
print(f"  Unique parcel labels (sample): {unique_parcels[:5]}...{unique_parcels[-3:]}")

# ── Helper functions ──────────────────────────────────────────────────────────

def convert_2d_to_parcels(subj_2d, brain_mask_nii, atlas_img, network_labels):
    """
    Convert 2D (TR × masked_voxels) subject data to (TR × networks).
    Matches the wsfc_3.ipynb convert_2d_to_parcels pipeline exactly:
      1. unmask: (TR, 112179) → (78, 93, 65, TR) using brain_mask_nii
      2. Average voxels within each atlas parcel → (n_parcels, TR)
      3. Average parcels within each network    → (n_networks, TR)
    Returns: (TR, n_networks)
    """
    # Step 1: unmask back to 3D+time image
    img_4d = unmask(subj_2d, brain_mask_nii)   # Nifti shape (78,93,65,TR)
    func_data = img_4d.get_fdata()              # (78, 93, 65, TR)

    # Step 2: average by parcel
    parcel_ids = np.unique(atlas_img)[1:]       # skip background label 0
    func_by_parcel = np.array([
        np.mean(func_data[atlas_img == pid, :], axis=0)
        for pid in parcel_ids
    ])  # (n_parcels, TR)

    # Step 3: average parcels into networks (preserve label order)
    unique_nets = list(dict.fromkeys(network_labels))
    net_data = []
    for net in unique_nets:
        net_idxs = [i for i, n in enumerate(network_labels) if n == net]
        valid = [i for i in net_idxs if i < func_by_parcel.shape[0]]
        if valid:
            net_data.append(np.mean(func_by_parcel[valid, :], axis=0))

    return np.column_stack(net_data)   # (TR, n_networks)


def compute_wsfc_group(data_3d, brain_mask_nii, atlas_img, network_labels):
    """
    Compute within-subject FC for each subject and Fisher-z transform.
    data_3d: (TR, voxels, subjects)
    Returns: (n_networks, n_networks, subjects)
    """
    n_subs = data_3d.shape[2]
    wsfc_all = []

    for s in range(n_subs):
        subj_2d = data_3d[..., s]   # (TR, voxels)
        net_ts = convert_2d_to_parcels(subj_2d, brain_mask_nii, atlas_img, network_labels)
        corr = np.corrcoef(net_ts.T)
        corr = np.clip(corr, -0.9999, 0.9999)
        wsfc_all.append(np.expand_dims(corr, axis=2))

    wsfc_stack = np.concatenate(wsfc_all, axis=2)
    return np.arctanh(wsfc_stack)   # Fisher z-transform


def av_wsfc_signed(mat3d, perm=False):
    """Average WSFC matrix across subjects, optionally with sign-flipping."""
    if perm:
        n_subs = mat3d.shape[2]
        sign_flips = np.random.choice([-1, 1], size=(n_subs, 1, 1), replace=True)
        mat3d = sign_flips * np.rollaxis(mat3d, axis=2)
        return np.nanmean(mat3d, axis=0)   # subjects at axis 0 after rollaxis
    return np.nanmean(mat3d, axis=2)       # subjects at axis 2 in original shape


# ── Load ISC data ─────────────────────────────────────────────────────────────
print("\n[2] Loading ISC voxel data...")
ext_isc = np.load(f'{ISC_DIR}/n39_ext_isc.npz', allow_pickle=True)
int_isc = np.load(f'{ISC_DIR}/n39_int_isc.npz', allow_pickle=True)

movie_list = list(ext_isc.files)
print(f"  External movies: {movie_list}")
print(f"  Internal movies: {list(int_isc.files)}")
print(f"  Sample shape (ext {movie_list[0]}): {ext_isc[movie_list[0]].shape}")
# Shape: (TRs, voxels, 4 runs, ~19 subs)

# Define which transitions to compute
if TRANSITION == 'all':
    transitions = [(1, 2), (2, 3), (3, 4)]
elif TRANSITION == '1':
    transitions = [(1, 2)]
elif TRANSITION == '2':
    transitions = [(2, 3)]
elif TRANSITION == '3':
    transitions = [(3, 4)]
else:
    raise ValueError(f"Unknown transition: {TRANSITION}. Use 1, 2, 3, or all.")

print(f"\n  Computing transitions: {transitions}")

# ── Main computation loop ──────────────────────────────────────────────────────
transition_results = {}

if N_PERMS == 0:
    print("\n  n_perms=0: skipping computation, loading saved results from disk.")
    transitions = []  # skip the loop entirely

for (run_a, run_b) in transitions:
    key = f'{run_a}to{run_b}'
    print(f"\n[3] Transition rep{run_a}→rep{run_b}")

    actual_cors_movs = []
    perms_movs = []
    int_cors_movs = []
    int_perms_movs = []
    ext_cors_movs = []
    ext_perms_movs = []

    for movie in movie_list:
        print(f"  Movie: {movie}")

        ext_data = ext_isc[movie]  # (TRs, voxels, 4, subs)
        int_data = int_isc[movie]

        # Extract the two run slices
        ext_run_a = ext_data[..., run_a - 1, :]  # (TRs, voxels, subs)
        ext_run_b = ext_data[..., run_b - 1, :]
        int_run_a = int_data[..., run_a - 1, :]
        int_run_b = int_data[..., run_b - 1, :]

        print(f"    ext run{run_a} shape: {ext_run_a.shape}")

        # Compute WSFC for each run × condition
        print(f"    Computing WSFC for ext run{run_a}...")
        m_ext_a = compute_wsfc_group(ext_run_a, brain_mask_nii, atlas_img, networks)
        print(f"    Computing WSFC for ext run{run_b}...")
        m_ext_b = compute_wsfc_group(ext_run_b, brain_mask_nii, atlas_img, networks)
        print(f"    Computing WSFC for int run{run_a}...")
        m_int_a = compute_wsfc_group(int_run_a, brain_mask_nii, atlas_img, networks)
        print(f"    Computing WSFC for int run{run_b}...")
        m_int_b = compute_wsfc_group(int_run_b, brain_mask_nii, atlas_img, networks)

        # ── Composite: Δ = (int_b − ext_b) − (int_a − ext_a) ────────────────
        diff_b = av_wsfc_signed(m_int_b) - av_wsfc_signed(m_ext_b)
        diff_a = av_wsfc_signed(m_int_a) - av_wsfc_signed(m_ext_a)
        actual_diff = diff_b - diff_a
        wsfc_c, iscs = squareform_isfc(actual_diff)
        actual_cors_movs.append(np.hstack((wsfc_c, iscs)))

        # ── Internal-only: int_b − int_a ──────────────────────────────────────
        int_diff = av_wsfc_signed(m_int_b) - av_wsfc_signed(m_int_a)
        ic, ii = squareform_isfc(int_diff)
        int_cors_movs.append(np.hstack((ic, ii)))

        # ── External-only: ext_b − ext_a ──────────────────────────────────────
        ext_diff = av_wsfc_signed(m_ext_b) - av_wsfc_signed(m_ext_a)
        ec, ei = squareform_isfc(ext_diff)
        ext_cors_movs.append(np.hstack((ec, ei)))

        # Permutation testing
        print(f"    Running {N_PERMS} permutations...")
        perm_vecs = []
        int_perm_vecs = []
        ext_perm_vecs = []
        for perm_i in range(N_PERMS):
            diff_b_perm = av_wsfc_signed(m_int_b, perm=True) - av_wsfc_signed(m_ext_b, perm=True)
            diff_a_perm = av_wsfc_signed(m_int_a, perm=True) - av_wsfc_signed(m_ext_a, perm=True)
            perm_diff = diff_b_perm - diff_a_perm
            perm_c, perm_isc = squareform_isfc(perm_diff)
            perm_vecs.append(np.hstack((perm_c, perm_isc)))

            ip = av_wsfc_signed(m_int_b, perm=True) - av_wsfc_signed(m_int_a, perm=True)
            ipc, ipd = squareform_isfc(ip)
            int_perm_vecs.append(np.hstack((ipc, ipd)))

            ep = av_wsfc_signed(m_ext_b, perm=True) - av_wsfc_signed(m_ext_a, perm=True)
            epc, epd = squareform_isfc(ep)
            ext_perm_vecs.append(np.hstack((epc, epd)))

        perms_movs.append(np.vstack(perm_vecs))
        int_perms_movs.append(np.vstack(int_perm_vecs))
        ext_perms_movs.append(np.vstack(ext_perm_vecs))

    def _build_result(cors_movs, perms_movs_list, d_shape, n_perms):
        """Average across movies, compute significance, reconstruct matrices."""
        av = np.mean(np.stack(cors_movs), axis=0)
        perm_av = np.mean(np.stack(perms_movs_list), axis=0)
        sig_vals = [(np.sum(av[i] < perm_av[:, i])) / n_perms
                    for i in range(len(av))]
        sig_binary = np.array([1 if v < 0.05 else 0 for v in sig_vals])
        try:
            _, sig_fdr, _, _ = multipletests(sig_vals, method='fdr_bh')
            sig_binary_fdr = (sig_fdr < 0.05).astype(int)
        except Exception:
            sig_fdr = np.ones(len(sig_vals))
            sig_binary_fdr = sig_binary.copy()
        actual_mat = squareform_isfc(av[:-d_shape], av[-d_shape:])
        sig_mat = squareform_isfc(sig_binary_fdr[:-d_shape], sig_binary_fdr[-d_shape:])
        return {
            'actual_mat': actual_mat,
            'sig_mat': sig_mat,
            'sig_vals': np.array(sig_vals),
            'sig_fdr': sig_fdr,
            'n_sig_uncorrected': int(np.sum(sig_binary)),
            'n_sig_fdr': int(np.sum(sig_binary_fdr)),
            'd_shape': d_shape
        }

    d_shape = N_NETS
    comp = _build_result(actual_cors_movs, perms_movs, d_shape, N_PERMS)
    int_res = _build_result(int_cors_movs, int_perms_movs, d_shape, N_PERMS)
    ext_res = _build_result(ext_cors_movs, ext_perms_movs, d_shape, N_PERMS)

    transition_results[key] = comp
    transition_results[f'{key}_int'] = int_res
    transition_results[f'{key}_ext'] = ext_res

    print(f"\n  Transition {key} (composite): "
          f"{comp['n_sig_uncorrected']} sig (uncorrected), "
          f"{comp['n_sig_fdr']} sig (FDR)")
    print(f"  Transition {key} (internal): "
          f"{int_res['n_sig_fdr']} sig (FDR)")
    print(f"  Transition {key} (external): "
          f"{ext_res['n_sig_fdr']} sig (FDR)")

    # Save intermediate results
    np.save(f'{OUT_DIR}/transition_{key}_wsfc.npy', comp)
    np.save(f'{OUT_DIR}/transition_{key}_int_wsfc.npy', int_res)
    np.save(f'{OUT_DIR}/transition_{key}_ext_wsfc.npy', ext_res)
    print(f"  Saved: {OUT_DIR}/transition_{key}_[composite|int|ext]_wsfc.npy")

# ── Plot if all transitions computed ─────────────────────────────────────────
saved_keys = list(transition_results.keys())
all_computed = all(f'{a}to{b}' in saved_keys for a, b in [(1,2),(2,3),(3,4)])

if all_computed:
    print("\n[4] Generating connectivity transition figure...")
    _plot_transitions = [(1,2), (2,3), (3,4)]
elif len(saved_keys) > 0:
    # Try to load any previously saved transitions
    for trans_key in ['1to2', '2to3', '3to4']:
        fp = f'{OUT_DIR}/transition_{trans_key}_wsfc.npy'
        if os.path.exists(fp) and trans_key not in transition_results:
            try:
                transition_results[trans_key] = np.load(fp, allow_pickle=True).item()
                print(f"  Loaded saved {trans_key}")
            except Exception:
                pass
    all_computed = all(f'{a}to{b}' in transition_results for a, b in [(1,2),(2,3),(3,4)])

if all(f'{a}to{b}' in transition_results for a, b in [(1,2),(2,3),(3,4)]):
    import seaborn as sns
    sns.set_style('white')

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    transition_labels = ['Rep 1 → Rep 2', 'Rep 2 → Rep 3', 'Rep 3 → Rep 4']
    vmin, vmax = -0.20, 0.20

    # unique network labels and their positions in the N_NETS x N_NETS matrix
    unique_nets = list(dict.fromkeys(network_labels))
    net_tick_pos = [network_labels.index(n) for n in unique_nets]

    im = None
    for ax, (run_a, run_b), label in zip(axes, [(1,2),(2,3),(3,4)], transition_labels):
        key = f'{run_a}to{run_b}'
        res = transition_results[key]
        mat = res['actual_mat']
        sig = res['sig_mat']

        im = ax.matshow(mat, cmap='RdBu_r', vmin=vmin, vmax=vmax)

        # Overlay FDR-significant connections in grey
        sig_overlay = np.where(sig == 1, 0.6, np.nan)
        ax.matshow(sig_overlay, cmap='Greys', alpha=0.5, vmin=0, vmax=1)

        # Network tick labels (match reference style)
        ax.set_xticks(net_tick_pos)
        ax.set_xticklabels(unique_nets, rotation=90, fontsize=7)
        ax.set_yticks(net_tick_pos)
        ax.set_yticklabels(unique_nets, fontsize=7)
        ax.tick_params(axis='y', which='both', length=0)
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        n_sig = int(res['n_sig_fdr'])
        ax.set_title(f'{label}\n(n sig FDR = {n_sig})', fontsize=11, pad=8)

    # Single shared colorbar
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04, label='Δ connectivity (z)')

    plt.suptitle(
        'By-Repetition Connectivity Change: Internal − External\n'
        'Δ[int−ext] for each consecutive transition, averaged across 6 movies\n'
        'Grey overlay = FDR-corrected significant connections (p < 0.05)',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/connectivity_transitions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/connectivity_transitions.png")
else:
    print(f"\n  Not all transitions computed yet. Run all 3 transitions to generate final figure.")
    print(f"  Computed so far: {list(transition_results.keys())}")

# ── Figure 2: Internal vs External condition WSFC (2×3 grid) ─────────────────
int_ext_ready = all(
    f'{a}to{b}_int' in transition_results and f'{a}to{b}_ext' in transition_results
    for a, b in [(1,2),(2,3),(3,4)]
)

if not int_ext_ready:
    # Try loading from disk
    for a, b in [(1,2),(2,3),(3,4)]:
        tk = f'{a}to{b}'
        for suffix in ['_int', '_ext']:
            fk = f'{tk}{suffix}'
            fp = f'{OUT_DIR}/transition_{tk}{suffix}_wsfc.npy'
            if os.path.exists(fp) and fk not in transition_results:
                try:
                    transition_results[fk] = np.load(fp, allow_pickle=True).item()
                except Exception:
                    pass
    int_ext_ready = all(
        f'{a}to{b}_int' in transition_results and f'{a}to{b}_ext' in transition_results
        for a, b in [(1,2),(2,3),(3,4)]
    )

if int_ext_ready:
    import seaborn as sns
    sns.set_style('white')
    print("\n[5] Generating internal vs external connectivity figure (2×3)...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 14),
                             gridspec_kw={'hspace': 0.45})
    transition_labels = ['Rep 1 → Rep 2', 'Rep 2 → Rep 3', 'Rep 3 → Rep 4']
    row_labels = ['Internal', 'External']
    suffixes = ['_int', '_ext']
    vmin_f, vmax_f = -0.20, 0.20

    unique_nets = list(dict.fromkeys(network_labels))
    net_tick_pos = [network_labels.index(n) for n in unique_nets]

    im = None
    for row, (suffix, row_label) in enumerate(zip(suffixes, row_labels)):
        for col, (run_a, run_b) in enumerate([(1,2),(2,3),(3,4)]):
            ax = axes[row, col]
            key = f'{run_a}to{run_b}{suffix}'
            res = transition_results[key]
            mat = res['actual_mat']
            sig = res['sig_mat']

            im = ax.matshow(mat, cmap='RdBu_r', vmin=vmin_f, vmax=vmax_f)

            # Outline significant cells with black border
            n_rows, n_cols = sig.shape
            for i in range(n_rows):
                for j in range(n_cols):
                    if sig[i, j] == 1:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor='black', linewidth=1.0, zorder=5
                        ))

            ax.set_xticks(net_tick_pos)
            ax.set_xticklabels(unique_nets, rotation=90, fontsize=7)
            ax.set_yticks(net_tick_pos)
            ax.set_yticklabels(unique_nets, fontsize=7)
            ax.tick_params(axis='y', which='both', length=0)
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()

            n_sig = int(res['n_sig_fdr'])
            title = f'{row_label}: {transition_labels[col]}\n(n sig FDR = {n_sig})'
            ax.set_title(title, fontsize=11, pad=8)

    # Colorbar along the bottom
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        orientation='horizontal', fraction=0.03, pad=0.08,
                        shrink=0.5, label='Δ connectivity (z)')

    plt.suptitle(
        'By-Repetition Connectivity Change by Condition\n'
        'WSFC change (run_b − run_a) averaged across 6 movies\n'
        'Dimmed cells = non-significant (FDR p ≥ 0.05)',
        fontsize=12, y=1.01
    )
    plt.savefig(f'{PLOT_DIR}/connectivity_transitions_by_condition.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/connectivity_transitions_by_condition.png")

    # ── V2: no grey overlay — significant cells outlined with black border ──
    import copy
    cmap_v2 = copy.copy(plt.cm.RdBu_r)
    cmap_v2.set_bad('white')          # NaN → white, not grey

    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 14),
                               gridspec_kw={'hspace': 0.45})
    fig2.patch.set_facecolor('white')
    im2 = None
    for row, (suffix, row_label) in enumerate(zip(suffixes, row_labels)):
        for col, (run_a, run_b) in enumerate([(1,2),(2,3),(3,4)]):
            ax = axes2[row, col]
            ax.set_facecolor('white')
            key = f'{run_a}to{run_b}{suffix}'
            res = transition_results[key]
            mat = res['actual_mat'].copy()
            sig = res['sig_mat']

            im2 = ax.matshow(mat, cmap=cmap_v2, vmin=vmin_f, vmax=vmax_f)

            # Outline significant cells with black border
            n_rows, n_cols = sig.shape
            for i in range(n_rows):
                for j in range(n_cols):
                    if sig[i, j] == 1:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor='black', linewidth=1.0,
                            zorder=5
                        ))

            ax.set_xticks(net_tick_pos)
            ax.set_xticklabels(unique_nets, rotation=90, fontsize=7)
            ax.set_yticks(net_tick_pos)
            ax.set_yticklabels(unique_nets, fontsize=7)
            ax.tick_params(axis='y', which='both', length=0)
            ax.xaxis.set_label_position('bottom')
            ax.xaxis.tick_bottom()

            n_sig = int(res['n_sig_fdr'])
            title = f'{row_label}: {transition_labels[col]}\n(n sig FDR = {n_sig})'
            ax.set_title(title, fontsize=11, pad=8)

    cbar2 = fig2.colorbar(im2, ax=axes2.ravel().tolist(),
                          orientation='horizontal', fraction=0.03, pad=0.08,
                          shrink=0.5, label='Δ connectivity (z)')

    plt.suptitle(
        'By-Repetition Connectivity Change by Condition\n'
        'WSFC change (run_b − run_a) averaged across 6 movies\n'
        'Black outlines = FDR-significant (p < 0.05)',
        fontsize=12, y=1.01
    )
    plt.savefig(f'{PLOT_DIR}/connectivity_transitions_by_condition_v2.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR}/connectivity_transitions_by_condition_v2.png")

print("\nAnalysis 2 complete (this transition).")
