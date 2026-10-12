"""
SIREN-PINN inversion of field ERT data.

Configurable single-file implementation. Driven by a YAML config so the
same code runs any of the four profiles in the paper.

Usage (from a terminal, not inside Jupyter):
    python scripts/run_profile1.py

Usage (from inside Jupyter or another Python script):
    from src.run_inversion import run
    run("configs/config_profile1.yaml")
"""

import os
import sys
import time
import gc
import re
import inspect
import multiprocessing
import warnings
from pathlib import Path

num_cores = multiprocessing.cpu_count()
target_threads = str(min(8, max(2, num_cores // 2)))
os.environ["OMP_NUM_THREADS"] = target_threads
os.environ["MKL_NUM_THREADS"] = target_threads
os.environ["OPENBLAS_NUM_THREADS"] = target_threads

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.sparse import coo_matrix
from scipy.interpolate import (interp1d,
                                LinearNDInterpolator,
                                NearestNDInterpolator)
from scipy.stats import pearsonr, norm as sp_norm
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.patches as mpatches
import yaml

warnings.filterwarnings('ignore')
torch.set_num_threads(2)
try:
    pg.setThreadCount(int(target_threads))
except Exception:
    pass


matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14, 'font.weight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'axes.linewidth': 2.5,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 8, 'ytick.major.size': 8,
    'xtick.major.width': 2.2, 'ytick.major.width': 2.2,
    'legend.fontsize': 13, 'legend.framealpha': 0.95,
    'legend.edgecolor': 'black', 'legend.fancybox': False,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'lines.linewidth': 2.2,
})

CMAP_RHO = plt.cm.rainbow
CMAP_MISS = 'RdBu_r'
CMAP_COND = plt.cm.viridis_r


def apply_tnr(fig):
    for obj in fig.findobj(matplotlib.text.Text):
        try:
            obj.set_fontfamily('serif')
            obj.set_fontweight('bold')
        except Exception:
            pass


def set_xlabel(ax, t, fs=16):
    ax.set_xlabel(t, fontsize=fs, fontweight='bold', fontfamily='serif')


def set_ylabel(ax, t, fs=16):
    ax.set_ylabel(t, fontsize=fs, fontweight='bold', fontfamily='serif')


def set_title(ax, t, fs=16):
    ax.set_title(t, fontsize=fs, fontweight='bold',
                 fontfamily='serif', pad=10)


def style_colorbar(cb, lbl, fs=14, ts=13):
    cb.set_label(lbl, fontsize=fs, fontweight='bold', fontfamily='serif')
    cb.ax.tick_params(labelsize=ts, width=2.0)
    for t in cb.ax.get_yticklabels():
        t.set_fontfamily('serif')
        t.set_fontweight('bold')
        t.set_fontsize(ts)
    cb.outline.set_linewidth(2.0)


def get_trap_corners(x_min, x_max, z_max, taper):
    c = 0.5 * (x_min + x_max)
    thw = 0.5 * (x_max - x_min)
    bhw = thw * (1.0 - taper)
    return [(c - thw, 0.0), (c + thw, 0.0),
            (c + bhw, z_max), (c - bhw, z_max)]


def draw_trap_outline(ax, x_min, x_max, z_max, taper, lw=2.0):
    corners = get_trap_corners(x_min, x_max, z_max, taper)
    corners = corners + [corners[0]]
    ax.plot([p[0] for p in corners], [p[1] for p in corners],
            '-', lw=lw, color='k', alpha=0.9, zorder=6)


def plot_trap_section(ax, data, xi, zi, x_min, x_max, z_max,
                      norm, cmap, taper,
                      elec_x=None, meas_x=None, meas_z=None,
                      show_electrodes=True, show_meas=True,
                      show_hull=True, title=None,
                      elec_offset=-0.25, interpolation='bilinear'):
    c = 0.5 * (x_min + x_max)
    thw = 0.5 * (x_max - x_min)
    X, Z = np.meshgrid(xi, zi)
    hw = thw * (1.0 - taper * (Z / z_max))
    mask = (np.abs(X - c) <= hw) & (Z <= z_max)
    d = data.copy().astype(float)
    d[~mask] = np.nan
    ext = [xi.min(), xi.max(), zi.max(), zi.min()]
    im = ax.imshow(d, extent=ext, aspect='auto', origin='upper',
                   norm=norm, cmap=cmap, interpolation=interpolation,
                   zorder=2)
    if show_hull:
        draw_trap_outline(ax, x_min, x_max, z_max, taper)
    if show_meas and meas_x is not None:
        ax.scatter(meas_x, meas_z, c='k', s=3, marker='.',
                   alpha=0.55, zorder=4)
    if show_electrodes and elec_x is not None:
        ax.scatter(elec_x, np.full_like(elec_x, elec_offset),
                   c='k', s=45, marker='v', zorder=7,
                   edgecolors='white', linewidths=0.7, label='Electrodes')
    ax.set_xlim(x_min - 1.5, x_max + 1.5)
    ax.set_ylim(z_max * 1.05, elec_offset - 0.3)
    if title:
        set_title(ax, title)
    return im


def build_interpolators(x_in, z_in, vals):
    pts = np.column_stack([x_in, z_in])
    return LinearNDInterpolator(pts, vals), NearestNDInterpolator(pts, vals)


def fast_grid(lin, near, XI, ZI):
    g = lin(XI, ZI)
    if np.any(np.isnan(g)):
        g = np.where(np.isnan(g), near(XI, ZI), g)
    return g


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=5.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = 1.0 / in_f if is_first else np.sqrt(6.0 / in_f) / w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


def build_siren(hidden, n_layers, w0, log_rho_mean, log_rho_std, device):
    class SirenNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_rho_mean = log_rho_mean
            self.log_rho_std = log_rho_std
            self.first = SineLayer(2, hidden, w0=w0, is_first=True)
            self.hidden = nn.ModuleList([
                SineLayer(hidden, hidden, w0=w0)
                for _ in range(n_layers - 1)
            ])
            self.out = nn.Linear(hidden, 1)
            with torch.no_grad():
                self.out.weight.uniform_(-0.05, 0.05)
                self.out.bias.zero_()

        def forward(self, x, z):
            xz = torch.stack([x, z], dim=1)
            h = self.first(xz)
            for layer in self.hidden:
                h = layer(h)
            raw = self.out(h).squeeze(-1)
            return self.log_rho_mean + self.log_rho_std * raw

    return SirenNet().to(device)





# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run(config_path):

    # ---------- load config ----------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    PROFILE_ID = cfg["profile_id"]
    INPUT_FILE = cfg["input_file"]
    OUTPUT_DIR_RAW = cfg["output_dir"]

    HIDDEN = cfg["hidden"]
    N_SIREN_LAYERS = cfg["n_siren_layers"]
    W0 = cfg["w0"]

    N_ITERS_MAX = cfg["n_iters_max"]
    MIN_ITERS_BEFORE_CONV = cfg["min_iters_before_conv"]
    CONV_WINDOW = cfg["conv_window"]
    CONV_TOL_REL = cfg["conv_tol_rel"]
    CONV_TOL_ABS = cfg["conv_tol_abs"]
    CONV_CHI2_PLATEAU = cfg["conv_chi2_plateau"]
    TARGET_CHI2 = cfg["target_chi2"]
    USE_TARGET_CHI2 = True
    CHI2_FLOOR = cfg["chi2_floor"]
    USE_CHI2_FLOOR = True

    LR = cfg["learning_rate"]
    LAMBDA_REG = cfg["lambda_reg"]
    ADAPT_LAMBDA = cfg["adapt_lambda"]
    GRAD_CLIP = cfg["grad_clip"]
    LAMBDA_FLOOR = cfg["lambda_floor"]
    LAMBDA_CEILING = cfg["lambda_ceiling"]

    ERR_REL = cfg["err_rel"]
    ERR_ABS = cfg["err_abs"]

    RHO_MIN = cfg["rho_min"]
    RHO_MAX = cfg["rho_max"]
    DEPTH_WEIGHT_ENABLE = cfg["depth_weight_enable"]
    DEPTH_WEIGHT_FACTOR = cfg["depth_weight_factor"]
    MIN_MEAS_PER_LEVEL = cfg["min_meas_per_level"]

    PARA_DEPTH = cfg["para_depth"]
    PARA_MAX_CELL = cfg["para_max_cell"]
    MESH_QUALITY = cfg["mesh_quality"]
    TRAPEZOID_TAPER = cfg["trapezoid_taper"]

    JAC_UPDATE_SCHEDULE = set(cfg["jac_update_schedule"])

    NORMALIZE_GRADIENTS = True
    TRAIN_TEST_SPLIT = cfg["train_test_split"]
    SPLIT_SEED = cfg["split_seed"]

    PLOT_NX, PLOT_NZ = 300, 100

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUTPUT_DIR = Path(re.sub(r'[\r\n\t]', '', str(OUTPUT_DIR_RAW)).strip())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = PROFILE_ID.replace(" ", "")

    print(f"  INPUT_FILE : {INPUT_FILE}")
    print(f"  PROFILE_ID : {PROFILE_ID}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR}")

    _test = OUTPUT_DIR / "_write_test.txt"
    try:
        _test.write_text("ok", encoding="utf-8")
        _test.unlink()
        print("  Folder writable")
    except Exception as e:
        raise RuntimeError(f"Output not writable: {e}")

    print("=" * 72)
    print(f"PINN INVERSION v5.0 - {PROFILE_ID}")
    print(f"  N_ITERS_MAX={N_ITERS_MAX} | MIN_ITERS={MIN_ITERS_BEFORE_CONV}")
    print(f"  TARGET_CHI2={TARGET_CHI2} | CHI2_FLOOR={CHI2_FLOOR}")
    print(f"  Error model: {ERR_REL*100:.1f}% relative")
    print("=" * 72)

    np.random.seed(42)
    torch.manual_seed(42)

    # ---------- STEP 1: LOAD DATA ----------
    print("\nSTEP 1 - LOAD DATA")
    df = pd.read_excel(INPUT_FILE, sheet_name='Sheet1')

    res_col = None
    for col in df.columns:
        cl = col.lower()
        if ('resistivity' in cl or 'resisitivty' in cl
                or 'rho' in cl):
            res_col = col
            break
    if res_col is None:
        raise ValueError(
            "Could not find apparent resistivity column in the "
            f"Excel file. Columns present: {list(df.columns)}"
        )

    rho_obs = df[res_col].values.astype(float)
    NMEAS = len(rho_obs)
    valid = np.isfinite(rho_obs) & (rho_obs > 0)
    if not np.all(valid):
        df = df[valid].reset_index(drop=True)
        rho_obs = rho_obs[valid]
        NMEAS = len(rho_obs)

    raw_Ax = df['Ax'].values.astype(float)
    raw_Bx = df['Bx'].values.astype(float)
    raw_Mx = df['Mx'].values.astype(float)
    raw_Nx = df['Nx'].values.astype(float)

    UNIT = 1.0
    print(f"  NMEAS = {NMEAS}")
    print(f"  rho_obs range : "
          f"[{rho_obs.min():.2f}, {rho_obs.max():.2f}] Ohm.m")

    # ---------- STEP 2: GEOMETRY ----------
    print("\nSTEP 2 - GEOMETRY")
    all_elec_sorted = np.sort(np.unique(
        np.concatenate([raw_Ax, raw_Bx, raw_Mx, raw_Nx])))
    N_SENSORS = len(all_elec_sorted)

    pos_to_idx = {round(float(x), 6): i
                  for i, x in enumerate(all_elec_sorted)}

    def pos2idx(x_arr):
        return np.array(
            [pos_to_idx[round(float(x), 6)] for x in x_arr],
            dtype=int)

    A_idx = pos2idx(raw_Ax)
    B_idx = pos2idx(raw_Bx)
    M_idx = pos2idx(raw_Mx)
    N_idx = pos2idx(raw_Nx)

    sx = all_elec_sorted.copy()
    X_MIN, X_MAX = sx.min(), sx.max()
    ARRAY_LEN = X_MAX - X_MIN

    AB = np.abs(raw_Bx - raw_Ax)
    AM = np.abs(raw_Mx - raw_Ax)
    mid_x_raw = 0.5 * (raw_Ax + raw_Bx)
    pseudo_z = 0.17 * AB
    INVEST_DEPTH = pseudo_z.max()

    OFFSET = X_MIN + ARRAY_LEN / 2.0
    elec_x_field = sx - OFFSET
    mid_x = mid_x_raw - OFFSET

    a_spacing = AM.copy()
    a_lv = np.round(a_spacing / UNIT).astype(int)
    unique_lvs = np.unique(a_lv)

    level_counts = {int(lv): int(np.sum(a_lv == lv))
                    for lv in unique_lvs}
    reliable_levels = sorted(
        [lv for lv, n in level_counts.items()
         if n >= MIN_MEAS_PER_LEVEL])
    reliable_mask = np.isin(a_lv, reliable_levels)
    n_reliable = int(reliable_mask.sum())

    np.random.seed(SPLIT_SEED)
    n_train = int(TRAIN_TEST_SPLIT * NMEAS)
    n_test = NMEAS - n_train
    perm = np.random.permutation(NMEAS)
    train_mask = np.zeros(NMEAS, dtype=bool)
    train_mask[perm[:n_train]] = True
    test_mask = ~train_mask

    print(f"  N sensors     : {N_SENSORS}")
    print(f"  Array length  : {ARRAY_LEN:.1f} m")
    print(f"  Investig dep  : {INVEST_DEPTH:.2f} m")
    print(f"  Wenner levels : {unique_lvs}")
    print(f"  Train/Test    : {n_train}/{n_test}")

    # ---------- STEP 3: BUILD FIELD SCHEME ----------
    print("\nSTEP 3 - BUILD FIELD SCHEME")
    scheme_field = pg.DataContainerERT()
    for x in sx:
        scheme_field.createSensor([float(x), 0.0, 0.0])

    scheme_field.resize(NMEAS)
    scheme_field['a'] = pg.Vector(A_idx.astype(float))
    scheme_field['b'] = pg.Vector(B_idx.astype(float))
    scheme_field['m'] = pg.Vector(M_idx.astype(float))
    scheme_field['n'] = pg.Vector(N_idx.astype(float))
    scheme_field['valid'] = pg.Vector(np.ones(NMEAS))
    scheme_field.markValid(scheme_field['valid'] > 0)
    scheme_field['rhoa'] = pg.Vector(rho_obs)
    scheme_field['err'] = pg.Vector(np.full(NMEAS, ERR_REL))

    try:
        K_use = np.array(
            ert.createGeometricFactors(scheme_field,
                                        numerical=True))
        print("  Numerical k factors")
    except Exception:
        K_use = 2.0 * np.pi * a_spacing
        print("  Analytical Wenner k = 2*pi*a")

    scheme_field['k'] = pg.Vector(K_use)

    assert scheme_field.sensorCount() == N_SENSORS
    assert max(A_idx.max(), B_idx.max(),
               M_idx.max(), N_idx.max()) < N_SENSORS
    print(f"  Sensors      : {scheme_field.sensorCount()}")
    print(f"  Measurements : {scheme_field.size()}")

    # ---------- STEP 4: MESH + FEM OPERATOR ----------
    print("\nSTEP 4 - MESH + FEM OPERATOR")
    grid = mt.createParaMesh(scheme_field,
                              paraDepth=PARA_DEPTH,
                              quality=MESH_QUALITY,
                              paraMaxCellSize=PARA_MAX_CELL,
                              paraBoundary=3, paraDX=0.3)

    mgr = ert.ERTManager(scheme_field, verbose=False)
    mgr.setMesh(grid, ignoreRegionManager=False)

    fop = mgr.fop
    para_mesh = fop.paraDomain
    N_PARAMS = fop.regionManager().parameterCount()
    N_CELLS = N_PARAMS

    centers = para_mesh.cellCenters()
    cx_cell = np.array(
        [c.x() for c in centers], dtype=np.float32)[:N_CELLS]
    cy_cell = np.array(
        [c.y() for c in centers], dtype=np.float32)[:N_CELLS]

    sort_idx = np.argsort(sx)
    surf_x = sx[sort_idx]
    surf_y = np.zeros_like(surf_x)
    _, uidx = np.unique(surf_x, return_index=True)
    surf_x, surf_y = surf_x[uidx], surf_y[uidx]
    surf_interp = interp1d(surf_x, surf_y, kind='cubic',
                            fill_value='extrapolate',
                            bounds_error=False)
    local_surf_y = surf_interp(cx_cell).astype(np.float32)
    cz_cell = np.maximum(
        local_surf_y - cy_cell, 0.0).astype(np.float32)

    X_CENTER = 0.5 * (cx_cell.min() + cx_cell.max())
    X_HALF = 0.5 * (cx_cell.max() - cx_cell.min()) + 1e-6
    Z_MAX = float(cz_cell.max()) + 1e-6

    elec_x_c = elec_x_field
    mid_x_c = mid_x

    def norm_x(x):
        return (x - X_CENTER) / X_HALF

    def norm_z(z):
        return 2.0 * z / Z_MAX - 1.0

    X_t = torch.tensor(norm_x(cx_cell),
                        dtype=torch.float32, device=DEVICE)
    Z_t = torch.tensor(norm_z(cz_cell),
                        dtype=torch.float32, device=DEVICE)

    log_rho_obs = np.log(rho_obs)
    LOG_RHO_MEAN = float(log_rho_obs.mean())
    LOG_RHO_STD = float(log_rho_obs.std())
    LOG_RHO_MIN = float(np.log(RHO_MIN))
    LOG_RHO_MAX = float(np.log(RHO_MAX))

    print(f"  Mesh cells    : {N_CELLS}")

    net = build_siren(HIDDEN, N_SIREN_LAYERS, W0,
                       LOG_RHO_MEAN, LOG_RHO_STD, DEVICE)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Network params: {total_params:,}")

    layer_info = [
        {'name': 'Input', 'neurons': 2, 'params': 0},
        {'name': 'Sine 1', 'neurons': HIDDEN,
         'params': 2 * HIDDEN + HIDDEN},
        {'name': 'Sine 2', 'neurons': HIDDEN,
         'params': HIDDEN * HIDDEN + HIDDEN},
        {'name': 'Sine 3', 'neurons': HIDDEN,
         'params': HIDDEN * HIDDEN + HIDDEN},
        {'name': 'Output', 'neurons': 1, 'params': HIDDEN + 1}
    ]

    # ---------- smoothness matrix ----------
    node_to_cells = {}
    cell_nodes = []
    for c_idx in range(N_CELLS):
        cell = para_mesh.cell(c_idx)
        cn = tuple(cell.node(n).id()
                   for n in range(cell.nodeCount()))
        cell_nodes.append(cn)
        for nid in cn:
            node_to_cells.setdefault(nid, []).append(c_idx)

    pair_set = set()
    for c_idx in range(N_CELLS):
        counts = {}
        for nid in cell_nodes[c_idx]:
            for other in node_to_cells.get(nid, []):
                if other != c_idx:
                    counts[other] = counts.get(other, 0) + 1
        for other, sh in counts.items():
            if sh >= 2:
                pair_set.add((min(c_idx, other),
                              max(c_idx, other)))

    pairs_arr = np.array(list(pair_set))
    pair_i, pair_j = pairs_arr[:, 0], pairs_arr[:, 1]
    n_pairs = len(pair_i)
    method_used = "shared_nodes"

    rows = np.concatenate([pair_i, pair_j, pair_i, pair_j])
    cols = np.concatenate([pair_i, pair_j, pair_j, pair_i])
    vals = np.concatenate([np.ones(n_pairs), np.ones(n_pairs),
                            -np.ones(n_pairs),
                            -np.ones(n_pairs)])
    C_sparse = coo_matrix((vals, (rows, cols)),
                          shape=(N_CELLS, N_CELLS)).tocsr()

    print(f"  Smoothness pairs: {n_pairs}")




















        # ---------- STEP 5: INVERSION LOOP ----------
    print("\n" + "=" * 72)
    print("STEP 5 - INVERSION LOOP")
    print("=" * 72)

    W_d_base = 1.0 / (ERR_REL + ERR_ABS
                       / np.maximum(rho_obs, 1e-6))
    if DEPTH_WEIGHT_ENABLE:
        depth_weight = 1.0 + DEPTH_WEIGHT_FACTOR * (
            pseudo_z / pseudo_z.max())
    else:
        depth_weight = 1.0
    W_d_log_np = W_d_base * depth_weight
    log_d_obs_np = np.log(rho_obs)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_ITERS_MAX, eta_min=LR / 10)

    lam = LAMBDA_REG
    history = {k: [] for k in [
        'iter', 'chi2', 'phi_smooth', 'phi_total', 'rms_log',
        'rrmse', 'lam', 'rho_min', 'rho_max', 'train_loss',
        'test_loss', 'train_rmse', 'test_rmse', 'train_r2',
        'test_r2', 'is_new_jac']}

    t0 = time.time()
    best_total_obj, best_state, best_metrics = np.inf, None, {}
    n_failed_iters = 0
    J_log_cached = None
    jacobian_computes = 0
    stopped_at_iter = None
    stop_reason = "max_iter"

    print(f"\n {'It':>4} {'chi2':>9} {'TrLoss':>9} "
          f"{'TeLoss':>9} {'TrRMSE%':>8} {'TeRMSE%':>8} "
          f"{'lam':>6} {'t(s)':>6} {'Jac':>5}")
    print(" " + "-" * 95)

    for it in range(N_ITERS_MAX):
        t_it = time.time()
        try:
            optimizer.zero_grad()
            log_rho = net(X_t, Z_t)
            if not torch.all(torch.isfinite(log_rho)):
                n_failed_iters += 1
                continue
            log_rho_np = np.clip(
                log_rho.detach().cpu().numpy(),
                LOG_RHO_MIN, LOG_RHO_MAX)
            rho_np = np.clip(np.exp(log_rho_np),
                              RHO_MIN, RHO_MAX)

            d_pred = np.array(fop.response(rho_np))
            if (not np.all(np.isfinite(d_pred))
                    or np.any(d_pred <= 0)):
                n_failed_iters += 1
                continue

            d_pred_safe = np.maximum(d_pred, 1e-6)
            resid = np.log(d_pred_safe) - log_d_obs_np

            jac_marker = "reuse"
            is_new_jac = False
            if it in JAC_UPDATE_SCHEDULE \
                    or J_log_cached is None:
                fop.createJacobian(rho_np)
                J_raw = np.array(fop.jacobian())
                if not np.all(np.isfinite(J_raw)):
                    n_failed_iters += 1
                    continue
                J_log_cached = ((1.0 / d_pred_safe[:, None])
                                * J_raw * rho_np[None, :])
                jacobian_computes += 1
                jac_marker = "NEW"
                is_new_jac = True

            W_d_train = W_d_log_np * train_mask.astype(float)
            g_data = J_log_cached.T @ (W_d_train ** 2 * resid)
            C_log_rho = C_sparse @ log_rho_np
            g_smooth = C_sparse @ log_rho_np

            if NORMALIZE_GRADIENTS:
                gd_n = np.linalg.norm(g_data) + 1e-12
                gs_n = np.linalg.norm(g_smooth) + 1e-12
                total_g = (g_data / gd_n
                           + lam * g_smooth / gs_n) * gd_n
            else:
                total_g = g_data + lam * g_smooth

            if not np.all(np.isfinite(total_g)):
                n_failed_iters += 1
                continue
            g_norm = np.linalg.norm(total_g)
            if g_norm > 1e6:
                total_g = total_g * (1e6 / g_norm)

            total_g_t = torch.tensor(total_g,
                                      dtype=torch.float32,
                                      device=DEVICE)
            log_rho.backward(total_g_t)
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                            GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            chi2 = float(np.mean((W_d_log_np * resid) ** 2))
            phi_smooth = float(0.5 * np.sum(C_log_rho ** 2))
            phi_total = chi2 + lam * phi_smooth
            rms_log = float(np.sqrt(np.mean(resid ** 2)))
            rrmse = float(
                np.sqrt(np.mean((d_pred - rho_obs) ** 2))
                / np.mean(rho_obs) * 100)

            train_loss = float(np.mean(
                (W_d_log_np[train_mask]
                 * resid[train_mask]) ** 2))
            train_rmse = float(
                np.sqrt(np.mean(
                    (d_pred[train_mask]
                     - rho_obs[train_mask]) ** 2))
                / np.mean(rho_obs[train_mask]) * 100)
            try:
                train_r2 = float(
                    1.0 - np.sum(
                        (d_pred[train_mask]
                         - rho_obs[train_mask]) ** 2)
                    / np.sum(
                        (rho_obs[train_mask]
                         - rho_obs[train_mask].mean()) ** 2))
            except Exception:
                train_r2 = 0.0

            test_loss = float(np.mean(
                (W_d_log_np[test_mask]
                 * resid[test_mask]) ** 2))
            test_rmse = float(
                np.sqrt(np.mean(
                    (d_pred[test_mask]
                     - rho_obs[test_mask]) ** 2))
                / np.mean(rho_obs[test_mask]) * 100)
            try:
                test_r2 = float(
                    1.0 - np.sum(
                        (d_pred[test_mask]
                         - rho_obs[test_mask]) ** 2)
                    / np.sum(
                        (rho_obs[test_mask]
                         - rho_obs[test_mask].mean()) ** 2))
            except Exception:
                test_r2 = 0.0

            for k_, v_ in [('iter', it + 1), ('chi2', chi2),
                            ('phi_smooth', phi_smooth),
                            ('phi_total', phi_total),
                            ('rms_log', rms_log), ('rrmse', rrmse),
                            ('lam', lam),
                            ('rho_min', float(rho_np.min())),
                            ('rho_max', float(rho_np.max())),
                            ('train_loss', train_loss),
                            ('test_loss', test_loss),
                            ('train_rmse', train_rmse),
                            ('test_rmse', test_rmse),
                            ('train_r2', train_r2),
                            ('test_r2', test_r2),
                            ('is_new_jac', is_new_jac)]:
                history[k_].append(v_)

            if phi_total < best_total_obj:
                best_total_obj = phi_total
                best_state = {k: v.clone().detach()
                              for k, v in net.state_dict().items()}
                best_metrics = {'iter': it + 1, 'chi2': chi2,
                                'phi_total': phi_total,
                                'rrmse': rrmse}

            stopped = False
            if it + 1 >= MIN_ITERS_BEFORE_CONV:
                if len(history['chi2']) >= 2 * CONV_WINDOW:
                    recent = np.array(
                        history['chi2'][-CONV_WINDOW:])
                    prior = np.array(
                        history['chi2']
                        [-2 * CONV_WINDOW:-CONV_WINDOW])
                    m_rec = float(np.mean(recent))
                    m_pri = float(np.mean(prior))
                    d_abs = abs(m_pri - m_rec)
                    d_rel = d_abs / max(abs(m_pri), 1e-12) * 100
                    if d_rel < CONV_TOL_REL:
                        stop_reason = (
                            f"rate-based dmean chi2 < "
                            f"{CONV_TOL_REL}% "
                            f"(drel={d_rel:.3f}%)")
                        stopped = True
                    elif (m_rec < CONV_CHI2_PLATEAU
                          and d_abs < CONV_TOL_ABS):
                        stop_reason = (
                            f"plateau chi2 < "
                            f"{CONV_CHI2_PLATEAU} "
                            f"& dabs < {CONV_TOL_ABS}")
                        stopped = True

                if not stopped and USE_TARGET_CHI2 \
                        and len(history['chi2']) >= 5:
                    rm = float(np.mean(history['chi2'][-5:]))
                    if rm < TARGET_CHI2:
                        stop_reason = (
                            f"target chi2 "
                            f"(5-iter mean={rm:.4f}) "
                            f"< {TARGET_CHI2}")
                        stopped = True

                if not stopped and USE_CHI2_FLOOR \
                        and len(history['chi2']) >= 5:
                    rm = float(np.mean(history['chi2'][-5:]))
                    if rm < CHI2_FLOOR:
                        stop_reason = (
                            f"chi2 floor "
                            f"({CHI2_FLOOR}) reached")
                        stopped = True

                if stopped:
                    stopped_at_iter = it + 1
                    print(f" {it+1:>4d} {chi2:>9.4f} "
                          f"{train_loss:>9.4f} "
                          f"{test_loss:>9.4f} "
                          f"{train_rmse:>7.2f}% "
                          f"{test_rmse:>7.2f}% {lam:>6.4f} "
                          f"{time.time()-t_it:>6.1f} "
                          f"{jac_marker:>5}")
                    print(f"\n  STOPPED at iter "
                          f"{stopped_at_iter} - {stop_reason}")
                    print(f"     chi2/N = {chi2:.4f}, "
                          f"RRMSE = {rrmse:.2f}%")
                    break

            print(f" {it+1:>4d} {chi2:>9.4f} "
                  f"{train_loss:>9.4f} {test_loss:>9.4f} "
                  f"{train_rmse:>7.2f}% "
                  f"{test_rmse:>7.2f}% {lam:>6.4f} "
                  f"{time.time()-t_it:>6.1f} {jac_marker:>5}")

            if ADAPT_LAMBDA and it > 5:
                if chi2 > 2.5:
                    lam = max(LAMBDA_FLOOR, lam * 0.90)
                elif chi2 < 0.8:
                    lam = min(LAMBDA_CEILING, lam * 1.05)

        except Exception as e:
            print(f" Iter {it+1} error: {e}")
            n_failed_iters += 1

    t_total = time.time() - t0
    if best_state is not None:
        net.load_state_dict(best_state)

    print(f"\n  Runtime: {t_total:.1f}s")
    print(f"  Stopped at: "
          f"{stopped_at_iter if stopped_at_iter else N_ITERS_MAX}")
    print(f"  Reason: {stop_reason}")
    print(f"  Jacobian computes: {jacobian_computes}")

    # ---------- STEP 6: METRICS ----------
    print("\nSTEP 6 - METRICS")
    net.eval()
    with torch.no_grad():
        log_rho_final = net(X_t, Z_t).cpu().numpy()
    log_rho_final = np.clip(log_rho_final,
                             LOG_RHO_MIN, LOG_RHO_MAX)
    rho_final = np.clip(np.exp(log_rho_final),
                         RHO_MIN, RHO_MAX)

    d_final = np.array(fop.response(rho_final))
    d_final_safe = np.maximum(d_final, 1e-6)
    resid_final = np.log(d_final_safe) - log_d_obs_np

    chi2_final = float(np.mean((W_d_log_np * resid_final) ** 2))
    rms_log_final = float(np.sqrt(np.mean(resid_final ** 2)))
    rms_final = float(np.sqrt(np.mean((d_final - rho_obs) ** 2)))
    rrmse_final = float(rms_final / np.mean(rho_obs) * 100)
    bias_final = float(np.mean(d_final - rho_obs))
    mae_final = float(np.mean(np.abs(d_final - rho_obs)))
    max_err_final = float(np.max(np.abs(d_final - rho_obs)))
    r2_final = float(
        1.0 - np.sum((d_final - rho_obs) ** 2)
        / np.sum((rho_obs - rho_obs.mean()) ** 2))

    train_rmse_final = float(
        np.sqrt(np.mean(
            (d_final[train_mask]
             - rho_obs[train_mask]) ** 2))
        / np.mean(rho_obs[train_mask]) * 100)
    test_rmse_final = float(
        np.sqrt(np.mean(
            (d_final[test_mask]
             - rho_obs[test_mask]) ** 2))
        / np.mean(rho_obs[test_mask]) * 100)
    train_r2_final = float(
        1.0 - np.sum(
            (d_final[train_mask]
             - rho_obs[train_mask]) ** 2)
        / np.sum(
            (rho_obs[train_mask]
             - rho_obs[train_mask].mean()) ** 2))
    test_r2_final = float(
        1.0 - np.sum(
            (d_final[test_mask]
             - rho_obs[test_mask]) ** 2)
        / np.sum(
            (rho_obs[test_mask]
             - rho_obs[test_mask].mean()) ** 2))

    log_r = (np.log(np.maximum(d_final, 1e-9))
             - np.log(np.maximum(rho_obs, 1e-9)))
    log_bias = float(np.mean(log_r))
    pct_v = 100.0 * (d_final - rho_obs) / rho_obs
    med_r = float(np.median(d_final / rho_obs))

    try:
        r_val, p_val = pearsonr(rho_obs, d_final)
        pearson_r = float(r_val)
    except Exception:
        pearson_r = 0.0
        p_val = 1.0

    mu_lr, sig_lr = sp_norm.fit(log_r)
    mu_pct, sig_pct = sp_norm.fit(pct_v)

    lv_list_sorted = sorted(np.unique(a_lv))
    level_stats = {}
    for lv in lv_list_sorted:
        mk = (a_lv == lv)
        if mk.sum() < 3:
            continue
        ro_l, rm_l = rho_obs[mk], d_final[mk]
        mf_l = rm_l - ro_l
        try:
            r2_l = float(pearsonr(ro_l, rm_l)[0] ** 2)
        except Exception:
            r2_l = 0.0
        level_stats[lv] = dict(
            n=int(mk.sum()),
            a_spacing_m=float(lv * UNIT),
            rms=float(np.sqrt(np.mean(mf_l ** 2))),
            rrmse=(100.0 * float(np.sqrt(np.mean(mf_l ** 2)))
                   / np.mean(ro_l)),
            r2=r2_l,
            bias=float(np.mean(mf_l)),
            mae=float(np.mean(np.abs(mf_l))),
            rms_log=float(np.sqrt(np.mean(
                (np.log(np.maximum(rm_l, 1e-9))
                 - np.log(np.maximum(ro_l, 1e-9))) ** 2))),
            mean_obs=float(np.mean(ro_l)),
            mean_sim=float(np.mean(rm_l)),
            reliable=lv in reliable_levels,
        )

    lv_arr = np.array(lv_list_sorted)
    x_pos = lv_arr * UNIT
    rr_arr = np.array([level_stats[l]['rrmse']
                       for l in lv_arr])
    r2_arr = np.array([level_stats[l]['r2']
                       for l in lv_arr])
    rms_arr = np.array([level_stats[l]['rms']
                        for l in lv_arr])
    mae_arr = np.array([level_stats[l]['mae']
                        for l in lv_arr])
    bias_arr = np.array([level_stats[l]['bias']
                         for l in lv_arr])
    reliable_arr = np.array([level_stats[l]['reliable']
                             for l in lv_arr])

    print(f"  R2      = {r2_final:.4f}")
    print(f"  RRMSE   = {rrmse_final:.2f}%")
    print(f"  chi2/N  = {chi2_final:.4f}")








        # ---------- STEP 7: EXPORT TABLES ----------
    print("\nSTEP 7 - EXPORT")
    tab1 = pd.DataFrame({
        'Metric': ['N', 'N_train', 'N_test', 'Pearson_R',
                   'R2_all', 'R2_train', 'R2_test',
                   'RMS', 'MAE', 'Max_error',
                   'RRMSE_all_%', 'RRMSE_train_%',
                   'RRMSE_test_%',
                   'RMS_log', 'Bias', 'Log_bias',
                   'Median_ratio',
                   'Mean_obs', 'Mean_sim', 'chi2_per_N',
                   'Model_rho_min', 'Model_rho_max',
                   'N_cells', 'Smoothness_pairs',
                   'Iterations', 'Stop_reason',
                   'Jacobian_computes', 'Runtime_s'],
        'Value': [NMEAS, n_train, n_test, pearson_r,
                  r2_final, train_r2_final, test_r2_final,
                  rms_final, mae_final, max_err_final,
                  rrmse_final, train_rmse_final,
                  test_rmse_final,
                  rms_log_final, bias_final, log_bias, med_r,
                  float(np.mean(rho_obs)),
                  float(np.mean(d_final)),
                  chi2_final,
                  float(rho_final.min()),
                  float(rho_final.max()),
                  N_CELLS, n_pairs,
                  (stopped_at_iter
                   if stopped_at_iter else N_ITERS_MAX),
                  stop_reason,
                  jacobian_computes, t_total],
    })
    tab1.to_csv(str(OUTPUT_DIR / f'Table1_{suffix}.csv'),
                index=False)

    tab2 = pd.DataFrame(level_stats).T.reset_index()
    tab2 = tab2.rename(columns={'index': 'Level'})
    tab2.to_csv(str(OUTPUT_DIR / f'Table2_{suffix}.csv'),
                index=False)

    pd.DataFrame(layer_info).to_csv(
        str(OUTPUT_DIR
            / f'Table6_Architecture_{suffix}.csv'),
        index=False)

    pd.DataFrame({
        'x_m': cx_cell,
        'depth_m': cz_cell,
        'rho_model': rho_final,
        'sigma_model_mSm': (1000.0
                             / np.maximum(rho_final, 1e-9)),
    }).to_excel(
        str(OUTPUT_DIR / f'PINN_Model_{suffix}.xlsx'),
        index=False)

    pd.DataFrame({
        'rho_obs': rho_obs,
        'rho_pred': d_final,
        'misfit': d_final - rho_obs,
    }).to_excel(
        str(OUTPUT_DIR / f'PINN_DataFit_{suffix}.xlsx'),
        index=False)

    sio.savemat(
        str(OUTPUT_DIR / f'PINN_Results_{suffix}.mat'),
        {'cx': cx_cell, 'cy': cy_cell, 'cz': cz_cell,
         'rho_model': rho_final,
         'sigma_model': (1000.0
                          / np.maximum(rho_final, 1e-9)),
         'rho_obs': rho_obs, 'rho_pred': d_final,
         'r2_all': r2_final, 'rrmse_all': rrmse_final,
         'chi2': chi2_final, 'runtime': t_total,
         'stopped_at_iter': (stopped_at_iter
                             if stopped_at_iter
                             else N_ITERS_MAX),
         'history_chi2': np.array(history['chi2']),
         'history_train_loss':
             np.array(history['train_loss']),
         'history_test_loss':
             np.array(history['test_loss'])})

    # ---------- STEP 8: GRIDDING ----------
    print("\nSTEP 8 - GRIDDING")

    def fill_grid(x_in, z_in, vals, nx=PLOT_NX,
                  nz=PLOT_NZ, x_min=None, x_max=None,
                  z_min=0.0, z_max=None):
        if x_min is None:
            x_min = x_in.min()
        if x_max is None:
            x_max = x_in.max()
        if z_max is None:
            z_max = z_in.max()
        xi = np.linspace(x_min, x_max, nx)
        zi = np.linspace(z_min, z_max, nz)
        XI, ZI = np.meshgrid(xi, zi)
        lin, near = build_interpolators(x_in, z_in, vals)
        return xi, zi, fast_grid(lin, near, XI, ZI)

    xi_p, zi_p, obs_g = fill_grid(mid_x_c, pseudo_z,
                                    rho_obs)
    xi_s, zi_s, sim_g = fill_grid(mid_x_c, pseudo_z,
                                    d_final)
    xi_r, zi_r, mis_g = fill_grid(mid_x_c, pseudo_z, log_r)

    X_MIN_F = float(mid_x_c.min())
    X_MAX_F = float(mid_x_c.max())
    Z_MAX_F = float(pseudo_z.max())
    dx_pad = 5.0
    xi_mdl = np.linspace(X_MIN_F - dx_pad,
                          X_MAX_F + dx_pad, PLOT_NX)
    zi_mdl = np.linspace(0, cz_cell.max() * 1.05,
                          PLOT_NZ)
    XIm, ZIm = np.meshgrid(xi_mdl, zi_mdl)
    lin_mdl, near_mdl = build_interpolators(
        cx_cell - X_CENTER, cz_cell, rho_final)
    rho_model_2d = fast_grid(lin_mdl, near_mdl, XIm, ZIm)

    # ---------- save one final figure ----------
    def savefig(name, fig=None):
        target = fig if fig is not None else plt.gcf()
        clean = re.sub(r'[^\w\-]', '_', str(name)).strip('_')
        for ext in ['.png', '.pdf']:
            target.savefig(
                str(OUTPUT_DIR / f"{clean}{ext}"),
                dpi=300, bbox_inches='tight')
        plt.close(target)
        del target
        gc.collect()
        print(f"  Saved {clean}")

    fig, ax = plt.subplots(figsize=(14, 6))
    norm_mod = Normalize(
        np.nanpercentile(rho_model_2d, 2),
        np.nanpercentile(rho_model_2d, 98))
    im = plot_trap_section(
        ax, rho_model_2d, xi_mdl, zi_mdl,
        X_MIN_F, X_MAX_F, Z_MAX_F,
        norm_mod, CMAP_RHO, TRAPEZOID_TAPER,
        elec_x=elec_x_c)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    style_colorbar(cb, 'Resistivity (Ohm-m)')
    set_xlabel(ax, 'Distance (m)')
    set_ylabel(ax, 'Depth (m)')
    set_title(ax, f'{PROFILE_ID}: recovered resistivity '
                  f'model - R2 = {r2_final:.4f}')
    savefig(f'Fig_Model_{suffix}', fig)

    # ---------- FINAL SUMMARY ----------
    print("\n" + "=" * 72)
    print(f"  COMPLETE - {PROFILE_ID}")
    print("=" * 72)
    print(f"  R2       = {r2_final:.4f}")
    print(f"  RRMSE    = {rrmse_final:.2f}%")
    print(f"  chi2/N   = {chi2_final:.4f}")
    print(f"  Stopped  : "
          f"{stopped_at_iter if stopped_at_iter else N_ITERS_MAX}")
    print(f"  Runtime  : {t_total:.1f}s")
    print(f"  Output   : {OUTPUT_DIR}")
    print("=" * 72)

    return {
        'rho_final': rho_final,
        'd_final': d_final,
        'chi2_final': chi2_final,
        'r2_final': r2_final,
        'rrmse_final': rrmse_final,
        'rms_log_final': rms_log_final,
        'history': history,
        'level_stats': level_stats,
        'cx_cell': cx_cell,
        'cy_cell': cy_cell,
        'cz_cell': cz_cell,
        'rho_model_2d': rho_model_2d,
        'xi_mdl': xi_mdl,
        'zi_mdl': zi_mdl,
        'output_dir': OUTPUT_DIR,
        'profile_id': PROFILE_ID,
    }


# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    cfg_path = (sys.argv[1] if len(sys.argv) > 1
                else "configs/config_profile1.yaml")
    run(cfg_path)