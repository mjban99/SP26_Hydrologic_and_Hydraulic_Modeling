"""
scripts/run_prior_mc.py
=======================
Stage 1 of the SIR workflow: build the prior parameter ensemble.

Workflow:
  1. Latin hypercube sample N_lhs parameter combinations (default 2000)
  2. Forward-simulate every sample on the storm of interest
  3. Apply behavioural filter (NSE_Q >= nse_q_min, default 0.3)
  4. Subsample to N_keep (default 500) for use as the SIR working ensemble

Output: a CSV file containing parameter values and per-sample metrics
(NSE_Q, NSE_C, KGE_Q, KGE_C, J = NSE_Q + 1.5 NSE_C).

Usage:
  python scripts/run_prior_mc.py --storm-id 33 --n-lhs 2000 --n-workers 8
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from uhf_sir import Model, metrics


# ── Prior bounds (Table A1 of the manuscript) ──────────────────────────
PRIOR_BOUNDS = {
    'K_s':         (1e-7,  1e-6,   'log'),       # m/s
    'psi':         (0.08,  0.40,   'log'),       # m
    'theta_d':     (0.08,  0.30,   'lin'),       # -
    'K_ex':        (1e-7,  5e-4,   'log'),       # 1/s
    'f_sat':       (0.05,  0.30,   'lin'),       # -
    'S_max':       (0.015, 0.10,   'log'),       # m
    'tau_drain':   (8100,  6e4,    'log'),       # s  (~2.25-16.6 hr)
    'uh_stretch':  (1.0,   8.0,    'lin'),       # -
    'n':           (0.035, 0.065,  'lin'),       # -
    'V_soil':      (0.05,  2.0,    'log'),       # m
    'alpha_soil':  (0.50,  1.00,   'lin'),       # -
    'Q_threshold': (0.15,  1.20,   'lin'),       # m^3/s
    'gamma':       (2.0,   5.0,    'lin'),       # -
    'lambda_sat':  (0.01,  0.99,   'lin'),       # -
    'baseflow_Q':  (0.03,  0.35,   'lin'),       # m^3/s
}
PARAM_NAMES = list(PRIOR_BOUNDS.keys())


def sample_lhs(n, seed=0):
    """
    Latin hypercube sample of all 16 prior parameters.

    Inputs:
    -------
    n : int
        Number of samples.
    seed : int
        Reproducibility seed.

    Returns:
    --------
    df : pd.DataFrame (n x 16)
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    u = sampler.random(n=n)
    df = pd.DataFrame(index=range(n), columns=PARAM_NAMES, dtype=float)
    for j, p in enumerate(PARAM_NAMES):
        lo, hi, scale = PRIOR_BOUNDS[p]
        if scale == 'log':
            df[p] = np.exp(u[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            df[p] = u[:, j] * (hi - lo) + lo
    return df


def _worker(args):
    """ProcessPool worker: simulate one parameter set + score."""
    idx, params, storm_data, site, Q_obs, C_obs_h, scoring_mask = args
    try:
        model = Model(storm_data, params, site, N=20, dt_fine=300.0)
        model.simulate(force_alpha_one=False)
        if not model.success:
            return idx, None
        Q_sim = model.Q_out[scoring_mask]
        C_sim = model.C_out[scoring_mask]
        Q_obs_s = Q_obs[scoring_mask]
        C_obs_s = C_obs_h[scoring_mask]
        result = {
            'NSE_Q': metrics.nse(Q_sim, Q_obs_s),
            'NSE_C': metrics.nse(C_sim, C_obs_s),
            'KGE_Q': metrics.kge(Q_sim, Q_obs_s),
            'KGE_C': metrics.kge(C_sim, C_obs_s),
        }
        return idx, result
    except Exception:
        return idx, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--storm-id', type=int, default=33)
    ap.add_argument('--n-lhs', type=int, default=2000)
    ap.add_argument('--n-keep', type=int, default=500)
    ap.add_argument('--nse-q-min', type=float, default=0.3)
    ap.add_argument('--n-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-csv', default='outputs/prior_mc_storm33.csv')
    args = ap.parse_args()

    # ── Load storm + site ──────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ts = pd.read_csv(data_dir / 'storm_timeseries_all.csv')
    cl = pd.read_csv(data_dir / 'storm_chloride_all.csv')
    ts33 = ts[ts['storm_id'] == args.storm_id].reset_index(drop=True)
    cl33 = cl[cl['storm_id'] == args.storm_id].reset_index(drop=True)

    crain = cl33[cl33['kind'] == 'rain'].sort_values('t_sec')
    cstream = cl33[cl33['kind'] == 'stream'].sort_values('t_sec')

    storm_data = {
        'rain_t':     ts33['t_sec'].values,
        'rain_flux':  ts33['rain_mmhr'].values,
        'rain_C':     np.interp(ts33['t_sec'].values,
                                crain['t_sec'].values, crain['Cl_mgL'].values),
        't_Q_hourly': ts33['t_sec'].values,
        'pre_Cl':     5.5,
    }
    Q_obs = ts33['Q_obs_m3s'].values
    # Observed stream Cl interpolated onto hourly grid
    C_obs_h = np.interp(ts33['t_sec'].values,
                         cstream['t_sec'].values, cstream['Cl_mgL'].values)
    scoring_mask = ts33['in_scoring'].values.astype(bool)
    # Site
    uh_npz = np.load(data_dir / 'uhf_empirical_uh.npz')
    site = {
        'catchment_area_km2': 1.219,
        'L_channel':          5153.124,
        'B_width':            2.0,
        'S_0':                0.07371,
        'uh_t_hr':            np.asarray(uh_npz['t_hr']),
        'uh_hr_inv':          np.asarray(uh_npz['uh_hr_inv']),
    }
    # ── LHS sample prior ──────────────────────────────────────────────
    print(f'LHS sampling {args.n_lhs} parameter sets...')
    prior = sample_lhs(args.n_lhs, seed=args.seed)
    print(f'  prior shape: {prior.shape}')
    # ── Forward simulate each sample ──────────────────────────────────
    print(f'Simulating {args.n_lhs} samples on storm {args.storm_id}...')
    tasks = [(i, {p: float(prior.iloc[i][p]) for p in PARAM_NAMES},
              storm_data, site, Q_obs, C_obs_h, scoring_mask)
             for i in range(args.n_lhs)]
    results = [None] * args.n_lhs
    t0 = time.time()
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
            futures = [exe.submit(_worker, t) for t in tasks]
            for j, f in enumerate(as_completed(futures)):
                idx, r = f.result()
                results[idx] = r
                if (j + 1) % max(1, args.n_lhs // 20) == 0:
                    el = time.time() - t0
                    print(f'  {j+1}/{args.n_lhs} done  ({el:.0f}s, {(j+1)/el:.1f}/s)')
    else:
        for j, task in enumerate(tasks):
            idx, r = _worker(task)
            results[idx] = r
            if (j + 1) % max(1, args.n_lhs // 20) == 0:
                el = time.time() - t0
                print(f'  {j+1}/{args.n_lhs} done  ({el:.0f}s, {(j+1)/el:.1f}/s)')
    print(f'  total {time.time()-t0:.0f}s')
    # ── Assemble result table ─────────────────────────────────────────
    out = prior.copy()
    out['success'] = [r is not None for r in results]
    for col in ['NSE_Q', 'NSE_C', 'KGE_Q', 'KGE_C']:
        out[col] = [r[col] if r is not None else np.nan for r in results]
    out['J'] = out['NSE_Q'] + 1.5 * out['NSE_C']
    # ── Behavioural filter ────────────────────────────────────────────
    ok = out[out['success'] & (out['NSE_Q'] >= args.nse_q_min)]
    print(f'  successful: {out["success"].sum()}/{args.n_lhs}')
    print(f'  behavioural (NSE_Q >= {args.nse_q_min}): {len(ok)}')
    # ── Subsample to N_keep ───────────────────────────────────────────
    rng = np.random.default_rng(42)
    if len(ok) > args.n_keep:
        keep_idx = rng.choice(len(ok), size=args.n_keep, replace=False)
        keep = ok.iloc[sorted(keep_idx)]
    else:
        keep = ok
        print(f'  WARNING: behavioural set ({len(ok)}) < N_keep ({args.n_keep})')
    keep['working_ensemble'] = True
    out['working_ensemble'] = out.index.isin(keep.index)
    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    print(f'  total rows = {len(out)},  working ensemble rows = {out["working_ensemble"].sum()}')


if __name__ == '__main__':
    main()
