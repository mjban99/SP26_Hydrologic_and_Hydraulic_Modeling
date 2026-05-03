"""
scripts/run_sir_sweep.py
========================
Stage 2 of the SIR workflow: run expanding-window SIR updates against
the working prior ensemble produced by run_prior_mc.py.

This script is a thin wrapper around the ``uhf_sir.SIR`` class. The
five-step procedure of the manuscript Section 2.3 is implemented in
``uhf_sir/sir.py``; for a step-by-step walkthrough with intermediate
visualisations see ``notebooks/04_sir_sweep.ipynb``.

Usage:
  python scripts/run_sir_sweep.py --storm-id 33 --n-workers 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from uhf_sir import SIR


# Default DA window snapshots (manuscript Section 2.3)
DEFAULT_SNAPSHOTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 19]
PARAM_NAMES = [
    'K_s', 'psi', 'theta_d', 'K_ex', 'f_sat', 'S_max', 'tau_drain',
    'uh_stretch', 'n', 'V_soil', 'alpha_soil',
    'Q_threshold', 'gamma', 'lambda_sat', 'baseflow_Q',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--storm-id', type=int, default=33)
    ap.add_argument('--prior', default='outputs/prior_mc_storm33.csv')
    ap.add_argument('--snapshots', nargs='+', type=float,
                    default=DEFAULT_SNAPSHOTS)
    ap.add_argument('--scenarios', nargs='+', default=['Q-only', 'Q+Cl'])
    ap.add_argument('--sigma-Q', type=float, default=None,
                    help='Default: 10%% of observed peak in scoring window')
    ap.add_argument('--sigma-C', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--n-workers', type=int, default=4)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-npz', default='outputs/sir_sweep_storm33.npz')
    args = ap.parse_args()

    # ── Load storm data and site ──────────────────────────────────────
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
    C_obs_h = np.interp(ts33['t_sec'].values,
                         cstream['t_sec'].values, cstream['Cl_mgL'].values)
    scoring_mask = ts33['in_scoring'].values.astype(bool)
    t_rain_start = float(ts33.loc[scoring_mask, 't_sec'].iloc[0])
    t_hr = (ts33['t_sec'].values - t_rain_start) / 3600.0

    uh_npz = np.load(data_dir / 'uhf_empirical_uh.npz')
    site = {
        'catchment_area_km2': 1.219,
        'L_channel':          5153.124,
        'B_width':            2.0,
        'S_0':                0.07371,
        'uh_t_hr':            np.asarray(uh_npz['t_hr']),
        'uh_hr_inv':          np.asarray(uh_npz['uh_hr_inv']),
    }

    # ── Load Stage 1 prior working ensemble ───────────────────────────
    prior_csv = pd.read_csv(args.prior)
    if 'working_ensemble' in prior_csv.columns:
        prior = prior_csv[prior_csv['working_ensemble']].reset_index(drop=True)
    else:
        prior = prior_csv.reset_index(drop=True)
    prior = prior[PARAM_NAMES]
    print(f'Working ensemble size: {len(prior)}')

    # ── Build SIR ─────────────────────────────────────────────────────
    sir = SIR(storm_data, site, prior, Q_obs, C_obs_h, t_hr, scoring_mask,
              sigma_Q=args.sigma_Q, sigma_C=args.sigma_C, seed=args.seed)
    print(f'sigma_Q = {sir.sigma_Q:.3f} m^3/s')
    print(f'sigma_C = {sir.sigma_C:.3f} mg/L')

    # ── Stage 1: prior forward simulation ─────────────────────────────
    print('\nStage 1: prior forward simulation')
    sir.simulate_prior(n_workers=args.n_workers)

    # ── Stage 2: SIR sweep ────────────────────────────────────────────
    print('\nStage 2: SIR sweep')
    sir.run_sweep(args.snapshots, scenarios=tuple(args.scenarios),
                   n_workers=args.n_workers)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sir.save(out_path)
    print(f'\nSaved: {out_path}')

    # ── N_eff summary ─────────────────────────────────────────────────
    print('\nN_eff summary:')
    print(f'  {"window":>8s}  {"Q-only":>10s}  {"Q+Cl":>10s}')
    for T in args.snapshots:
        sname = f'{int(round(T)):02d}h'
        n_q = sir.N_eff.get(f'{sname}__Q-only', np.nan)
        n_qc = sir.N_eff.get(f'{sname}__Q+Cl', np.nan)
        print(f'  {sname:>8s}  {n_q:>10.2f}  {n_qc:>10.2f}')


if __name__ == '__main__':
    main()
