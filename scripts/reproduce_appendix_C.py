"""
scripts/reproduce_appendix_C.py
================================
Reproduce Appendix C of the manuscript: ensemble distributions of the
three-layer model versus a simplified variant.

The simplified variant is obtained from the three-layer model by
forcing alpha(Q) = 1 and K_ex = 0. This removes the state-dependent
mixing closure and the interflow exfiltration term, leaving only
Green-Ampt surface runoff and a constant baseflow path.

The script samples two 500-member behavioural ensembles under matched
Monte Carlo procedures and reports:
  - Balanced-best NSE_Q and NSE_C in each ensemble
  - 5-95% chloride trough range
  - Number of simulations with trough < 4.5 mg/L

Usage:
  python scripts/reproduce_appendix_C.py --storm-id 33 --n-lhs 2000 --n-workers 8
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from uhf_sir import Model, metrics
from scripts.run_prior_mc import sample_lhs, PARAM_NAMES


def _worker(args):
    """ProcessPool worker: simulate one parameter set for the requested variant."""
    idx, params, storm_data, site, Q_obs, C_obs_h, scoring_mask, force_alpha = args
    try:
        # Apply the simplified-model parameter constraint at K_ex = 0
        if force_alpha:
            params = dict(params)
            params['K_ex'] = 0.0
        model = Model(storm_data, params, site, N=20, dt_fine=300.0)
        model.simulate(force_alpha_one=force_alpha)
        if not model.success:
            return idx, None
        Q_sim = model.Q_out
        C_sim = model.C_out
        Q_sim_s = Q_sim[scoring_mask]
        C_sim_s = C_sim[scoring_mask]
        return idx, {
            'NSE_Q':       metrics.nse(Q_sim_s, Q_obs[scoring_mask]),
            'NSE_C':       metrics.nse(C_sim_s, C_obs_h[scoring_mask]),
            'C_trough':    float(np.nanmin(C_sim_s)),
            'Q_peak':      float(np.nanmax(Q_sim_s)),
        }
    except Exception:
        return idx, None


def run_variant(prior, storm_data, site, Q_obs, C_obs_h, scoring_mask,
                force_alpha, n_workers, label):
    """Forward-simulate every prior sample under the requested variant."""
    n = len(prior)
    tasks = []
    for i in range(n):
        params_i = {p: float(prior.iloc[i][p]) for p in PARAM_NAMES}
        tasks.append((i, params_i, storm_data, site, Q_obs, C_obs_h,
                      scoring_mask, force_alpha))
    print(f'\nSimulating {n} samples for {label}...')
    results = [None] * n
    t0 = time.time()
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = [exe.submit(_worker, t) for t in tasks]
            for j, f in enumerate(as_completed(futures)):
                idx, r = f.result()
                results[idx] = r
                if (j + 1) % max(1, n // 10) == 0:
                    print(f'  [{label}] {j+1}/{n}  ({time.time()-t0:.0f}s)')
    else:
        for j, task in enumerate(tasks):
            idx, r = _worker(task)
            results[idx] = r
            if (j + 1) % max(1, n // 10) == 0:
                print(f'  [{label}] {j+1}/{n}  ({time.time()-t0:.0f}s)')
    df = prior.copy()
    df['success'] = [r is not None for r in results]
    for col in ['NSE_Q', 'NSE_C', 'C_trough', 'Q_peak']:
        df[col] = [r[col] if r is not None else np.nan for r in results]
    return df


def summarise_variant(df, label, obs_trough):
    """Print Appendix C Table C1 statistics for one variant."""
    ok = df[df['success']].copy()
    if len(ok) == 0:
        print(f'  [{label}] no successful samples')
        return
    ok['balanced'] = np.minimum(ok['NSE_Q'], ok['NSE_C'])
    best = ok.loc[ok['balanced'].idxmax()]
    troughs = ok['C_trough'].values
    p5, p95 = np.percentile(troughs, [5, 95])
    n_below = int(np.sum(troughs < 4.5))
    print(f'\n=== {label} ===')
    print(f'  Successful samples:                 {len(ok)} / {len(df)}')
    print(f'  Balanced-best NSE_Q:                {best["NSE_Q"]:+.3f}')
    print(f'  Balanced-best NSE_C:                {best["NSE_C"]:+.3f}')
    print(f'  5-95% chloride trough range (mg/L): [{p5:.2f}, {p95:.2f}]')
    print(f'  Observed trough (mg/L):             {obs_trough:.2f}')
    print(f'  Trough included in 5-95% envelope?  '
          f'{"Yes" if (p5 <= obs_trough <= p95) else "No"}')
    print(f'  Simulations with trough < 4.5 mg/L: {n_below} / {len(ok)} ({100*n_below/len(ok):.1f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--storm-id', type=int, default=33)
    ap.add_argument('--n-lhs', type=int, default=2000)
    ap.add_argument('--n-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-csv', default='outputs/appendix_C.csv')
    args = ap.parse_args()

    # ── Load storm + site ─────────────────────────────────────────────
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
    obs_trough = float(cstream[(cstream['t_sec'] >= ts33.loc[scoring_mask, 't_sec'].iloc[0]) &
                                (cstream['t_sec'] <= ts33.loc[scoring_mask, 't_sec'].iloc[-1])
                                ]['Cl_mgL'].min())
    uh_npz = np.load(data_dir / 'uhf_empirical_uh.npz')
    site = {
        'catchment_area_km2': 1.219,
        'L_channel':          5153.124,
        'B_width':            2.0,
        'S_0':                0.07371,
        'uh_t_hr':            np.asarray(uh_npz['t_hr']),
        'uh_hr_inv':          np.asarray(uh_npz['uh_hr_inv']),
    }
    # ── Sample shared prior ───────────────────────────────────────────
    print(f'LHS sampling {args.n_lhs} parameter sets (shared prior)...')
    prior = sample_lhs(args.n_lhs, seed=args.seed)
    # ── Run both variants ─────────────────────────────────────────────
    df_3layer = run_variant(prior, storm_data, site, Q_obs, C_obs_h,
                              scoring_mask, force_alpha=False,
                              n_workers=args.n_workers, label='3-layer')
    df_simple = run_variant(prior, storm_data, site, Q_obs, C_obs_h,
                              scoring_mask, force_alpha=True,
                              n_workers=args.n_workers, label='simplified')
    # ── Summarise ─────────────────────────────────────────────────────
    print(f'\nObserved chloride trough (storm {args.storm_id}): {obs_trough:.2f} mg/L')
    summarise_variant(df_3layer, '3-layer model',  obs_trough)
    summarise_variant(df_simple, 'Simplified model', obs_trough)
    # ── Save combined results ─────────────────────────────────────────
    df_3layer['variant'] = '3-layer'
    df_simple['variant'] = 'simplified'
    out = pd.concat([df_3layer, df_simple], axis=0, ignore_index=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
