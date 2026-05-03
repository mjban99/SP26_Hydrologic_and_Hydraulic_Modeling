import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from uhf_sir import metrics
from uhf_sir.model import Model


# Module-level worker for ProcessPoolExecutor (must be picklable)
def _simulate_one(args):
    """
    Forward-simulate a single parameter set.

    Returns (idx, Q_hourly, C_hourly), with both arrays None on failure.
    """
    idx, parameters, storm_data, site = args
    try:
        model = Model(storm_data, parameters, site, N=20, dt_fine=300.0)
        model.simulate(force_alpha_one=False)
        if not model.success:
            return idx, None, None
        return idx, model.Q_out, model.C_out
    except Exception:
        return idx, None, None


class SIR():
    """
    Sequential Importance Resampling particle filter for joint
    streamflow-chloride parameter inference.

    Inputs:
    -------
    storm_data : dict
        Storm-specific input data, identical to the dict accepted by
        ``uhf_sir.Model``.

    site : dict
        Site-specific fixed quantities, identical to the dict accepted
        by ``uhf_sir.Model``.

    prior : pd.DataFrame (N x 16)
        Prior parameter ensemble. Each row is a parameter sample; each
        column corresponds to one calibration parameter. Column names
        must match the keys expected by ``Model.parameters``.

    Q_obs : np.ndarray
        Observed streamflow on the hourly grid (m^3/s). Must align
        with ``storm_data['t_Q_hourly']``.

    C_obs : np.ndarray
        Observed stream chloride on the hourly grid (mg/L). Seven-hourly
        grab samples should be linearly interpolated onto the hourly
        grid by the caller before construction.

    t_hr : np.ndarray
        Time vector aligned with Q_obs and C_obs, in hours from rain
        start (negative during warm-up, zero at rain onset).

    scoring_mask : np.ndarray (bool)
        Boolean mask aligned with t_hr selecting the scoring window
        (typically [rain_start, rain_start + 84 h]).

    sigma_Q : float
        Streamflow observation error standard deviation (m^3/s).
        If None, defaults to 10% of observed peak in the scoring window.

    sigma_C : float
        Chloride observation error standard deviation (mg/L). Default
        0.8 mg/L (Plynlimon ion-chromatography RSD ~5%).

    perturb_frac : float
        Gaussian jitter fraction of prior std applied during resampling
        (default 0.05).

    log_params : iterable of str
        Parameter names sampled in log space (default
        {K_s, K_ex, tau_drain, S_max}). Identifiability is computed in
        linear space throughout, with log-space reduction reported
        separately for V_soil where it differs materially.

    seed : int
        Random seed for systematic resampling and perturbation
        (default 12345).

    Methods:
    --------
    simulate_prior : Forward-simulate every prior sample
    step           : Perform one SIR update over a single window/scenario
    run_sweep      : Run all (window, scenario) combinations
    save           : Write results to .npz

    Attributes:
    -----------
    prior_Q          : Prior-simulated streamflow ensemble (N x T)
    prior_C          : Prior-simulated chloride ensemble (N x T)
    posterior_params : Dict {key: pd.DataFrame} of posterior parameter samples
    posterior_Q      : Dict {key: np.ndarray} of posterior streamflow simulations
    posterior_C      : Dict {key: np.ndarray} of posterior chloride simulations
    weights          : Dict {key: np.ndarray} of normalized particle weights
    N_eff            : Dict {key: float} of effective sample sizes
    n_obs_used       : Dict {key: int} of likelihood-evaluated time steps

    Where each key is of the form '{window}h__{scenario}', e.g.
    '10h__Q+Cl'.
    """
    # Default log-space parameters (Table A1 of the manuscript)
    LOG_PARAMS = ('K_s', 'K_ex', 'tau_drain', 'S_max')
    # Default perturbation fraction (Section 2.3, step 5 of the manuscript)
    PERTURB_FRAC = 0.05

    def __init__(self, storm_data, site, prior, Q_obs, C_obs, t_hr,
                 scoring_mask, sigma_Q=None, sigma_C=0.8,
                 perturb_frac=PERTURB_FRAC, log_params=LOG_PARAMS,
                 seed=12345):
        # Copy inputs to prevent modification
        storm_data = copy.deepcopy(storm_data)
        site = copy.deepcopy(site)
        prior = copy.deepcopy(prior)
        # Save copied inputs to instance
        self.storm_data = storm_data
        self.site = site
        self.prior = prior.reset_index(drop=True)
        self.N = len(self.prior)
        self.param_names = list(self.prior.columns)
        # Cast observation arrays
        self._Q_obs = np.asarray(Q_obs, dtype=np.float64)
        self._C_obs = np.asarray(C_obs, dtype=np.float64)
        self._t_hr = np.asarray(t_hr, dtype=np.float64)
        self._scoring_mask = np.asarray(scoring_mask, dtype=bool)
        # Observation error
        if sigma_Q is None:
            sigma_Q = 0.10 * float(self._Q_obs[self._scoring_mask].max())
        self.sigma_Q = float(sigma_Q)
        self.sigma_C = float(sigma_C)
        # Perturbation
        self.perturb_frac = float(perturb_frac)
        self.log_params = set(log_params)
        # Seed
        self.seed = int(seed)
        # Initialize result containers (filled by simulate_prior, step, run_sweep)
        self.prior_Q = None
        self.prior_C = None
        self.posterior_params = {}
        self.posterior_Q = {}
        self.posterior_C = {}
        self.weights = {}
        self.N_eff = {}
        self.n_obs_used = {}

    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Prior forward simulation
    # ──────────────────────────────────────────────────────────────────

    def simulate_prior(self, n_workers=1, label='prior'):
        """
        Forward-simulate every prior parameter sample on the hourly grid.

        Inputs:
        -------
        n_workers : int
            Number of parallel processes. Default 1 (serial).
        label : str
            Progress-print label. Default 'prior'.
        """
        # Build task list
        T = len(self.storm_data['t_Q_hourly'])
        Q_mat = np.full((self.N, T), np.nan)
        C_mat = np.full((self.N, T), np.nan)
        tasks = []
        for i in range(self.N):
            params_i = {p: float(self.prior.iloc[i][p]) for p in self.param_names}
            tasks.append((i, params_i, self.storm_data, self.site))
        # Run
        t0 = time.time()
        n_ok = 0
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as exe:
                futures = [exe.submit(_simulate_one, t) for t in tasks]
                for j, f in enumerate(as_completed(futures)):
                    idx, Qh, Ch = f.result()
                    if Qh is not None:
                        Q_mat[idx] = Qh
                        C_mat[idx] = Ch
                        n_ok += 1
                    if (j + 1) % max(1, self.N // 5) == 0:
                        print(f'   [{label}] {j+1}/{self.N} done, {time.time()-t0:.0f}s')
        else:
            for j, task in enumerate(tasks):
                idx, Qh, Ch = _simulate_one(task)
                if Qh is not None:
                    Q_mat[idx] = Qh
                    C_mat[idx] = Ch
                    n_ok += 1
                if (j + 1) % max(1, self.N // 5) == 0:
                    print(f'   [{label}] {j+1}/{self.N} done, {time.time()-t0:.0f}s')
        print(f'   [{label}] success {n_ok}/{self.N}, {time.time()-t0:.0f}s')
        # Drop failed prior samples
        ok_mask = ~np.isnan(Q_mat[:, 0])
        if ok_mask.sum() < self.N:
            print(f'   [{label}] dropping {(~ok_mask).sum()} failed prior sims')
            self.prior = self.prior[ok_mask].reset_index(drop=True)
            Q_mat = Q_mat[ok_mask]
            C_mat = C_mat[ok_mask]
            self.N = len(self.prior)
        # Export
        self.prior_Q = Q_mat
        self.prior_C = C_mat

    # ──────────────────────────────────────────────────────────────────
    # SIR update primitives (Section 2.3 of the manuscript, steps 2-5)
    # ──────────────────────────────────────────────────────────────────

    def _log_likelihood(self, Q_sim, C_sim, t_end_hr, use_Q, use_C):
        """
        Sum Gaussian log-likelihood over the assimilation window.

        Eq. 6 of the manuscript:
            ln L_i = -1/2 sum_{t in W} [ (Q_s,i - Q_o)^2 / sigma_Q^2
                                        + (C_s,i - C_o)^2 / sigma_C^2 ]

        Inputs:
        -------
        Q_sim : np.ndarray (N x T)
            Simulated streamflow ensemble.
        C_sim : np.ndarray (N x T)
            Simulated chloride ensemble.
        t_end_hr : float
            Window end time (hours from rain start).
        use_Q : bool
            Include streamflow term in likelihood.
        use_C : bool
            Include chloride term in likelihood.

        Returns:
        --------
        log_w : np.ndarray (N,)
            Unnormalized log-likelihood per particle.
        n_used : int
            Number of likelihood-evaluated time steps.
        """
        # Import instance variables
        Q_obs = self._Q_obs
        C_obs = self._C_obs
        t_hr = self._t_hr
        mask = self._scoring_mask
        sigma_Q = self.sigma_Q
        sigma_C = self.sigma_C
        N = Q_sim.shape[0]
        # Identify failed prior simulations (NaN rows)
        valid = ~(np.isnan(Q_sim[:, 0]) | np.isnan(C_sim[:, 0]))
        log_w = np.zeros(N)
        n_used = 0
        # Sum over time steps inside the window AND inside the scoring mask
        for t in range(len(t_hr)):
            if not mask[t]:
                continue
            if t_hr[t] > t_end_hr:
                break
            if use_Q:
                diff = (Q_obs[t] - Q_sim[:, t]) ** 2
                log_w[valid] -= 0.5 * diff[valid] / sigma_Q ** 2
            if use_C:
                diff = (C_obs[t] - C_sim[:, t]) ** 2
                log_w[valid] -= 0.5 * diff[valid] / sigma_C ** 2
            n_used += 1
        # Failed particles get extremely low weight
        log_w[~valid] = -1e12
        return log_w, n_used

    @staticmethod
    def _normalize_weights(log_w):
        """
        Normalize log-weights to a probability vector.

        Subtract max for numerical stability, exponentiate, and divide
        by the sum. Falls back to uniform-on-valid if all weights
        underflow.
        """
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        if w.sum() < 1e-300:
            valid = log_w > -1e11
            w = np.where(valid, 1.0, 0.0)
        w = w / w.sum()
        return w

    def _systematic_resample(self, weights, N_new, rng):
        """
        Systematic resampling (Kitagawa, 1996).

        Generates N_new resampled indices with O(N_new) cost and lower
        Monte Carlo variance than multinomial resampling (Douc et al.,
        2005).

        Inputs:
        -------
        weights : np.ndarray (N,)
            Normalized particle weights.
        N_new : int
            Number of resampled particles to draw.
        rng : np.random.Generator
            Random generator for the single uniform offset.

        Returns:
        --------
        idx : np.ndarray (N_new,)
            Resampled particle indices into the original ensemble.
        """
        N = len(weights)
        # Single uniform offset, rest is deterministic
        positions = (np.arange(N_new) + rng.uniform()) / N_new
        cumw = np.cumsum(weights)
        idx = np.searchsorted(cumw, positions)
        idx = np.clip(idx, 0, N - 1)
        return idx

    def _perturb(self, ensemble_df, rng):
        """
        Apply Gaussian jitter to resampled particles in their native
        sampling space, clipped to prior bounds.

        Log-space parameters are perturbed in log space; linear-space
        parameters are perturbed in linear space. Jitter standard
        deviation is perturb_frac times the prior standard deviation
        in the corresponding space.
        """
        # Import instance variables
        prior = self.prior
        log_params = self.log_params
        perturb_frac = self.perturb_frac
        N_new = len(ensemble_df)
        new_df = ensemble_df.copy()
        for p in self.param_names:
            vals = prior[p].values
            pmin = float(np.nanmin(vals))
            pmax = float(np.nanmax(vals))
            if pmax <= pmin or not np.isfinite(pmax - pmin):
                continue
            if p in log_params and pmin > 0:
                # Log-space perturbation
                log_v = np.log(new_df[p].values.astype(float))
                log_std = float(np.nanstd(np.log(vals[vals > 0])))
                if log_std > 0:
                    noise = rng.normal(0, perturb_frac * log_std, N_new)
                    log_v = log_v + noise
                new_df[p] = np.exp(np.clip(log_v,
                                            np.log(pmin), np.log(pmax)))
            else:
                # Linear-space perturbation
                orig_std = float(np.nanstd(vals))
                if orig_std <= 0:
                    continue
                noise = rng.normal(0, perturb_frac * orig_std, N_new)
                new_df[p] = np.clip(new_df[p].values + noise, pmin, pmax)
        return new_df

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def step(self, t_end_hr, scenario='Q+Cl', N_new=None, n_workers=1):
        """
        Perform one SIR update for a single (window, scenario) combination.

        Implements the five-step SIR update of Section 2.3 of the
        manuscript:
            1. Prior simulation (assumed already done by simulate_prior)
            2. SIS log-likelihood weighting (Eq. 6)
            3. Effective sample size diagnostic (N_eff = 1 / sum w_i^2)
            4. Systematic resampling (Kitagawa, 1996)
            5. Perturbation and re-simulation

        Inputs:
        -------
        t_end_hr : float
            Assimilation window end time, in hours from rain start.
        scenario : str
            'Q-only' or 'Q+Cl'.
        N_new : int
            Posterior ensemble size. Defaults to the prior ensemble size.
        n_workers : int
            Number of parallel processes for re-simulation.

        Returns:
        --------
        key : str
            Result key '{tend}h__{scenario}', e.g. '10h__Q+Cl'.
        """
        # Validate scenario
        if scenario == 'Q-only':
            use_Q, use_C = True, False
        elif scenario == 'Q+Cl':
            use_Q, use_C = True, True
        else:
            raise ValueError(f"scenario must be 'Q-only' or 'Q+Cl', got '{scenario}'")
        # Validate prior simulation has been run
        if self.prior_Q is None:
            raise RuntimeError('Prior has not been simulated. Call simulate_prior() first.')
        # Default posterior size = prior size
        if N_new is None:
            N_new = self.N
        # Result key
        key = f'{int(round(t_end_hr)):02d}h__{scenario}'
        # Per-step RNG (reproducible across (window, scenario) combos)
        rng = np.random.default_rng(self.seed + hash(key) % (2 ** 31))
        print(f'\n  ── SIR  {key}  (DA = 0-{t_end_hr:.0f} h) ──')
        # Step 2: SIS log-likelihood weighting
        log_w, n_used = self._log_likelihood(
            self.prior_Q, self.prior_C, t_end_hr, use_Q, use_C)
        w = self._normalize_weights(log_w)
        # Step 3: Effective sample size diagnostic
        neff = metrics.effective_sample_size(w)
        print(f'     SIS: {n_used} obs points, N_eff = {neff:.0f} / {self.N}')
        # Step 4: Systematic resampling
        idx = self._systematic_resample(w, N_new, rng)
        new_df = self.prior.iloc[idx].reset_index(drop=True).copy()
        # Step 5: Perturbation
        new_df = self._perturb(new_df, rng)
        # Step 5 (continued): Re-simulation
        print(f'     Re-simulating {N_new} perturbed particles...')
        T = self.prior_Q.shape[1]
        Q_post = np.full((N_new, T), np.nan)
        C_post = np.full((N_new, T), np.nan)
        tasks = []
        for i in range(N_new):
            params_i = {p: float(new_df.iloc[i][p]) for p in self.param_names}
            tasks.append((i, params_i, self.storm_data, self.site))
        t0 = time.time()
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as exe:
                futures = [exe.submit(_simulate_one, t) for t in tasks]
                for f in as_completed(futures):
                    i, Qh, Ch = f.result()
                    if Qh is not None:
                        Q_post[i] = Qh
                        C_post[i] = Ch
        else:
            for task in tasks:
                i, Qh, Ch = _simulate_one(task)
                if Qh is not None:
                    Q_post[i] = Qh
                    C_post[i] = Ch
        n_ok = int(np.sum(~np.isnan(Q_post[:, 0])))
        print(f'     Re-sim: {n_ok}/{N_new} success, {time.time()-t0:.0f}s')
        # Export
        self.weights[key] = w
        self.N_eff[key] = float(neff)
        self.n_obs_used[key] = int(n_used)
        self.posterior_params[key] = new_df
        self.posterior_Q[key] = Q_post
        self.posterior_C[key] = C_post
        return key

    def run_sweep(self, snapshots, scenarios=('Q-only', 'Q+Cl'),
                  N_new=None, n_workers=1):
        """
        Run SIR updates for every (window, scenario) combination.

        Inputs:
        -------
        snapshots : iterable of float
            Assimilation window end times in hours from rain start.
            The manuscript uses {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12,
            14, 16, 19} h.
        scenarios : iterable of str
            Subset of {'Q-only', 'Q+Cl'}.
        N_new : int
            Posterior ensemble size per (window, scenario).
        n_workers : int
            Number of parallel processes.
        """
        for t_end_hr in snapshots:
            for scenario in scenarios:
                self.step(t_end_hr, scenario=scenario,
                          N_new=N_new, n_workers=n_workers)

    def save(self, path):
        """
        Write all results to a single .npz archive.

        The archive contains:
            - storm_data, site (as object arrays)
            - prior_params, prior_Q, prior_C
            - For each (window, scenario) key:
                {key}_w        : SIS weights (N,)
                {key}_neff     : effective sample size (scalar)
                {key}_n_used   : likelihood-evaluated time steps (int)
                {key}_params   : posterior parameter samples (N x P)
                {key}_Qpost    : posterior streamflow ensemble (N x T)
                {key}_Cpost    : posterior chloride ensemble (N x T)

        Inputs:
        -------
        path : str or pathlib.Path
            Output file path.
        """
        save_dict = {
            'param_names':   np.array(self.param_names, dtype=object),
            'prior_params':  self.prior[self.param_names].values,
            'prior_Q':       self.prior_Q,
            'prior_C':       self.prior_C,
            'Q_obs':         self._Q_obs,
            'C_obs':         self._C_obs,
            't_hr':          self._t_hr,
            'scoring_mask':  self._scoring_mask,
            'sigma_Q':       self.sigma_Q,
            'sigma_C':       self.sigma_C,
            'N':             self.N,
        }
        for key in self.posterior_params:
            tag = key.replace('-', '_').replace('+', '_')
            save_dict[f'{tag}_w']      = self.weights[key]
            save_dict[f'{tag}_neff']   = self.N_eff[key]
            save_dict[f'{tag}_n_used'] = self.n_obs_used[key]
            save_dict[f'{tag}_params'] = self.posterior_params[key][self.param_names].values
            save_dict[f'{tag}_Qpost']  = self.posterior_Q[key]
            save_dict[f'{tag}_Cpost']  = self.posterior_C[key]
        np.savez(path, **save_dict)
