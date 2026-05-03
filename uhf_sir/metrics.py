import numpy as np


def nse(sim, obs):
    """
    Nash-Sutcliffe efficiency.

    Inputs:
    -------
    sim : np.ndarray
        Simulated time series.
    obs : np.ndarray
        Observed time series.

    Returns:
    --------
    float
        NSE in (-inf, 1]. NSE = 1 is a perfect fit.
    """
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() < 2:
        return np.nan
    sim = sim[mask]
    obs = obs[mask]
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - obs.mean()) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - num / den


def kge(sim, obs):
    """
    Kling-Gupta efficiency

    Inputs:
    -------
    sim : np.ndarray
        Simulated time series.
    obs : np.ndarray
        Observed time series.

    Returns:
    --------
    float
        KGE in (-inf, 1]. KGE = 1 is a perfect fit.
    """
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() < 2:
        return np.nan
    sim = sim[mask]
    obs = obs[mask]
    if obs.std() <= 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def iqr_reduction(prior, posterior):
    """
    Relative reduction in interquartile range from prior to posterior.

    Inputs:
    -------
    prior : np.ndarray
        Prior parameter samples.
    posterior : np.ndarray
        Posterior parameter samples.

    Returns:
    --------
    float
        IQR reduction in percent. Positive values indicate posterior
        sharpening; negative values indicate posterior broadening or
        prior misspecification.
    """
    q_prior = np.percentile(prior, [25, 75])
    q_post = np.percentile(posterior, [25, 75])
    iqr_prior = q_prior[1] - q_prior[0]
    iqr_post = q_post[1] - q_post[0]
    if iqr_prior <= 0:
        return 0.0
    return 100.0 * (1.0 - iqr_post / iqr_prior)


def tracer_gain(prior, post_q_only, post_q_cl):
    """
    Marginal IQR reduction contributed by chloride observations.

    Inputs:
    -------
    prior : np.ndarray
        Prior parameter samples.
    post_q_only : np.ndarray
        Posterior under streamflow-only assimilation.
    post_q_cl : np.ndarray
        Posterior under joint streamflow-chloride assimilation.

    Returns:
    --------
    float
        Tracer gain in percentage points. Positive values identify
        parameters for which chloride adds identifiability beyond
        streamflow alone.
    """
    return iqr_reduction(prior, post_q_cl) - iqr_reduction(prior, post_q_only)


def effective_sample_size(weights):
    """
    Effective sample size of a weighted particle ensemble.

    Inputs:
    -------
    weights : np.ndarray
        Normalized particle weights (sum to 1).

    Returns:
    --------
    float
        N_eff = 1 / sum(w_i^2). Equals N for uniform weights and
        approaches 1 for fully-collapsed weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return 1.0 / np.sum(weights ** 2)
