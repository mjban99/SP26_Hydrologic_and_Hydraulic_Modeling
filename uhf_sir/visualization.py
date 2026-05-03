import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# Parameter groups for ordered display in heatmaps
PARAM_GROUPS = {
    'Infiltration':   ['K_s', 'psi', 'theta_d'],
    'Interflow':      ['K_ex'],
    'Baseflow':       ['baseflow_Q'],
    'Mixing':         ['Q_threshold', 'gamma', 'alpha_soil', 'V_soil', 'lambda_sat'],
    'Soil state':     ['theta_d', 'f_sat', 'S_max', 'tau_drain'],
    'Channel/Timing': ['n', 'uh_stretch'],
}

PARAM_DESCS = {
    'K_s':         'Saturated hydraulic conductivity',
    'K_ex':        'Interflow exfiltration coefficient',
    'baseflow_Q':  'Baseline groundwater discharge',
    'Q_threshold': 'Flow level at which mixing transitions',
    'alpha_soil':  'Soil-to-stream Cl scaling',
    'V_soil':      'Soil mixing pool depth',
    'gamma':       'Mixing transition sharpness',
    'lambda_sat':  'Max state-dependent mixing weight',
    'uh_stretch':  'UH stretch factor',
    'tau_drain':   'Drainage timescale',
    'psi':         'Wetting-front suction head',
    'theta_d':     'Initial soil-moisture deficit',
    'f_sat':       'Saturated area fraction',
    'S_max':       'Max shallow-storage depth',
    'n':           "Manning's roughness",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _band(mat):
    """Return (p2.5, p50, p97.5) along axis 0, or (None, None, None) if empty."""
    if mat.size == 0 or np.all(np.isnan(mat)):
        return None, None, None
    return (np.nanpercentile(mat, 2.5,  axis=0),
            np.nanpercentile(mat, 50,   axis=0),
            np.nanpercentile(mat, 97.5, axis=0))


def _iqr_reduction(prior, posterior):
    """Inline IQR reduction (avoid metrics import to keep this module light)."""
    q_prior = np.percentile(prior, [25, 75])
    q_post = np.percentile(posterior, [25, 75])
    iqr_prior = q_prior[1] - q_prior[0]
    iqr_post = q_post[1] - q_post[0]
    if iqr_prior <= 0:
        return 0.0
    return 100.0 * (1.0 - iqr_post / iqr_prior)


# ──────────────────────────────────────────────────────────────────────
# Figure 1 - Catchment outline + empirical UH (manuscript Figure 1)
# ──────────────────────────────────────────────────────────────────────

def plot_catchment_and_uh(uh_t_hr, uh_hr_inv, ax=None):
    """
    Plot the DEM-based empirical unit hydrograph.

    The catchment outline (left panel of manuscript Figure 1) is
    produced from a GIS shapefile and is not reproduced here. This
    function plots the right panel: u(t) ordinates against travel
    time, with the peak time annotated.

    Inputs:
    -------
    uh_t_hr : np.ndarray
        UH time points (hr).
    uh_hr_inv : np.ndarray
        UH ordinates (1/hr); the integral over uh_t_hr is 1.
    ax : matplotlib axis, optional
        Axis to plot into. If None, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    # Plot UH
    ax.fill_between(uh_t_hr, 0, uh_hr_inv, color='#2c7fb8', alpha=0.4,
                    label='DEM empirical UH (integral = 1)')
    ax.plot(uh_t_hr, uh_hr_inv, color='#2c7fb8', lw=1.5)
    # Annotate peak
    peak_idx = int(np.argmax(uh_hr_inv))
    t_peak = float(uh_t_hr[peak_idx])
    ax.axvline(t_peak, color='#d7191c', ls=':', lw=1, alpha=0.7)
    ax.scatter([t_peak], [uh_hr_inv[peak_idx]], color='#d7191c', s=40,
               zorder=5, label=f'peak at t = {t_peak:.2f} h')
    # Cosmetics
    ax.set_xlabel('Travel time t (hr)')
    ax.set_ylabel('UH ordinate u(t) (1/hr)')
    ax.set_xlim(0, max(uh_t_hr))
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ──────────────────────────────────────────────────────────────────────
# Figure 3 - SIR posterior envelopes (manuscript Figure 3)
# ──────────────────────────────────────────────────────────────────────

def plot_sir_envelopes(npz, max_window=12.0, mode='forecast',
                       show_tmin=-5, show_tmax=40, fig=None):
    """
    Plot 2x2 grid of SIR posterior envelopes across expanding windows.

    Layout:
        (a) Q-only DA - streamflow envelope
        (b) Q-only DA - chloride envelope
        (c) Q+Cl  DA - streamflow envelope
        (d) Q+Cl  DA - chloride envelope

    Inputs:
    -------
    npz : dict-like
        Loaded SIR_sweep NPZ (see SIR.save()).
    max_window : float
        Longest DA window to plot (hours). Windows beyond this are
        likely degenerate (N_eff < 10).
    mode : str
        'forecast' plots only t > T for each window (post-window
        prediction); 'full' plots the entire timeseries.
    show_tmin, show_tmax : float
        Display range in hours from rain start.
    fig : matplotlib figure, optional

    Returns:
    --------
    fig : matplotlib figure
    """
    # Import npz fields
    snapshots = dict(npz['snapshots']) if 'snapshots' in npz.files else None
    if snapshots is None:
        # Fallback: infer snapshots from key names in the new SIR.save() format
        snap_keys = [k for k in npz.files if k.endswith('_neff')]
        snapshots = {}
        for k in snap_keys:
            sname = k.split('__')[0]
            T = float(int(sname.replace('h', '')))
            snapshots[sname] = T
    t_hr = npz['t_hr']
    Q_obs = npz['Q_obs']
    C_obs_h = npz['C_obs_hourly'] if 'C_obs_hourly' in npz.files else npz['C_obs']
    # Sort and filter snapshots
    snap_items = sorted([(k, float(v)) for k, v in snapshots.items()],
                         key=lambda x: x[1])
    snap_items = [(k, v) for k, v in snap_items if v <= max_window]
    # Color gradient: light at small T, dark at large T
    T_max = max(v for _, v in snap_items)
    norms = [v / T_max for _, v in snap_items]
    # Figure
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    else:
        axes = fig.subplots(2, 2, sharex=True)
    in_show = (t_hr >= show_tmin) & (t_hr <= show_tmax)
    scenario_rows = {'Q-only': 0, 'Q+Cl': 1}
    # Plot
    for scen, row in scenario_rows.items():
        ax_Q = axes[row, 0]
        ax_C = axes[row, 1]
        for i, (sname, T) in enumerate(snap_items):
            tag = f'{sname}__{scen}'.replace('-', '_').replace('+', '_')
            key_Q = f'{tag}_Qpost'
            key_C = f'{tag}_Cpost'
            if key_Q not in npz.files:
                continue
            Q_post = npz[key_Q]
            C_post = npz[key_C]
            mask_fc = (t_hr > T) if mode == 'forecast' else np.ones_like(t_hr, dtype=bool)
            mask_show = mask_fc & in_show
            Qp2, _, Qp97 = _band(Q_post[:, mask_show])
            Cp2, _, Cp97 = _band(C_post[:, mask_show])
            if Qp2 is None:
                continue
            t_fc = t_hr[mask_show]
            col = cm.Blues(norms[i])
            ax_Q.fill_between(t_fc, Qp2, Qp97, color=col, lw=0)
            ax_C.fill_between(t_fc, Cp2, Cp97, color=col, lw=0)
        # Observations
        ax_Q.plot(t_hr[in_show], Q_obs[in_show], color='#d7191c', lw=1.2,
                   label='Obs Q')
        ax_C.plot(t_hr[in_show], C_obs_h[in_show], color='#d7191c', lw=1.2,
                   label='Obs Cl (1h interp)')
        # Cosmetics
        panel_letters = {0: ('a', 'b'), 1: ('c', 'd')}[row]
        ax_Q.set_title(f'({panel_letters[0]}) {scen} DA - Streamflow', loc='left')
        ax_C.set_title(f'({panel_letters[1]}) {scen} DA - Chloride', loc='left')
        ax_Q.set_ylabel('Q (m^3/s)')
        ax_C.set_ylabel('C (mg/L)')
        ax_Q.legend(loc='upper right', fontsize=8)
        ax_C.legend(loc='upper right', fontsize=8)
        ax_Q.grid(alpha=0.25)
        ax_C.grid(alpha=0.25)
    axes[1, 0].set_xlabel('Time t (hr from rain start)')
    axes[1, 1].set_xlabel('Time t (hr from rain start)')
    fig.suptitle('Posterior CI evolution as DA window expands',
                  fontsize=12, y=1.00)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Figure 4 / Table B1 - IQR-reduction heatmap (manuscript Figure 4 inset)
# ──────────────────────────────────────────────────────────────────────

def plot_iqr_heatmap(npz, max_window=10.0, fig=None):
    """
    Plot the IQR-reduction heatmap and tracer-gain heatmap (manuscript
    Figure 4 inset C, Table B1 source).

    Two side-by-side panels:
        Left  : IQR reduction under Q+Cl assimilation
        Right : Tracer gain = IQR_red(Q+Cl) - IQR_red(Q-only)

    The valid-window restriction (default 10 h) follows the
    N_eff >= 10 criterion of Section 3.2.

    Inputs:
    -------
    npz : dict-like
        Loaded SIR_sweep NPZ.
    max_window : float
        Longest DA window to include in the heatmap. Default 10 h.
    fig : matplotlib figure, optional

    Returns:
    --------
    fig : matplotlib figure
    """
    # Import npz fields
    param_names = list(npz['param_names'])
    param_names = [str(p) for p in param_names]
    prior_params = npz['prior_params']
    snapshots = dict(npz['snapshots']) if 'snapshots' in npz.files else {}
    if not snapshots:
        snap_keys = [k for k in npz.files if k.endswith('_neff')]
        for k in snap_keys:
            sname = k.split('__')[0]
            snapshots[sname] = float(int(sname.replace('h', '')))
    # Order parameters by group
    ordered_params = []
    for g, plist in PARAM_GROUPS.items():
        for p in plist:
            if p in param_names and p not in ordered_params:
                ordered_params.append(p)
    ordered_idx = [param_names.index(p) for p in ordered_params]
    # Filter snapshots within valid window
    snap_items = sorted([(k, float(v)) for k, v in snapshots.items()],
                         key=lambda x: x[1])
    snap_items = [(k, v) for k, v in snap_items if v <= max_window]
    snap_hours = [v for _, v in snap_items]
    n_snap = len(snap_items)
    n_par = len(ordered_params)
    # Compute IQR-reduction matrices
    mats = {}
    for scen in ('Q-only', 'Q+Cl'):
        M = np.full((n_par, n_snap), np.nan)
        for j, (sname, _) in enumerate(snap_items):
            tag = f'{sname}__{scen}'.replace('-', '_').replace('+', '_')
            key = f'{tag}_params'
            if key not in npz.files:
                continue
            post = npz[key]
            for i, pi in enumerate(ordered_idx):
                pr_vals = prior_params[:, pi]
                po_vals = post[:, pi]
                po_vals = po_vals[~np.isnan(po_vals)]
                if po_vals.size >= 2:
                    M[i, j] = _iqr_reduction(pr_vals, po_vals)
        mats[scen] = M
    # Figure
    if fig is None:
        fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(14, 8))
    else:
        ax_L, ax_R = fig.subplots(1, 2)
    panels = [
        (ax_L, 'IQR reduction (Q+Cl)',
         mats['Q+Cl'], plt.get_cmap('RdBu_r'),
         mcolors.TwoSlopeNorm(0, -40, 80), 'IQR reduction [%]'),
        (ax_R, 'Tracer gain (Q+Cl) - (Q-only)',
         mats['Q+Cl'] - mats['Q-only'], plt.get_cmap('PiYG'),
         mcolors.TwoSlopeNorm(0, -40, 40), 'D IQR reduction [pp]'),
    ]
    for idx, (ax, title, M, cmap, norm, cb_label) in enumerate(panels):
        im = ax.imshow(M, aspect='auto', cmap=cmap, norm=norm,
                        interpolation='nearest', origin='upper')
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(n_snap))
        ax.set_xticklabels([f'{int(h)}' for h in snap_hours], fontsize=9)
        ax.set_xlabel('DA window (h from rain start)')
        if idx == 0:
            labels = [f'{p}  |  {PARAM_DESCS.get(p, "")}' for p in ordered_params]
        else:
            labels = ordered_params
        ax.set_yticks(range(n_par))
        ax.set_yticklabels(labels, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cb.set_label(cb_label, fontsize=10)
        # Annotate cells with the value
        for i in range(n_par):
            for j in range(n_snap):
                if not np.isnan(M[i, j]):
                    txt_color = 'white' if abs(M[i, j]) >= 20 else 'black'
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center',
                            va='center', fontsize=8, color=txt_color)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Figure D1 - V_soil tertile decomposition (manuscript Figure D1)
# ──────────────────────────────────────────────────────────────────────

def plot_vsoil_tertile(npz, t_end_hr=10.0, fig=None):
    """
    Plot the V_soil tertile decomposition of the t_end_hr Q+Cl posterior.

    Three panels:
        (a) Stream chloride trajectory medians + 5-95% bands by tertile
        (b) alpha_soil posterior distribution by tertile
        (c) Q_threshold posterior distribution by tertile

    Tertile boundaries are set at the 33rd and 67th percentiles of
    the posterior V_soil distribution.

    Inputs:
    -------
    npz : dict-like
        Loaded SIR_sweep NPZ.
    t_end_hr : float
        Posterior window to partition. Default 10 h.
    fig : matplotlib figure, optional

    Returns:
    --------
    fig : matplotlib figure
    """
    # Import npz fields
    param_names = [str(p) for p in npz['param_names']]
    sname = f'{int(round(t_end_hr)):02d}h'
    tag = f'{sname}__Q_Cl'
    post_params = npz[f'{tag}_params']
    post_C = npz[f'{tag}_Cpost']
    t_hr = npz['t_hr']
    C_obs_h = npz['C_obs_hourly'] if 'C_obs_hourly' in npz.files else npz['C_obs']
    pre_Cl = float(C_obs_h[t_hr < 0][-24:].mean()) if (t_hr < 0).any() else 5.5
    # Partition into tertiles
    i_v = param_names.index('V_soil')
    i_a = param_names.index('alpha_soil')
    i_q = param_names.index('Q_threshold')
    V = post_params[:, i_v]
    a = post_params[:, i_a]
    q = post_params[:, i_q]
    t1, t2 = np.percentile(V, [33.33, 66.67])
    fast_mask = V <= t1
    med_mask = (V > t1) & (V <= t2)
    slow_mask = V > t2
    # Figure
    if fig is None:
        fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax_top = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_q = fig.add_subplot(gs[1, 1])
    # Display range
    in_show = (t_hr >= 0) & (t_hr <= 84)
    t_show = t_hr[in_show]
    # Panel (a): stream Cl trajectory by tertile
    colors = {'Fast': '#2b8cbe', 'Medium': '#7f7f7f', 'Slow': '#c0392b'}
    for label, mask in [('Fast', fast_mask), ('Medium', med_mask), ('Slow', slow_mask)]:
        sub = post_C[mask][:, in_show]
        if sub.size == 0:
            continue
        med = np.nanmedian(sub, axis=0)
        lo = np.nanpercentile(sub, 5, axis=0)
        hi = np.nanpercentile(sub, 95, axis=0)
        ax_top.fill_between(t_show, lo, hi, color=colors[label], alpha=0.25, lw=0)
        n = int(mask.sum())
        if label == 'Fast':
            rng_str = f'V_soil <= {t1:.2f} m'
        elif label == 'Medium':
            rng_str = f'{t1:.2f} < V_soil <= {t2:.2f} m'
        else:
            rng_str = f'V_soil > {t2:.2f} m'
        ax_top.plot(t_show, med, color=colors[label], lw=1.6,
                     label=f'{label} ({rng_str}, n={n})')
    # Observed stream Cl markers (raw 7-h grab samples if available)
    if 't_obs_hr' in npz.files and 'C_obs_raw' in npz.files:
        t_obs_show = npz['t_obs_hr']
        C_obs_show = npz['C_obs_raw']
        m_obs = (t_obs_show >= 0) & (t_obs_show <= 84)
        ax_top.scatter(t_obs_show[m_obs], C_obs_show[m_obs], color='black',
                        s=30, zorder=5, label='Observed stream Cl')
    ax_top.axhline(pre_Cl, color='gray', ls='--', lw=1, alpha=0.7,
                    label=f'Pre-storm baseline ({pre_Cl:.1f} mg/L)')
    ax_top.set_xlabel('Hours after rain onset')
    ax_top.set_ylabel('Stream chloride (mg/L)')
    ax_top.set_title(f'(a) Chloride trajectory by V_soil tertile  (t_end = {t_end_hr:.0f} h Q+Cl posterior)',
                      loc='left')
    ax_top.legend(fontsize=8, loc='lower right')
    ax_top.grid(alpha=0.3)
    # Panel (b): alpha_soil distribution by tertile
    bins_a = np.linspace(a.min(), a.max(), 20)
    for label, mask in [('Fast', fast_mask), ('Medium', med_mask), ('Slow', slow_mask)]:
        ax_a.hist(a[mask], bins=bins_a, color=colors[label], alpha=0.5, label=label)
    ax_a.set_xlabel('alpha_soil')
    ax_a.set_ylabel('Particle count')
    ax_a.set_title('(b) alpha_soil distribution by tertile', loc='left')
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)
    # Panel (c): Q_threshold distribution by tertile
    bins_q = np.linspace(q.min(), q.max(), 20)
    for label, mask in [('Fast', fast_mask), ('Medium', med_mask), ('Slow', slow_mask)]:
        ax_q.hist(q[mask], bins=bins_q, color=colors[label], alpha=0.5, label=label)
    ax_q.set_xlabel('Q_threshold (m^3/s)')
    ax_q.set_ylabel('Particle count')
    ax_q.set_title('(c) Q_threshold distribution by tertile', loc='left')
    ax_q.legend(fontsize=8)
    ax_q.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Figure E1 - Rainfall + stream chloride mass balance (manuscript Figure E1)
# ──────────────────────────────────────────────────────────────────────

def plot_mass_balance(storm_data, cl_rain_t, cl_rain, cl_stream_t, cl_stream,
                       pre_Cl=5.5, fig=None):
    """
    Plot rainfall + stream chloride for a single storm (manuscript Figure E1).

    Two stacked panels:
        (a) Rainfall depth (bars, mm/h) and rainfall Cl (markers, mg/L)
        (b) Streamflow (line, m^3/s) and stream Cl (markers, mg/L)

    Inputs:
    -------
    storm_data : dict
        Storm-data dict accepted by Model.
    cl_rain_t : np.ndarray
        Rainfall Cl sample times (hours from rain start).
    cl_rain : np.ndarray
        Rainfall Cl concentrations (mg/L).
    cl_stream_t : np.ndarray
        Stream Cl sample times (hours from rain start).
    cl_stream : np.ndarray
        Stream Cl concentrations (mg/L).
    pre_Cl : float
        Pre-storm stream Cl baseline (mg/L). Default 5.5 (storm 33).
    fig : matplotlib figure, optional

    Returns:
    --------
    fig : matplotlib figure
    """
    # Time axis: hours from rain start, restricted to [0, 84]
    t_h = (storm_data['t_Q_hourly'] - storm_data['t_Q_hourly'][0]) / 3600.0
    in_win = (t_h >= 0) & (t_h <= 84)
    rain_in = (cl_rain_t >= 0) & (cl_rain_t <= 84)
    stream_in = (cl_stream_t >= 0) & (cl_stream_t <= 84)
    # Rainfall-weighted mean rainfall Cl (per hour of rain)
    rain_flux_h = storm_data['rain_flux'][in_win]
    rain_t_h = t_h[in_win]
    weights, concs = [], []
    for t_s, c_s in zip(cl_rain_t[rain_in], cl_rain[rain_in]):
        bin_mask = (rain_t_h >= t_s - 3.5) & (rain_t_h < t_s + 3.5)
        rain_in_bin = rain_flux_h[bin_mask].sum()
        if rain_in_bin > 0:
            weights.append(rain_in_bin)
            concs.append(c_s)
    if weights:
        wmean = np.sum(np.array(weights) * np.array(concs)) / np.sum(weights)
    else:
        wmean = np.nan
    # Figure
    if fig is None:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8),
                                               sharex=True,
                                               gridspec_kw={'hspace': 0.3})
    else:
        ax_top, ax_bot = fig.subplots(2, 1, sharex=True)
    # Colors
    col_rain = '#3498db'
    col_rain_C = '#1f3a93'
    col_Q = '#34495e'
    col_str_C = '#c0392b'
    col_base = '#7f7f7f'
    # Panel (a) - rainfall + rainfall Cl
    ax_top.bar(rain_t_h, rain_flux_h, width=0.9, color=col_rain,
                edgecolor='none', alpha=0.7, label='Rainfall (hourly)')
    ax_top.set_ylabel('Rainfall (mm/h)', color=col_rain)
    ax_top.tick_params(axis='y', labelcolor=col_rain)
    ax_top.invert_yaxis()
    ax_top_r = ax_top.twinx()
    ax_top_r.scatter(cl_rain_t[rain_in], cl_rain[rain_in], color=col_rain_C,
                      s=50, zorder=5, edgecolor='white', linewidth=0.8,
                      label='Rainfall Cl (7h)')
    ax_top_r.set_ylabel('Rainfall Cl (mg/L)', color=col_rain_C)
    ax_top_r.tick_params(axis='y', labelcolor=col_rain_C)
    ax_top_r.set_ylim(0, max(16.0, float(cl_rain[rain_in].max()) * 1.15))
    ax_top.set_title(f'(a) Rainfall depth and rainfall Cl  '
                       f'(rainfall-weighted mean Cl = {wmean:.1f} mg/L)',
                       loc='left', fontsize=11)
    # Panel (b) - streamflow + stream Cl
    ax_bot.plot(t_h[in_win], storm_data.get('Q_obs',
                np.full(in_win.sum(), np.nan))[in_win] if 'Q_obs' in storm_data
                else np.full(in_win.sum(), np.nan),
                color=col_Q, lw=1.5, label='Streamflow (hourly)')
    ax_bot.set_xlabel('Hours after rain onset')
    ax_bot.set_ylabel('Streamflow (m^3/s)', color=col_Q)
    ax_bot.tick_params(axis='y', labelcolor=col_Q)
    ax_bot_r = ax_bot.twinx()
    ax_bot_r.scatter(cl_stream_t[stream_in], cl_stream[stream_in],
                      color=col_str_C, s=50, zorder=5,
                      edgecolor='white', linewidth=0.8,
                      label='Stream Cl (observed)')
    ax_bot_r.axhline(pre_Cl, color=col_base, ls='--', lw=1, alpha=0.85,
                       label=f'Pre-storm baseline ({pre_Cl:.1f} mg/L)')
    ax_bot_r.set_ylabel('Stream Cl (mg/L)', color=col_str_C)
    ax_bot_r.tick_params(axis='y', labelcolor=col_str_C)
    ax_bot_r.set_ylim(3.0, 6.5)
    ax_bot.set_title(f'(b) Streamflow and stream Cl  '
                       f'(observed trough = {float(cl_stream[stream_in].min()):.2f} mg/L)',
                       loc='left', fontsize=11)
    ax_bot.set_xlim(0, 84)
    ax_bot.grid(alpha=0.3)
    ax_top.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig
