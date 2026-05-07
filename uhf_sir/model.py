import copy
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class Model():
    """
    Coupled hillslope-channel chloride-aided rainfall-runoff model for
    the Upper Hafren catchment, Plynlimon, Wales.

    The hillslope module produces three runoff pathways from rainfall:
    Green-Ampt infiltration excess (Ban, 2026, Eq. A.1-A.2), interflow
    exfiltration from cumulative infiltration (Eq. A.4), and
    saturation-excess overflow from a shallow storage tank (Eq. A.3).
    A continuously stirred soil mixing pool tracks chloride
    (Eq. A.5). The total surface runoff is convolved with an empirical
    DTM-derived unit hydrograph to produce hillslope discharge
    Q_hill(t), which is distributed uniformly along the channel as
    lateral inflow q_L(t) = Q_hill / L. The channel routes water via
    a 1D Saint-Venant diffusion-wave equation (Eq. A.9-A.11; Q is
    diagnostic from h via the algebraic Manning closure under
    S_f = S_0 - dh/dx) and chloride via a 1D advection-dispersion
    equation (Eq. A.13). Two-endmember mixing combines high-Cl
    baseflow with the soil-pool Cl through a sigmoidal Q-dependent
    weight (Eq. A.6-A.8) with optional state-dependent attenuation by
    shallow saturation.

    The model is fifteen-parameter (Model B in the manuscript). A
    simplified variant (Appendix C of the manuscript) is reproduced
    by passing ``force_alpha_one=True`` to ``simulate`` together with
    K_ex = 0 in the parameter dictionary.

    Inputs:
    -------
    storm_data : dict
        Storm-specific input data. The following keys are required:

        |--------------+---------------+-------+----------------------|
        | Key          | Type          | Unit  | Description          |
        |--------------+---------------+-------+----------------------|
        | rain_t       | np.ndarray    | s     | Hourly forcing time  |
        | rain_flux    | np.ndarray    | mm/hr | Rainfall intensity   |
        | rain_C       | np.ndarray    | mg/L  | Rainfall chloride    |
        | t_Q_hourly   | np.ndarray    | s     | Hourly output time   |
        | pre_Cl       | float         | mg/L  | Pre-storm stream Cl  |
        |--------------+---------------+-------+----------------------|

    parameters : dict
        Sixteen calibration parameters. Required keys:

        |-------------+-------+-------+----------------------------------|
        | Key         | Type  | Unit  | Description                      |
        |-------------+-------+-------+----------------------------------|
        | K_s         | float | m/s   | Saturated hydraulic conductivity |
        | psi         | float | m     | Wetting-front suction head       |
        | theta_d     | float | -     | Initial soil-moisture deficit    |
        | K_ex        | float | 1/s   | Interflow exfiltration rate      |
        | f_sat       | float | -     | Saturated area fraction          |
        | S_max       | float | m     | Max shallow-storage depth        |
        | tau_drain   | float | s     | Shallow drainage timescale       |
        | uh_stretch  | float | -     | UH time-stretch factor           |
        | n           | float | -     | Manning's roughness              |
        | V_soil      | float | m     | Soil mixing pool depth           |
        | alpha_soil  | float | -     | Soil-pool initialization scale   |
        | Q_threshold | float | m^3/s | Mixing transition discharge      |
        | gamma       | float | -     | Mixing transition sharpness      |
        | lambda_sat  | float | -     | State-dep mixing strength        |
        | baseflow_Q  | float | m^3/s | Baseline groundwater discharge   |
        |-------------+-------+-------+----------------------------------|

    site : dict
        Site-specific fixed quantities. The following keys are required:

        |---------------------+-------+------+------------------------|
        | Key                 | Type  | Unit | Description            |
        |---------------------+-------+------+------------------------|
        | catchment_area_km2  | float | km^2 | Catchment drainage area|
        | L_channel           | float | m    | Channel length         |
        | B_width             | float | m    | Channel width          |
        | S_0                 | float | -    | Channel bed slope      |
        | uh_t_hr             | array | hr   | Empirical UH times     |
        | uh_hr_inv           | array | 1/hr | Empirical UH ordinates |
        |---------------------+-------+------+------------------------|

    N : int
        Number of channel cells (default 20).

    dt_fine : float
        Fine time step for hillslope ODE integration (default 300 s).

    Methods:
    --------
    simulate : Run the full forward simulation
    extract_at : Interpolate outlet Q, C onto a target time grid

    Attributes:
    -----------
    # Hillslope state (on dt_fine grid)
    F_t       : Cumulative infiltration depth (m)
    S_t       : Shallow storage depth (m)
    q_GA      : Green-Ampt-domain surface runoff rate (m/s)
    q_SAT     : Saturation-excess-domain surface runoff rate (m/s)
    q_surf    : Total surface runoff rate to UH (m/s) - Eq. (1)
    Q_hill    : Hillslope discharge after UH convolution (m^3/s)
    C_soil    : Soil-pool chloride concentration (mg/L)
    S_sat_norm: Normalized shallow storage in [0, 1]

    # Channel state (on hourly grid, at outlet cell)
    Q_out     : Outlet streamflow (m^3/s)
    C_out     : Outlet stream chloride (mg/L)

    # Bookkeeping
    t_fine    : Fine-grid time vector (s)
    t_hourly  : Hourly time vector (s)
    success   : True after a successful simulate() call
    """
    # Constants
    G = 9.81                               # gravitational acceleration (m/s^2)
    F0 = 1e-6                              # initial cumulative infiltration (m)
    S0_FRAC = 0.5                          # initial shallow-storage filling fraction
    D_0 = 0.05                             # base dispersion coefficient (m^2/s)
    BETA_DISP = 0.5                        # dispersion-Q exponent
    Q_REF = 1.0                            # dispersion reference discharge (m^3/s)
    K_DEC = 0.0                            # chloride decay rate (1/s; conservative tracer)

    def __init__(self, storm_data, parameters, site, N=20, dt_fine=300.0):
        # Copy input dicts to prevent modification
        storm_data = copy.deepcopy(storm_data)
        parameters = copy.deepcopy(parameters)
        site = copy.deepcopy(site)
        # Save copied inputs to instance
        self.storm_data = storm_data
        self.parameters = parameters
        self.site = site
        # Cast forcing arrays for stability
        self._rain_t = np.asarray(storm_data['rain_t'], dtype=np.float64)
        self._rain_flux = np.asarray(storm_data['rain_flux'], dtype=np.float64)
        self._rain_C = np.asarray(storm_data['rain_C'], dtype=np.float64)
        self._t_hourly = np.asarray(storm_data['t_Q_hourly'], dtype=np.float64)
        self._pre_Cl = float(storm_data['pre_Cl'])
        # Geometry
        self.area = float(site['catchment_area_km2']) * 1e6        # m^2
        self.L = float(site['L_channel'])                          # m
        self.B = float(site['B_width'])                            # m
        self.S_0 = float(site['S_0'])                              # -
        self.dx = self.L / N
        self.N = int(N)
        self.dt_fine = float(dt_fine)
        # Empirical UH (DEM-derived; integrates to 1 in 1/hr)
        self._uh_t_hr = np.asarray(site['uh_t_hr'], dtype=np.float64)
        self._uh_hr_inv = np.asarray(site['uh_hr_inv'], dtype=np.float64)
        # Fine grid
        t_end = float(max(self._t_hourly[-1], self._rain_t[-1]))
        self.t_fine = np.arange(0.0, t_end + dt_fine, dt_fine)
        self.t_hourly = self._t_hourly
        # Initial soil-pool concentration (Eq. 4 / Eq. A.5 initial condition)
        self._C_soil_0 = float(parameters['alpha_soil']) * self._pre_Cl
        # Initialize state arrays (filled by simulate)
        self.F_t = None
        self.S_t = None
        self.q_GA = None
        self.q_SAT = None
        self.q_surf = None
        self.Q_hill = None
        self.C_soil = None
        self.S_sat_norm = None
        self.Q_out = None
        self.C_out = None
        self.success = False

    # ──────────────────────────────────────────────────────────────────
    # Forcing interpolators
    # ──────────────────────────────────────────────────────────────────

    def _p_func(self, t):
        """Rainfall intensity at time t (m/s)."""
        p_mm_hr = float(np.interp(t, self._rain_t, self._rain_flux,
                                   left=0.0, right=0.0))
        return max(p_mm_hr, 0.0) * 1e-3 / 3600.0

    def _C_rain_func(self, t):
        """Rainfall chloride at time t (mg/L)."""
        return float(np.interp(t, self._rain_t, self._rain_C))

    # ──────────────────────────────────────────────────────────────────
    # Hillslope module: Eq. A.1-A.5 of the manuscript
    # ──────────────────────────────────────────────────────────────────

    def _solve_green_ampt(self):
        """
        Solve Green-Ampt cumulative infiltration ODE with interflow recovery.

        Eq. A.1: dF/dt = min(K_s (psi theta_d + F)/F, p(t)) - K_ex F
        Eq. A.2: q_GA(t) = max(0, p(t) - K_s (psi theta_d + F)/F)
        Eq. A.4: g(t)    = K_ex F(t)

        The total Green-Ampt-domain surface runoff is q_GA + g, with
        q_GA the infiltration-excess component and g the
        interflow-exfiltration component.
        """
        # Import instance variables
        p = self.parameters
        K_s = float(p['K_s'])                  # saturated hydraulic conductivity (m/s)
        psi = float(p['psi'])                  # wetting-front suction head (m)
        theta_d = float(p['theta_d'])          # initial soil-moisture deficit (-)
        K_ex = float(p['K_ex'])                # interflow exfiltration rate (1/s)
        t_fine = self.t_fine
        # Pre-evaluate rainfall on fine grid
        p_t = np.array([self._p_func(t) for t in t_fine], dtype=np.float64)
        # ODE RHS, Eq. A.1
        def _dF_dt(t, F):
            F_val = max(float(F[0]), 1e-8)
            f_pot = K_s * (psi * theta_d + F_val) / F_val
            f = min(f_pot, self._p_func(t))
            g = K_ex * F_val
            return [f - g]
        # Integrate over fine grid
        sol = solve_ivp(
            _dF_dt, (t_fine[0], t_fine[-1]), [self.F0],
            t_eval=t_fine, method='RK45',
            rtol=1e-6, atol=1e-10, max_step=600.0,
        )
        F_t = np.maximum(sol.y[0], 1e-8)
        # Diagnose infiltration capacity, Eq. A.2
        f_pot = K_s * (psi * theta_d + F_t) / F_t
        f_t = np.minimum(f_pot, p_t)
        g_t = K_ex * F_t
        q_excess = np.maximum(p_t - f_t, 0.0)
        # Total Green-Ampt-domain surface runoff: infiltration excess + interflow
        q_GA = q_excess + g_t
        # Export
        self.F_t = F_t
        self._p_t = p_t
        self.q_GA = q_GA
        self._g_t = g_t

    def _solve_saturation_tank(self):
        """
        Solve shallow saturation-excess storage tank ODE.

        Eq. A.3: dS/dt = p(t) - S/tau_drain
        Saturation overflow q_SAT activates when S would exceed S_max.

        Forward Euler on the fine grid is sufficient because the
        overflow check is intrinsically discrete.
        """
        # Import instance variables
        p = self.parameters
        S_max = float(p['S_max'])              # max shallow-storage depth (m)
        tau_drain = float(p['tau_drain'])      # drainage timescale (s)
        t_fine = self.t_fine
        n = len(t_fine)
        dt = self.dt_fine
        p_t = self._p_t                        # rainfall on fine grid (m/s)
        # Initial storage at S0_frac of capacity (per manuscript Section 2.2)
        S = np.zeros(n)
        S[0] = self.S0_FRAC * S_max
        q_drain_arr = np.zeros(n)
        q_sat_arr = np.zeros(n)
        # Forward Euler with overflow handling
        for i in range(n - 1):
            q_drain = S[i] / tau_drain
            S_tent = S[i] + (p_t[i] - q_drain) * dt
            if S_tent > S_max:
                q_sat = (S_tent - S_max) / dt
                S_new = S_max
            else:
                q_sat = 0.0
                S_new = max(S_tent, 0.0)
            S[i + 1] = S_new
            q_drain_arr[i] = q_drain
            q_sat_arr[i] = q_sat
        # Final-step drainage so arrays align
        q_drain_arr[-1] = S[-1] / tau_drain
        q_sat_arr[-1] = 0.0
        # Total saturation-domain surface runoff: drainage + overflow
        q_SAT = q_drain_arr + q_sat_arr
        # Normalized saturation state for state-dependent mixing
        S_sat_norm = np.clip(S / max(S_max, 1e-9), 0.0, 1.0)
        # Export
        self.S_t = S
        self.q_SAT = q_SAT
        self.S_sat_norm = S_sat_norm

    def _combine_surface_runoff(self):
        """
        Weight Green-Ampt and saturation-excess pathways by f_sat.

        Eq. (1): q_hill,surf(t) = (1 - f_sat) q_GA(t) + f_sat q_SAT(t)
        """
        f_sat = float(self.parameters['f_sat'])
        self.q_surf = (1.0 - f_sat) * self.q_GA + f_sat * self.q_SAT

    def _solve_soil_chloride_pool(self):
        """
        Solve the continuously stirred soil chloride mixing pool.

        Eq. A.5: d(V_soil C_soil)/dt = p(t) C_rain(t) - q_hill,surf(t) C_soil(t)

        Initial condition (Eq. 4): C_soil(t = -168 h) = alpha_soil * pre_Cl.
        With V_soil constant, this reduces to
            dC_soil/dt = (p C_rain - q_hill,surf C_soil) / V_soil.
        """
        # Import instance variables
        V_soil = float(self.parameters['V_soil'])
        t_fine = self.t_fine
        p_t = self._p_t                        # rainfall on fine grid (m/s)
        q_t = self.q_surf                      # total surface runoff on fine grid (m/s)
        C_rain_t = np.array([self._C_rain_func(t) for t in t_fine], dtype=np.float64)
        # ODE RHS, Eq. A.5
        def _dCsoil_dt(t, y):
            idx = min(max(int(np.searchsorted(t_fine, t)), 0), len(t_fine) - 1)
            return [(p_t[idx] * C_rain_t[idx] - q_t[idx] * y[0])
                    / max(V_soil, 1e-6)]
        # Integrate
        sol = solve_ivp(
            _dCsoil_dt, (t_fine[0], t_fine[-1]), [self._C_soil_0],
            t_eval=t_fine, method='LSODA',
            rtol=1e-3, atol=1e-4, max_step=self.dt_fine,
        )
        if not sol.success:
            # Fallback: hold at initial value
            C_soil = np.full_like(t_fine, self._C_soil_0)
        else:
            C_soil = np.clip(sol.y[0], 0.05, 100.0)
        # Export
        self.C_soil = C_soil

    def _convolve_unit_hydrograph(self):
        """
        Convolve total surface runoff with the empirical UH to produce Q_hill.

        The DEM-derived UH captures channel travel time only. The
        uh_stretch parameter time-stretches the UH while dividing
        ordinates by the same factor to preserve integral = 1 (mass
        conservation). This represents hillslope subsurface storage
        delay, which is not contained in the DEM-derived UH.
        """
        # Import instance variables
        uh_stretch = float(self.parameters['uh_stretch'])
        t_fine = self.t_fine
        dt = self.dt_fine
        # Excess rainfall depth increments per fine step (m)
        dF_excess = self.q_surf * dt
        # Stretch UH time axis, divide ordinates by stretch (mass-conservative)
        t_uh_stretched_hr = self._uh_t_hr * uh_stretch
        uh_stretched = self._uh_hr_inv / uh_stretch
        # Interpolate stretched UH onto fine grid
        t_uh_end_hr = float(t_uh_stretched_hr[-1])
        t_uh_sec = np.arange(0.0, t_uh_end_hr * 3600.0 + dt, dt)
        uh_on_dt = np.interp(t_uh_sec / 3600.0, t_uh_stretched_hr,
                             uh_stretched, left=0.0, right=0.0)
        # Convert UH to (m^3/s per m of excess rain): u(t) = A uh / 3600
        u = self.area * uh_on_dt / 3600.0
        # Convolve and clip to non-negative
        Q_hill_conv = np.convolve(dF_excess, u, mode='full')[:len(t_fine)]
        Q_hill_conv = np.maximum(Q_hill_conv, 0.0)
        # Export
        self.Q_hill = Q_hill_conv

    # ──────────────────────────────────────────────────────────────────
    # Channel module: Eq. A.6-A.12 of the manuscript
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _alpha_mixing(Q, Q_threshold, gamma):
        """
        Sigmoidal Q-dependent mixing weight, Eq. A.6.

        alpha(Q) = 1 / (1 + (Q/Q_threshold)^gamma)

        Returns 1 at low Q (baseflow-dominant) and approaches 0 at
        high Q (event-dominant).
        """
        Q_safe = np.maximum(np.abs(Q), 1e-10)
        return 1.0 / (1.0 + (Q_safe / Q_threshold) ** gamma)

    def _manning_h(self, Q_target, n):
        """
        Solve Manning's equation Q = (1/n) A R^(2/3) sqrt(S_0) for h
        in a rectangular channel of width B (Newton iteration).

        Used only for steady-state baseflow initialization, not within
        the time-stepping loop.
        """
        h = 0.05
        for _ in range(50):
            A = self.B * h
            P = self.B + 2.0 * h
            R = A / P
            Q_h = A * R ** (2.0 / 3.0) * np.sqrt(self.S_0) / n
            eps = max(1e-6, 0.001 * h)
            A2 = self.B * (h + eps)
            P2 = self.B + 2.0 * (h + eps)
            R2 = A2 / P2
            Q_h2 = A2 * R2 ** (2.0 / 3.0) * np.sqrt(self.S_0) / n
            dQ_dh = (Q_h2 - Q_h) / eps
            if abs(dQ_dh) < 1e-14:
                break
            h_new = max(h - (Q_h - Q_target) / dQ_dh, 1e-4)
            if abs(h_new - h) < 1e-10:
                h = h_new
                break
            h = h_new
        return h

    def _initial_channel_state(self):
        """
        Manning steady-state baseflow initial condition for the channel.

        State vector y0 = [h_1..h_N, (AC)_1..(AC)_N]  (2N states).
        Discharge Q is no longer a state; it is diagnosed from h via
        the diffusion-wave Manning closure (Eq. A.10-A.11) at every
        RHS evaluation.
        """
        # Import instance variables
        N = self.N
        n = float(self.parameters['n'])
        baseflow_Q = float(self.parameters['baseflow_Q'])
        # Lateral inflow per unit length under steady baseflow
        qL_base = baseflow_Q / self.L
        Q_init = qL_base * (self.dx * (np.arange(N) + 1))
        h_init = np.array([self._manning_h(Qi, n) for Qi in Q_init])
        AC_init = self.B * h_init * self._pre_Cl
        return np.concatenate([h_init, AC_init])

    def _diffusion_wave_Q(self, h, n):
        """
        Compute discharge Q from depth h under the diffusion-wave
        Manning closure (algebraic, Eq. A.10-A.11).

        Diffusion-wave assumes both ∂Q/∂t = 0 and ∂(Q^2/A)/∂x = 0 in
        the momentum equation, leaving the algebraic relation:
            S_f = S_0 - ∂h/∂x                            (Eq. A.10)
            Q   = sign(S_f) * (1/n) * A * R^(2/3) * |S_f|^(1/2)
                                                          (Eq. A.11)

        The sign of S_f is preserved so that the formulation admits
        upstream-directed flow when local backwater conditions occur
        (∂h/∂x > S_0). The square-root argument is floored at a small
        epsilon for numerical safety.
        """
        N = self.N
        B = self.B
        S_0 = self.S_0
        dx = self.dx
        EPS_SF = 1e-10
        # Geometry (h is already floored at 1e-6 by caller)
        h_safe = np.maximum(h, 1e-6)
        A = B * h_safe
        P = B + 2.0 * h_safe
        R = np.maximum(A / P, 1e-9)
        # Centered ∂h/∂x at cell centers; ghost cells at boundaries reflect
        # interior depth (zero-gradient) -> consistent with closed
        # upstream / free-outflow channel boundaries used elsewhere.
        h_ext = np.concatenate(([h_safe[0]], h_safe, [h_safe[-1]]))
        dhdx = (h_ext[2:] - h_ext[:-2]) / (2.0 * dx)
        # Diffusion-wave friction slope (Eq. A.10)
        Sf = S_0 - dhdx
        # Sign-preserving Manning closure (Eq. A.11)
        sign_Sf = np.sign(Sf)
        Q = sign_Sf * (1.0 / n) * A * R ** (2.0 / 3.0) * np.sqrt(np.abs(Sf) + EPS_SF)
        return Q

    def _channel_rhs(self, t, y, qL_interp, C_lat_interp, n):
        """
        RHS of coupled diffusion-wave + ADE system, Eq. A.9-A.13.

        State vector y = [h_1..h_N, (AC)_1..(AC)_N]  (2N states).

        Discharge Q is a diagnostic variable computed from h via the
        diffusion-wave Manning closure (Eq. A.10-A.11). This contrasts
        with a local-inertia formulation, which would carry Q as a
        third dynamical state under ∂Q/∂t = gA(-∂h/∂x + S_0 - S_f).
        Diffusion-wave eliminates the local acceleration term and
        therefore admits no gravity-wave oscillations, in line with
        Section 2.2 of the manuscript.
        """
        # Import instance variables
        N = self.N
        B = self.B
        dx = self.dx
        D_0 = self.D_0
        beta = self.BETA_DISP
        Q_ref = self.Q_REF
        k_dec = self.K_DEC
        # Unpack state (Q is no longer a state variable)
        h = np.maximum(y[:N], 1e-6)
        AC = y[N:2 * N]
        # Geometric quantities for rectangular channel
        A = B * h
        C = np.where(A > 1e-8, AC / np.maximum(A, 1e-8), 0.0)
        C = np.maximum(C, 0.0)
        # Diagnostic discharge from diffusion-wave Manning closure
        # (Eq. A.10-A.11)
        Q = self._diffusion_wave_Q(h, n)
        # Q at cell faces (staggered grid)
        Q_face = np.empty(N + 1)
        Q_face[0] = 0.0                        # closed upstream boundary (headwater)
        Q_face[1:N] = 0.5 * (Q[:-1] + Q[1:])
        Q_face[N] = Q[-1]                      # free outflow at outlet
        # Forcing at time t
        qL_t = float(qL_interp(t))
        C_lat_t = float(C_lat_interp(t))
        # Continuity, Eq. A.9
        dh = (qL_t - (Q_face[1:] - Q_face[:-1]) / dx) / B
        # ADE advection, Eq. A.13 (upwind based on Q_face sign)
        QC_face = np.empty(N + 1)
        QC_face[0] = 0.0
        pos = Q_face[1:N] >= 0
        QC_face[1:N] = np.where(pos, Q_face[1:N] * C[:-1], Q_face[1:N] * C[1:])
        QC_face[N] = Q_face[N] * C[-1]
        dAC = -(QC_face[1:] - QC_face[:-1]) / dx
        # ADE dispersion (central difference of A D dC/dx at interior faces)
        Q_abs = np.maximum(np.abs(Q), 1e-10)
        D_cell = D_0 * (Q_abs / Q_ref) ** beta
        A_face_int = 0.5 * (A[:-1] + A[1:])
        D_face_int = 0.5 * (D_cell[:-1] + D_cell[1:])
        grad_C = (C[1:] - C[:-1]) / dx
        flux_disp = A_face_int * D_face_int * grad_C
        dAC[:-1] += flux_disp / dx
        dAC[1:] -= flux_disp / dx
        # ADE source (lateral inflow) and decay (k_dec = 0 for chloride)
        dAC += qL_t * C_lat_t
        dAC -= k_dec * AC
        return np.concatenate([dh, dAC])

    def _solve_channel(self, force_alpha_one=False, n_mixing_iters=2,
                       rtol=1e-3, atol=1e-5, max_step=120.0):
        """
        Solve coupled diffusion-wave + ADE channel routing.

        State is (h, AC) of dimension 2N; discharge Q is a diagnostic
        variable computed from h at every RHS evaluation via the
        diffusion-wave Manning closure (Eq. A.10-A.11). A fixed-point
        iteration is used over the Q-dependent mixing: an outlet-Q
        guess feeds the mixing closure (Eq. A.6-A.8), the channel is
        solved, and the resulting outlet Q (computed diagnostically
        from h) updates the guess. Two iterations are sufficient for
        convergence at hourly output resolution.

        If ``force_alpha_one=True``, alpha(Q) is held at 1 for all Q,
        which reduces the closure to C_lat = C_b. This reproduces the
        simplified model variant of Appendix C.

        Note on tolerances: diffusion-wave is parabolic (no gravity
        waves to resolve), so ``max_step`` can be larger than would be
        safe for the local-inertia formulation; the defaults
        (rtol=1e-3, atol=1e-5, max_step=120 s) are tuned for hourly
        output resolution.
        """
        # Import instance variables
        p = self.parameters
        Q_threshold = float(p['Q_threshold'])
        gamma = float(p['gamma'])
        lambda_sat = float(p['lambda_sat'])
        n = float(p['n'])
        baseflow_Q = float(p['baseflow_Q'])
        # Lateral inflow Q_hill / L plus constant baseflow (m^2/s)
        qL_fine = self.Q_hill / self.L + baseflow_Q / self.L
        qL_interp = interp1d(self.t_fine, qL_fine, bounds_error=False,
                             fill_value=(qL_fine[0], qL_fine[-1]))
        # Soil-pool Cl on fine grid (event-water endmember in Eq. A.8)
        C_soil_interp = interp1d(self.t_fine, self.C_soil, bounds_error=False,
                                  fill_value=(self.C_soil[0], self.C_soil[-1]))
        # Initial channel state (h, AC) under Manning steady baseflow
        y0 = self._initial_channel_state()
        t_eval = self.t_hourly
        t_end = float(t_eval[-1])
        # Build dense Q-mixing grid for fixed-point iteration
        t_mix = np.linspace(0.0, t_end, 500)
        Q_guess = baseflow_Q * np.ones_like(t_mix)
        C_soil_mix = np.array([float(C_soil_interp(t)) for t in t_mix])
        S_sat_mix = np.interp(t_mix, self.t_fine, self.S_sat_norm)
        # Constant baseflow Cl endmember (Eq. A.8: C_b = pre_Cl)
        C_b = self._pre_Cl
        # Fixed-point iteration on Q-dependent mixing
        sol = None
        for _ in range(n_mixing_iters):
            # Q-dependent mixing weight, Eq. A.6
            if force_alpha_one:
                alpha_v = np.ones_like(Q_guess)
            else:
                alpha_v = self._alpha_mixing(Q_guess, Q_threshold, gamma)
                # State-dependent damping (Model B), Eq. A.7
                alpha_v = alpha_v * (1.0 - lambda_sat * S_sat_mix)
                alpha_v = np.clip(alpha_v, 0.0, 1.0)
            # Two-endmember mixing, Eq. A.8
            C_lat_mix = alpha_v * C_b + (1.0 - alpha_v) * C_soil_mix
            C_lat_interp = interp1d(t_mix, C_lat_mix, bounds_error=False,
                                     fill_value=(C_lat_mix[0], C_lat_mix[-1]))
            # Solve coupled diffusion-wave + ADE
            sol = solve_ivp(
                lambda t, y: self._channel_rhs(t, y, qL_interp, C_lat_interp, n),
                (0.0, t_end), y0, t_eval=t_eval, method='LSODA',
                rtol=rtol, atol=atol, max_step=max_step,
            )
            if not sol.success:
                break
            # Update Q guess from this solve. Q is diagnostic; recover
            # outlet Q timeseries by applying the diffusion-wave Manning
            # closure to the h-trajectory at each output time.
            h_traj = sol.y[:self.N]                # shape (N, len(t_eval))
            Q_traj = np.array([
                self._diffusion_wave_Q(h_traj[:, k], n)
                for k in range(h_traj.shape[1])
            ]).T                                    # shape (N, len(t_eval))
            Q_outlet = Q_traj[-1]
            Q_guess = np.interp(t_mix, t_eval, Q_outlet)
        # Extract outlet Q and C
        if sol is None or not sol.success:
            self.Q_out = np.full(len(t_eval), np.nan)
            self.C_out = np.full(len(t_eval), np.nan)
            return False
        # State indices under (h, AC) layout
        h_traj_final = sol.y[:self.N]
        AC_outlet = sol.y[2 * self.N - 1]
        h_outlet = h_traj_final[-1]
        # Outlet Q is diagnostic at every timestep
        Q_outlet_final = np.array([
            self._diffusion_wave_Q(h_traj_final[:, k], n)[-1]
            for k in range(h_traj_final.shape[1])
        ])
        A_outlet = self.B * np.maximum(h_outlet, 1e-6)
        C_outlet = np.where(A_outlet > 1e-8, AC_outlet / A_outlet, 0.0)
        # Export
        self.Q_out = Q_outlet_final
        self.C_out = np.maximum(C_outlet, 0.0)
        return True

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def simulate(self, force_alpha_one=False):
        """
        Run the full hillslope + channel forward simulation.

        Inputs:
        -------
        force_alpha_one : bool
            If True, force alpha(Q) = 1 for all Q. Combined with K_ex
            = 0 in the parameter dictionary, reproduces the simplified
            model variant of Appendix C of the manuscript.

        Returns:
        --------
        self : Model
            The model instance with all state attributes populated.
        """
        self._solve_green_ampt()
        self._solve_saturation_tank()
        self._combine_surface_runoff()
        self._solve_soil_chloride_pool()
        self._convolve_unit_hydrograph()
        ok = self._solve_channel(force_alpha_one=force_alpha_one)
        self.success = bool(ok)
        return self

    def extract_at(self, t_target):
        """
        Interpolate outlet Q and C onto a target observation time grid.

        Inputs:
        -------
        t_target : np.ndarray
            Target time vector (s).

        Returns:
        --------
        (Q_at, C_at) : tuple of np.ndarray
            Outlet Q (m^3/s) and C (mg/L) on the requested grid.
        """
        if not self.success:
            raise RuntimeError('Model has not been simulated. Call simulate() first.')
        Q_at = np.interp(t_target, self.t_hourly, self.Q_out)
        C_at = np.interp(t_target, self.t_hourly, self.C_out)
        return Q_at, C_at
