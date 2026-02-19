"""
Morris-Lecar model for excitable membranes with cable equation for spatial propagation.

Based on Chapter 8 of "Mathematical Modelling in Systems Biology" (Ingalls, 2012).
Implements the Morris-Lecar two-variable model coupled with the 1D cable equation
to simulate spatiotemporal action potential propagation along a neuronal axon.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MLParams:
    """All parameters for the Morris-Lecar model + cable equation.
    
    Membrane parameters from Figure 8.6 of Ingalls (2012).
    Cable parameters from Figure 8.12 of Ingalls (2012).
    """
    # --- Membrane capacitance ---
    C: float = 20.0          # µF/cm² — membrane capacitance

    # --- Maximal conductances ---
    g_Ca: float = 4.4        # mS/cm² — max calcium conductance
    g_K: float = 8.0         # mS/cm² — max potassium conductance
    g_leak: float = 2.0      # mS/cm² — leak conductance

    # --- Nernst (reversal) potentials ---
    E_Ca: float = 120.0      # mV — calcium Nernst potential
    E_K: float = -84.0       # mV — potassium Nernst potential
    E_leak: float = -60.0    # mV — leak Nernst potential

    # --- Gating function parameters ---
    V_m_star: float = -1.2   # mV — Ca²⁺ half-activation voltage
    V_w_star: float = 2.0    # mV — K⁺ half-activation voltage
    theta_m: float = 18.0    # mV — Ca²⁺ activation slope
    theta_w: float = 30.0    # mV — K⁺ activation slope

    # --- Potassium gating time constant ---
    phi: float = 0.05        # 1/ms — rate constant for w dynamics (τ_w = 1/phi = 20 ms)

    # --- Applied current ---
    I_applied: float = 0.0   # µA/cm² — externally applied current

    # --- Cable equation (spatial) parameters ---
    g_a: float = 30.0        # mS/cm — axial (cytoplasmic) conductivity
    r: float = 0.05          # cm — axon radius (500 µm)
    L: float = 30.0          # cm — axon length
    N: int = 500             # number of spatial grid points

    @property
    def tau_w(self):
        """Potassium gating time constant in ms."""
        return 1.0 / self.phi

    @property
    def dx(self):
        """Spatial step size in cm."""
        return self.L / self.N

    @property
    def D(self):
        """Effective diffusion coefficient: ḡ_a · r / (2C)  [cm²/ms]."""
        return self.g_a * self.r / (2.0 * self.C)


def m_inf(V: np.ndarray, params: MLParams) -> np.ndarray:
    """Steady-state fraction of open Ca²⁺ channels (Eq. 8.13).
    
    m∞(V) = 0.5 * (1 + tanh((V - V*_m) / θ_m))
    """
    return 0.5 * (1.0 + np.tanh((V - params.V_m_star) / params.theta_m))


def w_inf(V: np.ndarray, params: MLParams) -> np.ndarray:
    """Steady-state fraction of open K⁺ channels (Eq. 8.13).
    
    w∞(V) = 0.5 * (1 + tanh((V - V*_w) / θ_w))
    """
    return 0.5 * (1.0 + np.tanh((V - params.V_w_star) / params.theta_w))


def I_ion(V: np.ndarray, w: np.ndarray, params: MLParams) -> np.ndarray:
    """Total ionic transmembrane current density (positive = inward).
    
    I_ion = ḡ_Ca · m∞(V) · (E_Ca - V) + ḡ_K · w · (E_K - V) + g_leak · (E_leak - V)
    """
    I_Ca = params.g_Ca * m_inf(V, params) * (params.E_Ca - V)
    I_K = params.g_K * w * (params.E_K - V)
    I_L = params.g_leak * (params.E_leak - V)
    return I_Ca + I_K + I_L


# ──────────────────────────────────────────────────────────────────────────────
# Single-compartment (ODE) solver
# ──────────────────────────────────────────────────────────────────────────────

def _single_compartment_rhs(V, w, params):
    """Right-hand side of the Morris-Lecar ODE system (Eq. 8.12).
    
    Returns (dV/dt, dw/dt).
    """
    dVdt = (1.0 / params.C) * (I_ion(V, w, params) + params.I_applied)
    dwdt = (w_inf(V, params) - w) / params.tau_w
    return dVdt, dwdt


def simulate_single_compartment(params: MLParams, V0: float, w0: float,
                                 T: float, dt: float):
    """Simulate the Morris-Lecar model for a single membrane patch using RK4.
    
    Parameters
    ----------
    params : MLParams
        Model parameters.
    V0 : float
        Initial membrane voltage (mV).
    w0 : float
        Initial potassium gating variable (dimensionless, 0–1).
    T : float
        Total simulation time (ms).
    dt : float
        Time step (ms).
    
    Returns
    -------
    V_arr : ndarray, shape (n_steps+1,)
        Membrane voltage time series.
    w_arr : ndarray, shape (n_steps+1,)
        Potassium gating variable time series.
    t_arr : ndarray, shape (n_steps+1,)
        Time points.
    """
    n_steps = int(T / dt)
    t_arr = np.linspace(0, T, n_steps + 1)
    V_arr = np.zeros(n_steps + 1)
    w_arr = np.zeros(n_steps + 1)
    V_arr[0] = V0
    w_arr[0] = w0

    V = np.float64(V0)
    w = np.float64(w0)

    for i in range(n_steps):
        # RK4 integration
        k1v, k1w = _single_compartment_rhs(V, w, params)
        k2v, k2w = _single_compartment_rhs(V + 0.5*dt*k1v, w + 0.5*dt*k1w, params)
        k3v, k3w = _single_compartment_rhs(V + 0.5*dt*k2v, w + 0.5*dt*k2w, params)
        k4v, k4w = _single_compartment_rhs(V + dt*k3v, w + dt*k3w, params)

        V = V + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        w = w + (dt / 6.0) * (k1w + 2*k2w + 2*k3w + k4w)

        V_arr[i + 1] = V
        w_arr[i + 1] = w

    return V_arr, w_arr, t_arr


# ──────────────────────────────────────────────────────────────────────────────
# Spatially-extended (PDE) solver — cable equation + Morris-Lecar
# ──────────────────────────────────────────────────────────────────────────────

def _cable_rhs(V, w, params):
    """Right-hand side of the spatially-discretized cable equation.
    
    The nonlinear cable equation (textbook Section 8.4.3):
        ∂V/∂t = (1/C) * [ (ḡ_a · r / 2) · ∂²V/∂x² + I_ion(V,w) + I_applied ]
        ∂w/∂t = (w∞(V) - w) / τ_w
    
    Spatial derivative ∂²V/∂x² approximated by central differences.
    Neumann (zero-flux) boundary conditions.
    
    Parameters
    ----------
    V : ndarray, shape (N,)
        Voltage at each spatial grid point.
    w : ndarray, shape (N,)
        Potassium gating variable at each spatial grid point.
    params : MLParams
    
    Returns
    -------
    dVdt : ndarray, shape (N,)
    dwdt : ndarray, shape (N,)
    """
    dx = params.dx
    D_coeff = params.g_a * params.r / 2.0   # diffusion-like coefficient (before /C)

    # Central-difference Laplacian with Neumann BC (ghost nodes = neighbor)
    laplacian = np.zeros_like(V)
    laplacian[1:-1] = (V[:-2] - 2.0 * V[1:-1] + V[2:]) / (dx * dx)
    # Boundaries: ∂V/∂x = 0 → V_{-1} = V_0, V_{N} = V_{N-1}
    laplacian[0] = (V[1] - V[0]) / (dx * dx)           # V_{-1} = V_0
    laplacian[-1] = (V[-2] - V[-1]) / (dx * dx)         # V_{N} = V_{N-1}

    # Ionic + applied currents
    I_total = I_ion(V, w, params) + params.I_applied

    dVdt = (1.0 / params.C) * (D_coeff * laplacian + I_total)
    dwdt = (w_inf(V, params) - w) / params.tau_w

    return dVdt, dwdt


def simulate_cable(params: MLParams, V_init: np.ndarray, w_init: np.ndarray,
                   T: float, dt: float, save_every: int = 10):
    """Simulate action potential propagation along the axon using the cable equation + RK4.
    
    Parameters
    ----------
    params : MLParams
        Model parameters (includes spatial parameters N, L, g_a, r).
    V_init : ndarray, shape (N,)
        Initial voltage profile along the axon (mV).
    w_init : ndarray, shape (N,)
        Initial potassium gating profile along the axon.
    T : float
        Total simulation time (ms).
    dt : float
        Time step (ms).
    save_every : int
        Save state every this many time steps (to manage memory).
    
    Returns
    -------
    V_history : ndarray, shape (n_saved, N)
        Voltage at each saved time step.
    w_history : ndarray, shape (n_saved, N)
        Gating variable at each saved time step.
    t_saved : ndarray, shape (n_saved,)
        Saved time points (ms).
    x_grid : ndarray, shape (N,)
        Spatial positions (cm).
    """
    N = params.N
    n_steps = int(T / dt)
    x_grid = np.linspace(0, params.L, N)

    # Pre-allocate saved arrays
    n_saved = n_steps // save_every + 1
    V_history = np.zeros((n_saved, N))
    w_history = np.zeros((n_saved, N))
    t_saved = np.zeros(n_saved)

    V = V_init.copy().astype(np.float64)
    w = w_init.copy().astype(np.float64)

    V_history[0] = V
    w_history[0] = w
    t_saved[0] = 0.0
    save_idx = 1

    for step in range(1, n_steps + 1):
        # RK4 integration
        k1v, k1w = _cable_rhs(V, w, params)
        k2v, k2w = _cable_rhs(V + 0.5*dt*k1v, w + 0.5*dt*k1w, params)
        k3v, k3w = _cable_rhs(V + 0.5*dt*k2v, w + 0.5*dt*k2w, params)
        k4v, k4w = _cable_rhs(V + dt*k3v, w + dt*k3w, params)

        V = V + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        w = w + (dt / 6.0) * (k1w + 2*k2w + 2*k3w + k4w)

        if step % save_every == 0 and save_idx < n_saved:
            V_history[save_idx] = V
            w_history[save_idx] = w
            t_saved[save_idx] = step * dt
            save_idx += 1

    # Trim if fewer saves than expected
    V_history = V_history[:save_idx]
    w_history = w_history[:save_idx]
    t_saved = t_saved[:save_idx]

    return V_history, w_history, t_saved, x_grid


def default_params() -> MLParams:
    """Return the default textbook parameter set (Figure 8.6).
    
    These are the standard Morris-Lecar parameters for the
    single-compartment (ODE) excitable membrane model.
    """
    return MLParams()


def cable_params() -> MLParams:
    """Return parameters tuned for cable equation simulation (Figure 8.12).
    
    Key difference from default: g_leak = 0.5 mS/cm² (per Figure 8.12 caption:
    'other parameter values as in Figure 8.6 except g_leak = 0.5 mS/cm²').
    This reduced leak conductance allows action potentials to propagate
    spatially as travelling waves.
    """
    return MLParams(g_leak=0.5)


def resting_state(params: MLParams):
    """Compute approximate resting state (V_rest, w_rest).
    
    At rest, the leak current dominates (m∞ ≈ 0, w ≈ 0), so V_rest ≈ E_leak.
    We then set w_rest = w∞(V_rest).
    """
    V_rest = params.E_leak  # ≈ -60 mV
    # More accurate: find the fixed point numerically by iterating
    for _ in range(100):
        m = m_inf(np.float64(V_rest), params)
        w_r = w_inf(np.float64(V_rest), params)
        # At steady state: I_ion + I_applied = 0
        # Solve for V: g_Ca*m*(E_Ca - V) + g_K*w*(E_K - V) + g_leak*(E_leak - V) + I_app = 0
        num = (params.g_Ca * m * params.E_Ca +
               params.g_K * w_r * params.E_K +
               params.g_leak * params.E_leak +
               params.I_applied)
        den = params.g_Ca * m + params.g_K * w_r + params.g_leak
        V_rest = num / den
    w_rest = float(w_inf(np.float64(V_rest), params))
    return float(V_rest), w_rest


def initial_voltage_pulse(x_grid: np.ndarray, V_rest: float,
                          amplitude: float = 60.0, width_power: int = 8):
    """Create the initial voltage pulse for triggering an action potential.
    
    From Problem 8.6.9: V(x,0) = amplitude / (1 + x^width_power) + V_rest
    
    This produces a sharp depolarization centered at x = 0.
    """
    return amplitude / (1.0 + x_grid**width_power) + V_rest
