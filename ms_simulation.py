"""
Multiple Sclerosis demyelination model for the Morris-Lecar cable equation.

Generates random demyelinated lesions along the axon with varying severity
and simulates action potential propagation with spatially varying membrane
properties (capacitance C and leak conductance g_leak).

Lesion parameters:
- Length: Gaussian distribution (default mean=0.4 cm, std=0.1 cm)
- Severity: Uniform from {1=mild, 2=moderate, 3=severe}
- Position: Uniform along the axon
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from morris_lecar import MLParams, m_inf, w_inf


# ──────────────────────────────────────────────────────────────────────────────
# Lesion data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Lesion:
    """A single demyelinated lesion along the axon."""
    start: float    # cm — start position
    length: float   # cm — length of lesion
    severity: int   # 1=mild, 2=moderate, 3=severe


# ──────────────────────────────────────────────────────────────────────────────
# Lesion generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_lesions(
    axon_length: float = 30.0,
    n_lesions: int = 5,
    mean_length: float = 0.4,
    std_length: float = 0.1,
    seed: int = 42,
) -> List[Lesion]:
    """Generate random demyelinated lesions along the axon.

    Parameters
    ----------
    axon_length : float
        Length of the axon (cm).
    n_lesions : int
        Number of lesions to generate.
    mean_length : float
        Mean lesion length (cm). Default 0.4 cm = 4 mm.
    std_length : float
        Standard deviation of lesion length (cm). Default 0.1 cm = 1 mm.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    lesions : list of Lesion
    """
    rng = np.random.default_rng(seed)
    lesions = []

    for _ in range(n_lesions):
        length = np.clip(rng.normal(mean_length, std_length), 0.05, axon_length / 3)
        severity = int(rng.integers(1, 4))  # 1, 2, or 3
        start = rng.uniform(0, max(0.01, axon_length - length))
        lesions.append(Lesion(start=start, length=length, severity=severity))

    return lesions


# ──────────────────────────────────────────────────────────────────────────────
# Spatial parameter arrays
# ──────────────────────────────────────────────────────────────────────────────

def create_spatial_arrays(
    x_grid: np.ndarray,
    lesions: List[Lesion],
    base_C: float,
    base_g_leak: float,
    severity_C: dict,
    severity_g_leak: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spatially varying C and g_leak arrays.

    Parameters
    ----------
    x_grid : ndarray
        Spatial grid positions (cm).
    lesions : list of Lesion
        Demyelinated lesions.
    base_C : float
        Healthy membrane capacitance (µF/cm²).
    base_g_leak : float
        Healthy leak conductance (mS/cm²).
    severity_C : dict
        Mapping {1: C_mild, 2: C_moderate, 3: C_severe}.
    severity_g_leak : dict
        Mapping {1: g_leak_mild, 2: g_leak_moderate, 3: g_leak_severe}.

    Returns
    -------
    C_arr : ndarray
        Capacitance at each grid point.
    g_leak_arr : ndarray
        Leak conductance at each grid point.
    severity_map : ndarray of int
        Severity level at each grid point (0=healthy).
    """
    N = len(x_grid)
    C_arr = np.full(N, base_C)
    g_leak_arr = np.full(N, base_g_leak)
    severity_map = np.zeros(N, dtype=int)

    for lesion in lesions:
        mask = (x_grid >= lesion.start) & (x_grid <= lesion.start + lesion.length)
        indices = np.where(mask)[0]
        # More severe lesions override milder ones at overlapping positions
        override = indices[severity_map[indices] < lesion.severity]
        C_arr[override] = severity_C[lesion.severity]
        g_leak_arr[override] = severity_g_leak[lesion.severity]
        severity_map[override] = lesion.severity

    return C_arr, g_leak_arr, severity_map


# ──────────────────────────────────────────────────────────────────────────────
# Cable equation with spatially varying parameters
# ──────────────────────────────────────────────────────────────────────────────

def _cable_rhs_ms(V, w, params, C_arr, g_leak_arr):
    """Cable equation RHS with spatially varying C and g_leak.

    The only difference from the healthy model is that C and g_leak
    are arrays rather than scalars, varying by position along the axon.
    """
    dx = params.dx
    D_coeff = params.g_a * params.r / 2.0

    # Laplacian with Neumann (zero-flux) boundary conditions
    laplacian = np.zeros_like(V)
    laplacian[1:-1] = (V[:-2] - 2.0 * V[1:-1] + V[2:]) / (dx * dx)
    laplacian[0] = (V[1] - V[0]) / (dx * dx)
    laplacian[-1] = (V[-2] - V[-1]) / (dx * dx)

    # Ionic currents — g_leak varies spatially
    I_Ca = params.g_Ca * m_inf(V, params) * (params.E_Ca - V)
    I_K = params.g_K * w * (params.E_K - V)
    I_L = g_leak_arr * (params.E_leak - V)
    I_total = I_Ca + I_K + I_L + params.I_applied

    # C varies spatially
    dVdt = (1.0 / C_arr) * (D_coeff * laplacian + I_total)
    dwdt = (w_inf(V, params) - w) / params.tau_w

    return dVdt, dwdt


def simulate_cable_ms(
    params: MLParams,
    V_init: np.ndarray,
    w_init: np.ndarray,
    C_arr: np.ndarray,
    g_leak_arr: np.ndarray,
    T: float = 200.0,
    dt: float = 0.01,
    save_every: int = 10,
    progress_callback: Optional[Callable] = None,
):
    """Simulate cable equation with spatially varying MS parameters (RK4).

    Parameters
    ----------
    params : MLParams
        Base model parameters (g_Ca, g_K, g_a, r, etc.).
    V_init, w_init : ndarray, shape (N,)
        Initial conditions.
    C_arr, g_leak_arr : ndarray, shape (N,)
        Spatially varying capacitance and leak conductance.
    T : float
        Total simulation time (ms).
    dt : float
        Time step (ms).
    save_every : int
        Save state every this many steps.
    progress_callback : callable, optional
        Called with (fraction_done,) periodically.

    Returns
    -------
    V_history, w_history : ndarray, shape (n_saved, N)
    t_saved : ndarray, shape (n_saved,)
    x_grid : ndarray, shape (N,)
    """
    N = params.N
    n_steps = int(T / dt)
    x_grid = np.linspace(0, params.L, N)

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
        k1v, k1w = _cable_rhs_ms(V, w, params, C_arr, g_leak_arr)
        k2v, k2w = _cable_rhs_ms(V + 0.5*dt*k1v, w + 0.5*dt*k1w, params, C_arr, g_leak_arr)
        k3v, k3w = _cable_rhs_ms(V + 0.5*dt*k2v, w + 0.5*dt*k2w, params, C_arr, g_leak_arr)
        k4v, k4w = _cable_rhs_ms(V + dt*k3v, w + dt*k3w, params, C_arr, g_leak_arr)

        V = V + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        w = w + (dt / 6.0) * (k1w + 2*k2w + 2*k3w + k4w)

        if step % save_every == 0 and save_idx < n_saved:
            V_history[save_idx] = V
            w_history[save_idx] = w
            t_saved[save_idx] = step * dt
            save_idx += 1

            if progress_callback and save_idx % 50 == 0:
                progress_callback(step / n_steps)

    if progress_callback:
        progress_callback(1.0)

    return V_history[:save_idx], w_history[:save_idx], t_saved[:save_idx], x_grid
