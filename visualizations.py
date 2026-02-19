"""
Visualization suite for the Morris-Lecar action potential propagation simulation.

Provides 5 visualization types:
1. Spatiotemporal heatmap — V(x, t) as a 2D color map
2. Spatial snapshots — V(x) at selected time points
3. Temporal traces — V(t) at selected positions
4. Phase plane portrait — w vs V with nullclines
5. Animated propagation — V(x) evolving in time (saved as .gif)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from morris_lecar import MLParams, m_inf, w_inf, I_ion


# ──────────────────────────────────────────────────────────────────
# 1. Spatiotemporal Heatmap
# ──────────────────────────────────────────────────────────────────

def plot_heatmap(V_history, t_saved, x_grid, params=None, save_path=None):
    """2D heatmap of membrane voltage V(x, t).
    
    x-axis: position along axon, y-axis: time, color: voltage.
    The travelling wave appears as a red diagonal stripe, and the
    refractory hyperpolarization appears as a dark blue trail behind it.
    
    Uses a TwoSlopeNorm centered on the resting potential so that
    the sub-rest refractory period is clearly distinguishable from rest.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    extent = [x_grid[0], x_grid[-1], t_saved[0], t_saved[-1]]

    # Determine the resting potential (median voltage across all points/times)
    V_rest = np.median(V_history)
    V_min = np.min(V_history)
    V_max = np.max(V_history)

    # TwoSlopeNorm centers the colormap at V_rest, giving equal color
    # resolution to the depolarization (above rest) and hyperpolarization
    # (below rest, i.e. the refractory period)
    norm = mcolors.TwoSlopeNorm(vmin=V_min - 2, vcenter=V_rest, vmax=V_max + 2)

    im = ax.imshow(V_history, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', norm=norm)
    cbar = fig.colorbar(im, ax=ax, label='Membrane Voltage (mV)')
    # Mark the resting potential on the colorbar
    cbar.ax.axhline(y=V_rest, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Position along axon (cm)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Spatiotemporal Action Potential Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved heatmap → {save_path}")
    return fig, ax


# ──────────────────────────────────────────────────────────────────
# 2. Spatial Snapshots at Selected Times
# ──────────────────────────────────────────────────────────────────

def plot_spatial_snapshots(V_history, t_saved, x_grid, 
                           snapshot_times=None, save_path=None):
    """Overlay plots of V(x) at several time points.
    
    Shows the action potential profile maintaining its shape while
    translating rightward along the axon.
    """
    if snapshot_times is None:
        snapshot_times = [0, 10, 50, 100, 150]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshot_times)))

    for t_target, color in zip(snapshot_times, colors):
        idx = np.argmin(np.abs(t_saved - t_target))
        actual_t = t_saved[idx]
        ax.plot(x_grid, V_history[idx], color=color, linewidth=1.8,
                label=f't = {actual_t:.0f} ms')

    ax.set_xlabel('Position along axon (cm)', fontsize=12)
    ax.set_ylabel('Membrane Voltage (mV)', fontsize=12)
    ax.set_title('Action Potential Profile — Spatial Snapshots', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(-100, 60)
    ax.axhline(y=-60, color='gray', linestyle='--', alpha=0.4, label='Resting potential')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved spatial snapshots → {save_path}")
    return fig, ax


# ──────────────────────────────────────────────────────────────────
# 3. Temporal Traces at Selected Positions
# ──────────────────────────────────────────────────────────────────

def plot_temporal_traces(V_history, t_saved, x_grid,
                          probe_positions=None, save_path=None):
    """Plot V(t) for several positions along the axon.
    
    Each curve shows the action potential arriving with a time delay
    proportional to distance from the stimulus site.
    """
    if probe_positions is None:
        probe_positions = [0, 5, 10, 15, 20]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(probe_positions)))

    for x_target, color in zip(probe_positions, colors):
        idx = np.argmin(np.abs(x_grid - x_target))
        actual_x = x_grid[idx]
        ax.plot(t_saved, V_history[:, idx], color=color, linewidth=1.8,
                label=f'x = {actual_x:.1f} cm')

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Membrane Voltage (mV)', fontsize=12)
    ax.set_title('Action Potential Arrival — Temporal Traces', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(-100, 60)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved temporal traces → {save_path}")
    return fig, ax


# ──────────────────────────────────────────────────────────────────
# 4. Phase Plane Portrait (V vs w)
# ──────────────────────────────────────────────────────────────────

def plot_phase_plane(params: MLParams, trajectories=None, save_path=None):
    """Phase plane with V- and w-nullclines and optional trajectories.
    
    - V-nullcline: dV/dt = 0  →  w = -(g_Ca·m∞(V)·(E_Ca-V) + g_leak·(E_leak-V) + I_app) / (g_K·(E_K-V))
    - w-nullcline: dw/dt = 0  →  w = w∞(V)
    
    trajectories: list of (V_arr, w_arr) tuples to overlay on the phase plane.
    """
    V_range = np.linspace(-80, 60, 500)

    # w-nullcline: w = w∞(V)
    w_null = w_inf(V_range, params)

    # V-nullcline: solve dV/dt = 0 for w
    # 0 = g_Ca*m∞*(E_Ca - V) + g_K*w*(E_K - V) + g_leak*(E_leak - V) + I_app
    # w = -(g_Ca*m∞*(E_Ca-V) + g_leak*(E_leak-V) + I_app) / (g_K*(E_K - V))
    m_ss = m_inf(V_range, params)
    numerator = -(params.g_Ca * m_ss * (params.E_Ca - V_range) +
                  params.g_leak * (params.E_leak - V_range) +
                  params.I_applied)
    denominator = params.g_K * (params.E_K - V_range)

    # Avoid division by zero near E_K
    with np.errstate(divide='ignore', invalid='ignore'):
        V_null_w = numerator / denominator

    # Mask out extreme values
    valid = np.abs(V_null_w) < 2.0
    V_range_v = V_range[valid]
    V_null_w = V_null_w[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Nullclines
    ax.plot(V_range, w_null, 'b-', linewidth=2, label='w-nullcline (dw/dt = 0)')
    ax.plot(V_range_v, V_null_w, 'r-', linewidth=2, label='V-nullcline (dV/dt = 0)')

    # Trajectories
    if trajectories:
        traj_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(trajectories)))
        for i, (V_traj, w_traj) in enumerate(trajectories):
            label = f'Trajectory {i+1}' if len(trajectories) > 1 else 'Trajectory'
            ax.plot(V_traj, w_traj, color=traj_colors[i], linewidth=1.5,
                    alpha=0.8, label=label)
            ax.plot(V_traj[0], w_traj[0], 'o', color=traj_colors[i],
                    markersize=8, zorder=5)

    ax.set_xlabel('Membrane Voltage V (mV)', fontsize=12)
    ax.set_ylabel('Potassium Gating Variable w', fontsize=12)
    ax.set_title('Phase Plane Portrait — Morris-Lecar Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-80, 60)
    ax.set_ylim(-0.05, 0.6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved phase plane → {save_path}")
    return fig, ax


# ──────────────────────────────────────────────────────────────────
# 5. Animated Propagation
# ──────────────────────────────────────────────────────────────────

def create_animation(V_history, t_saved, x_grid, save_path=None,
                     frame_skip=2, fps=30):
    """Animated plot of V(x) evolving in real time, saved as .gif.
    
    Shows the action potential pulse travelling along the axon.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], 'b-', linewidth=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(-100, 60)
    ax.set_xlabel('Position along axon (cm)', fontsize=12)
    ax.set_ylabel('Membrane Voltage (mV)', fontsize=12)
    ax.set_title('Action Potential Propagation', fontsize=14, fontweight='bold')
    ax.axhline(y=-60, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3)

    frames = range(0, len(t_saved), frame_skip)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame_idx):
        i = list(frames)[frame_idx]
        line.set_data(x_grid, V_history[i])
        time_text.set_text(f't = {t_saved[i]:.1f} ms')
        return line, time_text

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=len(list(frames)),
                                    interval=1000//fps, blit=True)
    if save_path:
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print(f"  Saved animation → {save_path}")
    plt.close(fig)
    return anim
