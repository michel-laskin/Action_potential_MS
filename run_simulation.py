"""
Run the Morris-Lecar action potential propagation simulation.

Executes:
1. Single-compartment Morris-Lecar simulations (sub-threshold and supra-threshold)
2. Full cable equation simulation (spatial propagation)
3. Generates all 5 visualizations
"""

import os
import sys
import numpy as np
from morris_lecar import (
    MLParams, default_params, cable_params, resting_state,
    simulate_single_compartment, simulate_cable, initial_voltage_pulse, w_inf
)
from visualizations import (
    plot_heatmap, plot_spatial_snapshots, plot_temporal_traces,
    plot_phase_plane, create_animation
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run_single_compartment(params: MLParams):
    """Run single-compartment simulations and produce the phase plane portrait."""
    print("\n" + "="*60)
    print(" SINGLE-COMPARTMENT MORRIS-LECAR MODEL")
    print("="*60)

    V_rest, w_rest = resting_state(params)
    print(f"  Resting state: V_rest = {V_rest:.2f} mV, w_rest = {w_rest:.4f}")

    # --- Sub-threshold trajectory (V0 = -15 mV) ---
    print("\n  [1] Sub-threshold stimulus: V₀ = -15 mV")
    V_sub, w_sub, t_sub = simulate_single_compartment(params, -15.0, w_rest, T=200, dt=0.01)
    print(f"      Peak voltage: {np.max(V_sub):.2f} mV (no action potential)")

    # --- Supra-threshold trajectory (V0 = -13 mV) ---
    print("  [2] Supra-threshold stimulus: V₀ = -13 mV")
    V_sup, w_sup, t_sup = simulate_single_compartment(params, -13.0, w_rest, T=200, dt=0.01)
    print(f"      Peak voltage: {np.max(V_sup):.2f} mV (action potential!)")

    # --- Supra-threshold trajectory (V0 = +5 mV) ---
    print("  [3] Strong stimulus: V₀ = +5 mV")
    V_strong, w_strong, t_strong = simulate_single_compartment(params, 5.0, w_rest, T=200, dt=0.01)
    print(f"      Peak voltage: {np.max(V_strong):.2f} mV")

    # --- Phase plane with all trajectories ---
    print("\n  Generating phase plane portrait...")
    trajectories = [
        (V_sub, w_sub),
        (V_sup, w_sup),
        (V_strong, w_strong),
    ]
    plot_phase_plane(params, trajectories=trajectories,
                     save_path=os.path.join(OUTPUT_DIR, "phase_plane.png"))

    return V_rest, w_rest


def run_cable_simulation(params: MLParams, V_rest: float, w_rest: float):
    """Run the spatially-extended cable equation simulation."""
    print("\n" + "="*60)
    print(" CABLE EQUATION — SPATIAL PROPAGATION")
    print("="*60)

    # Create spatial grid and initial conditions
    x_grid = np.linspace(0, params.L, params.N)
    V_init = initial_voltage_pulse(x_grid, V_rest, amplitude=60.0, width_power=8)
    w_init = np.full(params.N, w_rest)

    print(f"  Axon length: {params.L} cm,  Grid points: {params.N},  dx = {params.dx:.4f} cm")
    print(f"  Time step: 0.01 ms,  Total time: 200 ms")
    print(f"  Diffusion coefficient D = ḡ_a·r/(2C) = {params.D:.4f} cm²/ms")
    print(f"  Initial pulse: V(0,0) = {V_init[0]:.2f} mV,  V(L,0) = {V_init[-1]:.2f} mV\n")

    print("  Running simulation... ", end="", flush=True)
    V_history, w_history, t_saved, x_grid = simulate_cable(
        params, V_init, w_init, T=200, dt=0.01, save_every=10
    )
    print(f"Done! ({len(t_saved)} saved frames)")
    print(f"  Peak voltage in domain: {np.max(V_history):.2f} mV")
    print(f"  Voltage at far end at final time: {V_history[-1, -1]:.2f} mV")

    return V_history, w_history, t_saved, x_grid


def generate_visualizations(V_history, t_saved, x_grid, params):
    """Generate all 5 visualization types."""
    print("\n" + "="*60)
    print(" GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Spatiotemporal heatmap
    print("\n  [1/5] Spatiotemporal heatmap...")
    plot_heatmap(V_history, t_saved, x_grid,
                 save_path=os.path.join(OUTPUT_DIR, "heatmap.png"))

    # 2. Spatial snapshots
    print("  [2/5] Spatial snapshots...")
    plot_spatial_snapshots(V_history, t_saved, x_grid,
                           snapshot_times=[0, 20, 50, 100, 150],
                           save_path=os.path.join(OUTPUT_DIR, "spatial_snapshots.png"))

    # 3. Temporal traces
    print("  [3/5] Temporal traces...")
    plot_temporal_traces(V_history, t_saved, x_grid,
                          probe_positions=[0, 5, 10, 15, 20],
                          save_path=os.path.join(OUTPUT_DIR, "temporal_traces.png"))

    # 4. Phase plane (already generated in single-compartment section)
    print("  [4/5] Phase plane — already generated.")

    # 5. Animation
    print("  [5/5] Animated propagation (this may take a moment)...")
    create_animation(V_history, t_saved, x_grid,
                     save_path=os.path.join(OUTPUT_DIR, "propagation.gif"),
                     frame_skip=3, fps=25)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Single-compartment uses default params (g_leak = 2.0, Figure 8.6)
    params_ode = default_params()
    # Cable simulation uses modified params (g_leak = 0.5, Figure 8.12)
    params_cable = cable_params()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   MORRIS-LECAR ACTION POTENTIAL PROPAGATION SIMULATION   ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Parameters from Ingalls (2012), Chapter 8, Figures 8.6/8.12")

    # Part 1: Single compartment (uses g_leak = 2.0)
    V_rest, w_rest = run_single_compartment(params_ode)

    # Part 2: Spatial propagation (uses g_leak = 0.5 per Fig 8.12)
    V_rest_cable, w_rest_cable = resting_state(params_cable)
    print(f"\n  Cable resting state: V_rest = {V_rest_cable:.2f} mV, w_rest = {w_rest_cable:.4f}")
    V_history, w_history, t_saved, x_grid = run_cable_simulation(
        params_cable, V_rest_cable, w_rest_cable
    )

    # Part 3: Visualizations
    generate_visualizations(V_history, t_saved, x_grid, params_cable)

    print("\n" + "="*60)
    print(f" ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
