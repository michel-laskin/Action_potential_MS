"""
PyQt5 GUI for exploring Multiple Sclerosis action potential propagation.

Provides interactive controls for demyelination parameters and side-by-side
visualizations comparing healthy vs MS-affected action potential propagation.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QGroupBox, QLabel, QDoubleSpinBox,
    QSpinBox, QPushButton, QProgressBar, QScrollArea, QFrame,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, TwoSlopeNorm

from morris_lecar import (
    MLParams, cable_params, resting_state, initial_voltage_pulse,
    w_inf, simulate_cable
)
from ms_simulation import (
    generate_lesions, create_spatial_arrays, simulate_cable_ms
)


# ──────────────────────────────────────────────────────────────────────────────
# Color scheme
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    'bg': '#1e1e2e',
    'panel': '#2a2a3d',
    'card': '#333350',
    'fg': '#e0e0e0',
    'fg_dim': '#8888aa',
    'accent': '#6c9fff',
    'healthy': '#4caf50',
    'mild': '#ffd700',
    'moderate': '#ff8c00',
    'severe': '#ff3b3b',
    'border': '#444466',
}

SEVERITY_COLORS = {
    0: '#555577',
    1: COLORS['mild'],
    2: COLORS['moderate'],
    3: COLORS['severe'],
}

SEVERITY_NAMES = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}

# Matplotlib style for embedded figures
MPL_STYLE = {
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': '#2a2a3d',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'text.color': '#000000',
    'grid.color': '#3a3a55',
    'grid.alpha': 0.5,
}

TEXT_COLOR = '#000000'
TICK_COLOR = '#000000'


# ──────────────────────────────────────────────────────────────────────────────
# Simulation worker thread
# ──────────────────────────────────────────────────────────────────────────────

class SimulationWorker(QThread):
    """Runs both healthy and MS simulations in a background thread."""
    progress = pyqtSignal(float, str)   # (fraction, status_text)
    finished = pyqtSignal(dict)          # results dict
    error = pyqtSignal(str)

    def __init__(self, params, lesions, severity_C, severity_g_leak, sim_time):
        super().__init__()
        self.params = params
        self.lesions = lesions
        self.severity_C = severity_C
        self.severity_g_leak = severity_g_leak
        self.sim_time = sim_time

    def run(self):
        try:
            params = self.params
            V_rest, w_rest = resting_state(params)
            x_grid = np.linspace(0, params.L, params.N)

            C_arr, g_leak_arr, severity_map = create_spatial_arrays(
                x_grid, self.lesions,
                params.C, params.g_leak,
                self.severity_C, self.severity_g_leak,
            )

            V_init = initial_voltage_pulse(x_grid, V_rest, amplitude=60.0, width_power=8)
            w_init = np.full(params.N, w_rest)
            T = self.sim_time

            # Healthy simulation
            self.progress.emit(0.05, "Running healthy simulation…")
            V_h, w_h, t_h, x_h = simulate_cable(
                params, V_init.copy(), w_init.copy(), T=T, dt=0.01, save_every=10
            )

            # MS simulation
            self.progress.emit(0.30, "Running MS simulation…")

            def ms_cb(frac):
                self.progress.emit(0.30 + 0.65 * frac,
                                   f"MS simulation: {int(frac*100)}%")

            V_ms, w_ms, t_ms, x_ms = simulate_cable_ms(
                params, V_init.copy(), w_init.copy(),
                C_arr, g_leak_arr,
                T=T, dt=0.01, save_every=10,
                progress_callback=ms_cb,
            )

            self.progress.emit(1.0, "Done!")
            self.finished.emit({
                'V_healthy': V_h, 't_healthy': t_h, 'x_healthy': x_h,
                'V_ms': V_ms, 't_ms': t_ms, 'x_ms': x_ms,
                'C_arr': C_arr, 'g_leak_arr': g_leak_arr,
                'severity_map': severity_map, 'x_grid': x_grid,
                'lesions': self.lesions,
            })

        except Exception as e:
            self.error.emit(str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Main GUI window
# ──────────────────────────────────────────────────────────────────────────────

class MSSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ MS Action Potential Propagation Simulator")
        self.setMinimumSize(1300, 750)
        self.resize(1500, 870)
        self._apply_dark_theme()
        self._build_ui()
        self.worker = None

    # ── Theme ──

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(COLORS['bg']))
        palette.setColor(QPalette.WindowText, QColor(COLORS['fg']))
        palette.setColor(QPalette.Base, QColor(COLORS['card']))
        palette.setColor(QPalette.AlternateBase, QColor(COLORS['panel']))
        palette.setColor(QPalette.Text, QColor(COLORS['fg']))
        palette.setColor(QPalette.Button, QColor(COLORS['card']))
        palette.setColor(QPalette.ButtonText, QColor(COLORS['fg']))
        palette.setColor(QPalette.Highlight, QColor(COLORS['accent']))
        palette.setColor(QPalette.HighlightedText, QColor('#ffffff'))
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QMainWindow {{ background: {COLORS['bg']}; }}
            QGroupBox {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 14px;
                padding: 12px 8px 8px 8px;
                font-weight: bold;
                color: #000000;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
            QLabel {{ color: #000000; }}
            QDoubleSpinBox, QSpinBox {{
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 3px 6px;
                color: #ffffff;
                min-height: 22px;
            }}
            QPushButton#simulate {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5b7fff, stop:1 #7c5cff);
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                min-height: 30px;
            }}
            QPushButton#simulate:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6b8fff, stop:1 #8c6cff);
            }}
            QPushButton#simulate:disabled {{
                background: {COLORS['border']};
                color: #888;
            }}
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                text-align: center;
                color: white;
                background: {COLORS['card']};
                min-height: 18px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5b7fff, stop:1 #7c5cff);
                border-radius: 3px;
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background: {COLORS['bg']};
            }}
            QTabBar::tab {{
                background: {COLORS['card']};
                border: 1px solid {COLORS['border']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: {COLORS['fg_dim']};
            }}
            QTabBar::tab:selected {{
                background: {COLORS['panel']};
                color: {COLORS['accent']};
                font-weight: bold;
            }}
            QScrollArea {{ border: none; background: {COLORS['bg']}; }}
        """)

    # ── UI construction ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: scrollable control panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        ctrl_widget = QWidget()
        self.ctrl_layout = QVBoxLayout(ctrl_widget)
        self.ctrl_layout.setSpacing(8)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # Right: tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        self._build_controls()
        self._build_tabs()

    def _make_int_spin(self, val, lo, hi, step=1):
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setSingleStep(step)
        return sb

    def _make_dbl_spin(self, val, lo, hi, step=0.1, decimals=2):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        return sb

    def _add_row(self, grid, row, label, widget, unit=""):
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)
        if unit:
            u = QLabel(unit)
            u.setStyleSheet("color: #000000; font-size: 11px;")
            grid.addWidget(u, row, 2)

    def _build_controls(self):
        # ── Lesion Generation ──
        grp = QGroupBox("🧬  Lesion Generation")
        grid = QGridLayout()
        grp.setLayout(grid)

        self.spin_n_lesions = self._make_int_spin(5, 1, 30)
        self.spin_seed = self._make_int_spin(42, 0, 99999)
        self.spin_mean_len = self._make_dbl_spin(0.4, 0.05, 5.0, 0.05)
        self.spin_std_len = self._make_dbl_spin(0.1, 0.01, 2.0, 0.01)

        self._add_row(grid, 0, "# Lesions:", self.spin_n_lesions)
        self._add_row(grid, 1, "Seed:", self.spin_seed)
        self._add_row(grid, 2, "Mean length:", self.spin_mean_len, "cm")
        self._add_row(grid, 3, "Std length:", self.spin_std_len, "cm")
        self.ctrl_layout.addWidget(grp)

        # ── Severity parameter groups ──
        sev_config = [
            (1, "🟡  Mild  (Severity 1)", 40.0, 2.0),
            (2, "🟠  Moderate  (Severity 2)", 60.0, 4.0),
            (3, "🔴  Severe  (Severity 3)", 100.0, 8.0),
        ]
        self.sev_spins = {}  # {severity: {'C': spin, 'g_leak': spin}}

        for sev, title, c_def, gl_def in sev_config:
            grp = QGroupBox(title)
            grid = QGridLayout()
            grp.setLayout(grid)

            sp_c = self._make_dbl_spin(c_def, 20.0, 200.0, 5.0, 1)
            sp_gl = self._make_dbl_spin(gl_def, 0.5, 20.0, 0.5, 1)
            self.sev_spins[sev] = {'C': sp_c, 'g_leak': sp_gl}

            self._add_row(grid, 0, "C:", sp_c, "µF/cm²")
            self._add_row(grid, 1, "g_leak:", sp_gl, "mS/cm²")
            self.ctrl_layout.addWidget(grp)

        # ── Simulation ──
        grp = QGroupBox("⚙️  Simulation")
        vbox = QVBoxLayout()
        grp.setLayout(vbox)

        time_grid = QGridLayout()
        self.spin_sim_time = self._make_dbl_spin(200, 50, 1000, 50, 0)
        self._add_row(time_grid, 0, "Time:", self.spin_sim_time, "ms")
        vbox.addLayout(time_grid)

        self.btn_simulate = QPushButton("▶   Simulate")
        self.btn_simulate.setObjectName("simulate")
        self.btn_simulate.clicked.connect(self._on_simulate)
        vbox.addWidget(self.btn_simulate)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        vbox.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #000000; font-size: 11px;")
        vbox.addWidget(self.status_label)
        self.ctrl_layout.addWidget(grp)

        # ── Healthy baseline info ──
        grp = QGroupBox("ℹ️  Healthy Baseline")
        vbox = QVBoxLayout()
        grp.setLayout(vbox)
        info = QLabel("C = 20.0 µF/cm²\ng_leak = 0.5 mS/cm²\ng_Ca = 4.4 mS/cm²\n"
                       "g_K = 8.0 mS/cm²\nAxon: 30 cm, 500 pts")
        info.setFont(QFont("Consolas", 9))
        info.setStyleSheet("color: #000000;")
        vbox.addWidget(info)
        self.ctrl_layout.addWidget(grp)

        self.ctrl_layout.addStretch()

    def _build_tabs(self):
        self.fig_map = {}
        self.canvas_map = {}
        tab_names = ['Lesion Map', 'Heatmap', 'Spatial Snapshots', 'Temporal Traces']

        for name in tab_names:
            widget = QWidget()
            vbox = QVBoxLayout(widget)
            vbox.setContentsMargins(4, 4, 4, 4)

            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, widget)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas)

            self.fig_map[name] = fig
            self.canvas_map[name] = canvas
            self.tabs.addTab(widget, f"  {name}  ")

            # Placeholder text
            with plt.rc_context(MPL_STYLE):
                ax = fig.add_subplot(111)
                ax.set_facecolor(COLORS['panel'])
                ax.text(0.5, 0.5, 'Press "Simulate" to generate visualizations',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, color=COLORS['fg_dim'], style='italic')
                ax.set_xticks([])
                ax.set_yticks([])
            canvas.draw()

    # ── Simulation ──

    def _on_simulate(self):
        if self.worker and self.worker.isRunning():
            return

        self.btn_simulate.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Generating lesions…")

        params = cable_params()
        lesions = generate_lesions(
            axon_length=params.L,
            n_lesions=self.spin_n_lesions.value(),
            mean_length=self.spin_mean_len.value(),
            std_length=self.spin_std_len.value(),
            seed=self.spin_seed.value(),
        )

        severity_C = {s: self.sev_spins[s]['C'].value() for s in [1, 2, 3]}
        severity_g_leak = {s: self.sev_spins[s]['g_leak'].value() for s in [1, 2, 3]}

        self.worker = SimulationWorker(params, lesions, severity_C,
                                        severity_g_leak, self.spin_sim_time.value())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, frac, status):
        self.progress.setValue(int(frac * 100))
        self.status_label.setText(status)

    def _on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        self.btn_simulate.setEnabled(True)

    def _on_finished(self, results):
        self.progress.setValue(100)
        self.status_label.setText("Rendering plots…")
        QApplication.processEvents()

        self._plot_lesion_map(results)
        self._plot_heatmaps(results)
        self._plot_snapshots(results)
        self._plot_traces(results)

        self.status_label.setText("Done ✓")
        self.btn_simulate.setEnabled(True)

    # ── Plotting ──

    def _plot_lesion_map(self, r):
        fig = self.fig_map['Lesion Map']
        fig.clear()

        with plt.rc_context(MPL_STYLE):
            gs = fig.add_gridspec(3, 1, hspace=0.45, left=0.08, right=0.95,
                                  top=0.92, bottom=0.08)

            # 1) Axon diagram with lesions
            ax1 = fig.add_subplot(gs[0])
            x = r['x_grid']
            smap = r['severity_map']
            ax1.axhspan(0, 1, color=COLORS['healthy'], alpha=0.25, label='Healthy')

            for sev, color in [(1, COLORS['mild']), (2, COLORS['moderate']),
                               (3, COLORS['severe'])]:
                mask = smap == sev
                if mask.any():
                    for start, end in self._contiguous_ranges(x, mask):
                        ax1.axvspan(start, end, 0.0, 1.0, color=color, alpha=0.75)

            # Legend patches
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=COLORS['healthy'], alpha=0.3, label='Healthy'),
                       mpatches.Patch(color=COLORS['mild'], alpha=0.75, label='Mild'),
                       mpatches.Patch(color=COLORS['moderate'], alpha=0.75, label='Moderate'),
                       mpatches.Patch(color=COLORS['severe'], alpha=0.75, label='Severe')]
            ax1.legend(handles=patches, loc='upper right', fontsize=8,
                       facecolor=COLORS['panel'], edgecolor=COLORS['border'])
            ax1.set_xlim(x[0], x[-1])
            ax1.set_yticks([])
            ax1.set_title('Demyelination Lesion Map', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Position along axon (cm)')
            self._style_ax(ax1)

            # 2) C(x) profile
            ax2 = fig.add_subplot(gs[1])
            ax2.fill_between(x, r['C_arr'], alpha=0.5, color=COLORS['accent'])
            ax2.plot(x, r['C_arr'], color=COLORS['accent'], lw=1.2)
            ax2.set_ylabel('C (µF/cm²)')
            ax2.set_xlabel('Position (cm)')
            ax2.set_xlim(x[0], x[-1])
            ax2.set_title('Membrane Capacitance Profile', fontsize=10)
            ax2.grid(True, alpha=0.3)
            self._style_ax(ax2)

            # 3) g_leak(x) profile
            ax3 = fig.add_subplot(gs[2])
            ax3.fill_between(x, r['g_leak_arr'], alpha=0.5, color='#ff7b7b')
            ax3.plot(x, r['g_leak_arr'], color='#ff7b7b', lw=1.2)
            ax3.set_ylabel('g_leak (mS/cm²)')
            ax3.set_xlabel('Position (cm)')
            ax3.set_xlim(x[0], x[-1])
            ax3.set_title('Leak Conductance Profile', fontsize=10)
            ax3.grid(True, alpha=0.3)
            self._style_ax(ax3)

        self.canvas_map['Lesion Map'].draw()

    def _plot_heatmaps(self, r):
        fig = self.fig_map['Heatmap']
        fig.clear()

        with plt.rc_context(MPL_STYLE):
            gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.93,
                                  top=0.90, bottom=0.10)

            all_min = min(r['V_healthy'].min(), r['V_ms'].min())
            all_max = max(r['V_healthy'].max(), r['V_ms'].max())
            v_rest = -60.0
            norm = TwoSlopeNorm(vmin=all_min, vcenter=v_rest, vmax=all_max)

            for idx, (label, Vkey, tkey, xkey) in enumerate([
                ('Healthy', 'V_healthy', 't_healthy', 'x_healthy'),
                ('MS (Demyelinated)', 'V_ms', 't_ms', 'x_ms'),
            ]):
                ax = fig.add_subplot(gs[idx])
                V = r[Vkey]
                t = r[tkey]
                x = r[xkey]
                im = ax.pcolormesh(x, t, V, cmap='RdBu_r', norm=norm,
                                   shading='auto', rasterized=True)
                ax.set_xlabel('Position (cm)')
                ax.set_ylabel('Time (ms)')
                ax.set_title(label, fontsize=12, fontweight='bold')
                self._style_ax(ax)
                cb = fig.colorbar(im, ax=ax, label='V (mV)', shrink=0.85)
                cb.ax.yaxis.set_tick_params(color=TICK_COLOR)
                cb.ax.yaxis.label.set_color(TEXT_COLOR)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color=TICK_COLOR)

        self.canvas_map['Heatmap'].draw()

    def _plot_snapshots(self, r):
        fig = self.fig_map['Spatial Snapshots']
        fig.clear()
        snapshot_times = [0, 20, 50, 100, 150]
        colors_cycle = ['#6c9fff', '#ff6b9d', '#ffd93d', '#6bcb77', '#c36cff']

        with plt.rc_context(MPL_STYLE):
            gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.95,
                                  top=0.90, bottom=0.10)

            for idx, (label, Vkey, tkey, xkey) in enumerate([
                ('Healthy', 'V_healthy', 't_healthy', 'x_healthy'),
                ('MS', 'V_ms', 't_ms', 'x_ms'),
            ]):
                ax = fig.add_subplot(gs[idx])
                V, t, x = r[Vkey], r[tkey], r[xkey]
                for i, st in enumerate(snapshot_times):
                    tidx = np.argmin(np.abs(t - st))
                    ax.plot(x, V[tidx], color=colors_cycle[i % len(colors_cycle)],
                            lw=1.5, label=f't = {st} ms')
                ax.set_xlabel('Position (cm)')
                ax.set_ylabel('V (mV)')
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, facecolor=COLORS['panel'],
                          edgecolor=COLORS['border'], labelcolor=TEXT_COLOR)
                ax.grid(True, alpha=0.3)
                self._style_ax(ax)

        self.canvas_map['Spatial Snapshots'].draw()

    def _plot_traces(self, r):
        fig = self.fig_map['Temporal Traces']
        fig.clear()
        probe_positions = [0, 5, 10, 15, 20]
        colors_cycle = ['#6c9fff', '#ff6b9d', '#ffd93d', '#6bcb77', '#c36cff']

        with plt.rc_context(MPL_STYLE):
            gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.95,
                                  top=0.90, bottom=0.10)

            for idx, (label, Vkey, tkey, xkey) in enumerate([
                ('Healthy', 'V_healthy', 't_healthy', 'x_healthy'),
                ('MS', 'V_ms', 't_ms', 'x_ms'),
            ]):
                ax = fig.add_subplot(gs[idx])
                V, t, x = r[Vkey], r[tkey], r[xkey]
                for i, pos in enumerate(probe_positions):
                    xidx = np.argmin(np.abs(x - pos))
                    ax.plot(t, V[:, xidx], color=colors_cycle[i % len(colors_cycle)],
                            lw=1.2, label=f'x = {pos} cm')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('V (mV)')
                ax.set_title(label, fontsize=12, fontweight='bold')
                ax.legend(fontsize=8, facecolor=COLORS['panel'],
                          edgecolor=COLORS['border'], labelcolor=TEXT_COLOR)
                ax.grid(True, alpha=0.3)
                self._style_ax(ax)

        self.canvas_map['Temporal Traces'].draw()

    # ── Helpers ──

    @staticmethod
    def _style_ax(ax):
        """Force bright text on every axis element for readability."""
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(axis='x', colors=TICK_COLOR, labelsize=9)
        ax.tick_params(axis='y', colors=TICK_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#8888aa')

    @staticmethod
    def _contiguous_ranges(x, mask):
        """Yield (start, end) x-values for contiguous True regions in mask."""
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask)]
        for s, e in zip(starts, ends):
            yield x[s], x[min(e, len(x)-1)]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MSSimulatorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
