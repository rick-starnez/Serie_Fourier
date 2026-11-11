import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle


# ---------- Generadores de señales ----------
def gen_square(N):
    x = np.ones(N)
    x[N // 2:] = -1.0
    return x


def gen_sine(N, freq=1):
    t = np.linspace(0, 1, N, endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def gen_triangle(N, freq=1):
    t = np.linspace(0, 1, N, endpoint=False)
    return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1


def gen_random(N, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, N)


# ---------- Reconstrucciones ----------
def fourier_trig(x, K):
    N = len(x)
    t = np.arange(N) / N
    a0 = 2.0 / N * np.sum(x)
    y = np.full(N, a0 / 2.0)
    for k in range(1, K + 1):
        ak = 2.0 / N * np.sum(x * np.cos(2 * np.pi * k * t))
        bk = 2.0 / N * np.sum(x * np.sin(2 * np.pi * k * t))
        y += ak * np.cos(2 * np.pi * k * t) + bk * np.sin(2 * np.pi * k * t)
    return y


def fourier_complex(x, K):
    N = len(x)
    t = np.arange(N) / N
    c = np.fft.fft(x) / N
    y = np.zeros(N, dtype=complex)
    y += c[0]
    for k in range(1, K + 1):
        y += c[k] * np.exp(2j * np.pi * k * t)
        y += c[-k] * np.exp(-2j * np.pi * k * t)
    return np.real(y)


# ---------- Gestor de zoom por arrastre directo ----------
class DragZoom:
    """
    Zoom por arrastre con el botón izquierdo sobre un eje.
    - Dibuja un rectángulo mientras arrastras.
    - Al soltar, hace zoom a ese rectángulo (en X e Y).
    - Doble clic: reset del eje.
    """
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.x0 = None
        self.y0 = None
        self.rect = None
        self.dragging = False

        cid_press = self.fig.canvas.mpl_connect(
            'button_press_event', self._on_press)
        cid_release = self.fig.canvas.mpl_connect(
            'button_release_event', self._on_release)
        cid_motion = self.fig.canvas.mpl_connect(
            'motion_notify_event', self._on_motion)

        # Guardamos referencias para no desconectar
        self._cids = (cid_press, cid_release, cid_motion)

    def _on_press(self, event):
        # Solo dentro de nuestro eje y con botón izquierdo
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Doble clic → reset
        if event.dblclick:
            self._reset()
            return

        # Iniciar arrastre
        self.dragging = True
        self.x0, self.y0 = event.xdata, event.ydata

        # Rectángulo visual
        if self.rect is not None:
            try:
                self.rect.remove()
            except Exception:
                pass
        self.rect = Rectangle(
            (self.x0, self.y0), 0, 0,
            facecolor="tab:red", alpha=0.15,
            edgecolor="tab:red", linewidth=1,
        )
        self.ax.add_patch(self.rect)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = min(self.x0, event.xdata)
        y = min(self.y0, event.ydata)
        w = abs(event.xdata - self.x0)
        h = abs(event.ydata - self.y0)
        self.rect.set_xy((x, y))
        self.rect.set_width(w)
        self.rect.set_height(h)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False

        if event.button != 1:
            self._remove_rect()
            return

        # Si el release cae fuera del eje, igual usamos las coords capturadas
        if event.xdata is None or event.ydata is None:
            self._remove_rect()
            return

        x1, y1 = self.x0, self.y0
        x2, y2 = event.xdata, event.ydata

        self._remove_rect()

        # Ignorar si el rectángulo es despreciable
        if abs(x2 - x1) < 1e-9 or abs(y2 - y1) < 1e-12:
            return

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.fig.canvas.draw_idle()

    def _remove_rect(self):
        if self.rect is not None:
            try:
                self.rect.remove()
            except Exception:
                pass
            self.rect = None
            self.fig.canvas.draw_idle()

    def _reset(self):
        data = getattr(self.ax, "_zoom_data", None)
        if data is None:
            self.ax.relim()
            self.ax.autoscale()
        else:
            (x0, x1), (y0, y1) = data
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
        self.fig.canvas.draw_idle()


# ---------- Aplicación ----------
class FourierSeriesSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Serie Trigonométrica vs Serie Compleja de Fourier")
        self.root.geometry("1400x880")

        # ---------- Barra superior ----------
        top = ttk.Frame(root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Señal:").pack(side=tk.LEFT, padx=(0, 6))
        self.signal_var = tk.StringVar(value="Cuadrada")
        combo = ttk.Combobox(
            top, textvariable=self.signal_var,
            values=["Cuadrada", "Seno", "Triangular", "Random"],
            state="readonly", width=15,
        )
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        ttk.Label(top, text="   N muestras:").pack(side=tk.LEFT, padx=(12, 6))
        self.N_var = tk.IntVar(value=512)
        N_spin = ttk.Spinbox(
            top, from_=64, to=4096, increment=64,
            textvariable=self.N_var, width=8,
            command=self.update_plots,
        )
        N_spin.pack(side=tk.LEFT)
        N_spin.bind("<Return>", lambda e: self.update_plots())

        ttk.Label(top, text="   Coeficientes K:").pack(side=tk.LEFT, padx=(12, 6))
        self.K_var = tk.IntVar(value=5)
        K_spin = ttk.Spinbox(
            top, from_=1, to=200, increment=1,
            textvariable=self.K_var, width=8,
            command=self.update_plots,
        )
        K_spin.pack(side=tk.LEFT)
        K_spin.bind("<Return>", lambda e: self.update_plots())

        ttk.Button(top, text="Reset zoom", command=self.reset_zoom).pack(
            side=tk.LEFT, padx=(16, 0))

        self.info_label = ttk.Label(top, text="")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        hint = ("Zoom: arrastra con el botón izquierdo sobre la gráfica de reconstrucción  |  "
                "Reset: doble clic o botón 'Reset zoom'")
        ttk.Label(root, text=hint, foreground="#666").pack(
            side=tk.TOP, anchor="w", padx=14, pady=(0, 4))

        # ---------- Dos columnas ----------
        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Serie Trigonométrica de Fourier")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        right = ttk.LabelFrame(body, text="Serie Compleja de Fourier")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig_trig, self.ax_trig_rec, self.ax_trig_mag = self._make_figure()
        self.fig_cplx, self.ax_cplx_rec, self.ax_cplx_mag = self._make_figure()

        self.canvas_trig = FigureCanvasTkAgg(self.fig_trig, master=left)
        self.canvas_trig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas_cplx = FigureCanvasTkAgg(self.fig_cplx, master=right)
        self.canvas_cplx.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Instalamos el zoom por arrastre solo en las gráficas de reconstrucción
        self._zoom_trig = DragZoom(self.ax_trig_rec)
        self._zoom_cplx = DragZoom(self.ax_cplx_rec)

        self.update_plots()

    def _make_figure(self):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        fig.tight_layout(pad=2.0)
        return fig, axes[0], axes[1]

    # ---------- Reset global ----------
    def reset_zoom(self):
        for z in (self._zoom_trig, self._zoom_cplx):
            z._reset()

    # ---------- Señal ----------
    def _make_signal(self, name, N):
        if name == "Cuadrada":
            return gen_square(N)
        elif name == "Seno":
            return gen_sine(N, freq=3)
        elif name == "Triangular":
            return gen_triangle(N, freq=2)
        else:
            return gen_random(N, seed=7)

    # ---------- Actualización ----------
    def update_plots(self):
        import time

        name = self.signal_var.get()
        N = int(self.N_var.get())
        K = max(1, int(self.K_var.get()))
        K = min(K, N // 2 - 1)
        self.K_var.set(K)

        x = self._make_signal(name, N)
        t = np.arange(N) / N

        t0 = time.perf_counter()
        y_trig = fourier_trig(x, K)
        t_trig = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        y_cplx = fourier_complex(x, K)
        t_cplx = (time.perf_counter() - t0) * 1000

        err = np.max(np.abs(y_trig - y_cplx))
        self.info_label.config(
            text=f"Trig: {t_trig:.2f} ms | Cplx: {t_cplx:.2f} ms | err: {err:.2e}"
        )

        # Coeficientes trigonométricos
        a0 = 2.0 / N * np.sum(x)
        ks = np.arange(1, K + 1)
        ak = np.array([2.0 / N * np.sum(x * np.cos(2 * np.pi * k * t)) for k in ks])
        bk = np.array([2.0 / N * np.sum(x * np.sin(2 * np.pi * k * t)) for k in ks])
        mag_trig = np.sqrt(ak**2 + bk**2)

        # Coeficientes complejos — SOLO k >= 0 para la gráfica
        c = np.fft.fft(x) / N
        ks_c = np.arange(0, K + 1)                # 0, 1, ..., K
        c_sel = np.array([c[k] for k in ks_c])    # c_0, c_1, ..., c_K (positivos)
        mag_cplx = np.abs(c_sel)

        # --- Trigonométrica (reconstrucción) ---
        ax = self.ax_trig_rec
        ax.clear()
        ax.plot(t, x, color="gray", linewidth=1.2, label="Original")
        ax.plot(t, y_trig, color="tab:blue", linewidth=1.6,
                label=f"Reconstrucción (K={K})")
        ax.set_title(f"Serie Trigonométrica — {name} (N={N}, K={K})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self._save_full_limits(ax, [x, y_trig], t)

        # --- Trigonométrica (magnitud) ---
        ax = self.ax_trig_mag
        ax.clear()
        ax.stem(ks, mag_trig, basefmt=" ",
                linefmt="tab:blue", markerfmt="o")
        ax.set_title("Magnitud √(aₖ² + bₖ²)")
        ax.set_xlabel("k")
        ax.set_ylabel("Magnitud")
        ax.grid(True, alpha=0.3)

        # --- Compleja (reconstrucción) ---
        ax = self.ax_cplx_rec
        ax.clear()
        ax.plot(t, x, color="gray", linewidth=1.2, label="Original")
        ax.plot(t, y_cplx, color="tab:orange", linewidth=1.6,
                label=f"Reconstrucción (K={K})")
        ax.set_title(f"Serie Compleja — {name} (N={N}, K={K})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self._save_full_limits(ax, [x, y_cplx], t)

        # --- Compleja (magnitud) — solo k >= 0 ---
        ax = self.ax_cplx_mag
        ax.clear()
        ax.stem(ks_c, mag_cplx, basefmt=" ",
                linefmt="tab:orange", markerfmt="o")
        ax.set_title("Magnitud |cₖ| (k ≥ 0)")
        ax.set_xlabel("k")
        ax.set_ylabel("|cₖ|")
        ax.grid(True, alpha=0.3)

        self.fig_trig.tight_layout()
        self.fig_cplx.tight_layout()
        self.canvas_trig.draw()
        self.canvas_cplx.draw()

    def _save_full_limits(self, ax, y_list, t):
        y_all = np.concatenate(y_list)
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        pad = (y_max - y_min) * 0.1 or 1e-3
        ax._zoom_data = (
            (float(t[0]), float(t[-1])),
            (y_min - pad, y_max + pad),
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = FourierSeriesSimulator(root)
    root.mainloop()