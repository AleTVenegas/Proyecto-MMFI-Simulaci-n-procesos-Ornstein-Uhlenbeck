import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from PIL import ImageTk, Image
import tkinter.messagebox as mb
from scipy.stats import norm
from scipy.stats import gaussian_kde


def graphs(parametros):
    """Generate two series for plotting from the given parameter dict.

    Returns:
    - data1: (x, y, title) - Mean trajectory
    - data2: (x, y_emp, y_an, title) - Empirical vs analytical distributions
    - trajectories: array of all simulated trajectories
    """
    # read parameters with sensible defaults
    try:
        R = float(parametros.get("Resistencia R (Ω)", 2000.0))
        C = float(parametros.get("Capacitancia C (µF)", 1000.0)) * 1e-6
        V0 = float(parametros.get("Voltaje inicial V₀ (V)", 0.0))
        Vf = float(parametros.get("Voltaje de la fuente Vε (V)", 10.0))
        sigma = float(parametros.get("Intensidad del ruido σ", 1.0))
        dt = float(parametros.get("Paso temporal Δt (s)", 0.01))
        T = float(parametros.get("Tiempo total T (s)", 10.0))
        n_traj = int(max(1, int(parametros.get("Cantidad de corridas", 20))))
    except Exception:
        # fallback defaults if parsing fails
        R, C, V0, Vf, sigma, dt, T, n_traj = 2000.0, 1e-3, 0.0, 10.0, 1.0, 0.01, 10.0, 20

    # compute derived quantities
    if C <= 0:
        C = 1e-6
    theta = 1.0 / (R * C) if R * C != 0 else 1.0

    N = max(2, int(np.ceil(T / dt)))
    x = np.linspace(0, T, N)

    # Vectorized simulation: shape (n_traj, N)
    trajectories = np.zeros((n_traj, N))
    trajectories[:, 0] = V0
    # random increments for all trajectories and timesteps (N-1)
    rng = np.random.default_rng()
    normals = rng.standard_normal(size=(n_traj, N - 1))

    sqrt_dt = np.sqrt(dt)
    for t in range(1, N):
        trajectories[:, t] = (
            trajectories[:, t - 1]
            + theta * (Vf - trajectories[:, t - 1]) * dt
            + sigma * sqrt_dt * normals[:, t - 1]
        )

    mean_traj = np.mean(trajectories, axis=0)

    # Data1: Mean trajectory over time
    data1 = (x, mean_traj, f"Media de {n_traj} trayectorias (OU)")

    # Data2: Empirical vs analytical distributions
    X_all = trajectories.flatten()
    if len(X_all) < 2:
        X_all = np.array([X_all[0], X_all[0]+1e-6])
    
    kde = gaussian_kde(X_all)
    x_vals = np.linspace(np.min(X_all)-0.1, np.max(X_all)+0.1, 300)
    pdf_sim = kde(x_vals)

    mean_an = Vf + (V0 - Vf)*np.exp(-theta*T)
    var_an = (sigma**2)/(2*theta)*(1 - np.exp(-2*theta*T))
    pdf_an = norm.pdf(x_vals, mean_an, np.sqrt(var_an))

    dx = x_vals[1] - x_vals[0]
    L2_error = np.sqrt(np.sum((pdf_sim - pdf_an)**2) * dx)
    overlap = np.sum(np.minimum(pdf_sim, pdf_an) * dx)

    # Store both empirical and analytical PDFs for plotting
    data2 = {
        'x': x_vals,
        'pdf_sim': pdf_sim,
        'pdf_an': pdf_an,
        'L2_error': L2_error,
        'n_traj': n_traj,
        'overlap': overlap
    }

    return data1, data2, trajectories

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto MMFI: Simulación de Procesos Ornstein-Uhlenbeck")
        self.configure(bg="#222020")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Fullscreen mode
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind("<F11>", self._toggle_fullscreen)

        # Shared data
        self.initial_values = {}
        self.canvas = None
        self.toolbar = None
        
        self.key_sequence = ""
        self.bind("<Key>", self._on_key_press)

        # --- Create frames (screens) ---
        self.frames = {}
        for F in (PresentationScreen, InputScreen, GraphScreen, ImplementacionesAdicionalesScreen):
            # pass controller (self) into frames so they can access shared state
            frame = F(self, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(PresentationScreen)

    def show_frame(self, screen):
        frame = self.frames[screen]
        # Track previous frame if navigating to Implementaciones Adicionales
        if screen == ImplementacionesAdicionalesScreen:
            current_frame = None
            for F, f in self.frames.items():
                if f.winfo_viewable():
                    current_frame = F
                    break
            frame.previous_frame = current_frame if current_frame else InputScreen
        frame.tkraise()

    def _toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode on F11."""
        current_state = self.attributes("-fullscreen")
        self.attributes("-fullscreen", not current_state)

    def display_figure(self, data_list, compare=False, extra_trajs=None, median_color="blue", empirical_color="blue", analytical_color="red"):
        """Display matplotlib figure with optional extra trajectories.

        Parameters:
        - data_list: list of (x, y, title) tuples
        - compare: if True, plot side-by-side subplots for two items
        - extra_trajs: optional ndarray of shape (n_traj, N) to plot faintly behind the main plot
        - median_color: color for the median line
        """
        frame = self.frames[GraphScreen].graph_display_frame
        label = self.frames[GraphScreen].processing_label

        # --- Clear any previous graph before showing processing label ---
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None

        # --- Show processing label centered ---
        label.place(relx=0.5, rely=0.5, anchor="center")
        label.config(text="Generando gráfica...", foreground="red")
        label.lift()
        
        # --- Disable graph buttons during rendering ---
        self.frames[GraphScreen].btn_media.config(state="disabled")
        self.frames[GraphScreen].btn_pdf.config(state="disabled")
        self.frames[GraphScreen].btn_comparar.config(state="disabled")

        # --- Animate label while processing ---
        running = True

        def animate_label():
            dots = 0
            while running:
                text = "Generando gráfica" + "." * (dots % 4)
                label.config(text=text)
                dots += 1
                time.sleep(0.4)

        anim_thread = threading.Thread(target=animate_label, daemon=True)
        anim_thread.start()

        # --- Render plot in background ---
        def render_plot():
            self.frames[GraphScreen].is_rendering = True
            time.sleep(0.4)  # short fake delay

            fig = Figure(dpi=100)
            
            # Helper to safely get opacity and linewidth
            def get_style_params():
                try:
                    alpha_val = float(self.frames[GraphScreen].opacity_var.get())
                    alpha_val = max(0.0, min(1.0, alpha_val))
                except Exception:
                    alpha_val = 0.15
                try:
                    lw_val = float(self.frames[GraphScreen].lw_var.get())
                    lw_val = max(0.1, lw_val)
                except Exception:
                    lw_val = 0.8
                return alpha_val, lw_val
            
            if compare:
                axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
                
                # First subplot: data1 with optional trajectories
                ax = axes[0]
                x, y, title = data_list[0]
                if extra_trajs is not None:
                    alpha_val, lw_val = get_style_params()
                    try:
                        for traj in extra_trajs:
                            ax.plot(x, traj, color='gray', alpha=alpha_val, linewidth=lw_val)
                    except Exception:
                        pass
                ax.plot(x, y, color=median_color, linewidth=2)
                ax.set_title(title)
                ax.set_xlabel("Tiempo T (s)")
                ax.set_ylabel("Voltaje V (V)")
                
                # Second subplot: data2 (distribution)
                ax = axes[1]
                item2 = data_list[1]
                
                if isinstance(item2, dict) and 'pdf_sim' in item2:
                    x_vals = item2['x']
                    pdf_sim = item2['pdf_sim']
                    pdf_an = item2['pdf_an']
                    L2_error = item2['L2_error']
                    overlap = item2['overlap']
                    n_traj = item2['n_traj']
                    ax.plot(x_vals, pdf_sim, color=empirical_color, label=f'Empírica, %Overlap={overlap*100:.3f}%')
                    ax.plot(x_vals, pdf_an, color=analytical_color, linestyle='--', lw=2, label='Analítica')
                    ax.set_title(f'Densidad OU a partir de {n_traj} trayectorias')
                    ax.set_xlabel("Voltaje V (V)")
                    ax.set_ylabel("Densidades de probabilidad")
                    ax.legend()
            else:
                ax = fig.add_subplot(111)
                item = data_list[0]
                
                if isinstance(item, dict) and 'pdf_sim' in item:
                    x_vals = item['x']
                    pdf_sim = item['pdf_sim']
                    pdf_an = item['pdf_an']
                    L2_error = item['L2_error']
                    overlap = item['overlap']
                    n_traj = item['n_traj']
                    ax.plot(x_vals, pdf_sim, color=empirical_color, label=f'Empírica, %Overlap={overlap*100:.3f}%')
                    ax.plot(x_vals, pdf_an, color=analytical_color, linestyle='--', lw=2, label='Analítica')
                    ax.set_title(f'Densidad OU a partir de {n_traj} trayectorias')
                    ax.set_xlabel("Voltaje V (V)")
                    ax.set_ylabel("Densidades de probabilidad")
                    ax.legend()
                else:
                    # Regular trajectory data with optional faint runs
                    x, y, title = item
                    
                    # Plot faint trajectories only if extra_trajs is explicitly provided (not None)
                    if extra_trajs is not None:
                        alpha_val, lw_val = get_style_params()
                        try:
                            for traj in extra_trajs:
                                ax.plot(x, traj, color='gray', alpha=alpha_val, linewidth=lw_val)
                        except Exception:
                            pass

                    ax.plot(x, y, color=median_color, linewidth=2)
                    ax.set_title(title)
                    ax.set_xlabel("Tiempo T (s)")
                    ax.set_ylabel("Voltaje V (V)")

            fig.tight_layout()

            # Embed new figure
            self.canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.canvas.draw()

            # Toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
            self.toolbar.update()
            self.toolbar.pack(fill=tk.X)

            # Stop animation and hide label
            nonlocal running
            running = False
            label.place_forget()
            self.frames[GraphScreen].is_rendering = False
            
            # --- Re-enable graph buttons after rendering ---
            self.frames[GraphScreen].btn_media.config(state="normal")
            self.frames[GraphScreen].btn_pdf.config(state="normal")
            self.frames[GraphScreen].btn_comparar.config(state="normal")

        threading.Thread(target=render_plot, daemon=True).start()

    def _on_key_press(self, event):
        """Track key presses for implementaciones adicionales sequence."""
        if event.char.isalpha():
            self.key_sequence += event.char.lower()
            # Keep only last 19 characters (length of "losverdaderosheroes")
            if len(self.key_sequence) > 19:
                self.key_sequence = self.key_sequence[-19:]
            
            # Check for the magic sequence
            if self.key_sequence.endswith("losverdaderosheroes"):
                self.show_frame(ImplementacionesAdicionalesScreen)
                self.key_sequence = ""  # Reset after triggering

# --- Presentation screen ---
class PresentationScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)

        self.controller = controller

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        frame_btn_exit = tk.Frame(self, bg="white", bd=2)
        frame_btn_exit.grid(row=0, column=0, sticky="ne", padx=10, pady=10)
        btn_exit = tk.Button(
            frame_btn_exit,
            text="✖ Salir",
            command=master.destroy,
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_exit.pack()

        frame_titulos = tk.Frame(self, bg="#222020")
        frame_titulos.grid(row=0, column=0, pady=(40,10))

        tk.Label(
            frame_titulos,
            text="Métodos Matemáticos para Física e Ingeniería I",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 34, "bold")
        ).pack(side="top", pady=5)

        tk.Label(
            frame_titulos,
            text="Proyecto de Investigación",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 28, "bold")
        ).pack(side="top", pady=5)

        tk.Label(
            frame_titulos,
            text="Simulación de Proceso Ornstein Uhlenbeck en Circuito RC",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 24, "bold")
        ).pack(side="top", pady=5)

        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, sticky="ew", padx=20, pady=(0,20))

        im1 = Image.open('circuito.png').resize((400, 350))
        circuito_img = ImageTk.PhotoImage(im1)

        frame_btn_calc = tk.Frame(self, bg="white", bd=2)
        frame_btn_calc.grid(row=2, column=0, pady=40)

        btn_calc = tk.Button(
            frame_btn_calc,
            text="Ir al Menú de Simulación",
            image=circuito_img,
            compound="top",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            relief="raised",
            cursor="hand2",
            padx=10,
            pady=10,
            command=lambda: controller.show_frame(InputScreen),
            activebackground="#222020",
            activeforeground="white",
        )
        btn_calc.image = circuito_img
        btn_calc.pack()

        frame_integrantes = tk.Frame(self, bg="#222020")
        frame_integrantes.place(relx=0.01, rely=0.98, anchor="sw")

        tk.Label(
            frame_integrantes,
            text="Integrantes",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0,5))

        integrantes = [
            "Chaves Sebastián",
            "Granados John",
            "Navarro Carlos",
            "Torres Alejandro"
        ]

        for i, nombre in enumerate(integrantes, start=1):
            tk.Label(
                frame_integrantes,
                text=nombre,
                bg="#222020",
                fg="white",
                font=("Segoe UI", 12)
            ).grid(row=i, column=0, sticky="w")

        frame_semestre = tk.Frame(self, bg="#222020")
        frame_semestre.place(relx=0.98, rely=0.98, anchor="se")

        tk.Label(
            frame_semestre,
            text="II Semestre",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, sticky="e", pady=(0,2))

        tk.Label(
            frame_semestre,
            text="2025",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).grid(row=1, column=0, sticky="e", pady=(0,5))

# --- Input screen ---
class InputScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)

        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")

        # Buttons and title on row 0
        frame_btn_regresar = tk.Frame(self, bg="white", bd=2)
        frame_btn_regresar.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
        btn_regresar = tk.Button(
            frame_btn_regresar,
            text="← Regresar",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2",
            command=lambda: master.show_frame(PresentationScreen)
        )
        btn_regresar.pack()

        tk.Label(self, text="Simulación del Proceso Ornstein–Uhlenbeck",
                bg="#222020", fg="white",
                font=("Bahnschrift SemiBold Condensed", 28, "bold")).grid(row=0, column=1, sticky="nsew", pady=10)

        frame_btn_salir = tk.Frame(self, bg="white", bd=2)
        frame_btn_salir.grid(row=0, column=2, sticky="ne", padx=10, pady=10)
        btn_salir = tk.Button(
            frame_btn_salir,
            text="✖ Salir",
            command=master.destroy,
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_salir.pack()

        # Line separator on row 1
        linea_titulo = tk.Frame(self, bg="#FFFFFF", height=3)
        linea_titulo.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20, pady=(0,10))

        # Content frame on row 2 with two columns
        frame_contenido = tk.Frame(self, bg="#222020")
        frame_contenido.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        frame_contenido.columnconfigure(0, weight=7)  # Left gets 7x width (siete octavos)
        frame_contenido.columnconfigure(1, weight=1)  # Right gets 1x width (un octavo)
        frame_contenido.rowconfigure(0, weight=1)

        frame_izquierda = tk.Frame(frame_contenido, bg="#222020")
        frame_izquierda.grid(row=0, column=0, sticky="nsew", padx=(40, 15), pady=0)

        im1 = Image.open('circuito.png').resize((400, 350))
        circuito_img = ImageTk.PhotoImage(im1)
        lbl_circuito = tk.Label(frame_izquierda, image=circuito_img, bg="#222020")
        lbl_circuito.image = circuito_img  
        lbl_circuito.pack(expand=True)

        frame_derecha = tk.Frame(frame_contenido, bg="#222020")
        frame_derecha.grid(row=0, column=1, sticky="nsew", padx=(15, 40), pady=0)
        frame_derecha.columnconfigure(0, weight=1)
        frame_derecha.columnconfigure(1, weight=1)

        frame_param_cuadro = tk.Frame(frame_derecha, bg="#333333", bd=2, relief="ridge")
        frame_param_cuadro.pack(padx=50, pady=10, fill="x", expand=True)
        
        # Configure equal column widths for centered layout
        frame_param_cuadro.columnconfigure(0, weight=1)
        frame_param_cuadro.columnconfigure(1, weight=1)

        tk.Label(
            frame_param_cuadro,
            text="Parámetros a ingresar",
            bg="#333333",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 18, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(10,15))

        # parameter labels (these match the keys used by graficas.py)
        parametros_col1 = [
            "Voltaje de la fuente Vε (V)",
            "Voltaje inicial V₀ (V)",
            "Capacitancia C (µF)",
            "Resistencia R (Ω)"
        ]

        parametros_col2 = [
            "Intensidad del ruido σ",
            "Paso temporal Δt (s)",
            "Tiempo total T (s)",
            "Cantidad de corridas"
        ]

        # create a Tk variable for each parameter and place entries in two columns
        self.entries = {}
        all_params = parametros_col1 + parametros_col2
        
        # Define default values for each parameter
        default_values = {
            "Voltaje de la fuente Vε (V)": 10.0,
            "Voltaje inicial V₀ (V)": 0.0,
            "Capacitancia C (µF)": 1000.0,
            "Resistencia R (Ω)": 2000.0,
            "Intensidad del ruido σ": 1.0,
            "Paso temporal Δt (s)": 0.01,
            "Tiempo total T (s)": 100.0,
            "Cantidad de corridas": 20
        }
        
        for idx, nombre in enumerate(all_params):
            col = 0 if idx < len(parametros_col1) else 1
            row_label = (idx % len(parametros_col1)) * 2 + 1
            row_entry = row_label + 1

            tk.Label(
                frame_param_cuadro,
                text=nombre,
                bg="#333333",
                fg="white",
                font=("Segoe UI", 14)
            ).grid(row=row_label, column=col, padx=15, pady=(5,0), sticky="ew")

            # Use IntVar for 'Cantidad de corridas', DoubleVar otherwise
            if nombre == "Cantidad de corridas":
                var = tk.IntVar(value=default_values.get(nombre, 1))
            else:
                var = tk.DoubleVar(value=default_values.get(nombre, 0.0))

            entry = tk.Entry(
                frame_param_cuadro,
                font=("Segoe UI", 14),
                width=15,
                justify="center",
                textvariable=var
            )
            entry.grid(row=row_entry, column=col, padx=15, pady=(0,10), sticky="ew")
            self.entries[nombre] = var

        frame_btn_simular = tk.Frame(frame_param_cuadro, bg="#222020")
        frame_btn_simular.grid(row=12, column=0, columnspan=2, pady=(15,30))

        btn_simular = tk.Button(
            frame_btn_simular,
            text="Iniciar Simulación",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 16, "bold"),
            relief="raised",
            cursor="hand2",
            padx=15,
            pady=10,
            command=lambda: simular()
        )
        btn_simular.pack()

        def simular():
            try:
                # validate and store into controller.shared dict
                for nombre, var in self.entries.items():
                    if nombre == "Cantidad de corridas":
                        valor_num = int(var.get())
                        if valor_num <= 0:
                            raise ValueError(f"{nombre} debe ser un número positivo.")
                    else:
                        # Allow Voltaje inicial to be zero or negative; other params should be > 0
                        valor_num = float(var.get())
                        if nombre not in ["Voltaje inicial V₀ (V)", "Voltaje promedio V_prom (V)"] and valor_num <= 0:
                            raise ValueError(f"{nombre} debe ser un número positivo.")
                    # store validated numeric value in the app's shared dict
                    self.controller.initial_values[nombre] = valor_num

                # switch to graph screen and generate
                self.controller.show_frame(GraphScreen)
                self.controller.frames[GraphScreen].generate_graphs()
            except Exception:
                mb.showerror("Error", f"Error en los valores ingresados:\nPor favor ingrese valores numéricos válidos.")


# --- Graph screen ---
class GraphScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        
        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")
        self.controller = controller
        self.is_rendering = False  # Flag to prevent concurrent renders
        self.pending_render_id = None  # Store scheduled render callback

        # Configure grid weights for full window coverage
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)  # Title column expands to center
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=7)  # Graph area: 9/10 width
        self.columnconfigure(4, weight=1)  # Customization area: 1/10 of width
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)  # Graph display frame expands

        # Buttons and title on row 0 (Regresar left, Title center, Salir right)
        frame_btn_regresar = tk.Frame(self, bg="white", bd=2)
        frame_btn_regresar.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
        btn_regresar = tk.Button(
            frame_btn_regresar,
            text="← Regresar",
            command=lambda: master.show_frame(InputScreen),
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_regresar.pack()

        tk.Label(
            self,
            text="Resultados",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 28, "bold")
        ).grid(row=0, column=1, columnspan=3, sticky="nsew", pady=10)

        frame_btn_salir = tk.Frame(self, bg="white", bd=2)
        frame_btn_salir.grid(row=0, column=4, sticky="ne", padx=10, pady=10)
        btn_salir = tk.Button(
            frame_btn_salir,
            text="✖ Salir",
            command=master.destroy,
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_salir.pack()

        # Line separator on row 1
        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, columnspan=5, sticky="ew", padx=20, pady=(0, 10))
        
        # Update rowconfigure for new layout
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)

        button_frame = ttk.Frame(self, style="Custom.TFrame")
        button_frame.grid(row=2, column=0, columnspan=5, pady=10)

        btn_media = tk.Button(button_frame, text="Media", command=self.show_graph1, 
                              bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                              relief="raised", cursor="hand2", padx=15, pady=2)
        btn_media.grid(row=0, column=0, padx=10)
        
        btn_pdf = tk.Button(button_frame, text="PDF", command=self.show_graph2,
                            bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                            relief="raised", cursor="hand2", padx=15, pady=2)
        btn_pdf.grid(row=0, column=1, padx=10)
        
        btn_comparar = tk.Button(button_frame, text="Comparar", command=self.show_compare,
                                 bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                                 relief="raised", cursor="hand2", padx=15, pady=2)
        btn_comparar.grid(row=0, column=2, padx=10)
        
        # Store button references for enabling/disabling during rendering
        self.btn_media = btn_media
        self.btn_pdf = btn_pdf
        self.btn_comparar = btn_comparar

        # Graph display frame on row 3, columns 3-4 (15/16 of width)
        self.graph_display_frame = ttk.Frame(self)
        self.graph_display_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)

        # Customization panel on row 3, column 4 (1/10 of width)
        custom_panel = tk.Frame(self, bg="#333333", bd=2, relief="ridge")
        custom_panel.grid(row=3, column=4, sticky="nsew", padx=(0, 10), pady=5)
        custom_panel.rowconfigure(0, weight=0)
        custom_panel.rowconfigure(1, weight=1)
        custom_panel.columnconfigure(0, weight=1)

        # Title for customization panel
        tk.Label(
            custom_panel,
            text="Controles",
            bg="#333333",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 20, "bold")
        ).grid(row=0, column=0, pady=(5, 10), sticky="ew")

        # Scrollable frame for controls
        self.controls_frame = tk.Frame(custom_panel, bg="#333333")
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.controls_frame.columnconfigure(0, weight=1)
        # Set minimum width to keep consistent regardless of content
        self.controls_frame.winfo_reqwidth()
        custom_panel.update_idletasks()
        min_width = 150
        self.controls_frame.grid_propagate(False)
        self.controls_frame.configure(width=min_width)

        # Initialize control widgets (will be shown/hidden based on graph type)
        self.show_runs_var = tk.BooleanVar(value=True)
        self.opacity_var = tk.DoubleVar(value=0.15)
        self.lw_var = tk.DoubleVar(value=0.8)
        self.median_color_var = tk.StringVar(value="azul")
        self.empirical_color_var = tk.StringVar(value="azul")
        self.analytical_color_var = tk.StringVar(value="rojo")

        # Media graph controls
        self.chk_show_runs = ttk.Checkbutton(self.controls_frame, text="Mostrar corridas", variable=self.show_runs_var,
                        command=self._on_toggle_show_runs)

        self.lbl_opacity = ttk.Label(self.controls_frame, text="Opacidad:")
        self.opacity_spin = ttk.Spinbox(self.controls_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.opacity_var, width=8, command=self._debounce_redraw)

        self.lbl_linewidth = ttk.Label(self.controls_frame, text="Grosor:")
        self.lw_spin = ttk.Spinbox(self.controls_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.lw_var, width=8, command=self._debounce_redraw)

        self.lbl_median_color = ttk.Label(self.controls_frame, text="Color media:")
        self.median_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.median_color_var, 
                                   values=["azul", "rojo", "verde", "negro", "púrpura", "naranja", "marrón"], 
                                   width=10, state="readonly")
        self.median_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_toggle_show_runs())

        # PDF graph controls
        self.lbl_empirical_color = ttk.Label(self.controls_frame, text="Color empírica:")
        self.empirical_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.empirical_color_var, 
                                   values=["azul", "rojo", "verde", "negro", "púrpura", "naranja", "marrón"], 
                                   width=10, state="readonly")
        self.empirical_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_pdf_colors_changed())

        self.lbl_analytical_color = ttk.Label(self.controls_frame, text="Color analítica:")
        self.analytical_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.analytical_color_var, 
                                   values=["azul", "rojo", "verde", "negro", "púrpura", "naranja", "marrón"], 
                                   width=10, state="readonly")
        self.analytical_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_pdf_colors_changed())

        # Centered processing label (overlay)
        self.processing_label = ttk.Label(
            self.graph_display_frame,
            text="",
            font=("Helvetica", 14, "italic")
        )

        self.data1 = None
        self.data2 = None

    def generate_graphs(self):
        # Generate real OU-based series using the parameters collected in the input screen
        data1, data2, trajectories = graphs(self.master.initial_values)
        self.data1 = data1
        self.data2 = data2
        # store trajectories so the display function can plot them faintly if requested
        self.trajectories = trajectories
        # show default view (Graph A) and mark current mode
        self.current_mode = 'a'
        self.show_graph1()

    def show_graph1(self):
        # only pass trajectories if the user enabled 'Mostrar corridas'
        self.current_mode = 'a'
        self._show_media_controls()
        show_runs = bool(self.show_runs_var.get())
        extra = getattr(self, 'trajectories', None) if show_runs else None
        self.master.display_figure([self.data1], extra_trajs=extra, median_color=self._get_color_value(self.median_color_var.get()))

    def show_graph2(self):
        self.current_mode = 'b'
        self._show_pdf_controls()
        self.master.display_figure([self.data2], empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def show_compare(self):
        # when comparing, show faint runs on the left subplot only if enabled
        self.current_mode = 'compare'
        self._show_compare_controls()
        show_runs = bool(self.show_runs_var.get())
        extra = getattr(self, 'trajectories', None) if show_runs else None
        self.master.display_figure([self.data1, self.data2], compare=True, extra_trajs=extra, median_color=self._get_color_value(self.median_color_var.get()), empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def _get_color_value(self, spanish_color):
        """Convert Spanish color names to matplotlib color names."""
        color_map = {
            "azul": "blue",
            "rojo": "red",
            "verde": "green",
            "negro": "black",
            "púrpura": "purple",
            "naranja": "orange",
            "marrón": "brown"
        }
        return color_map.get(spanish_color, "blue")

    def _show_media_controls(self):
        """Show Media graph controls and hide PDF controls."""
        # Clear the frame
        for widget in self.controls_frame.winfo_children():
            widget.grid_forget()
        
        # Grid Media controls
        self.chk_show_runs.grid(row=0, column=0, pady=5, sticky="ew")
        self.lbl_opacity.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        self.opacity_spin.grid(row=2, column=0, pady=(0, 5), sticky="ew")
        self.lbl_linewidth.grid(row=3, column=0, pady=(10, 0), sticky="ew")
        self.lw_spin.grid(row=4, column=0, pady=(0, 5), sticky="ew")
        self.lbl_median_color.grid(row=5, column=0, pady=(10, 0), sticky="ew")
        self.median_color_menu.grid(row=6, column=0, pady=(0, 5), sticky="ew")

    def _show_pdf_controls(self):
        """Show PDF graph controls and hide Media controls."""
        # Clear the frame
        for widget in self.controls_frame.winfo_children():
            widget.grid_forget()
        
        # Grid PDF controls
        self.lbl_empirical_color.grid(row=0, column=0, pady=(10, 0), sticky="ew")
        self.empirical_color_menu.grid(row=1, column=0, pady=(0, 5), sticky="ew")
        self.lbl_analytical_color.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        self.analytical_color_menu.grid(row=3, column=0, pady=(0, 5), sticky="ew")

    def _show_compare_controls(self):
        """Show both Media and PDF controls for Compare view, properly separated."""
        # Clear the frame
        for widget in self.controls_frame.winfo_children():
            widget.grid_forget()
        
        # Grid Media controls section
        tk.Label(self.controls_frame, text="Media", bg="#333333", fg="white", 
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, pady=(0, 5), sticky="ew")
        self.chk_show_runs.grid(row=1, column=0, pady=5, sticky="ew")
        self.lbl_opacity.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        self.opacity_spin.grid(row=3, column=0, pady=(0, 5), sticky="ew")
        self.lbl_linewidth.grid(row=4, column=0, pady=(10, 0), sticky="ew")
        self.lw_spin.grid(row=5, column=0, pady=(0, 5), sticky="ew")
        self.lbl_median_color.grid(row=6, column=0, pady=(10, 0), sticky="ew")
        self.median_color_menu.grid(row=7, column=0, pady=(0, 10), sticky="ew")
        
        # Separator
        separator = tk.Frame(self.controls_frame, bg="#555555", height=1)
        separator.grid(row=8, column=0, sticky="ew", pady=5)
        
        # Grid PDF controls section
        tk.Label(self.controls_frame, text="PDF", bg="#333333", fg="white", 
                 font=("Segoe UI", 10, "bold")).grid(row=9, column=0, pady=(5, 5), sticky="ew")
        self.lbl_empirical_color.grid(row=10, column=0, pady=(10, 0), sticky="ew")
        self.empirical_color_menu.grid(row=11, column=0, pady=(0, 5), sticky="ew")
        self.lbl_analytical_color.grid(row=12, column=0, pady=(10, 0), sticky="ew")
        self.analytical_color_menu.grid(row=13, column=0, pady=(0, 5), sticky="ew")

    def _on_pdf_colors_changed(self):
        """Refresh PDF view when colors change."""
        if self.is_rendering:
            return
        self.master.display_figure([self.data2], empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def _debounce_redraw(self):
        """Debounce rapid spinbox changes to avoid concurrent renders."""
        # Cancel any pending redraw
        if self.pending_render_id is not None:
            self.after_cancel(self.pending_render_id)
        
        # Schedule redraw after 200ms of inactivity
        self.pending_render_id = self.after(200, self._on_toggle_show_runs)

    def _on_toggle_show_runs(self):
        """Refresh current view only if not already rendering."""
        if self.is_rendering:
            return
        
        try:
            if getattr(self, 'current_mode', 'a') == 'a':
                self.show_graph1()
            elif self.current_mode == 'b':
                self.show_graph2()
            else:
                self.show_compare()
        except Exception:
            pass

# --- Implementaciones Adicionales Screen ---
class ImplementacionesAdicionalesScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        
        self.controller = controller
        self.previous_frame = None  # Store previous frame for back button
        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        
        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")
        
        # Back button (left) and Exit button (right)
        frame_btn_back = tk.Frame(self, bg="white", bd=2)
        frame_btn_back.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
        btn_back = tk.Button(
            frame_btn_back,
            text="← Regresar",
            command=self._go_back,
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_back.pack()
        
        frame_btn_exit = tk.Frame(self, bg="white", bd=2)
        frame_btn_exit.grid(row=0, column=0, sticky="ne", padx=10, pady=10)
        btn_exit = tk.Button(
            frame_btn_exit,
            text="✖ Salir",
            command=master.destroy,
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_exit.pack()
        
        # Title
        frame_titulo = tk.Frame(self, bg="#222020")
        frame_titulo.grid(row=0, column=0, pady=(40, 10))
        
        tk.Label(
            frame_titulo,
            text="Los verdaderos héroes",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 34, "bold")
        ).pack(side="top", pady=5)
        
        # Separator line
        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        # Content frame with images side by side
        content_frame = tk.Frame(self, bg="#222020")
        content_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.columnconfigure(2, weight=1)
        content_frame.columnconfigure(3, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Image filenames and display names mapping
        images_data = [
            ("zzz/sebastian.jpg", "Sebasss (◕ᴗ◕✿)"),
            ("zzz/john.jpg", "Johncito (｡•̀ᴗ-)✧"),
            ("zzz/carlos.jpg", "Carlitoss ( ꈍᴗꈍ)"),
            ("zzz/alejandro.jpg", "Aleee ┌(・。・)┘♪")
        ]
        
        # Function to calculate and load images dynamically
        def load_images():
            # Get window dimensions and calculate available width per column
            master.update_idletasks()
            available_width = master.winfo_width() - 40  # Subtract padding
            col_width = (available_width - 20) // 4  # 4 columns, subtract inter-column padding
            # Image should take ~90% of column width, maintain aspect ratio
            img_width = max(100, int(col_width * 0.85))
            img_height = int(img_width * 1.33)  # 3:4 aspect ratio
            
            for col, (img_path, name) in enumerate(images_data):
                try:
                    img = Image.open(img_path).resize((img_width, img_height))
                    photo = ImageTk.PhotoImage(img)
                    
                    # Container for image and caption
                    img_container = tk.Frame(content_frame, bg="#222020")
                    img_container.grid(row=0, column=col, padx=10, pady=15, sticky="nsew")
                    
                    # Image
                    lbl_img = tk.Label(img_container, image=photo, bg="#222020")
                    lbl_img.image = photo  # Keep a reference
                    lbl_img.pack()
                    
                    # Caption
                    tk.Label(
                        img_container,
                        text=name,
                        bg="#222020",
                        fg="white",
                        font=("Segoe UI", 24, "bold"),
                        wraplength=img_width - 20
                    ).pack(pady=(10, 0))
                    
                except Exception as e:
                    # If image fails to load, show placeholder
                    tk.Label(
                        content_frame,
                        text=f"{name}\n(Imagen no\nencontrada)",
                        bg="#333333",
                        fg="white",
                        font=("Segoe UI", 10),
                        width=15,
                        height=12
                    ).grid(row=0, column=col, padx=10, pady=15, sticky="nsew")
        
        # Schedule image loading after widget is fully rendered
        self.after(100, load_images)

    def _go_back(self):
        """Go back to the previous frame (InputScreen or GraphScreen)."""
        previous = getattr(self, 'previous_frame', InputScreen)
        self.controller.show_frame(previous)

if __name__ == "__main__":
    app = App()
    app.mainloop()