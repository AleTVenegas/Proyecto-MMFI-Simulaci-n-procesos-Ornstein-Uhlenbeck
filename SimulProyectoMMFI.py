##################  PROYECTO MMFI: SIMULACIÓN DE PROCESOS ORNSTEIN-UHLENBECK EN CIRCUITO RC  ##################

## Realizado por:
# Sebastián Chaves
# John Granados
# Carlos Navarro
# Alejandro Torres

# I semestre 2025

## Instituto Tecnológico de Costa Rica


#Librerías
import tkinter as tk
import matplotlib.pyplot as plt
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

# Configuración global de matplotlib para las gráficas
plt.rcParams.update({
    'font.size': 16,          # tamaño general
    'axes.titlesize': 18,     # títulos
    'axes.labelsize': 16,     # nombres de ejes
    'legend.fontsize': 14,    # leyenda
    'xtick.labelsize': 14,    # ticks en x
    'ytick.labelsize': 14     # ticks en y
    })

#Definiciones para las gráficas
def graphs(parametros):
    # Leer los parámetros. Si no los encuentra, usa estos valores por defecto.
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
        # Caso de Fallo. (Luego se tiene otra validación en la interfaz gráfica, entonces no debería pasar, pero mejor prevenir que lamentar)
        R, C, V0, Vf, sigma, dt, T, n_traj = 2000.0, 1e-3, 0.0, 10.0, 1.0, 0.01, 10.0, 20

    theta = 1.0 / (R * C) if R * C != 0 else 1.0  #Cálculo del coeficiente theta. Evita división por cero.

    N = max(2, int(np.ceil(T / dt)))  #Cantidad de pasos de tiempo
    x = np.linspace(0, T, N)          #Define el espacio de tiempo

    # Simulación de las trayectorias individuales del proceso OU
    trajectories = np.zeros((n_traj, N))
    trajectories[:, 0] = V0
    # Incrementos aleatorios normales para todas las corridas
    rng = np.random.default_rng()
    normals = rng.standard_normal(size=(n_traj, N - 1))

    # Simulación usando el método de Euler-Maruyama
    for t in range(1, N):
        trajectories[:, t] = (
            trajectories[:, t - 1]
            + theta * (Vf - trajectories[:, t - 1]) * dt
            + sigma * np.sqrt(dt) * normals[:, t - 1]
        )

    mean_traj = np.mean(trajectories, axis=0) # Trayectoria media

    # Trayectorio media (gráfica 1)
    data1 = (x, mean_traj, f"Media de {n_traj} trayectorias (OU)")

    ### Gráfica 2: Densidad de probabilidad al tiempo T con el método KDE

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
    L2_error = np.sqrt(np.sum((pdf_sim - pdf_an)**2) * dx) # Cálculo del error L2. Al final se optó por usar el % de sobrelape, pero igual se tenía calculado.
    overlap = np.sum(np.minimum(pdf_sim, pdf_an) * dx)     # Cálculo del % de sobrelape entre las dos distribuciones.

    # Guarda los datos para la gráfica 2
    data2 = {
        'x': x_vals,
        'pdf_sim': pdf_sim,
        'pdf_an': pdf_an,
        'L2_error': L2_error,
        'n_traj': n_traj,
        'overlap': overlap
    }

    return data1, data2, trajectories

###### Acá empieza la aplicación de Tkinter ######

class App(tk.Tk):
    def __init__(self):
        # Configuración de la ventana principal
        super().__init__()
        self.title("Proyecto MMFI: Simulación de Procesos Ornstein-Uhlenbeck")
        self.configure(bg="#222020")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        self.bind("<F11>", self._toggle_fullscreen)

        # Definir las variables compartidas entre pantallas
        self.initial_values = {}
        self.canvas = None
        self.toolbar = None
        
        self.key_sequence = ""
        self.bind("<Key>", self._on_key_press)

        # Creación de las pantallas
        self.frames = {}
        for F in (PresentationScreen, InputScreen, GraphScreen, ImplementacionesAdicionalesScreen):
            frame = F(self, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(PresentationScreen)

    # Función para mostrar una patalla específica. (todas está una sobre la otra, pero solo se ve la que se "eleva" al frente)
    def show_frame(self, screen):
        frame = self.frames[screen]
        frame.tkraise()

    # Función para alternar el modo de pantalla completa (luego se implemte con F11)
    def _toggle_fullscreen(self, event=None):
        current_state = self.attributes("-fullscreen")
        self.attributes("-fullscreen", not current_state)

    ## La parte bonita :D, mostrarar las gráficas con matplotlib dentro de Tkinter
    def display_figure(self, data_list, compare=False, extra_trajs=None, median_color="blue", empirical_color="blue", analytical_color="red"):

        frame = self.frames[GraphScreen].graph_display_frame # El frame donde se mostrará la gráfica
        label = self.frames[GraphScreen].processing_label    

        # Revisa si hay una gráfica previa y si sí la elimina
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None

        # Texto temporal de "Generando gráfica..."
        label.place(relx=0.5, rely=0.5, anchor="center")
        label.config(text="Generando gráfica...", foreground="red")
        label.lift()
        
        # Desactivar botones mientras se genera la gráfica (se hacía un despelote si se apretaba un botón mientras procesaba una gráfica D: )
        self.frames[GraphScreen].btn_media.config(state="disabled")
        self.frames[GraphScreen].btn_pdf.config(state="disabled")
        self.frames[GraphScreen].btn_comparar.config(state="disabled")

        # Pegueña animación de puntos suspensivos
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

        #  Graficar en el fondo
        def render_plot():
            self.frames[GraphScreen].is_rendering = True
            time.sleep(0.4)  # Pequeño atraso para que parezca que tarda un poquito (cuando son pesadas no hace falta jajaj, pero para las rápidas sí)

            fig = Figure(dpi=100) #Crea la figura y establece el tamaño/resolución
            
            # Para obtener los parámetros de personalización
            def get_style_params():
                try:
                    alpha_val = float(self.frames[GraphScreen].opacity_var.get()) # Transparencia
                    alpha_val = max(0.0, min(1.0, alpha_val)) # intervalo [0,1] pq ni modo q sea 200% opaco, o con transparencia negativa
                except Exception:
                    alpha_val = 0.15 # valor por defecto por si acaso
                try:
                    lw_val = float(self.frames[GraphScreen].lw_var.get()) # Grosor de línea
                    lw_val = max(0.1, lw_val) # mínimo 0.1 para que se vea algo jsjs, si no las quieren ver que desactiven la otra opción
                except Exception:
                    lw_val = 0.8 # valor por defecto por si acaso
                return alpha_val, lw_val
            
            if compare: # Si se pone la opción de comparación (dos subplots)
                axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
                
                # Primera subgráfica: data1 (trayectoria media)
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
                
                # Segunda subgráfica: data2 (PDF)
                ax = axes[1]
                item2 = data_list[1]
                
                if isinstance(item2, dict) and 'pdf_sim' in item2: # Asegura que es el formato esperado (diccionario)
                    x_vals = item2['x']
                    pdf_sim = item2['pdf_sim']
                    pdf_an = item2['pdf_an']
                    L2_error = item2['L2_error']    # al final no lo usamos, pero sí se había implementado
                    overlap = item2['overlap']
                    n_traj = item2['n_traj']
                    ax.plot(x_vals, pdf_sim, color=empirical_color, label=f'Empírica, %Overlap={overlap*100:.3f}%')
                    ax.plot(x_vals, pdf_an, color=analytical_color, linestyle='--', lw=2, label='Analítica')
                    ax.set_title(f'Densidad OU a partir de {n_traj} trayectorias')
                    ax.set_xlabel("Voltaje V (V)")
                    ax.set_ylabel("Densidades de probabilidad")
                    ax.legend()
            else: # Solo una gráfica
                ax = fig.add_subplot(111)
                item = data_list[0]
                
                if isinstance(item, dict) and 'pdf_sim' in item: # Revisar si item es diccionario
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
                    x, y, title = item
                    
                    # Agrega las gráficas de las corridas individuales si la opción está habilitada
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

            fig.tight_layout(pad=0.1, w_pad=0.3) # Ajusta el layout para que no se sobrepongan elementos
            fig.subplots_adjust(wspace=0.15,left=0.04) # Reduce el espacio entre subplots

            # Agraga la figura al canvas de Tkinter
            self.canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.canvas.draw()

            # Agrega el toolbar <3 (me encantaa)
            self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
            self.toolbar.update()
            self.toolbar.pack(fill=tk.X)

            # detener la animación y quitar el label de "Generando gráfica..."
            nonlocal running
            running = False
            label.place_forget()
            self.frames[GraphScreen].is_rendering = False
            
            # Des-desactivar los botones
            self.frames[GraphScreen].btn_media.config(state="normal")
            self.frames[GraphScreen].btn_pdf.config(state="normal")
            self.frames[GraphScreen].btn_comparar.config(state="normal")
        
        # Renderizar en un hilo aparte para no congelar la interfaz. Debatible lo bien que funciona, pero se hace el intento jaja
        threading.Thread(target=render_plot, daemon=True).start() 

    def _on_key_press(self, event):
        if event.char.isalpha():
            self.key_sequence += event.char.lower()
            if len(self.key_sequence) > 19:
                self.key_sequence = self.key_sequence[-19:]
            if self.key_sequence.endswith("losverdaderosheroes"):
                self.show_frame(ImplementacionesAdicionalesScreen)
                self.key_sequence = ""

# Source - https://stackoverflow.com/a     # líneas 315-352
# Posted by squareRoot17, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-16, License - CC BY-SA 4.0

## código para implementar advertencia en el botón de simulación.
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                    background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                    font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

    

### Pantalla de presentación ###

class PresentationScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)

        self.controller = controller # para acceder a datos compartidos entre pantallas

        #Configuación inicial
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")
        
        # Botón de salida
        frame_btn_exit = tk.Frame(self, bg="white", bd=2)
        frame_btn_exit.grid(row=0, column=0, sticky="ne", padx=10, pady=10)
        btn_exit = tk.Button(
            frame_btn_exit,
            text="✖ Salir",
            command=master.destroy, #Mata al amo (cierra la aplicación)
            bg="#222020",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2"
        )
        btn_exit.pack()

        # Título principal
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

        # divisor
        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, sticky="ew", padx=20, pady=(0,20))
        
        # Botón para ir al menú de simulación con imagen del circuito (para q no se desperdicie el tiempo en tikz jsjs)
        im1 = Image.open('circuito.png').resize((400, 350))
        circuito_img = ImageTk.PhotoImage(im1)

        frame_btn_calc = tk.Frame(self, bg="white", bd=2)
        frame_btn_calc.grid(row=2, column=0, pady=40)

        # no me va a creer qué hace este botón :0
        btn_calc = tk.Button(
            frame_btn_calc,
            text="Ir al Menú de Simulación", # me arruinan la sorpresa
            image=circuito_img,
            compound="top",
            bg="#222020",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            relief="raised",
            cursor="hand2",
            padx=10,
            pady=10,
            command=lambda: controller.show_frame(InputScreen), #Función que muestra el menú de ingreso de parámetros
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
            "Chaves Sebastián", # Bello
            "Granados John",    # Guapo
            "Navarro Carlos",   # Bizarro (ver segunda definición de la RAE)
            "Torres Alejandro"  # Ale
        ]

        #Coloca todos los créditos
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
            text="II Semestre", # ojalá se acabe pronto
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

### Pantalla de entrada de parámetros ###
class InputScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        self.controller = controller # para acceder a datos compartidos entre pantallas (para este punto debería sabérselo)

        # Configuración inicial
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)

        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")

        # Botón de regresar (este sí es nuevo)
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
            command=lambda: master.show_frame(PresentationScreen) #Muestra la pantalla principal
        )
        btn_regresar.pack()

        tk.Label(self, text="Simulación del Proceso Ornstein–Uhlenbeck",
                bg="#222020", fg="white",
                font=("Bahnschrift SemiBold Condensed", 28, "bold")).grid(row=0, column=1, sticky="nsew", pady=10)

        # Botón de salir (igual q antes)
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

        # separador de línea (para un diseño serio y profesional)
        linea_titulo = tk.Frame(self, bg="#FFFFFF", height=3)
        linea_titulo.grid(row=1, column=0, columnspan=3, sticky="ew", padx=20, pady=(0,10))

        # Dividir la pantalla en dos: izquierda con la imagen (sí, otra vez. está linda :D) y derecha con los parámetros
        frame_contenido = tk.Frame(self, bg="#222020")
        frame_contenido.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        frame_contenido.columnconfigure(0, weight=7)  # Izquierda con el séptuple (?) del ancho = 7/8 de pantalla
        frame_contenido.columnconfigure(1, weight=1)  # Derecha con 1/8 de pantalla
        frame_contenido.rowconfigure(0, weight=1)

        frame_izquierda = tk.Frame(frame_contenido, bg="#222020")
        frame_izquierda.grid(row=0, column=0, sticky="nsew", padx=(40, 15), pady=0)

        im1 = Image.open('circuito.png').resize((400, 350)) # la bella imagen del circuito
        circuito_img = ImageTk.PhotoImage(im1)
        lbl_circuito = tk.Label(frame_izquierda, image=circuito_img, bg="#222020")
        lbl_circuito.image = circuito_img  
        lbl_circuito.pack(expand=True)

        frame_derecha = tk.Frame(frame_contenido, bg="#222020")
        frame_derecha.grid(row=0, column=1, sticky="nsew", padx=(15, 40), pady=0)
        frame_derecha.columnconfigure(0, weight=1)
        frame_derecha.columnconfigure(1, weight=1)

        # Cuadro de parámetros (cuadro dentro de cuadro dentro de cuadro jaja)
        frame_param_cuadro = tk.Frame(frame_derecha, bg="#333333", bd=2, relief="ridge")
        frame_param_cuadro.pack(padx=50, pady=10, fill="x", expand=True)
        
        # Aquí sí van con igual grosor las columnas
        frame_param_cuadro.columnconfigure(0, weight=1)
        frame_param_cuadro.columnconfigure(1, weight=1)

        tk.Label(
            frame_param_cuadro,
            text="Parámetros a ingresar",
            bg="#333333",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 18, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(10,15))

        # Parámetros de la primera columna
        parametros_col1 = [
            "Voltaje de la fuente Vε (V)",
            "Voltaje inicial V₀ (V)",
            "Capacitancia C (µF)",
            "Resistencia R (Ω)"
        ]
        # Parámetros de la segunda columna
        parametros_col2 = [
            "Intensidad del ruido σ",
            "Paso temporal Δt (s)",
            "Tiempo total T (s)",
            "Cantidad de corridas"
        ]

        # Crear una entrada para cada parámetro
        self.entries = {}
        all_params = parametros_col1 + parametros_col2 # juntar las variables
        
        # Definir valores por defecto. Como por tercera vez, ik
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
        
        # Botón que activa la magia
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
            command=lambda: simular() #yay
        )
        btn_simular.pack()

        for idx, nombre in enumerate(all_params): # recorrer todos los parámetros
            col = 0 if idx < len(parametros_col1) else 1 # ver en cuál columna le toca
            row_label = (idx % len(parametros_col1)) * 2 + 1 # ver en cual fila le toca
            row_entry = row_label + 1 # empezar abajo de la etiqueta

            tk.Label(
                frame_param_cuadro,
                text=nombre,
                bg="#333333",
                fg="white",
                font=("Segoe UI", 14)
            ).grid(row=row_label, column=col, padx=15, pady=(5,0), sticky="ew")

            # Cantidad de corridas necesita ser entero, lo demás mejor como flotante
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

        # Pequeña función para advertir al usuario si se coloca parámetros muy pesados
        def callback():
            try:
                peso_simul = self.entries["Cantidad de corridas"].get()*self.entries["Tiempo total T (s)"].get()/self.entries["Paso temporal Δt (s)"].get()
                if peso_simul >= 1e6 and peso_simul < 1e9:
                    btn_simular.config(bg="yellow", fg="black")
                    CreateToolTip(btn_simular, text = f'Se realizarán {peso_simul:.2e} cálculos. \nEsto puede tardar un poco.')
                elif peso_simul >= 1e9 and peso_simul < 1e18:
                    btn_simular.config(bg="red", fg="white")
                    CreateToolTip(btn_simular, text = f'Se realizarán {peso_simul:.2e} cálculos. \nEsto puede tardar mucho tiempo o incluso\nhacer que la aplicación deje de responder.')
                elif peso_simul >= 1e18:
                    btn_simular.config(bg="darkred", fg="black")
                    CreateToolTip(btn_simular, text = f'Se realizarán {peso_simul:.2e} cálculos. \nSi no quiere a su compu,\nse merece lo que le pase.')
                else:
                    btn_simular.config(bg="#222020", fg="white")
                    CreateToolTip(btn_simular, text = f'Se realizarán {peso_simul:.2e} cálculos.')
            except Exception:
                pass
            return True

        # Realiza la validación al salir de cada entry
        vcmd = (frame_param_cuadro.register(callback), )
        for entry_widget in frame_param_cuadro.winfo_children():
            if isinstance(entry_widget, tk.Entry):
                entry_widget.config(validate="focusout", validatecommand=vcmd)

        # Función para iniciar la simulación :D
        def simular():
            try:
                # revisar valores
                for nombre, var in self.entries.items():
                    if nombre == "Cantidad de corridas":
                        valor_num = int(var.get()) # Si se pone un flotante, solo se redondea hacia abajo
                        if valor_num <= 0:
                            raise ValueError(f"{nombre} debe ser un número positivo.") # Eso sí, tiene que ser positivo
                    else:
                        # Fuera de los voltajes, todo debe ser positivo
                        valor_num = float(var.get())
                        if nombre not in ["Voltaje inicial V₀ (V)", "Voltaje promedio V_prom (V)"] and valor_num <= 0:
                            raise ValueError(f"{nombre} debe ser un número positivo.")
                    # guardar el valor en el diccionario de valores iniciales
                    self.controller.initial_values[nombre] = valor_num
                self.controller.show_frame(GraphScreen)
                self.controller.frames[GraphScreen].generate_graphs()
            except Exception:
                mb.showerror("Error", f"Error en los valores ingresados:\nPor favor ingrese valores numéricos válidos.")

# Interfaz de graficación
class GraphScreen(ttk.Frame):
    def __init__(self, master, controller):
        super().__init__(master)
        
        s = ttk.Style()
        s.configure("Custom.TFrame", background="#222020")
        self.configure(style="Custom.TFrame")
        self.controller = controller
        self.is_rendering = False  # colorcar indicador para saber si se está renderizando
        self.pending_render_id = None  # ID del render pendiente si se solicita uno durante el render actual

        # Configuaración del layout
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)  
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=7)  # Área de graficación: 7/8 del ancho
        self.columnconfigure(4, weight=1)  # Área de personalización: 1/8 del ancho
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)  # La altura de la gráfica se expande

        # Botón de regresar
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

        # Botón de salir
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

        # Separador de línea
        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, columnspan=5, sticky="ew", padx=20, pady=(0, 10))
        
        # configuración de filas
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)

        button_frame = ttk.Frame(self, style="Custom.TFrame")
        button_frame.grid(row=2, column=0, columnspan=5, pady=10)

        #solo mostrar la función de media
        btn_media = tk.Button(button_frame, text="Media", command=self.show_graph1, 
                              bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                              relief="raised", cursor="hand2", padx=15, pady=2)
        btn_media.grid(row=0, column=0, padx=10)
        #solo mostrar la función de pdf
        btn_pdf = tk.Button(button_frame, text="PDF", command=self.show_graph2,
                            bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                            relief="raised", cursor="hand2", padx=15, pady=2)
        btn_pdf.grid(row=0, column=1, padx=10)
        #mostrar ambas
        btn_comparar = tk.Button(button_frame, text="Comparar", command=self.show_compare,
                                 bg="#222020", fg="white", font=("Segoe UI", 14, "bold"),
                                 relief="raised", cursor="hand2", padx=15, pady=2)
        btn_comparar.grid(row=0, column=2, padx=10)
        
        #Guardar referencias a los botones (sirve luego para desactivarlos mientras se calcula la gráfica)
        self.btn_media = btn_media
        self.btn_pdf = btn_pdf
        self.btn_comparar = btn_comparar

        # cuadro para la gráfica
        self.graph_display_frame = ttk.Frame(self)
        self.graph_display_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)

        #cuadro para la personalización
        custom_panel = tk.Frame(self, bg="#333333", bd=2, relief="ridge")
        custom_panel.grid(row=3, column=4, sticky="nsew", padx=(0, 10), pady=5)
        custom_panel.rowconfigure(0, weight=0)
        custom_panel.rowconfigure(1, weight=1)
        custom_panel.columnconfigure(0, weight=1)

        # título de panel de personalización
        tk.Label(
            custom_panel,
            text="Controles",
            bg="#333333",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 20, "bold")
        ).grid(row=0, column=0, pady=(5, 10), sticky="ew")

        # cuadro de controles
        self.controls_frame = tk.Frame(custom_panel, bg="#333333")
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.controls_frame.columnconfigure(0, weight=1)
        #ancho mínimo del panel de controles para que nunca quede muy delgado
        self.controls_frame.winfo_reqwidth()
        custom_panel.update_idletasks()
        min_width = 150
        self.controls_frame.grid_propagate(False)
        self.controls_frame.configure(width=min_width)

        # Variables de control
        self.show_runs_var = tk.BooleanVar(value=True)          # Mostrar corridas por defecto
        self.opacity_var = tk.DoubleVar(value=0.15)             # Opacidad de las corridas individuales por defecto
        self.lw_var = tk.DoubleVar(value=0.8)                   # Grosor de las corridas individuales por defecto
        self.median_color_var = tk.StringVar(value="Azul")      # Color resaltado por defecto
        self.empirical_color_var = tk.StringVar(value="Azul")   # Color empírica por defecto
        self.analytical_color_var = tk.StringVar(value="Rojo")  # Color analítica por defecto

        # toggle de mostrar corridas
        self.chk_show_runs = ttk.Checkbutton(self.controls_frame, text="Mostrar corridas", variable=self.show_runs_var,
                        command=self._on_toggle_show_runs)
        # controles de opacidad para corridas individuales
        self.lbl_opacity = ttk.Label(self.controls_frame, text="Opacidad:")
        self.opacity_spin = ttk.Spinbox(self.controls_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.opacity_var, width=8, command=self._debounce_redraw)
        # controles de grosor para corridas individuales
        self.lbl_linewidth = ttk.Label(self.controls_frame, text="Grosor:")
        self.lw_spin = ttk.Spinbox(self.controls_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.lw_var, width=8, command=self._debounce_redraw)
        # color de la media
        self.lbl_median_color = ttk.Label(self.controls_frame, text="Color media:")
        self.median_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.median_color_var, 
                                   values=["Azul", "Rojo", "Verde", "Negro", "Morado", "Naranja", "Café"], 
                                   width=10, state="readonly")
        self.median_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_toggle_show_runs())

        # colores de distribución empírica
        self.lbl_empirical_color = ttk.Label(self.controls_frame, text="Color empírica:")
        self.empirical_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.empirical_color_var, 
                                   values=["Azul", "Rojo", "Verde", "Negro", "Morado", "Naranja", "Café"], 
                                   width=10, state="readonly")
        self.empirical_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_pdf_colors_changed())
        # colores de distribución analítica
        self.lbl_analytical_color = ttk.Label(self.controls_frame, text="Color analítica:")
        self.analytical_color_menu = ttk.Combobox(self.controls_frame, textvariable=self.analytical_color_var, 
                                   values=["Azul", "Rojo", "Verde", "Negro", "Morado", "Naranja", "Café"], 
                                   width=10, state="readonly")
        self.analytical_color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_pdf_colors_changed())

        # label de "Generando gráfica..."
        self.processing_label = ttk.Label(
            self.graph_display_frame,
            text="",
            font=("Helvetica", 14, "italic")
        )

        self.data1 = None
        self.data2 = None

    def generate_graphs(self):
        # llama a la función de generación de gráficos y almacena los datos
        self.data1, self.data2, trajectories = graphs(self.master.initial_values)
        # guarda las corridas individuales para mostrarlas si el usuario lo desea
        self.trajectories = trajectories
        # mostrar la gráfica de media por defecto y definir ese como el estado actual
        self.current_mode = 'media'
        self.show_graph1()

    def show_graph1(self):
        # define estado actual como media y muestra controles correspondientes
        self.current_mode = 'media'
        self._show_media_controls()
        show_runs = bool(self.show_runs_var.get())
        extra = getattr(self, 'trajectories', None) if show_runs else None
        self.master.display_figure([self.data1], extra_trajs=extra, median_color=self._get_color_value(self.median_color_var.get()))

    def show_graph2(self):
        # define estado actual como pdf y muestra controles correspondientes
        self.current_mode = 'pdf'
        self._show_pdf_controls()
        self.master.display_figure([self.data2], empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def show_compare(self):
        # define el estado actual como comparacion y muestra controles correspondientes
        self.current_mode = 'comparacion'
        self._show_compare_controls()
        show_runs = bool(self.show_runs_var.get())
        extra = getattr(self, 'trajectories', None) if show_runs else None
        self.master.display_figure([self.data1, self.data2], compare=True, extra_trajs=extra, median_color=self._get_color_value(self.median_color_var.get()), empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def _get_color_value(self, spanish_color):
        # Convierte los nombres a español pq el grosero solo en english trabaja
        color_map = {
            "Azul": "blue",
            "Rojo": "red",
            "Verde": "green",
            "Negro": "black",
            "Morado": "purple",
            "Naranja": "orange",
            "Café": "brown"
        }
        return color_map.get(spanish_color, "blue")

    def _show_media_controls(self): # Mostrar controles para media
        # Primero limpiar el cuadro
        for widget in self.controls_frame.winfo_children():  # volver a los hijos palomas
            widget.grid_forget()
        
        # colocar los botones de configuración de media
        self.chk_show_runs.grid(row=0, column=0, pady=5, sticky="ew")
        self.lbl_opacity.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        self.opacity_spin.grid(row=2, column=0, pady=(0, 5), sticky="ew")
        self.lbl_linewidth.grid(row=3, column=0, pady=(10, 0), sticky="ew")
        self.lw_spin.grid(row=4, column=0, pady=(0, 5), sticky="ew")
        self.lbl_median_color.grid(row=5, column=0, pady=(10, 0), sticky="ew")
        self.median_color_menu.grid(row=6, column=0, pady=(0, 5), sticky="ew")

    def _show_pdf_controls(self): # Mostrar controles para PDF
        # igual, primero limpar el cuadro
        for widget in self.controls_frame.winfo_children(): # volver a los hijos palomas
            widget.grid_forget()
        
        # colocar los botones de configuración de PDF
        self.lbl_empirical_color.grid(row=0, column=0, pady=(10, 0), sticky="ew")
        self.empirical_color_menu.grid(row=1, column=0, pady=(0, 5), sticky="ew")
        self.lbl_analytical_color.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        self.analytical_color_menu.grid(row=3, column=0, pady=(0, 5), sticky="ew")

    def _show_compare_controls(self): # Mostrar controles para comparación
        # ya se la sabe; limpiar y matar bebés
        for widget in self.controls_frame.winfo_children():
            widget.grid_forget()
        
        # Colocar los botones de configuración de media ...
        tk.Label(self.controls_frame, text="Media", bg="#333333", fg="white", 
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, pady=(0, 5), sticky="ew")
        self.chk_show_runs.grid(row=1, column=0, pady=5, sticky="ew")
        self.lbl_opacity.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        self.opacity_spin.grid(row=3, column=0, pady=(0, 5), sticky="ew")
        self.lbl_linewidth.grid(row=4, column=0, pady=(10, 0), sticky="ew")
        self.lw_spin.grid(row=5, column=0, pady=(0, 5), sticky="ew")
        self.lbl_median_color.grid(row=6, column=0, pady=(10, 0), sticky="ew")
        self.median_color_menu.grid(row=7, column=0, pady=(0, 10), sticky="ew")
        
        # ...y...
        separator = tk.Frame(self.controls_frame, bg="#555555", height=1)
        separator.grid(row=8, column=0, sticky="ew", pady=5)
        
        # ...los de PDF
        tk.Label(self.controls_frame, text="PDF", bg="#333333", fg="white", 
                 font=("Segoe UI", 10, "bold")).grid(row=9, column=0, pady=(5, 5), sticky="ew")
        self.lbl_empirical_color.grid(row=10, column=0, pady=(10, 0), sticky="ew")
        self.empirical_color_menu.grid(row=11, column=0, pady=(0, 5), sticky="ew")
        self.lbl_analytical_color.grid(row=12, column=0, pady=(10, 0), sticky="ew")
        self.analytical_color_menu.grid(row=13, column=0, pady=(0, 5), sticky="ew")

    def _on_pdf_colors_changed(self): #función para cambio de colores en PDF
        if self.is_rendering:
            return
        self.master.display_figure([self.data2], empirical_color=self._get_color_value(self.empirical_color_var.get()), analytical_color=self._get_color_value(self.analytical_color_var.get()))

    def _debounce_redraw(self): 
        # evita que se hagan múltiples redraws al cambiar valores rápidamente
        if self.pending_render_id is not None:
            self.after_cancel(self.pending_render_id)
        
        # redibujar hasta 200 ms después del último cambio (porecito el pobre si no)
        self.pending_render_id = self.after(200, self._on_toggle_show_runs)

    def _on_toggle_show_runs(self):
        # evita llamadas adicionales si ya se está procesando
        if self.is_rendering:
            return
        
        try: # intenta mostrar la gráfica según el modo actual
            if getattr(self, 'current_mode', 'media') == 'media':
                self.show_graph1()
            elif self.current_mode == 'pdf':
                self.show_graph2()
            else:
                self.show_compare()
        except Exception: # si algo falla, no hace nada (igualito a mí)
            pass

class ImplementacionesAdicionalesScreen(ttk.Frame):
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
        frame_btn_back = tk.Frame(self, bg="white", bd=2)
        frame_btn_back.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
        btn_back = tk.Button(
            frame_btn_back,
            text="← Regresar",
            command= lambda: master.show_frame(PresentationScreen),
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
        frame_titulo = tk.Frame(self, bg="#222020")
        frame_titulo.grid(row=0, column=0, pady=(40, 10))
        tk.Label(
            frame_titulo,
            text="Los verdaderos héroes",
            bg="#222020",
            fg="white",
            font=("Bahnschrift SemiBold Condensed", 34, "bold")
        ).pack(side="top", pady=5)
        linea = tk.Frame(self, bg="#FFFFFF", height=3)
        linea.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        content_frame = tk.Frame(self, bg="#222020")
        content_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.columnconfigure(2, weight=1)
        content_frame.columnconfigure(3, weight=1)
        content_frame.rowconfigure(0, weight=1)
        images_data = [
            ("zzz/sebastian.jpg", "Sebasss (◕ᴗ◕✿)"),
            ("zzz/john.jpg", "Johncito (｡•̀ᴗ-)✧"),
            ("zzz/carlos.jpg", "Carlitoss ( ꈍᴗꈍ)"),
            ("zzz/alejandro.jpg", "Aleee ┌(・。・)┘♪")
        ]
        def load_images():
            master.update_idletasks()
            available_width = master.winfo_width() - 40 
            col_width = (available_width - 20) // 4
            img_width = max(100, int(col_width * 0.85))
            img_height = int(img_width * 1.33)
            for col, (img_path, name) in enumerate(images_data):
                try:
                    img = Image.open(img_path).resize((img_width, img_height))
                    photo = ImageTk.PhotoImage(img)
                    img_container = tk.Frame(content_frame, bg="#222020")
                    img_container.grid(row=0, column=col, padx=10, pady=15, sticky="nsew")
                    lbl_img = tk.Label(img_container, image=photo, bg="#222020")
                    lbl_img.image = photo
                    lbl_img.pack()
                    tk.Label(
                        img_container,
                        text=name,
                        bg="#222020",
                        fg="white",
                        font=("Segoe UI", 24, "bold"),
                        wraplength=img_width - 20
                    ).pack(pady=(10, 0))
                except Exception as e:
                    tk.Label(
                        content_frame,
                        text=f"{name}\n(Imagen no\nencontrada)",
                        bg="#333333",
                        fg="white",
                        font=("Segoe UI", 10),
                        width=15,
                        height=12
                    ).grid(row=0, column=col, padx=10, pady=15, sticky="nsew")
        self.after(100, load_images)


######## Finalmente correr la aplicación :D ########
if __name__ == "__main__":
    app = App()
    app.mainloop()
# Muchas gracias si llegó hasta acá :)