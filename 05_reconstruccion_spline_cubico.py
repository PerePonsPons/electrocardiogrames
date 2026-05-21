import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ============================================================
# 1. PARÁMETROS
# ============================================================

FILTERED_FILE = "ecg_filtrado.npz"
ANNOTATIONS_FILE = "100annotations.txt"

# Intervalo que queremos reconstruir
START_TIME = 0       # segundos
DURATION = 3         # segundos

# Factor de refinamiento temporal.
# Si Fs = 360 Hz y UPSAMPLE_FACTOR = 30,
# representamos la reconstrucción a 10800 Hz.
UPSAMPLE_FACTOR = 30

# Tipo de condición de contorno para el spline cúbico.
#
# "natural":
#     S''(t_0) = S''(t_N) = 0
#
# "not-a-knot":
#     condición por defecto habitual en scipy.
#
BOUNDARY_CONDITION = "natural"

# Archivo de salida
OUTPUT_FILE = "reconstruccion_spline_cubico.npz"

# Mostrar anotaciones
SHOW_ANNOTATIONS = True


# ============================================================
# 2. CARGAR SEÑAL FILTRADA
# ============================================================

def load_filtered_signal(filename):
    """
    Carga el archivo generado por 02_filtrar_ecg.py.

    Contiene:
        sample_numbers
        t
        x
        y
        Fs
        Ts
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"No se ha encontrado el archivo '{filename}'. "
            "Primero debes ejecutar 02_filtrar_ecg.py."
        )

    data = np.load(filename)

    sample_numbers = data["sample_numbers"]
    t = data["t"]
    x = data["x"]
    y = data["y"]
    Fs = float(data["Fs"])
    Ts = float(data["Ts"])

    return sample_numbers, t, x, y, Fs, Ts


# ============================================================
# 3. CARGAR ANOTACIONES
# ============================================================

def load_annotations(filename):
    """
    Carga las anotaciones del ECG.

    Formato esperado:

          Time   Sample #  Type  Sub Chan  Num    Aux
        0:00.050       18     +    0    0    0    (N
        0:00.214       77     N    0    0    0
        0:01.028      370     N    0    0    0

    Devuelve un DataFrame con columnas:
        time_text, sample, type, sub, chan, num, aux
    """

    if not os.path.exists(filename):
        print(f"No se ha encontrado '{filename}'. Se dibujará sin anotaciones.")
        return pd.DataFrame(
            columns=["time_text", "sample", "type", "sub", "chan", "num", "aux"]
        )

    annotations = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("Time"):
            continue

        parts = line.split()

        if len(parts) < 6:
            continue

        time_text = parts[0]
        sample = int(parts[1])
        annotation_type = parts[2]
        sub = int(parts[3])
        chan = int(parts[4])
        num = int(parts[5])
        aux = " ".join(parts[6:]) if len(parts) > 6 else ""

        annotations.append({
            "time_text": time_text,
            "sample": sample,
            "type": annotation_type,
            "sub": sub,
            "chan": chan,
            "num": num,
            "aux": aux
        })

    return pd.DataFrame(annotations)


# ============================================================
# 4. EXTRAER FRAGMENTO
# ============================================================

def extract_fragment(t, y, sample_numbers, start_time, duration):
    """
    Extrae un fragmento temporal de la señal.
    """

    end_time = start_time + duration

    mask = (t >= start_time) & (t <= end_time)

    if np.sum(mask) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    t_fragment = t[mask]
    y_fragment = y[mask]
    sample_numbers_fragment = sample_numbers[mask]

    return t_fragment, y_fragment, sample_numbers_fragment


# ============================================================
# 5. CONSTRUIR MALLA TEMPORAL FINA
# ============================================================

def build_dense_time_grid(start_time, duration, Fs, upsample_factor):
    """
    Construye una malla temporal fina en el intervalo escogido.
    """

    Fs_dense = Fs * upsample_factor
    Ts_dense = 1 / Fs_dense

    end_time = start_time + duration

    t_dense = np.arange(start_time, end_time, Ts_dense)

    return t_dense, Fs_dense, Ts_dense


# ============================================================
# 6. RECONSTRUCCIÓN MEDIANTE SPLINE CÚBICO
# ============================================================

def cubic_spline_reconstruction(t_samples, y_samples, t_eval,
                                boundary_condition="natural"):
    """
    Reconstruye una señal mediante spline cúbico.

    Dadas las muestras:

        (t_0, y_0), (t_1, y_1), ..., (t_N, y_N),

    construimos una función S(t) tal que:

        S(t_k) = y_k

    y en cada intervalo [t_k, t_{k+1}], S(t) es un polinomio
    de grado menor o igual que 3.

    Además, el spline cúbico cumple:

        S ∈ C^2,

    es decir, S, S' y S'' son continuas.

    Entrada:
        t_samples: tiempos originales
        y_samples: valores discretos
        t_eval: tiempos donde queremos evaluar el spline
        boundary_condition: condición de contorno

    Salida:
        y_spline: reconstrucción evaluada en t_eval
        spline_function: objeto CubicSpline
    """

    if boundary_condition == "natural":
        bc_type = "natural"
    elif boundary_condition == "not-a-knot":
        bc_type = "not-a-knot"
    else:
        raise ValueError(
            "Condición de contorno no válida. "
            "Usa 'natural' o 'not-a-knot'."
        )

    spline_function = CubicSpline(
        t_samples,
        y_samples,
        bc_type=bc_type
    )

    y_spline = spline_function(t_eval)

    return y_spline, spline_function


# ============================================================
# 7. COMPARACIÓN CON INTERPOLACIÓN LINEAL
# ============================================================

def linear_interpolation(t_samples, y_samples, t_eval):
    """
    Interpolación lineal para comparar con el spline cúbico.
    """

    return np.interp(t_eval, t_samples, y_samples)


# ============================================================
# 8. DIBUJAR ANOTACIONES
# ============================================================

def draw_annotations(annotations, sample_numbers, y, Fs, start_time, end_time):
    """
    Dibuja las anotaciones sobre la gráfica actual.
    """

    if annotations.empty:
        return

    start_sample = int(start_time * Fs)
    end_sample = int(end_time * Fs)

    annotations_plot = annotations[
        (annotations["sample"] >= start_sample) &
        (annotations["sample"] <= end_sample)
    ].copy()

    # Quitamos anotaciones auxiliares como "+"
    beat_annotations = annotations_plot[
        annotations_plot["type"] != "+"
    ].copy()

    for _, row in beat_annotations.iterrows():

        sample = row["sample"]
        ann_type = row["type"]

        t_ann = sample / Fs

        idx = np.searchsorted(sample_numbers, sample)

        if idx >= len(y):
            continue

        plt.axvline(t_ann, linestyle="--", linewidth=0.8)

        plt.text(
            t_ann,
            y[idx],
            ann_type,
            fontsize=8,
            ha="center",
            va="bottom"
        )


# ============================================================
# 9. REPRESENTACIÓN GRÁFICA
# ============================================================

def plot_spline_reconstruction(t_fragment, y_fragment,
                               sample_numbers_fragment,
                               t_dense, y_spline, y_linear,
                               annotations, Fs,
                               start_time, duration):
    """
    Representa la reconstrucción mediante spline cúbico.
    También compara con interpolación lineal.
    """

    end_time = start_time + duration

    plt.figure(figsize=(14, 8))

    # --------------------------------------------------------
    # Gráfica completa del fragmento
    # --------------------------------------------------------

    plt.subplot(2, 1, 1)

    plt.plot(
        t_dense,
        y_spline,
        label=r"Spline cúbico $\widetilde y_{\mathrm{spline}}(t)$",
        linewidth=1.4
    )

    plt.plot(
        t_dense,
        y_linear,
        label=r"Interpolación lineal",
        linewidth=1.0,
        linestyle="--"
    )

    plt.scatter(
        t_fragment,
        y_fragment,
        s=12,
        label=r"Muestras filtradas $y_k$"
    )

    plt.title("Reconstrucción D/A mediante spline cúbico")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()

    if SHOW_ANNOTATIONS:
        draw_annotations(
            annotations=annotations,
            sample_numbers=sample_numbers_fragment,
            y=y_fragment,
            Fs=Fs,
            start_time=start_time,
            end_time=end_time
        )

    # --------------------------------------------------------
    # Zoom del primer segundo
    # --------------------------------------------------------

    zoom_duration = min(1.0, duration)
    zoom_end = start_time + zoom_duration

    mask_samples_zoom = (t_fragment >= start_time) & (t_fragment <= zoom_end)
    mask_dense_zoom = (t_dense >= start_time) & (t_dense <= zoom_end)

    plt.subplot(2, 1, 2)

    plt.plot(
        t_dense[mask_dense_zoom],
        y_spline[mask_dense_zoom],
        label=r"Spline cúbico",
        linewidth=1.4
    )

    plt.plot(
        t_dense[mask_dense_zoom],
        y_linear[mask_dense_zoom],
        label=r"Interpolación lineal",
        linewidth=1.0,
        linestyle="--"
    )

    plt.scatter(
        t_fragment[mask_samples_zoom],
        y_fragment[mask_samples_zoom],
        s=14,
        label=r"Muestras filtradas $y_k$"
    )

    plt.title("Zoom: spline cúbico frente a interpolación lineal")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend()

    if SHOW_ANNOTATIONS:
        draw_annotations(
            annotations=annotations,
            sample_numbers=sample_numbers_fragment,
            y=y_fragment,
            Fs=Fs,
            start_time=start_time,
            end_time=zoom_end
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Cargar señal filtrada
    # --------------------------------------------------------

    sample_numbers, t, x, y, Fs, Ts = load_filtered_signal(FILTERED_FILE)
    annotations = load_annotations(ANNOTATIONS_FILE)

    # --------------------------------------------------------
    # Extraer fragmento
    # --------------------------------------------------------

    t_fragment, y_fragment, sample_numbers_fragment = extract_fragment(
        t=t,
        y=y,
        sample_numbers=sample_numbers,
        start_time=START_TIME,
        duration=DURATION
    )

    # --------------------------------------------------------
    # Construir malla fina
    # --------------------------------------------------------

    t_dense, Fs_dense, Ts_dense = build_dense_time_grid(
        start_time=START_TIME,
        duration=DURATION,
        Fs=Fs,
        upsample_factor=UPSAMPLE_FACTOR
    )

    # --------------------------------------------------------
    # Reconstrucción por spline cúbico
    # --------------------------------------------------------

    y_spline, spline_function = cubic_spline_reconstruction(
        t_samples=t_fragment,
        y_samples=y_fragment,
        t_eval=t_dense,
        boundary_condition=BOUNDARY_CONDITION
    )

    # --------------------------------------------------------
    # Interpolación lineal para comparar
    # --------------------------------------------------------

    y_linear = linear_interpolation(
        t_samples=t_fragment,
        y_samples=y_fragment,
        t_eval=t_dense
    )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    np.savez(
        OUTPUT_FILE,
        t_fragment=t_fragment,
        y_fragment=y_fragment,
        sample_numbers_fragment=sample_numbers_fragment,
        t_dense=t_dense,
        y_spline=y_spline,
        y_linear=y_linear,
        Fs=Fs,
        Ts=Ts,
        Fs_dense=Fs_dense,
        Ts_dense=Ts_dense,
        boundary_condition=BOUNDARY_CONDITION
    )

    # --------------------------------------------------------
    # Información por pantalla
    # --------------------------------------------------------

    print("Reconstrucción mediante spline cúbico calculada correctamente.")
    print()
    print(f"Intervalo reconstruido: [{START_TIME}, {START_TIME + DURATION}] s")
    print(f"Frecuencia original: Fs = {Fs:.0f} Hz")
    print(f"Periodo original: Ts = {Ts:.8f} s")
    print()
    print(f"Factor de refinamiento: {UPSAMPLE_FACTOR}")
    print(f"Frecuencia densa: Fs_dense = {Fs_dense:.0f} Hz")
    print(f"Periodo denso: Ts_dense = {Ts_dense:.8f} s")
    print()
    print(f"Condición de contorno del spline: {BOUNDARY_CONDITION}")
    print()
    print(f"Archivo guardado: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Representar
    # --------------------------------------------------------

    plot_spline_reconstruction(
        t_fragment=t_fragment,
        y_fragment=y_fragment,
        sample_numbers_fragment=sample_numbers_fragment,
        t_dense=t_dense,
        y_spline=y_spline,
        y_linear=y_linear,
        annotations=annotations,
        Fs=Fs,
        start_time=START_TIME,
        duration=DURATION
    )