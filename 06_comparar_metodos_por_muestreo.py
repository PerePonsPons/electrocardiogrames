import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ============================================================
# SHANNON (SINC)
# ============================================================

def sinc(x):
    return np.sinc(x)


def sinc_interpolation(t_samples, y_samples, t_eval):
    """
    Interpolació de Shannon (sinc).
    """

    T = t_samples[1] - t_samples[0]

    y_rec = np.zeros_like(t_eval, dtype=float)

    for i, t in enumerate(t_eval):
        y_rec[i] = np.sum(
            y_samples * sinc((t - t_samples) / T)
        )

    return y_rec
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
# 4. EXTRAER FRAGMENTO DE REFERENCIA
# ============================================================

def extract_reference_fragment(t, y, sample_numbers, start_time, duration):
    """
    Extrae un fragmento de la señal filtrada.

    Este fragmento será nuestra referencia:

        y_ref(t_j)

    evaluada en los instantes originales de la base de datos.
    """

    end_time = start_time + duration

    mask = (t >= start_time) & (t <= end_time)

    if np.sum(mask) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    t_ref = t[mask]
    y_ref = y[mask]
    sample_numbers_ref = sample_numbers[mask]

    return t_ref, y_ref, sample_numbers_ref


# ============================================================
# 5. SIMULAR UNA SEÑAL CON MENOS MUESTRAS
# ============================================================

def downsample_signal(t_ref, y_ref, sample_numbers_ref, factor):
    """
    Simula una frecuencia de muestreo menor tomando una muestra
    cada 'factor' muestras.

    Si la referencia está a Fs = 360 Hz y factor = 4, entonces:

        Fs_low = 360 / 4 = 90 Hz.

    Entrada:
        t_ref: tiempos de referencia
        y_ref: señal de referencia
        sample_numbers_ref: índices de muestra originales
        factor: factor de submuestreo

    Salida:
        t_low: tiempos de la señal submuestreada
        y_low: valores de la señal submuestreada
        sample_numbers_low: índices de muestra submuestreados
    """

    if factor < 1:
        raise ValueError("El factor de submuestreo debe ser mayor o igual que 1.")

    t_low = t_ref[::factor]
    y_low = y_ref[::factor]
    sample_numbers_low = sample_numbers_ref[::factor]

    return t_low, y_low, sample_numbers_low

# ============================================================
# 6. RECONSTRUCCIÓN POR ORDEN CERO
# ============================================================

def zero_order_hold(t_samples, y_samples, t_eval):
    """
    Retención de orden cero:

        y_tilde(t) = y_k,
        t in [t_k, t_{k+1}).
    """

    indices = np.searchsorted(t_samples, t_eval, side="right") - 1
    indices = np.clip(indices, 0, len(y_samples) - 1)

    return y_samples[indices]

# ============================================================
# 7. RECONSTRUCCIÓN LINEAL
# ============================================================

def linear_interpolation(t_samples, y_samples, t_eval):
    """
    Interpolación lineal.
    """

    return np.interp(t_eval, t_samples, y_samples)


# ============================================================
# 8. RECONSTRUCCIÓN MEDIANTE SPLINE CÚBICO
# ============================================================

def cubic_spline_reconstruction(t_samples, y_samples, t_eval, bc_type="natural"):
    """
    Reconstrucción mediante spline cúbico.

    Construye S(t) tal que:

        S(t_k) = y_k,

    y S es de clase C^2.
    """

    spline = CubicSpline(
        t_samples,
        y_samples,
        bc_type=bc_type
    )

    return spline(t_eval)


# ============================================================
# 9. NEWTON LOCAL
# ============================================================

def newton_divided_differences(t_nodes, y_nodes):
    """
    Calcula los coeficientes de Newton mediante diferencias divididas.
    """

    t_nodes = np.asarray(t_nodes, dtype=float)
    coeffs = np.asarray(y_nodes, dtype=float).copy()

    n = len(t_nodes)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeffs[i] = (
                (coeffs[i] - coeffs[i - 1])
                / (t_nodes[i] - t_nodes[i - j])
            )

    return coeffs


def newton_evaluate(t_nodes, coeffs, t_value):
    """
    Evalúa el polinomio de Newton usando la forma de Horner.
    """

    value = coeffs[-1]

    for j in range(len(coeffs) - 2, -1, -1):
        value = coeffs[j] + (t_value - t_nodes[j]) * value

    return value


def choose_local_window(t_samples, t_value, m):
    """
    Escoge una ventana local de m+1 nodos alrededor de t_value.
    """

    n_samples = len(t_samples)
    n_nodes = m + 1

    if n_nodes > n_samples:
        raise ValueError(
            f"No hay suficientes muestras para usar Newton con m = {m}."
        )

    k = np.searchsorted(t_samples, t_value, side="right") - 1
    k = np.clip(k, 0, n_samples - 1)

    left_nodes = m // 2

    start = k - left_nodes

    start = max(0, start)
    start = min(start, n_samples - n_nodes)

    end = start + n_nodes

    return start, end


def newton_local_reconstruction(t_samples, y_samples, t_eval, m):
    """
    Reconstrucción mediante polinomios de Newton locales.

    Para cada t:
        1. Escogemos m+1 nodos cercanos.
        2. Construimos el polinomio de Newton.
        3. Evaluamos en t.
    """

    y_newton = np.zeros_like(t_eval, dtype=float)

    for i, t_value in enumerate(t_eval):

        start, end = choose_local_window(
            t_samples=t_samples,
            t_value=t_value,
            m=m
        )

        t_nodes = t_samples[start:end]
        y_nodes = y_samples[start:end]

        coeffs = newton_divided_differences(
            t_nodes=t_nodes,
            y_nodes=y_nodes
        )

        y_newton[i] = newton_evaluate(
            t_nodes=t_nodes,
            coeffs=coeffs,
            t_value=t_value
        )

    return y_newton


# ============================================================
# 10. MÉTRICAS DE ERROR
# ============================================================

def compute_metrics(y_ref, y_rec):
    """
    Calcula métricas de error entre la referencia y la reconstrucción.

    Error puntual:

        e_j = y_rec(t_j) - y_ref(t_j)

    RMSE:

        sqrt( mean(e_j^2) )

    MAE:

        mean(|e_j|)

    Error máximo:

        max |e_j|

    PRD:

        100 * sqrt( sum(e_j^2) / sum(y_ref_j^2) )
    """

    error = y_rec - y_ref

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))

    denominator = np.sum(y_ref ** 2)

    if denominator == 0:
        prd = np.nan
    else:
        prd = 100 * np.sqrt(np.sum(error ** 2) / denominator)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MaxAbsError": max_error,
        "PRD_percent": prd
    }

# ============================================================
# 11. DIBUJAR ANOTACIONES
# ============================================================

def draw_annotations(annotations, y_ref, t_ref, Fs, start_time, end_time):
    """
    Dibuja anotaciones sobre la gráfica actual.
    """

    if annotations.empty:
        return

    start_sample = int(start_time * Fs)
    end_sample = int(end_time * Fs)

    annotations_plot = annotations[
        (annotations["sample"] >= start_sample) &
        (annotations["sample"] <= end_sample)
    ].copy()

    beat_annotations = annotations_plot[
        annotations_plot["type"] != "+"
    ].copy()

    for _, row in beat_annotations.iterrows():

        sample = row["sample"]
        ann_type = row["type"]

        t_ann = sample / Fs

        idx = np.searchsorted(t_ref, t_ann)

        if idx >= len(y_ref):
            continue

        plt.axvline(t_ann, linestyle="--", linewidth=0.8)

        plt.text(
            t_ann,
            y_ref[idx],
            ann_type,
            fontsize=8,
            ha="center",
            va="bottom"
        )


# ============================================================
# 12. REPRESENTACIÓN GRÁFICA
# ============================================================

def plot_comparison(t_ref, y_ref, t_low, y_low,
                    reconstructions, metrics_df,
                    annotations, Fs,
                    start_time, duration):
    """
    Dibuja:
        1. Comparación de reconstrucciones.
        2. Zoom de un segundo.
        3. Errores puntuales.
        4. Tabla visual de RMSE.
    """

    end_time = start_time + duration

    plt.figure(figsize=(15, 12))

# ============================================================
# RECONSTRUCCIONS
# ============================================================
# ============================================================
# 1. PARÁMETROS
# ============================================================

FILTERED_FILE = "ecg_filtrado.npz"
ANNOTATIONS_FILE = "archive/100annotations.txt"

# Intervalo temporal que queremos estudiar
START_TIME = 0       # segundos
DURATION = 8         # segundos

# Simulación de una frecuencia de muestreo más baja.
#
# Si la señal de referencia está a Fs = 360 Hz:
#
# DOWNSAMPLE_FACTOR = 2  ->  Fs_low = 180 Hz
# DOWNSAMPLE_FACTOR = 3  ->  Fs_low = 120 Hz
# DOWNSAMPLE_FACTOR = 4  ->  Fs_low = 90 Hz
# DOWNSAMPLE_FACTOR = 6  ->  Fs_low = 60 Hz
#
DOWNSAMPLE_FACTOR = 4

# Grados de Newton local que queremos comparar
NEWTON_M_VALUES = [1, 3, 5, 7]

# Condición de contorno para el spline cúbico
SPLINE_BC_TYPE = "natural"

# Archivos de salida
OUTPUT_FILE = "comparacion_metodos.npz"
METRICS_FILE = "metricas_comparacion.csv"

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

# ... (NO CANVIES RES del teu codi anterior fins aquí)

# ============================================================
# 13. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------

    sample_numbers, t, x, y, Fs, Ts = load_filtered_signal(FILTERED_FILE)
    annotations = load_annotations(ANNOTATIONS_FILE)

    # --------------------------------------------------------
    # Referencia
    # --------------------------------------------------------

    t_ref, y_ref, sample_numbers_ref = extract_reference_fragment(
        t=t,
        y=y,
        sample_numbers=sample_numbers,
        start_time=START_TIME,
        duration=DURATION
    )

    # --------------------------------------------------------
    # Simular una señal menos muestreada
    # --------------------------------------------------------

    t_low, y_low, sample_numbers_low = downsample_signal(
        t_ref=t_ref,
        y_ref=y_ref,
        sample_numbers_ref=sample_numbers_ref,
        factor=DOWNSAMPLE_FACTOR
    )

    Fs_low = Fs / DOWNSAMPLE_FACTOR
    Ts_low = 1 / Fs_low


    reconstructions = {}

    print("Orden cero...")
    reconstructions["Orden cero"] = zero_order_hold(
        t_low, y_low, t_ref
    )

    print("Lineal...")
    reconstructions["Lineal"] = linear_interpolation(
        t_low, y_low, t_ref
    )

    print("Spline cúbico...")
    reconstructions["Spline cúbico"] = cubic_spline_reconstruction(
        t_low, y_low, t_ref
    )

    print("Shannon (sinc)...")
    reconstructions["Shannon"] = sinc_interpolation(
        t_low, y_low, t_ref
    )

    for m in NEWTON_M_VALUES:
        print(f"Newton m={m}...")
        reconstructions[f"Newton m={m}"] = newton_local_reconstruction(
            t_low, y_low, t_ref, m
        )

    # ============================================================
    # MÉTRICAS
    # ============================================================

    metrics_rows = []

    for method_name, y_rec in reconstructions.items():

        metrics = compute_metrics(
            y_ref=y_ref,
            y_rec=y_rec
        )

        metrics_rows.append({
            "Metodo": method_name,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MaxAbsError": metrics["MaxAbsError"],
            "PRD_percent": metrics["PRD_percent"]
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(by="RMSE").reset_index(drop=True)

    print(metrics_df)

    # ============================================================
    # PLOTS (igual que abans)
    # ============================================================

    plot_comparison(
        t_ref=t_ref,
        y_ref=y_ref,
        t_low=t_low,
        y_low=y_low,
        reconstructions=reconstructions,
        metrics_df=metrics_df,
        annotations=annotations,
        Fs=Fs,
        start_time=START_TIME,
        duration=DURATION
    )