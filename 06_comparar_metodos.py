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

    # --------------------------------------------------------
    # 1. Reconstrucciones en todo el fragmento
    # --------------------------------------------------------

    plt.subplot(4, 1, 1)

    plt.plot(
        t_ref,
        y_ref,
        linewidth=1.5,
        label=r"Referencia $y_{\mathrm{ref}}$"
    )

    plt.scatter(
        t_low,
        y_low,
        s=15,
        label=r"Muestras disponibles $y_k$"
    )

    for method_name, y_rec in reconstructions.items():
        plt.plot(
            t_ref,
            y_rec,
            linewidth=1.0,
            label=method_name
        )

    plt.title("Comparación de métodos de reconstrucción D/A")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc="upper right")

    if SHOW_ANNOTATIONS:
        draw_annotations(
            annotations=annotations,
            y_ref=y_ref,
            t_ref=t_ref,
            Fs=Fs,
            start_time=start_time,
            end_time=end_time
        )

    # --------------------------------------------------------
    # 2. Zoom del primer segundo
    # --------------------------------------------------------

    zoom_duration = min(1.0, duration)
    zoom_end = start_time + zoom_duration

    mask_ref_zoom = (t_ref >= start_time) & (t_ref <= zoom_end)
    mask_low_zoom = (t_low >= start_time) & (t_low <= zoom_end)

    plt.subplot(4, 1, 2)

    plt.plot(
        t_ref[mask_ref_zoom],
        y_ref[mask_ref_zoom],
        linewidth=1.5,
        label=r"Referencia $y_{\mathrm{ref}}$"
    )

    plt.scatter(
        t_low[mask_low_zoom],
        y_low[mask_low_zoom],
        s=20,
        label=r"Muestras disponibles $y_k$"
    )

    for method_name, y_rec in reconstructions.items():
        plt.plot(
            t_ref[mask_ref_zoom],
            y_rec[mask_ref_zoom],
            linewidth=1.0,
            label=method_name
        )

    plt.title("Zoom local")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.legend(loc="upper right")

    if SHOW_ANNOTATIONS:
        draw_annotations(
            annotations=annotations,
            y_ref=y_ref,
            t_ref=t_ref,
            Fs=Fs,
            start_time=start_time,
            end_time=zoom_end
        )

    # --------------------------------------------------------
    # 3. Errores puntuales
    # --------------------------------------------------------

    plt.subplot(4, 1, 3)

    for method_name, y_rec in reconstructions.items():
        error = y_rec - y_ref

        plt.plot(
            t_ref,
            error,
            linewidth=1.0,
            label=method_name
        )

    plt.axhline(0, linewidth=0.8)
    plt.title(r"Errores puntuales $e(t_j)=\widetilde y(t_j)-y_{\mathrm{ref}}(t_j)$")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend(loc="upper right")

    # --------------------------------------------------------
    # 4. RMSE por método
    # --------------------------------------------------------

    plt.subplot(4, 1, 4)

    plt.bar(
        metrics_df["Metodo"],
        metrics_df["RMSE"]
    )

    plt.title("RMSE por método")
    plt.xlabel("Método")
    plt.ylabel("RMSE")
    plt.grid(True, axis="y")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.show()


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

    # --------------------------------------------------------
    # Reconstrucciones
    # --------------------------------------------------------

    reconstructions = {}

    print("Calculando retención de orden cero...")
    reconstructions["Orden cero"] = zero_order_hold(
        t_samples=t_low,
        y_samples=y_low,
        t_eval=t_ref
    )

    print("Calculando interpolación lineal...")
    reconstructions["Lineal"] = linear_interpolation(
        t_samples=t_low,
        y_samples=y_low,
        t_eval=t_ref
    )

    print("Calculando spline cúbico...")
    reconstructions["Spline cúbico"] = cubic_spline_reconstruction(
        t_samples=t_low,
        y_samples=y_low,
        t_eval=t_ref,
        bc_type=SPLINE_BC_TYPE
    )

    for m in NEWTON_M_VALUES:
        print(f"Calculando Newton local con m = {m}...")

        method_name = f"Newton m={m}"

        reconstructions[method_name] = newton_local_reconstruction(
            t_samples=t_low,
            y_samples=y_low,
            t_eval=t_ref,
            m=m
        )

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

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

    metrics_df = metrics_df.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Guardar métricas
    # --------------------------------------------------------

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Guardar reconstrucciones
    # --------------------------------------------------------

    save_dict = {
        "t_ref": t_ref,
        "y_ref": y_ref,
        "sample_numbers_ref": sample_numbers_ref,
        "t_low": t_low,
        "y_low": y_low,
        "sample_numbers_low": sample_numbers_low,
        "Fs": Fs,
        "Ts": Ts,
        "Fs_low": Fs_low,
        "Ts_low": Ts_low,
        "DOWNSAMPLE_FACTOR": DOWNSAMPLE_FACTOR
    }

    for method_name, y_rec in reconstructions.items():
        key = method_name.lower()
        key = key.replace(" ", "_")
        key = key.replace("=", "_")
        key = key.replace("ú", "u")
        save_dict[key] = y_rec

    np.savez(
        OUTPUT_FILE,
        **save_dict
    )

    # --------------------------------------------------------
    # Mostrar resumen
    # --------------------------------------------------------

    print()
    print("Comparación terminada correctamente.")
    print()
    print(f"Intervalo estudiado: [{START_TIME}, {START_TIME + DURATION}] s")
    print(f"Frecuencia de referencia: Fs = {Fs:.0f} Hz")
    print(f"Frecuencia simulada: Fs_low = {Fs_low:.0f} Hz")
    print(f"Factor de submuestreo: {DOWNSAMPLE_FACTOR}")
    print()
    print("Métricas de error:")
    print(metrics_df)
    print()
    print(f"Archivo de métricas guardado: {METRICS_FILE}")
    print(f"Archivo de reconstrucciones guardado: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Dibujar resultados
    # --------------------------------------------------------

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