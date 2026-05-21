import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PARÁMETROS
# ============================================================

FILTERED_FILE = "ecg_filtrado.npz"
ANNOTATIONS_FILE = "100annotations.txt"

# Intervalo que queremos representar
START_TIME = 0       # segundos
DURATION = 3         # segundos

# Factor de refinamiento temporal para simular una señal continua
# Si Fs = 360 Hz y UPSAMPLE_FACTOR = 30, representamos a 10800 Hz.
UPSAMPLE_FACTOR = 30

# Archivo de salida
OUTPUT_FILE = "reconstruccion_basica.npz"

# Mostrar o no las anotaciones
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
        return pd.DataFrame(columns=["time_text", "sample", "type", "sub", "chan", "num", "aux"])

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
# 4. RECONSTRUCCIÓN POR RETENCIÓN DE ORDEN CERO
# ============================================================

def zero_order_hold(t_samples, y_samples, t_eval):
    """
    Reconstrucción por retención de orden cero.

    Para cada t en [t_k, t_{k+1}), se define:

        y_tilde(t) = y_k

    Entrada:
        t_samples: tiempos originales t_k
        y_samples: valores discretos y_k
        t_eval: tiempos donde queremos evaluar la reconstrucción

    Salida:
        y_zoh: reconstrucción evaluada en t_eval
    """

    indices = np.searchsorted(t_samples, t_eval, side="right") - 1

    indices = np.clip(indices, 0, len(y_samples) - 1)

    y_zoh = y_samples[indices]

    return y_zoh


# ============================================================
# 5. RECONSTRUCCIÓN POR INTERPOLACIÓN LINEAL
# ============================================================

def linear_interpolation(t_samples, y_samples, t_eval):
    """
    Reconstrucción por interpolación lineal.

    Para t en [t_k, t_{k+1}], se define:

        y_tilde(t)
        =
        y_k + (y_{k+1} - y_k) / (t_{k+1} - t_k) * (t - t_k)

    Como el muestreo es uniforme:

        t_{k+1} - t_k = Ts

    Entrada:
        t_samples: tiempos originales t_k
        y_samples: valores discretos y_k
        t_eval: tiempos donde queremos evaluar la reconstrucción

    Salida:
        y_linear: reconstrucción evaluada en t_eval
    """

    y_linear = np.interp(t_eval, t_samples, y_samples)

    return y_linear


# ============================================================
# 6. CONSTRUIR MALLA TEMPORAL FINA
# ============================================================

def build_dense_time_grid(t, Fs, upsample_factor):
    """
    Construye una malla temporal fina para representar una señal continua.

    Si la señal original tiene frecuencia Fs, la nueva malla tendrá frecuencia:

        Fs_dense = upsample_factor * Fs
    """

    Fs_dense = upsample_factor * Fs
    Ts_dense = 1 / Fs_dense

    t_start = t[0]
    t_end = t[-1]

    t_dense = np.arange(t_start, t_end, Ts_dense)

    return t_dense, Fs_dense, Ts_dense


# ============================================================
# 7. REPRESENTACIÓN GRÁFICA
# ============================================================

def plot_reconstructions(t, y, t_dense, y_zoh, y_linear,
                         sample_numbers, annotations,
                         Fs, start_time, duration):
    """
    Representa la señal discreta y_k y sus dos reconstrucciones:
        - retención de orden cero
        - interpolación lineal
    """

    end_time = start_time + duration

    mask_discrete = (t >= start_time) & (t <= end_time)
    mask_dense = (t_dense >= start_time) & (t_dense <= end_time)

    t_plot = t[mask_discrete]
    y_plot = y[mask_discrete]

    t_dense_plot = t_dense[mask_dense]
    y_zoh_plot = y_zoh[mask_dense]
    y_linear_plot = y_linear[mask_dense]

    if len(t_plot) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    plt.figure(figsize=(14, 8))

    # --------------------------------------------------------
    # Retención de orden cero
    # --------------------------------------------------------

    plt.subplot(2, 1, 1)

    plt.plot(
        t_dense_plot,
        y_zoh_plot,
        label=r"Reconstrucción por orden cero $\widetilde y_0(t)$"
    )

    plt.scatter(
        t_plot,
        y_plot,
        s=12,
        label=r"Muestras filtradas $y_k$"
    )

    plt.ylabel("Amplitud")
    plt.title("Reconstrucción D/A por retención de orden cero")
    plt.grid(True)
    plt.legend()

    if SHOW_ANNOTATIONS:
        draw_annotations(annotations, sample_numbers, y, Fs, start_time, end_time)

    # --------------------------------------------------------
    # Interpolación lineal
    # --------------------------------------------------------

    plt.subplot(2, 1, 2)

    plt.plot(
        t_dense_plot,
        y_linear_plot,
        label=r"Reconstrucción lineal $\widetilde y_1(t)$"
    )

    plt.scatter(
        t_plot,
        y_plot,
        s=12,
        label=r"Muestras filtradas $y_k$"
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title("Reconstrucción D/A por interpolación lineal")
    plt.grid(True)
    plt.legend()

    if SHOW_ANNOTATIONS:
        draw_annotations(annotations, sample_numbers, y, Fs, start_time, end_time)

    plt.tight_layout()
    plt.show()


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
# 8. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Cargar señal filtrada
    # --------------------------------------------------------

    sample_numbers, t, x, y, Fs, Ts = load_filtered_signal(FILTERED_FILE)

    annotations = load_annotations(ANNOTATIONS_FILE)

    # --------------------------------------------------------
    # Construir malla fina
    # --------------------------------------------------------

    t_dense, Fs_dense, Ts_dense = build_dense_time_grid(
        t=t,
        Fs=Fs,
        upsample_factor=UPSAMPLE_FACTOR
    )

    # --------------------------------------------------------
    # Reconstrucciones D/A
    # --------------------------------------------------------

    y_zoh = zero_order_hold(
        t_samples=t,
        y_samples=y,
        t_eval=t_dense
    )

    y_linear = linear_interpolation(
        t_samples=t,
        y_samples=y,
        t_eval=t_dense
    )

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    np.savez(
        OUTPUT_FILE,
        t=t,
        y=y,
        t_dense=t_dense,
        y_zoh=y_zoh,
        y_linear=y_linear,
        Fs=Fs,
        Ts=Ts,
        Fs_dense=Fs_dense,
        Ts_dense=Ts_dense
    )

    # --------------------------------------------------------
    # Información por pantalla
    # --------------------------------------------------------

    print("Reconstrucciones básicas calculadas correctamente.")
    print()
    print(f"Frecuencia original: Fs = {Fs:.0f} Hz")
    print(f"Periodo original: Ts = {Ts:.8f} s")
    print()
    print(f"Factor de refinamiento: {UPSAMPLE_FACTOR}")
    print(f"Frecuencia de representación continua: Fs_dense = {Fs_dense:.0f} Hz")
    print(f"Periodo de representación continua: Ts_dense = {Ts_dense:.8f} s")
    print()
    print(f"Número de muestras originales: {len(y)}")
    print(f"Número de puntos reconstruidos: {len(t_dense)}")
    print()
    print(f"Archivo guardado: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Representar
    # --------------------------------------------------------

    plot_reconstructions(
        t=t,
        y=y,
        t_dense=t_dense,
        y_zoh=y_zoh,
        y_linear=y_linear,
        sample_numbers=sample_numbers,
        annotations=annotations,
        Fs=Fs,
        start_time=START_TIME,
        duration=DURATION
    )