import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1. PARÁMETROS
# ============================================================

Fs = 360  # Hz, frecuencia de muestreo de MIT-BIH
Ts = 1 / Fs

ECG_FILE = "archive/100.csv"
ANNOTATIONS_FILE = "archive/100annotations.txt"

CHANNEL = "MLII"

# Filtro pasa-banda típico para ECG
LOWCUT = 0.5   # Hz
HIGHCUT = 40   # Hz
FILTER_ORDER = 4

# Fragmento a representar
START_TIME = 0       # segundos
DURATION = 10        # segundos

# Archivo de salida para usar en los siguientes programas
OUTPUT_FILE = "ecg_filtrado.npz"


# ============================================================
# 2. CARGAR ECG
# ============================================================

def load_ecg_csv(filename, channel):
    """
    Carga un archivo CSV tipo MIT-BIH.

    Formato esperado:

        'sample #','MLII','V5'
        0,995,1011
        1,995,1011
        2,995,1011
        ...

    Devuelve:
        sample_numbers: índices de muestra
        t: tiempos en segundos
        x: señal ECG del canal escogido
    """

    df = pd.read_csv(filename)

    # Limpiar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    if "sample #" not in df.columns:
        raise ValueError(
            f"No se ha encontrado la columna 'sample #'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    if channel not in df.columns:
        raise ValueError(
            f"No se ha encontrado el canal '{channel}'. "
            f"Canales disponibles: {list(df.columns)}"
        )

    sample_numbers = df["sample #"].to_numpy(dtype=int)
    x = df[channel].to_numpy(dtype=float)

    t = sample_numbers / Fs

    return sample_numbers, t, x


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
# 4. DISEÑAR Y APLICAR FILTRO PASA-BANDA
# ============================================================

def bandpass_filter(x, fs, lowcut, highcut, order):
    """
    Aplica un filtro digital pasa-banda Butterworth.

    Entrada:
        x: señal discreta x_k
        fs: frecuencia de muestreo
        lowcut: frecuencia inferior del pasa-banda
        highcut: frecuencia superior del pasa-banda
        order: orden del filtro

    Salida:
        y: señal filtrada y_k
    """

    nyquist = fs / 2

    low = lowcut / nyquist
    high = highcut / nyquist

    if low <= 0:
        raise ValueError("La frecuencia inferior debe ser positiva.")

    if high >= 1:
        raise ValueError("La frecuencia superior debe ser menor que Fs/2.")

    sos = butter(
        N=order,
        Wn=[low, high],
        btype="bandpass",
        output="sos"
    )

    # sosfiltfilt aplica el filtro hacia delante y hacia atrás.
    # Esto evita desfase en la señal filtrada.
    y = sosfiltfilt(sos, x)

    return y


# ============================================================
# 5. REPRESENTAR x_k E y_k CON ANOTACIONES
# ============================================================

def plot_original_and_filtered(sample_numbers, t, x, y, annotations,
                               start_time, duration):
    """
    Dibuja la señal original x_k y la señal filtrada y_k
    en un intervalo temporal dado.
    """

    end_time = start_time + duration

    mask = (t >= start_time) & (t <= end_time)

    t_plot = t[mask]
    x_plot = x[mask]
    y_plot = y[mask]
    sample_plot = sample_numbers[mask]

    if len(t_plot) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    annotations_plot = annotations[
        (annotations["sample"] >= sample_plot[0]) &
        (annotations["sample"] <= sample_plot[-1])
    ].copy()

    # Quitamos anotaciones auxiliares como "+"
    beat_annotations = annotations_plot[
        annotations_plot["type"] != "+"
    ].copy()

    plt.figure(figsize=(14, 7))

    # --------------------------------------------------------
    # Señal original
    # --------------------------------------------------------

    plt.subplot(2, 1, 1)
    plt.plot(t_plot, x_plot, label=r"Señal original $x_k$")
    plt.ylabel("Amplitud")
    plt.title(r"ECG original $x_k$")
    plt.grid(True)
    plt.legend()

    for _, row in beat_annotations.iterrows():
        sample = row["sample"]
        ann_type = row["type"]
        t_ann = sample / Fs

        idx = np.searchsorted(sample_numbers, sample)

        if idx < len(x):
            plt.axvline(t_ann, linestyle="--", linewidth=0.8)
            plt.text(
                t_ann,
                x[idx],
                ann_type,
                fontsize=8,
                ha="center",
                va="bottom"
            )

    # --------------------------------------------------------
    # Señal filtrada
    # --------------------------------------------------------

    plt.subplot(2, 1, 2)
    plt.plot(t_plot, y_plot, label=r"Señal filtrada $y_k$")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title(
        rf"ECG filtrado $y_k$ con pasa-banda "
        rf"${LOWCUT}\,\mathrm{{Hz}} \leq f \leq {HIGHCUT}\,\mathrm{{Hz}}$"
    )
    plt.grid(True)
    plt.legend()

    for _, row in beat_annotations.iterrows():
        sample = row["sample"]
        ann_type = row["type"]
        t_ann = sample / Fs

        idx = np.searchsorted(sample_numbers, sample)

        if idx < len(y):
            plt.axvline(t_ann, linestyle="--", linewidth=0.8)
            plt.text(
                t_ann,
                y[idx],
                ann_type,
                fontsize=8,
                ha="center",
                va="bottom"
            )

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # Cargar datos
    sample_numbers, t, x = load_ecg_csv(ECG_FILE, CHANNEL)
    annotations = load_annotations(ANNOTATIONS_FILE)

    # Aplicar filtro
    y = bandpass_filter(
        x=x,
        fs=Fs,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        order=FILTER_ORDER
    )

    # Mostrar información básica
    print("ECG cargado y filtrado correctamente.")
    print(f"Canal usado: {CHANNEL}")
    print(f"Frecuencia de muestreo: Fs = {Fs} Hz")
    print(f"Periodo de muestreo: Ts = {Ts:.6f} s")
    print(f"Número de muestras: {len(x)}")
    print(f"Duración aproximada: {t[-1]:.2f} s")
    print()
    print("Filtro pasa-banda:")
    print(f"Frecuencia inferior: {LOWCUT} Hz")
    print(f"Frecuencia superior: {HIGHCUT} Hz")
    print(f"Orden del filtro: {FILTER_ORDER}")

    # Guardar señal filtrada para los siguientes programas
    np.savez(
        OUTPUT_FILE,
        sample_numbers=sample_numbers,
        t=t,
        x=x,
        y=y,
        Fs=Fs,
        Ts=Ts
    )

    print()
    print(f"Archivo guardado: {OUTPUT_FILE}")

    # Representar
    plot_original_and_filtered(
        sample_numbers=sample_numbers,
        t=t,
        x=x,
        y=y,
        annotations=annotations,
        start_time=START_TIME,
        duration=DURATION
    )