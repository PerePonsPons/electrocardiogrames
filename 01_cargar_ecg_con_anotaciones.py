import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PARÁMETROS
# ============================================================

# Frecuencia de muestreo de MIT-BIH Arrhythmia Database
Fs = 360  # Hz

# Período de muestreo
Ts = 1 / Fs

# Archivos
ECG_FILE = "archive/100.csv"
ANNOTATIONS_FILE = "archive/100annotations.txt"

# Canal que queremos estudiar
# En este archivo aparecen, por ejemplo, "MLII" y "V5"
CHANNEL = "MLII"

# Fragmento que queremos representar
START_TIME = 0       # segundos
DURATION = 10        # segundos


# ============================================================
# 2. CARGAR LA SEÑAL ECG
# ============================================================

def load_ecg_csv(filename, channel):
    """
    Carga una señal ECG en formato CSV tipo MIT-BIH.

    El archivo tiene una estructura del tipo:

        'sample #','MLII','V5'
        0,995,1011
        1,995,1011
        2,995,1011
        ...

    Devuelve:
        sample_numbers: array con los índices de muestra
        t: array de tiempos en segundos
        x: array con la señal escogida
    """

    df = pd.read_csv(filename)

    # Limpiamos nombres de columnas por si vienen con comillas o espacios
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

    # Tiempo asociado a cada muestra
    t = sample_numbers / Fs

    return sample_numbers, t, x


# ============================================================
# 3. CARGAR LAS ANOTACIONES
# ============================================================

def load_annotations(filename):
    """
    Carga las anotaciones del ECG.

    El archivo tiene una estructura del tipo:

          Time   Sample #  Type  Sub Chan  Num    Aux
        0:00.050       18     +    0    0    0    (N
        0:00.214       77     N    0    0    0
        0:01.028      370     N    0    0    0
        ...

    Devuelve un DataFrame con columnas:
        time_text, sample, type, sub, chan, num, aux
    """

    annotations = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Saltar líneas vacías o cabecera
        if not line:
            continue

        if line.startswith("Time"):
            continue

        parts = line.split()

        # Una línea válida debe tener al menos:
        # Time, Sample, Type, Sub, Chan, Num
        if len(parts) < 6:
            continue

        time_text = parts[0]
        sample = int(parts[1])
        annotation_type = parts[2]
        sub = int(parts[3])
        chan = int(parts[4])
        num = int(parts[5])

        # El campo Aux puede no existir
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
# 4. REPRESENTAR ECG CON ANOTACIONES
# ============================================================

def plot_ecg_with_annotations(sample_numbers, t, x, annotations,
                              start_time, duration, title):
    """
    Representa un fragmento del ECG junto con sus anotaciones.
    """

    end_time = start_time + duration

    # Máscara del fragmento temporal
    mask = (t >= start_time) & (t <= end_time)

    t_plot = t[mask]
    x_plot = x[mask]
    sample_plot = sample_numbers[mask]

    if len(t_plot) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    # Anotaciones dentro del intervalo
    annotations_plot = annotations[
        (annotations["sample"] >= sample_plot[0]) &
        (annotations["sample"] <= sample_plot[-1])
    ].copy()

    # Opcional: ignorar anotaciones auxiliares como "+"
    beat_annotations = annotations_plot[
        annotations_plot["type"] != "+"
    ].copy()

    plt.figure(figsize=(14, 5))

    plt.plot(t_plot, x_plot, label=f"ECG canal {CHANNEL}")

    # Dibujar anotaciones
    for _, row in beat_annotations.iterrows():
        sample = row["sample"]
        ann_type = row["type"]

        # Posición temporal de la anotación
        t_ann = sample / Fs

        # Buscamos la muestra más cercana para obtener la amplitud
        idx = np.searchsorted(sample_numbers, sample)

        if idx >= len(x):
            continue

        y_ann = x[idx]

        plt.scatter(t_ann, y_ann, marker="o")
        plt.text(
            t_ann,
            y_ann,
            ann_type,
            fontsize=9,
            ha="center",
            va="bottom"
        )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    sample_numbers, t, x = load_ecg_csv(ECG_FILE, CHANNEL)

    annotations = load_annotations(ANNOTATIONS_FILE)

    print("Señal cargada correctamente.")
    print(f"Número de muestras: {len(x)}")
    print(f"Frecuencia de muestreo: Fs = {Fs} Hz")
    print(f"Duración aproximada: {t[-1]:.2f} s")
    print(f"Canal usado: {CHANNEL}")
    print()

    print("Primeras muestras:")
    print(pd.DataFrame({
        "sample #": sample_numbers[:10],
        "t": t[:10],
        CHANNEL: x[:10]
    }))
    print()

    print("Primeras anotaciones:")
    print(annotations.head(10))

    plot_ecg_with_annotations(
        sample_numbers=sample_numbers,
        t=t,
        x=x,
        annotations=annotations,
        start_time=START_TIME,
        duration=DURATION,
        title=f"ECG {CHANNEL} con anotaciones"
    )