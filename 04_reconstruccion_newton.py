import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PARÁMETROS
# ============================================================

FILTERED_FILE = "ecg_filtrado.npz"
ANNOTATIONS_FILE = "archive/100annotations.txt"

# Intervalo de la señal que queremos reconstruir
START_TIME = 0       # segundos
DURATION = 3         # segundos

# Factor de refinamiento temporal.
# Si Fs = 360 Hz y UPSAMPLE_FACTOR = 30,
# entonces representamos a 10800 Hz.
UPSAMPLE_FACTOR = 30

# Grados de los polinomios de Newton que queremos comparar
# m = 1 equivale a interpolación lineal local.
M_VALUES = [1, 2, 3, 5, 7]

# Archivo de salida
OUTPUT_FILE = "reconstruccion_newton.npz"

# Mostrar anotaciones de latidos
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
# 4. DIVIDED DIFFERENCES DE NEWTON
# ============================================================

def newton_divided_differences(t_nodes, y_nodes):
    """
    Calcula los coeficientes del polinomio de Newton.

    Dados los nodos:

        t_0, t_1, ..., t_m

    y los valores:

        y_0, y_1, ..., y_m

    el polinomio de Newton es:

        P_m(t)
        =
        c_0
        + c_1 (t - t_0)
        + c_2 (t - t_0)(t - t_1)
        + ...
        + c_m (t - t_0)...(t - t_{m-1})

    donde los coeficientes c_j son diferencias divididas.
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


def newton_evaluate(t_nodes, coeffs, t_eval):
    """
    Evalúa el polinomio de Newton mediante una forma de Horner.

    Entrada:
        t_nodes: nodos usados en el polinomio
        coeffs: coeficientes de Newton
        t_eval: punto donde evaluar

    Salida:
        P_m(t_eval)
    """

    n = len(coeffs)

    value = coeffs[-1]

    for j in range(n - 2, -1, -1):
        value = coeffs[j] + (t_eval - t_nodes[j]) * value

    return value


# ============================================================
# 5. VENTANA LOCAL PARA NEWTON
# ============================================================

def choose_local_window(t_samples, t_value, m):
    """
    Escoge una ventana local de m+1 nodos alrededor de t_value.

    Si t_value está aproximadamente entre t_k y t_{k+1},
    escogemos una ventana centrada alrededor de ese intervalo.

    Para m = 1:
        usamos dos nodos, aproximadamente t_k y t_{k+1}.

    Para m = 3:
        usamos cuatro nodos, aproximadamente
        t_{k-1}, t_k, t_{k+1}, t_{k+2}.
    """

    n_samples = len(t_samples)
    n_nodes = m + 1

    if n_nodes > n_samples:
        raise ValueError(
            f"No hay suficientes muestras para construir una ventana de "
            f"{n_nodes} nodos."
        )

    # Índice k tal que t_k <= t_value < t_{k+1}
    k = np.searchsorted(t_samples, t_value, side="right") - 1

    # Evitamos índices fuera de rango
    k = np.clip(k, 0, n_samples - 1)

    # Número de nodos a la izquierda del punto central
    left_nodes = m // 2

    start = k - left_nodes

    # Ajustamos para que la ventana esté dentro de la señal
    start = max(0, start)
    start = min(start, n_samples - n_nodes)

    end = start + n_nodes

    return start, end


def newton_local_reconstruction(t_samples, y_samples, t_eval, m):
    """
    Reconstruye la señal mediante interpolación local de Newton.

    Para cada instante t en t_eval:
        1. Escoge m+1 muestras cercanas.
        2. Construye el polinomio de Newton de grado m.
        3. Evalúa el polinomio en t.

    Entrada:
        t_samples: tiempos originales t_k
        y_samples: señal discreta y_k
        t_eval: tiempos donde evaluar la reconstrucción
        m: grado del polinomio de Newton

    Salida:
        y_newton: reconstrucción evaluada en t_eval
    """

    if m < 0:
        raise ValueError("El grado m debe ser mayor o igual que 0.")

    y_newton = np.zeros_like(t_eval, dtype=float)

    for i, t_value in enumerate(t_eval):

        start, end = choose_local_window(
            t_samples=t_samples,
            t_value=t_value,
            m=m
        )

        t_nodes = t_samples[start:end]
        y_nodes = y_samples[start:end]

        coeffs = newton_divided_differences(t_nodes, y_nodes)

        y_newton[i] = newton_evaluate(
            t_nodes=t_nodes,
            coeffs=coeffs,
            t_eval=t_value
        )

    return y_newton


# ============================================================
# 6. CONSTRUIR FRAGMENTO Y MALLA DENSA
# ============================================================

def extract_fragment(t, y, sample_numbers, start_time, duration):
    """
    Extrae un fragmento temporal de la señal.
    """

    end_time = start_time + duration

    mask = (t >= start_time) & (t <= end_time)

    if np.sum(mask) == 0:
        raise ValueError("El intervalo escogido no contiene muestras.")

    return t[mask], y[mask], sample_numbers[mask]


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
# 7. DIBUJAR ANOTACIONES
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
# 8. REPRESENTACIÓN GRÁFICA
# ============================================================

def plot_newton_reconstructions(t_fragment, y_fragment,
                                sample_numbers_fragment,
                                t_dense, reconstructions,
                                annotations, Fs,
                                start_time, duration):
    """
    Representa las reconstrucciones de Newton para distintos grados m.
    """

    end_time = start_time + duration

    plt.figure(figsize=(14, 8))

    # --------------------------------------------------------
    # Gráfica conjunta
    # --------------------------------------------------------

    plt.subplot(2, 1, 1)

    plt.scatter(
        t_fragment,
        y_fragment,
        s=12,
        label=r"Muestras filtradas $y_k$"
    )

    for m, y_newton in reconstructions.items():
        plt.plot(
            t_dense,
            y_newton,
            linewidth=1.2,
            label=rf"Newton local, $m={m}$"
        )

    plt.title("Reconstrucción D/A mediante polinomios de Newton locales")
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
    # Zoom del primer segundo del fragmento
    # --------------------------------------------------------

    zoom_duration = min(1.0, duration)
    zoom_end = start_time + zoom_duration

    mask_samples_zoom = (t_fragment >= start_time) & (t_fragment <= zoom_end)
    mask_dense_zoom = (t_dense >= start_time) & (t_dense <= zoom_end)

    plt.subplot(2, 1, 2)

    plt.scatter(
        t_fragment[mask_samples_zoom],
        y_fragment[mask_samples_zoom],
        s=14,
        label=r"Muestras filtradas $y_k$"
    )

    for m, y_newton in reconstructions.items():
        plt.plot(
            t_dense[mask_dense_zoom],
            y_newton[mask_dense_zoom],
            linewidth=1.2,
            label=rf"Newton local, $m={m}$"
        )

    plt.title("Zoom: comparación local de los interpoladores")
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
# 9. PROGRAMA PRINCIPAL
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
    # Construir malla temporal fina
    # --------------------------------------------------------

    t_dense, Fs_dense, Ts_dense = build_dense_time_grid(
        start_time=START_TIME,
        duration=DURATION,
        Fs=Fs,
        upsample_factor=UPSAMPLE_FACTOR
    )

    # --------------------------------------------------------
    # Calcular reconstrucciones de Newton
    # --------------------------------------------------------

    reconstructions = {}

    for m in M_VALUES:
        print(f"Calculando reconstrucción de Newton con m = {m}...")

        y_newton = newton_local_reconstruction(
            t_samples=t,
            y_samples=y,
            t_eval=t_dense,
            m=m
        )

        reconstructions[m] = y_newton

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    save_dict = {
        "t_fragment": t_fragment,
        "y_fragment": y_fragment,
        "sample_numbers_fragment": sample_numbers_fragment,
        "t_dense": t_dense,
        "Fs": Fs,
        "Ts": Ts,
        "Fs_dense": Fs_dense,
        "Ts_dense": Ts_dense,
        "M_VALUES": np.array(M_VALUES, dtype=int)
    }

    for m, y_newton in reconstructions.items():
        save_dict[f"y_newton_m_{m}"] = y_newton

    np.savez(
        OUTPUT_FILE,
        **save_dict
    )

    # --------------------------------------------------------
    # Información por pantalla
    # --------------------------------------------------------

    print()
    print("Reconstrucciones de Newton calculadas correctamente.")
    print()
    print(f"Intervalo reconstruido: [{START_TIME}, {START_TIME + DURATION}] s")
    print(f"Frecuencia original: Fs = {Fs:.0f} Hz")
    print(f"Periodo original: Ts = {Ts:.8f} s")
    print()
    print(f"Factor de refinamiento: {UPSAMPLE_FACTOR}")
    print(f"Frecuencia densa: Fs_dense = {Fs_dense:.0f} Hz")
    print(f"Periodo denso: Ts_dense = {Ts_dense:.8f} s")
    print()
    print(f"Grados usados: {M_VALUES}")
    print(f"Archivo guardado: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Representar
    # --------------------------------------------------------

    plot_newton_reconstructions(
        t_fragment=t_fragment,
        y_fragment=y_fragment,
        sample_numbers_fragment=sample_numbers_fragment,
        t_dense=t_dense,
        reconstructions=reconstructions,
        annotations=annotations,
        Fs=Fs,
        start_time=START_TIME,
        duration=DURATION
    )