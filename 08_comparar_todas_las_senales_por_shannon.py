import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1. PARAMETROS
# ============================================================

ARCHIVE_DIR = "archive"

# Si es None, procesa todos los CSV de ARCHIVE_DIR.
# Ejemplo para limitar: RECORD_IDS = ["100", "101"]
RECORD_IDS = None

# Si es None, procesa todos los canales de cada CSV salvo "sample #".
# Ejemplo para limitar: CHANNELS = ["MLII"]
CHANNELS = None

Fs = 360
Ts = 1 / Fs

LOWCUT = 0.5
HIGHCUT = 40
FILTER_ORDER = 4

START_TIME = 0
DURATION = 8
DOWNSAMPLE_FACTOR = 4

NEWTON_M_VALUES = [1, 3, 5, 7]
SPLINE_BC_TYPE = "natural"

METRICS_FILE = "metricas_todas_senales_con_shannon.csv"
OUTPUT_FILE = "comparacion_todas_senales_con_shannon.npz"


# ============================================================
# 2. CARGAR FUNCIONES DE RECONSTRUCCION DEL PROGRAMA 07
# ============================================================

def load_shannon_module():
    """
    Carga 07_comparar_metodos_por_shannon.py como modulo auxiliar.

    El nombre del archivo empieza por un numero, por eso usamos
    importlib en lugar de un import normal.
    """

    module_path = Path(__file__).with_name("07_comparar_metodos_por_shannon.py")
    spec = importlib.util.spec_from_file_location("comparar_por_shannon", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shannon = load_shannon_module()


# ============================================================
# 3. CARGA Y FILTRADO
# ============================================================

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )
    return df


def load_ecg_csv(filename):
    """
    Carga un CSV tipo MIT-BIH y devuelve el DataFrame con columnas limpias.
    """

    df = pd.read_csv(filename)
    df = clean_columns(df)

    if "sample #" not in df.columns:
        raise ValueError(
            f"No se ha encontrado la columna 'sample #' en {filename}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    return df


def bandpass_filter(x, fs, lowcut, highcut, order):
    """
    Aplica un filtro digital pasa-banda Butterworth sin desfase.
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

    return sosfiltfilt(sos, x)


# ============================================================
# 4. PROCESAMIENTO DE UNA SENAL
# ============================================================

def compute_signal_comparison(sample_numbers, t, y_filtered):
    """
    Ejecuta la comparacion de metodos para una unica senal filtrada.
    """

    t_ref, y_ref, sample_numbers_ref = shannon.extract_reference_fragment(
        t=t,
        y=y_filtered,
        sample_numbers=sample_numbers,
        start_time=START_TIME,
        duration=DURATION
    )

    t_low, y_low, sample_numbers_low = shannon.downsample_signal(
        t_ref=t_ref,
        y_ref=y_ref,
        sample_numbers_ref=sample_numbers_ref,
        factor=DOWNSAMPLE_FACTOR
    )

    t_ref, y_ref, sample_numbers_ref = shannon.crop_to_low_interval(
        t_ref=t_ref,
        y_ref=y_ref,
        sample_numbers_ref=sample_numbers_ref,
        t_low=t_low
    )

    y_shannon = shannon.sinc_interpolation(
        t_samples=t_low,
        y_samples=y_low,
        t_eval=t_ref
    )

    reconstructions = {
        "Whittaker-Shannon": y_shannon,
        "Orden cero": shannon.zero_order_hold(
            t_samples=t_low,
            y_samples=y_low,
            t_eval=t_ref
        ),
        "Lineal": shannon.linear_interpolation(
            t_samples=t_low,
            y_samples=y_low,
            t_eval=t_ref
        ),
        "Spline cubico": shannon.cubic_spline_reconstruction(
            t_samples=t_low,
            y_samples=y_low,
            t_eval=t_ref,
            bc_type=SPLINE_BC_TYPE
        )
    }

    for m in NEWTON_M_VALUES:
        reconstructions[f"Newton m={m}"] = shannon.newton_local_reconstruction(
            t_samples=t_low,
            y_samples=y_low,
            t_eval=t_ref,
            m=m
        )

    metrics_rows = []

    for method_name, y_rec in reconstructions.items():
        if method_name == "Whittaker-Shannon":
            continue

        metrics = shannon.compute_metrics(
            y_reference=y_shannon,
            y_rec=y_rec
        )

        metrics_rows.append({
            "Metodo": method_name,
            "Referencia": "Whittaker-Shannon",
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MaxAbsError": metrics["MaxAbsError"],
            "PRD_percent": metrics["PRD_percent"]
        })

    return {
        "t_ref": t_ref,
        "y_highrate": y_ref,
        "t_low": t_low,
        "y_low": y_low,
        "sample_numbers_ref": sample_numbers_ref,
        "sample_numbers_low": sample_numbers_low,
        "y_shannon": y_shannon,
        "reconstructions": reconstructions,
        "metrics_rows": metrics_rows
    }


# ============================================================
# 5. UTILIDADES
# ============================================================

def discover_csv_files(archive_dir, record_ids):
    archive_path = Path(archive_dir)

    if record_ids is None:
        return sorted(archive_path.glob("*.csv"))

    return [archive_path / f"{record_id}.csv" for record_id in record_ids]


def available_channels(df, requested_channels):
    signal_channels = [column for column in df.columns if column != "sample #"]

    if requested_channels is None:
        return signal_channels

    missing = [channel for channel in requested_channels if channel not in signal_channels]
    if missing:
        print(f"Canales no encontrados y omitidos: {missing}")

    return [channel for channel in requested_channels if channel in signal_channels]


def safe_key(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return text.strip("_")


def add_signal_to_save_dict(save_dict, record_id, channel, result):
    prefix = f"record_{safe_key(record_id)}__channel_{safe_key(channel)}"

    save_dict[f"{prefix}__t_ref"] = result["t_ref"]
    save_dict[f"{prefix}__y_highrate"] = result["y_highrate"]
    save_dict[f"{prefix}__t_low"] = result["t_low"]
    save_dict[f"{prefix}__y_low"] = result["y_low"]
    save_dict[f"{prefix}__sample_numbers_ref"] = result["sample_numbers_ref"]
    save_dict[f"{prefix}__sample_numbers_low"] = result["sample_numbers_low"]
    save_dict[f"{prefix}__y_shannon"] = result["y_shannon"]

    for method_name, y_rec in result["reconstructions"].items():
        method_key = safe_key(method_name)
        save_dict[f"{prefix}__{method_key}"] = y_rec


# ============================================================
# 6. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    csv_files = discover_csv_files(
        archive_dir=ARCHIVE_DIR,
        record_ids=RECORD_IDS
    )

    if not csv_files:
        raise FileNotFoundError(f"No se han encontrado CSV en '{ARCHIVE_DIR}'.")

    all_metrics_rows = []
    save_dict = {
        "Fs": Fs,
        "Ts": Ts,
        "Fs_low": Fs / DOWNSAMPLE_FACTOR,
        "Ts_low": 1 / (Fs / DOWNSAMPLE_FACTOR),
        "DOWNSAMPLE_FACTOR": DOWNSAMPLE_FACTOR,
        "START_TIME": START_TIME,
        "DURATION": DURATION,
        "LOWCUT": LOWCUT,
        "HIGHCUT": HIGHCUT,
        "FILTER_ORDER": FILTER_ORDER
    }

    processed_signals = 0
    failed_signals = []

    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Archivo no encontrado y omitido: {csv_file}")
            continue

        record_id = csv_file.stem
        print(f"\n=== Registro {record_id} ===")

        df = load_ecg_csv(csv_file)
        sample_numbers = df["sample #"].to_numpy(dtype=int)
        t = sample_numbers / Fs
        channels = available_channels(df, CHANNELS)

        if not channels:
            print(f"No hay canales procesables en {csv_file}.")
            continue

        for channel in channels:
            print(f"Procesando canal {channel}...")

            try:
                x = df[channel].to_numpy(dtype=float)
                y_filtered = bandpass_filter(
                    x=x,
                    fs=Fs,
                    lowcut=LOWCUT,
                    highcut=HIGHCUT,
                    order=FILTER_ORDER
                )

                result = compute_signal_comparison(
                    sample_numbers=sample_numbers,
                    t=t,
                    y_filtered=y_filtered
                )

                for row in result["metrics_rows"]:
                    all_metrics_rows.append({
                        "Registro": record_id,
                        "Canal": channel,
                        **row
                    })

                add_signal_to_save_dict(
                    save_dict=save_dict,
                    record_id=record_id,
                    channel=channel,
                    result=result
                )

                processed_signals += 1

            except Exception as exc:
                failed_signals.append({
                    "Registro": record_id,
                    "Canal": channel,
                    "Error": str(exc)
                })
                print(f"Error en {record_id} / {channel}: {exc}")

    metrics_df = pd.DataFrame(all_metrics_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            by=["Registro", "Canal", "RMSE"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    metrics_df.to_csv(METRICS_FILE, index=False)

    if failed_signals:
        failed_df = pd.DataFrame(failed_signals)
        failed_df.to_csv("errores_todas_senales_con_shannon.csv", index=False)

    np.savez(OUTPUT_FILE, **save_dict)

    print("\nComparacion para todas las senales terminada.")
    print(f"Senales procesadas correctamente: {processed_signals}")
    print(f"Senales con error: {len(failed_signals)}")
    print(f"Archivo de metricas: {METRICS_FILE}")
    print(f"Archivo de reconstrucciones: {OUTPUT_FILE}")

    if failed_signals:
        print("Archivo de errores: errores_todas_senales_con_shannon.csv")
