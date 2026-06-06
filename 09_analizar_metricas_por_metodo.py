import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PARAMETROS
# ============================================================

INPUT_METRICS_FILE = "metricas_todas_senales_con_shannon.csv"

SUMMARY_FILE = "resumen_metricas_por_metodo.csv"
PLOT_FILE = "grafica_errores_medios_por_metodo.png"
SHOW_PLOT = False

ERROR_COLUMNS = [
    "RMSE",
    "MAE",
    "MaxAbsError",
    "PRD_percent"
]


# ============================================================
# 2. CARGA Y VALIDACION
# ============================================================

def load_metrics(filename):
    """
    Carga el CSV generado por 08_comparar_todas_las_senales_por_shannon.py.
    Cada fila representa el resultado de un metodo para una senal
    independiente, identificada por Registro y Canal.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"No se ha encontrado '{filename}'. "
            "Primero ejecuta 08_comparar_todas_las_senales_por_shannon.py."
        )

    metrics_df = pd.read_csv(filename)

    required_columns = ["Registro", "Canal", "Metodo", "Referencia", *ERROR_COLUMNS]
    missing_columns = [
        column for column in required_columns
        if column not in metrics_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en '{filename}': {missing_columns}"
        )

    for column in ERROR_COLUMNS:
        metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")

    return metrics_df


# ============================================================
# 3. ANALISIS AGRUPADO POR METODO
# ============================================================

def summarize_by_method(metrics_df):
    """
    Agrupa los resultados por metodo y calcula la media de cada error.

    Los canales de un mismo registro se tratan como senales distintas,
    asi que cada fila del CSV de entrada cuenta como una observacion
    independiente para el metodo correspondiente.
    """

    summary_df = (
        metrics_df
        .groupby("Metodo", as_index=False)
        .agg(
            N_senales=("RMSE", "count"),
            RMSE_medio=("RMSE", "mean"),
            MAE_medio=("MAE", "mean"),
            MaxAbsError_medio=("MaxAbsError", "mean"),
            PRD_percent_medio=("PRD_percent", "mean")
        )
    )

    summary_df = summary_df.sort_values(
        by="RMSE_medio",
        ascending=True
    ).reset_index(drop=True)

    return summary_df


# ============================================================
# 4. GRAFICA DE BARRAS
# ============================================================

def plot_mean_errors(summary_df, output_file, show_plot=False):
    """
    Dibuja una grafica de rectangulos con los errores medios por metodo.
    """

    methods = summary_df["Metodo"].to_numpy()

    error_series = [
        ("RMSE medio", "RMSE_medio"),
        ("MAE medio", "MAE_medio"),
        ("MaxAbsError medio", "MaxAbsError_medio"),
        ("PRD medio (%)", "PRD_percent_medio")
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 9),
        constrained_layout=True
    )

    axes = axes.ravel()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for ax, (title, column), color in zip(axes, error_series, colors):
        values = summary_df[column].to_numpy(dtype=float)
        x = np.arange(len(methods))

        bars = ax.bar(
            x,
            values,
            color=color,
            edgecolor="black",
            linewidth=0.6
        )

        ax.set_title(title)
        ax.set_ylabel("Error medio")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    fig.suptitle(
        "Errores medios por metodo respecto a Whittaker-Shannon",
        fontsize=14
    )

    fig.savefig(output_file, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    metrics_df = load_metrics(INPUT_METRICS_FILE)
    summary_df = summarize_by_method(metrics_df)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("Resumen de errores medios por metodo:")
    print(summary_df)
    print()
    print(f"Archivo resumen guardado: {SUMMARY_FILE}")

    plot_mean_errors(
        summary_df=summary_df,
        output_file=PLOT_FILE,
        show_plot=SHOW_PLOT
    )

    print(f"Grafica guardada: {PLOT_FILE}")
