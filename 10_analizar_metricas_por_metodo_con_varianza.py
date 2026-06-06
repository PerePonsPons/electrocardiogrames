import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PARAMETROS
# ============================================================

INPUT_METRICS_FILE = "metricas_todas_senales_con_shannon.csv"

SUMMARY_FILE = "resumen_metricas_por_metodo_con_varianza.csv"
PLOT_FILE = "grafica_media_varianza_errores_por_metodo.png"
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

    Cada fila representa un metodo aplicado a una senal independiente.
    Los canales de un mismo registro se consideran senales distintas.
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
    Calcula media y varianza de cada tipo de error agrupando por metodo.

    La varianza se calcula como varianza muestral, con ddof=1, que es el
    comportamiento por defecto de pandas.DataFrame.var().
    """

    summary_df = (
        metrics_df
        .groupby("Metodo")
        .agg(
            N_senales=("RMSE", "count"),
            RMSE_medio=("RMSE", "mean"),
            RMSE_varianza=("RMSE", "var"),
            MAE_medio=("MAE", "mean"),
            MAE_varianza=("MAE", "var"),
            MaxAbsError_medio=("MaxAbsError", "mean"),
            MaxAbsError_varianza=("MaxAbsError", "var"),
            PRD_percent_medio=("PRD_percent", "mean"),
            PRD_percent_varianza=("PRD_percent", "var")
        )
        .reset_index()
    )

    summary_df = summary_df.sort_values(
        by="RMSE_medio",
        ascending=True
    ).reset_index(drop=True)

    return summary_df


# ============================================================
# 4. GRAFICA DE MEDIA Y VARIANZA
# ============================================================

def plot_mean_and_variance(summary_df, output_file, show_plot=False):
    """
    Dibuja barras con media y varianza para cada metodo y tipo de error.
    """

    methods = summary_df["Metodo"].to_numpy()
    x = np.arange(len(methods))

    error_series = [
        ("RMSE", "RMSE_medio", "RMSE_varianza"),
        ("MAE", "MAE_medio", "MAE_varianza"),
        ("MaxAbsError", "MaxAbsError_medio", "MaxAbsError_varianza"),
        ("PRD (%)", "PRD_percent_medio", "PRD_percent_varianza")
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(16, 16),
        constrained_layout=True
    )

    mean_color = "#4C78A8"
    variance_color = "#F58518"

    for row_index, (metric_name, mean_column, var_column) in enumerate(error_series):
        ax_mean = axes[row_index, 0]
        ax_var = axes[row_index, 1]

        mean_values = summary_df[mean_column].to_numpy(dtype=float)
        var_values = summary_df[var_column].to_numpy(dtype=float)

        mean_bars = ax_mean.bar(
            x,
            mean_values,
            color=mean_color,
            edgecolor="black",
            linewidth=0.6
        )
        var_bars = ax_var.bar(
            x,
            var_values,
            color=variance_color,
            edgecolor="black",
            linewidth=0.6
        )

        ax_mean.set_title(f"{metric_name}: media")
        ax_var.set_title(f"{metric_name}: varianza")

        for ax in (ax_mean, ax_var):
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(True, axis="y", alpha=0.3)

        add_value_labels(ax_mean, mean_bars, mean_values)
        add_value_labels(ax_var, var_bars, var_values)

    fig.suptitle(
        "Media y varianza de los errores por metodo respecto a Whittaker-Shannon",
        fontsize=14
    )

    fig.savefig(output_file, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def add_value_labels(ax, bars, values):
    """
    Anade etiquetas numericas encima de las barras.
    """

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7
        )


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    metrics_df = load_metrics(INPUT_METRICS_FILE)
    summary_df = summarize_by_method(metrics_df)

    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("Resumen de errores por metodo con media y varianza:")
    print(summary_df)
    print()
    print(f"Archivo resumen guardado: {SUMMARY_FILE}")

    plot_mean_and_variance(
        summary_df=summary_df,
        output_file=PLOT_FILE,
        show_plot=SHOW_PLOT
    )

    print(f"Grafica guardada: {PLOT_FILE}")
