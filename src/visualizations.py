"""
visualizations.py
-----------------
Genera todos los gráficos del simulador con estilo oscuro de portfolio.
Nuevos gráficos: varianza, consistencia, puntos de terceros, camino al título.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import OUTPUTS_DIR

DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
TEXT_COL = "#E6EDF3"
MUTED    = "#8B949E"
GRID_COL = "#21262D"

def _fig(w=10, h=6):
    plt.rcParams.update({"figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
                          "text.color": TEXT_COL, "axes.labelcolor": TEXT_COL,
                          "xtick.color": MUTED, "ytick.color": TEXT_COL,
                          "axes.spines.top": False, "axes.spines.right": False,
                          "axes.spines.left": False, "axes.spines.bottom": False,
                          "grid.color": GRID_COL, "grid.linestyle": "--"})
    return plt.subplots(figsize=(w, h))


# \\\\\\\\\\\
# Gráfico 1: Probabilidades de campeón
# \\\\\\\\\\\

def plot_champion_probabilities(team_probs: pd.DataFrame, top_n=16) -> Path:
    data = team_probs.nlargest(top_n, "champion_pct").sort_values("champion_pct")
    fig, ax = _fig(10, 7)
    bars = ax.barh(data["team"], data["champion_pct"], color="#238636", edgecolor="none")
    for bar, val in zip(bars, data["champion_pct"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color=TEXT_COL)
    ax.set_xlabel("Probabilidad (%)", labelpad=10)
    ax.set_title("Probabilidad de ser Campeón del Mundial 2026",
                 pad=15, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, data["champion_pct"].max() * 1.25)
    fig.tight_layout()
    path = OUTPUTS_DIR / "champion_probabilities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico 2: Probabilidades de semifinal
# \\\\\\\\\\\

def plot_semifinal_probabilities(team_probs: pd.DataFrame, top_n=16) -> Path:
    data = team_probs.nlargest(top_n, "reach_semis_pct").sort_values("reach_semis_pct")
    fig, ax = _fig(10, 7)
    bars = ax.barh(data["team"], data["reach_semis_pct"], color="#1F6FEB", edgecolor="none")
    for bar, val in zip(bars, data["reach_semis_pct"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color=TEXT_COL)
    ax.set_xlabel("Probabilidad (%)", labelpad=10)
    ax.set_title("Probabilidad de llegar a Semifinales — Mundial 2026",
                 pad=15, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, data["reach_semis_pct"].max() * 1.25)
    fig.tight_layout()
    path = OUTPUTS_DIR / "semifinal_probabilities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico 3: Dificultad de grupos
# \\\\\\\\\\\

def plot_group_difficulty(group_summary: pd.DataFrame) -> Path:
    data   = group_summary.sort_values("average_rating", ascending=False)
    q75    = data["average_rating"].quantile(0.75)
    colors = ["#DA3633" if r >= q75 else "#388BFD" for r in data["average_rating"]]
    fig, ax = _fig(10, 5)
    bars = ax.bar(data["group"].apply(lambda g: f"Grupo {g}"),
                  data["average_rating"], color=colors, edgecolor="none")
    for bar, val in zip(bars, data["average_rating"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT_COL)
    ax.set_ylabel("Rating medio", labelpad=10)
    ax.set_title("Dificultad media por grupo — Mundial 2026\n(rojo = grupos más difíciles)",
                 pad=15, fontsize=12, fontweight="bold")
    ax.set_ylim(data["average_rating"].min() * 0.95, data["average_rating"].max() * 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUTPUTS_DIR / "group_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico 4 (NUEVO): Mapa de probabilidades por fase — heatmap
# \\\\\\\\\\\

def plot_phase_heatmap(team_probs: pd.DataFrame, top_n=20) -> Path:
    """
    Heatmap de probabilidades por fase para los N equipos más fuertes.
    Permite comparar visualmente la 'forma' de cada equipo en el torneo.
    """
    cols  = ["pass_group_stage_pct", "reach_round16_pct", "reach_quarters_pct",
             "reach_semis_pct", "reach_final_pct", "champion_pct"]
    labels = ["Pasa grupos", "Octavos", "Cuartos", "Semis", "Final", "Campeón"]

    data = team_probs.nlargest(top_n, "champion_pct").set_index("team")[cols]

    fig, ax = _fig(11, 8)
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=data.values.max())

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)

    for i in range(len(data)):
        for j in range(len(cols)):
            val = data.values[i, j]
            color = "black" if val > data.values.max() * 0.6 else TEXT_COL
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Probabilidades por fase — Top 20 equipos",
                 pad=15, fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Probabilidad (%)", shrink=0.8)
    fig.tight_layout()
    path = OUTPUTS_DIR / "phase_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico 5 (NUEVO): Varianza vs probabilidad de campeón (scatter)
# \\\\\\\\\\\

def plot_variance_scatter(variance_df: pd.DataFrame) -> Path:
    """
    Scatter: eje X = probabilidad de campeón, eje Y = % eliminado en grupos.
    Cuadrante superior derecho = alto potencial + alta inconsistencia (impredecibles).
    Cuadrante inferior derecho = favoritos consistentes.
    """
    fig, ax = _fig(10, 7)

    sc = ax.scatter(
        variance_df["champion_pct"],
        variance_df["group_exit_pct"],
        s=70, alpha=0.75,
        c=variance_df["consistency_index"],
        cmap="RdYlGn",
    )
    plt.colorbar(sc, ax=ax, label="Índice de consistencia", shrink=0.8)

    # Etiquetar top 12 equipos
    top12 = variance_df.nlargest(12, "champion_pct")
    for _, row in top12.iterrows():
        ax.annotate(row["team"],
                    (row["champion_pct"], row["group_exit_pct"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=8, color=TEXT_COL)

    # Líneas de referencia
    ax.axhline(variance_df["group_exit_pct"].median(), color=MUTED,
               linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(variance_df["champion_pct"].median(), color=MUTED,
               linestyle="--", alpha=0.4, linewidth=0.8)

    ax.set_xlabel("% de ser Campeón", labelpad=10)
    ax.set_ylabel("% de caer en Fase de Grupos", labelpad=10)
    ax.set_title("Varianza de equipos — Potencial vs Inconsistencia",
                 pad=15, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = OUTPUTS_DIR / "variance_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico 6 (NUEVO): Distribución de puntos de los mejores terceros
# \\\\\\\\\\\

def plot_third_place_points(third_stats: pd.DataFrame) -> Path:
    """
    Histograma de puntos con los que los terceros se clasifican o son eliminados.
    Responde directamente: ¿con cuántos puntos pasas siendo tercero?
    """
    data = third_stats[~third_stats["categoria"].str.startswith("RESUMEN")].copy()
    if data.empty:
        return None

    fig, ax = _fig(9, 5)

    for categoria, color in [("Clasificado (top 8)", "#238636"), ("Eliminado", "#DA3633")]:
        sub = data[data["categoria"] == categoria]
        if sub.empty:
            continue
        ax.bar(sub["puntos"].astype(str), sub["frecuencia_pct"],
               label=categoria, color=color, alpha=0.85, edgecolor="none")

    ax.set_xlabel("Puntos al final de la fase de grupos", labelpad=10)
    ax.set_ylabel("Frecuencia (%)", labelpad=10)
    ax.set_title("Puntos de los terceros clasificados vs eliminados\n(¿Con cuántos puntos pasas siendo tercero?)",
                 pad=15, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUTPUTS_DIR / "third_place_points.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Función principal: genera todos los gráficos
# \\\\\\\\\\\

def generate_all_charts(
    team_probs: pd.DataFrame,
    group_summary: pd.DataFrame,
    third_stats: pd.DataFrame = None,
    variance_df: pd.DataFrame = None,
) -> list[Path]:
    paths = [
        plot_champion_probabilities(team_probs),
        plot_semifinal_probabilities(team_probs),
        plot_group_difficulty(group_summary),
        plot_phase_heatmap(team_probs),
        plot_variance_scatter(variance_df if variance_df is not None else team_probs),
    ]
    if third_stats is not None and not third_stats.empty:
        p = plot_third_place_points(third_stats)
        if p:
            paths.append(p)
    return [p for p in paths if p]
