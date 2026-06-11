"""
visualizations.py
-----------------
Genera gráficos de barras con los resultados del simulador.
Guarda las imágenes en outputs/ para uso en el dashboard y el portfolio.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from pathlib import Path

from src.config import OUTPUTS_DIR

# \\\\\\\\\\\
# Estilo global de los gráficos
# \\\\\\\\\\\

STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.labelcolor":  "#E6EDF3",
    "text.color":       "#E6EDF3",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#E6EDF3",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "grid.color":       "#21262D",
    "grid.linestyle":   "--",
}


def _apply_style():
    """Aplica el estilo oscuro de portafolio a todos los gráficos."""
    plt.rcParams.update(STYLE)


# \\\\\\\\\\\
# Gráfico: probabilidades de campeón (Top N favoritos)
# \\\\\\\\\\\

def plot_champion_probabilities(team_probs: pd.DataFrame, top_n: int = 16) -> Path:
    """
    Crea gráfico de barras horizontal con los N equipos más probables
    campeones del torneo.

    Returns:
        Path del archivo PNG guardado.
    """
    _apply_style()
    data = team_probs.nlargest(top_n, "champion_probability").sort_values("champion_probability")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["team"], data["champion_probability"], color="#238636", edgecolor="none")

    # Etiquetas de valor en cada barra
    for bar, val in zip(bars, data["champion_probability"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#E6EDF3")

    ax.set_xlabel("Probabilidad (%)", labelpad=10)
    ax.set_title("Probabilidad de ser Campeón del Mundial 2026", pad=15, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, data["champion_probability"].max() * 1.25)

    fig.tight_layout()
    path = OUTPUTS_DIR / "champion_probabilities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico: probabilidades de semifinal (Top N)
# \\\\\\\\\\\

def plot_semifinal_probabilities(team_probs: pd.DataFrame, top_n: int = 16) -> Path:
    """
    Crea gráfico de barras horizontal con los N equipos más probables
    semifinalistas del torneo.
    """
    _apply_style()
    data = team_probs.nlargest(top_n, "semifinalist_probability").sort_values("semifinalist_probability")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["team"], data["semifinalist_probability"], color="#1F6FEB", edgecolor="none")

    for bar, val in zip(bars, data["semifinalist_probability"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#E6EDF3")

    ax.set_xlabel("Probabilidad (%)", labelpad=10)
    ax.set_title("Probabilidad de llegar a Semifinales - Mundial 2026", pad=15, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, data["semifinalist_probability"].max() * 1.25)

    fig.tight_layout()
    path = OUTPUTS_DIR / "semifinal_probabilities.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Gráfico: dificultad de grupos
# \\\\\\\\\\\

def plot_group_difficulty(group_summary: pd.DataFrame) -> Path:
    """
    Crea gráfico de barras con el rating medio de cada grupo
    para visualizar los grupos de la muerte.
    """
    _apply_style()
    data = group_summary.sort_values("average_rating", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#DA3633" if r > data["average_rating"].quantile(0.75) else "#388BFD"
              for r in data["average_rating"]]
    bars = ax.bar(data["group"].apply(lambda g: f"Grupo {g}"), data["average_rating"],
                  color=colors, edgecolor="none")

    for bar, val in zip(bars, data["average_rating"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#E6EDF3")

    ax.set_ylabel("Rating medio", labelpad=10)
    ax.set_title("Dificultad media por grupo - Mundial 2026\n(rojo = grupos más difíciles)",
                 pad=15, fontsize=12, fontweight="bold")
    ax.set_ylim(data["average_rating"].min() * 0.95, data["average_rating"].max() * 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUTS_DIR / "group_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# \\\\\\\\\\\
# Función principal: genera todos los gráficos en una sola llamada
# \\\\\\\\\\\

def generate_all_charts(team_probs: pd.DataFrame, group_summary: pd.DataFrame) -> list[Path]:
    """Genera todos los gráficos disponibles y devuelve sus rutas."""
    paths = [
        plot_champion_probabilities(team_probs),
        plot_semifinal_probabilities(team_probs),
        plot_group_difficulty(group_summary),
    ]
    return paths
