"""
analysis.py
-----------
Transforma los contadores crudos de Monte Carlo en probabilidades,
tablas de finales, resúmenes de grupos y conclusiones en texto.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Any


# \\\\\\\\\\\
# Conversión de contadores a tabla de probabilidades por equipo
# \\\\\\\\\\\

def generate_probabilities(
    counters: dict,
    n_simulations: int,
    tournament_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la probabilidad de llegar a cada fase del torneo.
    Incluye metadatos del equipo (grupo, confederación, ratings).

    Returns:
        DataFrame ordenado por probabilidad de campeón descendente.
    """
    def pct(counter: defaultdict, team: str) -> float:
        """Convierte conteos en porcentaje redondeado."""
        return round(counter[team] / n_simulations * 100, 2)

    rows = []
    for _, row in tournament_df.iterrows():
        team = row["team"]
        rows.append({
            "team":                    team,
            "group":                   row["group"],
            "confederation":           row["confederation"],
            "overall_rating":          row["overall_rating"],
            "champion_probability":    pct(counters["champion"],        team),
            "finalist_probability":    pct(counters["finalist"],        team),
            "semifinalist_probability":pct(counters["semifinalist"],    team),
            "quarterfinalist_prob":    pct(counters["quarterfinalist"], team),
            "round_of_16_probability": pct(counters["round_of_16"],     team),
            "round_of_32_probability": pct(counters["round_of_32"],     team),
            "group_exit_probability":  pct(counters["group_exit"],      team),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("champion_probability", ascending=False).reset_index(drop=True)
    return df


# \\\\\\\\\\\
# Tabla de finales más repetidas
# \\\\\\\\\\\

def generate_finals_table(counters: dict, n_simulations: int) -> pd.DataFrame:
    """
    Devuelve las 20 finales más frecuentes con su probabilidad estimada.
    """
    pairs = counters["finals_pairs"]
    rows  = [
        {"final": pair, "count": count, "probability_pct": round(count / n_simulations * 100, 2)}
        for pair, count in pairs.items()
    ]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).head(20)
    return df.reset_index(drop=True)


# \\\\\\\\\\\
# Resumen de dificultad de grupos
# \\\\\\\\\\\

def generate_group_summary(tournament_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la dificultad media de cada grupo basándose en los ratings.
    Útil para identificar los grupos de la muerte.
    """
    rows = []
    for group_name, group_df in tournament_df.groupby("group"):
        rows.append({
            "group":          group_name,
            "teams":          " / ".join(group_df["team"].tolist()),
            "average_rating": round(group_df["overall_rating"].mean(), 1),
            "max_rating":     group_df["overall_rating"].max(),
            "min_rating":     group_df["overall_rating"].min(),
            "rating_range":   group_df["overall_rating"].max() - group_df["overall_rating"].min(),
        })

    df = pd.DataFrame(rows).sort_values("average_rating", ascending=False)
    return df.reset_index(drop=True)


# \\\\\\\\\\\
# Generación de conclusiones automáticas en texto
# \\\\\\\\\\\

def generate_conclusions(
    team_probs: pd.DataFrame,
    finals_df: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    """
    Genera un texto con conclusiones automáticas del simulador.
    Basado exclusivamente en los datos generados por el modelo.
    """
    lines = ["=" * 60, "CONCLUSIONES DEL SIMULADOR MONTE CARLO - MUNDIAL 2026", "=" * 60, ""]

    # Favorito al título
    top = team_probs.iloc[0]
    lines.append(f"1. FAVORITO AL TÍTULO: {top['team']} ({top['champion_probability']}% de probabilidad).")

    # Equipo más consistente llegando a la final
    top_finalist = team_probs.sort_values("finalist_probability", ascending=False).iloc[0]
    lines.append(f"2. MÁS VECES EN LA FINAL: {top_finalist['team']} ({top_finalist['finalist_probability']}%).")

    # Mejor semifinalista que no es el favorito
    non_top = team_probs[team_probs["team"] != top["team"]]
    semi_surprise = non_top.sort_values("semifinalist_probability", ascending=False).iloc[0]
    lines.append(f"3. SEMIFINALISTA MÁS CONSISTENTE (no favorito): {semi_surprise['team']} ({semi_surprise['semifinalist_probability']}%).")

    # Selección con más varianza (alta probabilidad de campeón pero también de caer en grupos)
    team_probs["variance_score"] = team_probs["champion_probability"] * team_probs["group_exit_probability"]
    risky = team_probs[team_probs["variance_score"] > 0].sort_values("variance_score", ascending=False)
    if not risky.empty:
        rv = risky.iloc[0]
        lines.append(f"4. MAYOR VARIANZA: {rv['team']} (puede ganar pero también caer en grupos).")

    # Grupo más difícil
    hardest = group_summary.iloc[0]
    lines.append(f"5. GRUPO MÁS DIFÍCIL: Grupo {hardest['group']} (rating medio {hardest['average_rating']}).")
    lines.append(f"   Equipos: {hardest['teams']}.")

    # Grupo más accesible
    easiest = group_summary.iloc[-1]
    lines.append(f"6. GRUPO MÁS ACCESIBLE: Grupo {easiest['group']} (rating medio {easiest['average_rating']}).")

    # Final más probable
    if not finals_df.empty:
        top_final = finals_df.iloc[0]
        lines.append(f"7. FINAL MÁS REPETIDA: {top_final['final']} ({top_final['probability_pct']}% de las simulaciones).")

    # Posibles sorpresas (top 10 en cuartos pero rating < 80)
    lines.append("")
    lines.append("8. POSIBLES SORPRESAS (rating < 80, pero con opciones de cuartos):")
    surprises = team_probs[
        (team_probs["overall_rating"] < 80) &
        (team_probs["quarterfinalist_prob"] > 5)
    ].head(3)
    if surprises.empty:
        lines.append("   Ninguna selección de bajo rating llega frecuentemente a cuartos.")
    else:
        for _, s in surprises.iterrows():
            lines.append(f"   - {s['team']}: {s['quarterfinalist_prob']}% de llegar a cuartos (rating {s['overall_rating']}).")

    # Advertencia metodológica
    lines.extend([
        "",
        "-" * 60,
        "NOTA: Estos resultados son probabilísticos, no predicciones.",
        "Están condicionados a los datos y supuestos del modelo.",
        "Un cambio en los ratings modifica significativamente los resultados.",
        "-" * 60,
    ])

    return "\n".join(lines)
