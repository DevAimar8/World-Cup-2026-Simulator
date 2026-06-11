"""
knockout_stage.py
-----------------
Simula las fases eliminatorias del Mundial 2026:
    Dieciseisavos (32 → 16)
    Octavos      (16 →  8)
    Cuartos      ( 8 →  4)
    Semifinales  ( 4 →  2)
    Final        ( 2 →  1)

El bracket se construye por seeds (posición en grupo) para simplificar.
En una versión futura se puede adaptar al bracket oficial del Mundial.
"""

import pandas as pd
import numpy as np
from typing import Any

from src.match_simulator import simulate_knockout_match

# \\\\\\\\\\\
# Construcción del bracket inicial con 32 equipos
# \\\\\\\\\\\

def build_bracket(
    qualified: dict[str, list[str]],
    tournament_df: pd.DataFrame,
) -> list[str]:
    """
    Crea el bracket de 32 equipos ordenados por seed.
    Orden: [1°G1, 2°G2, 1°G3, 2°G4, ..., mejores_terceros...]

    Devuelve lista de 32 nombres en el orden del bracket.
    """
    firsts      = qualified["first"]       # 12 primeros
    seconds     = qualified["second"]      # 12 segundos
    third_best  = qualified["third_best"]  # 8 mejores terceros

    # Intercalar primeros y segundos de grupos opuestos para evitar rematches
    bracket = []
    half    = len(firsts) // 2

    for i in range(half):
        bracket.append(firsts[i])
        bracket.append(seconds[-(i + 1)])

    for i in range(half, len(firsts)):
        bracket.append(firsts[i])
        bracket.append(seconds[-(i + 1)])

    # Añadir los 8 mejores terceros (rellenar hasta 32)
    bracket.extend(third_best)

    # Garantizar exactamente 32 equipos
    bracket = bracket[:32]
    return bracket


# \\\\\\\\\\\
# Obtención de datos de un equipo por nombre desde el DataFrame
# \\\\\\\\\\\

def _get_team_row(team_name: str, tournament_df: pd.DataFrame) -> pd.Series:
    """Devuelve la fila del DataFrame de torneo para un equipo dado."""
    row = tournament_df[tournament_df["team"] == team_name]
    if row.empty:
        # Fallback: fila mínima para no romper el simulador
        return pd.Series({
            "team": team_name, "overall_rating": 70,
            "attack_rating": 70, "defense_rating": 70,
            "form_factor": 1.0, "is_host": 0,
        })
    return row.iloc[0]


# \\\\\\\\\\\
# Simulación de una ronda eliminatoria completa
# \\\\\\\\\\\

def simulate_round(
    teams: list[str],
    tournament_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Simula una ronda del torneo enfrentando equipos de dos en dos.
    Devuelve (ganadores, eliminados).
    """
    winners   = []
    eliminated = []

    for i in range(0, len(teams), 2):
        if i + 1 >= len(teams):
            # Número impar: pasa directamente (no debería ocurrir en formato correcto)
            winners.append(teams[i])
            continue

        team_a = _get_team_row(teams[i],     tournament_df)
        team_b = _get_team_row(teams[i + 1], tournament_df)
        winner = simulate_knockout_match(team_a, team_b)

        winners.append(winner)
        loser = teams[i] if winner == teams[i + 1] else teams[i + 1]
        eliminated.append(loser)

    return winners, eliminated


# \\\\\\\\\\\
# Simulación completa de la fase eliminatoria (dieciseisavos → final)
# \\\\\\\\\\\

def simulate_knockout_stage(
    qualified: dict[str, list[str]],
    tournament_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Ejecuta todas las rondas eliminatorias del Mundial 2026.

    Devuelve un diccionario con:
        champion          → str
        finalist          → str (subcampeón)
        semifinalists     → list[str] (4 semifinalistas)
        quarterfinalists  → list[str] (8 cuartofinalistas)
        round_of_16       → list[str] (16 equipos en octavos)
        round_of_32       → list[str] (32 equipos en dieciseisavos)
    """
    # Bracket de dieciseisavos
    bracket_32 = build_bracket(qualified, tournament_df)

    # Dieciseisavos (32 → 16)
    round_of_16, _  = simulate_round(bracket_32, tournament_df)

    # Octavos (16 → 8)
    quarter_finals, _ = simulate_round(round_of_16, tournament_df)

    # Cuartos (8 → 4)
    semi_finals, _  = simulate_round(quarter_finals, tournament_df)

    # Semifinales (4 → 2)
    finalists, semi_losers = simulate_round(semi_finals, tournament_df)

    # Final (2 → 1)
    final_teams = finalists[:]
    final_winner, _ = simulate_round(final_teams, tournament_df)
    champion = final_winner[0]
    runner_up = finalists[1] if champion == finalists[0] else finalists[0]

    return {
        "champion":        champion,
        "finalist":        runner_up,
        "semifinalists":   semi_finals,          # 4 equipos
        "quarterfinalists": quarter_finals,       # 8 equipos
        "round_of_16":     round_of_16,          # 16 equipos
        "round_of_32":     bracket_32,           # 32 equipos
    }
