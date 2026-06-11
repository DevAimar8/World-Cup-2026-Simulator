"""
knockout_stage.py
-----------------
Eliminatorias del Mundial 2026 (32→16→8→4→2→1).
Ahora también registra el camino del campeón (rivales por ronda)
para el análisis de 'path to title'.
"""

import pandas as pd
from src.match_simulator import simulate_knockout_match


def _get_row(name: str, df: pd.DataFrame) -> pd.Series:
    row = df[df["team"] == name]
    if row.empty:
        return pd.Series({"team": name, "overall_rating": 70,
                          "attack_rating": 70, "defense_rating": 70,
                          "attack_coef": 1.0, "defense_coef": 1.0,
                          "form_factor": 1.0, "is_host": 0, "confederation": "UEFA"})
    return row.iloc[0]


def _sim_round(teams: list, df: pd.DataFrame) -> tuple[list, list]:
    winners, losers = [], []
    for i in range(0, len(teams) - 1, 2):
        a = _get_row(teams[i],     df)
        b = _get_row(teams[i + 1], df)
        w = simulate_knockout_match(a, b)
        winners.append(w)
        losers.append(teams[i] if w == teams[i + 1] else teams[i + 1])
    return winners, losers


# \\\\\\\\\\\
# Construcción del bracket de 32 equipos
# \\\\\\\\\\\

def build_bracket(qualified: dict) -> list:
    firsts     = qualified["first"]
    seconds    = qualified["second"]
    third_best = qualified["third_best"]
    bracket    = []
    half       = len(firsts) // 2
    for i in range(half):
        bracket.append(firsts[i])
        bracket.append(seconds[-(i + 1)])
    for i in range(half, len(firsts)):
        bracket.append(firsts[i])
        bracket.append(seconds[-(i + 1)])
    bracket.extend(third_best)
    return bracket[:32]


# \\\\\\\\\\\
# Simulación completa de eliminatorias con seguimiento del camino
# \\\\\\\\\\\

def simulate_knockout_stage(qualified: dict, df: pd.DataFrame) -> dict:
    """
    Simula todas las rondas y registra el camino del campeón.
    Devuelve:
      champion, finalist, semifinalists, quarterfinalists,
      round_of_16, round_of_32, champion_path (dict ronda→rival)
    """
    bracket_32 = build_bracket(qualified)

    r16,  _  = _sim_round(bracket_32, df)   # Dieciseisavos
    qf,   _  = _sim_round(r16,        df)   # Octavos
    sf,   _  = _sim_round(qf,         df)   # Cuartos
    fin, sf_losers = _sim_round(sf,    df)   # Semifinales
    result, _ = _sim_round(fin,        df)   # Final

    champion  = result[0]
    runner_up = fin[1] if champion == fin[0] else fin[0]

    # \\\\\\\\\\\
    # Reconstruir el camino del campeón ronda a ronda
    # \\\\\\\\\\\

    def _find_opponent(team: str, bracket: list) -> str:
        """Encuentra el rival emparejado del equipo en un bracket dado."""
        for i in range(0, len(bracket) - 1, 2):
            if bracket[i] == team:
                return bracket[i + 1]
            if bracket[i + 1] == team:
                return bracket[i]
        return "unknown"

    champion_path = {
        "r32": _find_opponent(champion, bracket_32),
        "r16": _find_opponent(champion, r16),
        "qf":  _find_opponent(champion, qf),
        "sf":  _find_opponent(champion, sf),
    }

    return {
        "champion":         champion,
        "finalist":         runner_up,
        "semifinalists":    sf,
        "quarterfinalists": qf,
        "round_of_16":      r16,
        "round_of_32":      bracket_32,
        "champion_path":    champion_path,
    }
