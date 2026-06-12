"""
group_stage.py
--------------
Fase de grupos: todos contra todos en 12 grupos de 4 equipos.
Ddevuelve los puntos de los mejores terceros para
calcular la media de puntos con la que se clasifican.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from src.match_simulator import simulate_group_match


def _df_to_team_lookup(teams: pd.DataFrame) -> dict:
    return {row["team"]: pd.Series(row) for _, row in teams.iterrows()}


def _create_table(team_names: list, group_name: str) -> dict:
    return {
        name: {"team": name, "group": group_name,
               "played": 0, "won": 0, "drawn": 0, "lost": 0,
               "goals_for": 0, "goals_against": 0, "goal_diff": 0, "points": 0}
        for name in team_names
    }


def _update(table: dict, a: str, b: str, ga: int, gb: int) -> None:
    for name, gf, gc in [(a, ga, gb), (b, gb, ga)]:
        r = table[name]
        r["played"] += 1; r["goals_for"] += gf; r["goals_against"] += gc
        r["goal_diff"] += (gf - gc)
        if gf > gc:   r["won"] += 1;   r["points"] += 3
        elif gf == gc: r["drawn"] += 1; r["points"] += 1
        else:          r["lost"] += 1


def _sort_table(table: dict) -> pd.DataFrame:
    rows = list(table.values())
    for r in rows:
        r["tiebreak"] = np.random.random()
    df = pd.DataFrame(rows).sort_values(
        ["points", "goal_diff", "goals_for", "tiebreak"], ascending=False
    ).drop(columns=["tiebreak"])
    df["position"] = range(1, len(df) + 1)
    return df.set_index("team")


# \\\\\\\\\\\
# Simulación de un grupo completo
# \\\\\\\\\\\

def simulate_group(group_name: str, teams: pd.DataFrame) -> pd.DataFrame:
    names  = teams["team"].tolist()
    table  = _create_table(names, group_name)
    lookup = _df_to_team_lookup(teams)
    for a, b in combinations(names, 2):
        ga, gb = simulate_group_match(lookup[a], lookup[b])
        _update(table, a, b, ga, gb)
    return _sort_table(table)


def simulate_all_groups(tournament_df: pd.DataFrame) -> dict:
    return {g: simulate_group(g, gdf) for g, gdf in tournament_df.groupby("group")}


# \\\\\\\\\\\
# Clasificados + puntos de terceros para estadísticas
# \\\\\\\\\\\

def get_qualified_teams(
    group_tables: dict,
    return_third_points: bool = False,
) -> tuple | dict:
    """
    Devuelve los 32 clasificados.
    Si return_third_points=True, devuelve también la lista de puntos
    de los 8 mejores terceros para calcular la media de clasificación.
    """
    firsts, seconds, thirds = [], [], []

    for group_name, table in group_tables.items():
        firsts.append(table.index[0])
        seconds.append(table.index[1])
        third_row = table.iloc[2].copy()
        third_row["group_name"] = group_name
        thirds.append(third_row)

    thirds_df = pd.DataFrame(thirds).sort_values(
        ["points", "goal_diff", "goals_for"], ascending=False
    )
    best_thirds     = thirds_df.head(8)
    rejected_thirds = thirds_df.iloc[8:]

    qualified = {
        "first":      firsts,
        "second":     seconds,
        "third_best": best_thirds.index.tolist(),
    }

    if return_third_points:
        # Puntos de los 8 clasificados y del primero eliminado (umbral)
        pts_classified = best_thirds["points"].tolist()
        pts_eliminated = rejected_thirds["points"].tolist() if len(rejected_thirds) else []
        return qualified, {"classified": pts_classified, "eliminated": pts_eliminated}

    return qualified
