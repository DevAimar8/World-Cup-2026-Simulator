"""
group_stage.py
--------------
Simula la fase de grupos del Mundial 2026 (12 grupos × 4 equipos).
Genera la tabla de cada grupo, aplica las reglas de clasificación y
selecciona los 8 mejores terceros para completar los 32 clasificados.

Optimizado: usa dicts internos en lugar de accesos pandas por fila.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from typing import Any

from src.match_simulator import simulate_group_match

# \\\\\\\\\\\
# Conversión de grupo DataFrame → lista de dicts para acceso rápido
# \\\\\\\\\\\

def _df_to_team_dicts(teams: pd.DataFrame) -> list[dict]:
    """Convierte el grupo a lista de dicts para evitar lookups pandas en el bucle."""
    return teams.to_dict(orient="records")


# \\\\\\\\\\\
# Creación de tabla vacía para un grupo (dict de dicts)
# \\\\\\\\\\\

def _create_group_table(team_names: list[str], group_name: str) -> dict:
    """Inicializa la tabla con estadísticas en cero para cada equipo del grupo."""
    return {
        name: {
            "team": name, "group": group_name,
            "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "goals_for": 0, "goals_against": 0, "goal_diff": 0, "points": 0,
        }
        for name in team_names
    }


# \\\\\\\\\\\
# Actualización de tabla tras un partido (operaciones sobre dicts)
# \\\\\\\\\\\

def _update_table(table: dict, name_a: str, name_b: str, ga: int, gb: int) -> None:
    """Actualiza puntos, goles y victorias de ambos equipos."""
    for name, gf, gc in [(name_a, ga, gb), (name_b, gb, ga)]:
        row = table[name]
        row["played"]        += 1
        row["goals_for"]     += gf
        row["goals_against"] += gc
        row["goal_diff"]     += (gf - gc)
        if gf > gc:
            row["won"]    += 1
            row["points"] += 3
        elif gf == gc:
            row["drawn"]  += 1
            row["points"] += 1
        else:
            row["lost"]   += 1


# \\\\\\\\\\\
# Ordenación de la tabla del grupo según criterios FIFA
# \\\\\\\\\\\

def _sort_group_table(table: dict) -> pd.DataFrame:
    """
    Convierte la tabla a DataFrame y ordena por:
    puntos → diferencia de goles → goles a favor → aleatorio.
    """
    rows = list(table.values())
    for row in rows:
        row["tiebreak"] = np.random.random()   # desempate final aleatorio

    df = pd.DataFrame(rows).sort_values(
        by=["points", "goal_diff", "goals_for", "tiebreak"],
        ascending=False,
    ).drop(columns=["tiebreak"])
    df["position"] = range(1, len(df) + 1)
    return df.set_index("team")


# \\\\\\\\\\\
# Simulación de un grupo completo (todos contra todos)
# \\\\\\\\\\\

def simulate_group(group_name: str, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Simula los 6 partidos de un grupo de 4 equipos.
    Devuelve la tabla final ordenada con posición de clasificación.
    """
    team_dicts = _df_to_team_dicts(teams)
    team_names = [t["team"] for t in team_dicts]
    table      = _create_group_table(team_names, group_name)

    # Crear dict de acceso rápido por nombre
    team_lookup = {t["team"]: pd.Series(t) for t in team_dicts}

    for name_a, name_b in combinations(team_names, 2):
        ga, gb = simulate_group_match(team_lookup[name_a], team_lookup[name_b])
        _update_table(table, name_a, name_b, ga, gb)

    return _sort_group_table(table)


# \\\\\\\\\\\
# Simulación de todos los grupos del Mundial
# \\\\\\\\\\\

def simulate_all_groups(tournament_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Simula los 12 grupos del Mundial 2026.
    Devuelve un diccionario {nombre_grupo: tabla_ordenada}.
    """
    group_tables = {}
    for group_name, group_teams in tournament_df.groupby("group"):
        group_tables[group_name] = simulate_group(group_name, group_teams)
    return group_tables


# \\\\\\\\\\\
# Selección de clasificados: 1°, 2° y 8 mejores terceros
# \\\\\\\\\\\

def get_qualified_teams(group_tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """
    Extrae los clasificados de la fase de grupos según el formato del Mundial 2026:
    - 1° y 2° de cada uno de los 12 grupos  → 24 equipos
    - 8 mejores terceros de los 12 grupos    → 8 equipos
    Total: 32 equipos para la fase eliminatoria.
    """
    firsts, seconds, thirds = [], [], []

    for group_name, table in group_tables.items():
        firsts.append(table.index[0])
        seconds.append(table.index[1])
        third_row         = table.iloc[2].copy()
        third_row["group_name"] = group_name
        thirds.append(third_row)

    # Seleccionar los 8 mejores terceros
    thirds_df = pd.DataFrame(thirds).sort_values(
        by=["points", "goal_diff", "goals_for"], ascending=False
    ).head(8)

    return {
        "first":      firsts,
        "second":     seconds,
        "third_best": thirds_df.index.tolist(),
    }
