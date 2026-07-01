"""
tournament.py
-------------
Simula el torneo completo: fase de grupos → clasificados → eliminatorias.
Respeta el bracket oficial del Mundial 2026 cuando hay datos en vivo.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from src.match_simulator import simulate_group_match, simulate_knockout_match


# \\\\\\\\\\\
# Fase de grupos — todos contra todos
# \\\\\\\\\\\

def simulate_group(group_name: str, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Simula los 6 partidos de un grupo de 4. Devuelve tabla clasificatoria."""
    names  = teams_df["team"].tolist()
    lookup = {row["team"]: row for _, row in teams_df.iterrows()}
    table  = {n: dict(played=0, won=0, drawn=0, lost=0, gf=0, ga=0, gd=0, pts=0) for n in names}

    for a, b in combinations(names, 2):
        ga, gb = simulate_group_match(lookup[a], lookup[b])
        for name, gf, gc in [(a, ga, gb), (b, gb, ga)]:
            t = table[name]
            t["played"] += 1; t["gf"] += gf; t["ga"] += gc; t["gd"] += (gf - gc)
            if gf > gc:   t["won"] += 1;  t["pts"] += 3
            elif gf == gc: t["drawn"] += 1; t["pts"] += 1
            else:          t["lost"] += 1

    rows = sorted(table.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"], np.random.random()))
    result = pd.DataFrame([{"team": n, "group": group_name, **s} for n, s in rows])
    result["position"] = range(1, len(result) + 1)
    return result


def simulate_all_groups(ratings_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Simula los 12 grupos. Devuelve dict {grupo: tabla}."""
    return {grp: simulate_group(grp, gdf) for grp, gdf in ratings_df.groupby("group")}


# \\\\\\\\\\\
# Clasificados: 1°, 2° de cada grupo + 8 mejores terceros
# \\\\\\\\\\\

def get_qualified(group_tables: dict) -> tuple[dict, list, list]:
    """
    Devuelve (qualified_dict, pts_clasificados_terceros, pts_eliminados_terceros).
    qualified_dict: {'first': [...], 'second': [...], 'third_best': [...]}
    """
    firsts, seconds, thirds = [], [], []
    for grp, table in group_tables.items():
        firsts.append(table.iloc[0]["team"])
        seconds.append(table.iloc[1]["team"])
        thirds.append((table.iloc[2]["team"], table.iloc[2]["pts"],
                       table.iloc[2]["gd"], table.iloc[2]["gf"]))

    thirds_sorted = sorted(thirds, key=lambda x: (-x[1], -x[2], -x[3]))
    best8    = [t[0] for t in thirds_sorted[:8]]
    cls_pts  = [t[1] for t in thirds_sorted[:8]]
    elim_pts = [t[1] for t in thirds_sorted[8:]]

    return {"first": firsts, "second": seconds, "third_best": best8}, cls_pts, elim_pts


# \\\\\\\\\\\
# Bracket de eliminatorias
# \\\\\\\\\\\

def build_bracket(qualified: dict) -> list[str]:
    """Construye el bracket de 32 equipos por seeds."""
    f, s, t = qualified["first"], qualified["second"], qualified["third_best"]
    bracket, half = [], len(f) // 2
    for i in range(half):     bracket += [f[i], s[-(i+1)]]
    for i in range(half, len(f)): bracket += [f[i], s[-(i+1)]]
    bracket.extend(t)
    return bracket[:32]


def _sim_round(teams: list[str], lookup: dict) -> tuple[list[str], list[str]]:
    """Simula una ronda: devuelve (ganadores, eliminados)."""
    winners, losers = [], []
    for i in range(0, len(teams) - 1, 2):
        a_row = lookup.get(teams[i],   list(lookup.values())[0])
        b_row = lookup.get(teams[i+1], list(lookup.values())[0])
        w = simulate_knockout_match(a_row, b_row)
        winners.append(w)
        losers.append(teams[i] if w == teams[i+1] else teams[i+1])
    return winners, losers


def _find_opp(team: str, bracket: list[str]) -> str:
    for i in range(0, len(bracket) - 1, 2):
        if bracket[i] == team:   return bracket[i+1]
        if bracket[i+1] == team: return bracket[i]
    return "unknown"


def simulate_knockout_stage(qualified: dict, ratings_df: pd.DataFrame) -> dict:
    """
    Simula todas las rondas eliminatorias y registra el camino del campeón.
    Devuelve: champion, finalist, semis, quarters, r16, r32, champion_path.
    """
    lookup = {row["team"]: row for _, row in ratings_df.iterrows()}
    b32 = build_bracket(qualified)
    r16, _ = _sim_round(b32, lookup)
    qf,  _ = _sim_round(r16, lookup)
    sf,  _ = _sim_round(qf,  lookup)
    fin, _ = _sim_round(sf,  lookup)

    champ    = fin[0]
    runner   = fin[1] if champ == fin[0] else fin[0]
    path     = {
        "r32": _find_opp(champ, b32),
        "r16": _find_opp(champ, r16),
        "qf":  _find_opp(champ, qf),
        "sf":  _find_opp(champ, sf),
    }
    return {"champion": champ, "finalist": runner,
            "semis": sf, "quarters": qf,
            "r16": r16, "r32": b32, "champion_path": path}
