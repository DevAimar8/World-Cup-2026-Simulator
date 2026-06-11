"""
analysis.py
-----------
Transforma los contadores Monte Carlo en todas las tablas de análisis:

  - Probabilidades por fase (todas las fases)
  - Finales más repetidas
  - Resumen de grupos con dificultad
  - Media de puntos de los terceros clasificados
  - Camino más probable al título por equipo
  - Varianza y consistencia por equipo
  - Conclusiones automáticas en texto
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# \\\\\\\\\\\
# Tabla principal: probabilidades por fase para los 48 equipos
# \\\\\\\\\\\

def generate_probabilities(
    counters: dict,
    n_simulations: int,
    tournament_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera la tabla maestra de probabilidades.
    Incluye probabilidad de clasificarse de grupos, de pasar cada ronda
    y de ser eliminado en fase de grupos.
    """
    def pct(counter, team):
        return round(counter[team] / n_simulations * 100, 2)

    rows = []
    for _, row in tournament_df.iterrows():
        t = row["team"]

        # Probabilidad de clasificarse de grupos = no estar en group_exit
        pass_groups = round((1 - counters["group_exit"][t] / n_simulations) * 100, 2)

        rows.append({
            "team":                     t,
            "group":                    row["group"],
            "confederation":            row["confederation"],
            "overall_rating":           round(float(row["overall_rating"]), 1),
            "attack_coef":              round(float(row.get("attack_coef", 1.0)), 3),
            "defense_coef":             round(float(row.get("defense_coef", 1.0)), 3),

            # Probabilidades acumuladas por fase
            "pass_group_stage_pct":     pass_groups,
            "reach_round32_pct":        pct(counters["round_of_32"],     t),
            "reach_round16_pct":        pct(counters["round_of_16"],     t),
            "reach_quarters_pct":       pct(counters["quarterfinalist"], t),
            "reach_semis_pct":          pct(counters["semifinalist"],    t),
            "reach_final_pct":          pct(counters["finalist"],        t),
            "champion_pct":             pct(counters["champion"],        t),
            "group_exit_pct":           pct(counters["group_exit"],      t),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("champion_pct", ascending=False).reset_index(drop=True)
    return df


# \\\\\\\\\\\
# Finales más repetidas en las N simulaciones
# \\\\\\\\\\\

def generate_finals_table(counters: dict, n_simulations: int) -> pd.DataFrame:
    rows = [
        {"final": pair, "count": c,
         "probability_pct": round(c / n_simulations * 100, 2)}
        for pair, c in counters["finals_pairs"].items()
    ]
    return (pd.DataFrame(rows)
            .sort_values("count", ascending=False)
            .head(25)
            .reset_index(drop=True))


# \\\\\\\\\\\
# Resumen de dificultad de grupos
# \\\\\\\\\\\

def generate_group_summary(tournament_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, gdf in tournament_df.groupby("group"):
        rows.append({
            "group":          g,
            "teams":          " / ".join(gdf["team"].tolist()),
            "average_rating": round(gdf["overall_rating"].mean(), 1),
            "max_rating":     round(gdf["overall_rating"].max(), 1),
            "min_rating":     round(gdf["overall_rating"].min(), 1),
            "rating_range":   round(gdf["overall_rating"].max() - gdf["overall_rating"].min(), 1),
        })
    return (pd.DataFrame(rows)
            .sort_values("average_rating", ascending=False)
            .reset_index(drop=True))


# \\\\\\\\\\\
# Media de puntos con los que se clasifican los mejores terceros
# \\\\\\\\\\\

def generate_third_place_stats(counters: dict) -> pd.DataFrame:
    """
    Responde: ¿con qué media de puntos se clasifica un tercero de grupo?

    Devuelve distribución de puntos de los terceros clasificados y eliminados,
    con el umbral estadístico de clasificación.
    """
    data = counters.get("third_place_points", [])
    if not data:
        return pd.DataFrame(columns=["categoria", "puntos", "frecuencia_pct"])

    classified = []
    eliminated = []
    for d in data:
        classified.extend(d.get("classified", []))
        eliminated.extend(d.get("eliminated", []))

    rows = []
    for pts_list, label in [(classified, "Clasificado (top 8)"), (eliminated, "Eliminado")]:
        if not pts_list:
            continue
        arr = np.array(pts_list)
        for pts_val in sorted(set(arr)):
            rows.append({
                "categoria":      label,
                "puntos":         int(pts_val),
                "frecuencia_pct": round((arr == pts_val).mean() * 100, 2),
            })

    # Estadísticas resumen
    if classified:
        cl_arr = np.array(classified)
        summary_rows = [
            {"categoria": "RESUMEN - Clasificados", "puntos": round(float(cl_arr.mean()), 2),
             "frecuencia_pct": float("nan")},
            {"categoria": "RESUMEN - Mínimo clasificado", "puntos": int(cl_arr.min()),
             "frecuencia_pct": float("nan")},
            {"categoria": "RESUMEN - Máximo clasificado", "puntos": int(cl_arr.max()),
             "frecuencia_pct": float("nan")},
        ]
        if eliminated:
            el_arr = np.array(eliminated)
            summary_rows.append({
                "categoria": "RESUMEN - Máximo eliminado",
                "puntos": int(el_arr.max()), "frecuencia_pct": float("nan")
            })
        rows = summary_rows + rows

    return pd.DataFrame(rows)


# \\\\\\\\\\\
# Camino más probable al título por equipo favorito
# \\\\\\\\\\\

def generate_path_to_title(counters: dict, n_simulations: int) -> pd.DataFrame:
    """
    Para cada equipo, muestra el rival más frecuente en cada ronda
    cuando ese equipo gana el torneo.

    Ejemplo: España → R32: Bolivia | R16: Países Bajos | QF: Francia...
    """
    rounds = ["r32", "r16", "qf", "sf"]
    round_labels = {"r32": "Dieciseisavos", "r16": "Octavos",
                    "qf": "Cuartos", "sf": "Semifinal"}

    # Recopilar todos los equipos que alguna vez fueron campeones
    champions = set(counters["champion"].keys())
    rows = []

    for champ in champions:
        n_titles = counters["champion"][champ]
        if n_titles == 0:
            continue
        row = {"team": champ, "titles": n_titles,
               "champion_pct": round(n_titles / n_simulations * 100, 2)}

        for rnd in rounds:
            key       = f"champion_{rnd}_opp"
            opp_counter = counters.get(key, defaultdict(int))
            # Filtrar solo los pares que incluyen a este campeón
            filtered = {k: v for k, v in opp_counter.items() if k.startswith(champ + " vs ")}
            if filtered:
                most_common = max(filtered, key=filtered.get)
                rival = most_common.replace(champ + " vs ", "")
                freq  = round(filtered[most_common] / n_titles * 100, 1)
                row[f"{round_labels[rnd]}_rival"]    = rival
                row[f"{round_labels[rnd]}_freq_pct"] = freq
            else:
                row[f"{round_labels[rnd]}_rival"]    = "-"
                row[f"{round_labels[rnd]}_freq_pct"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    return df


# \\\\\\\\\\\
# Varianza y consistencia por equipo
# \\\\\\\\\\\

def generate_variance_table(team_probs: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el índice de varianza de cada equipo.

    Un equipo con alta probabilidad de campeón pero también alta
    probabilidad de caer en grupos tiene MUCHA varianza (impredecible).
    Un equipo que llega consistentemente a cuartos tiene POCA varianza.

    Índice de consistencia = reach_semis_pct / (group_exit_pct + 1)
    """
    df = team_probs.copy()

    df["variance_index"] = (
        df["champion_pct"] * df["group_exit_pct"]
    ).round(3)

    df["consistency_index"] = (
        df["reach_semis_pct"] / (df["group_exit_pct"] + 1)
    ).round(3)

    result = df[[
        "team", "group", "overall_rating",
        "champion_pct", "reach_semis_pct", "group_exit_pct",
        "variance_index", "consistency_index",
    ]].sort_values("consistency_index", ascending=False).reset_index(drop=True)

    return result


# \\\\\\\\\\\
# Conclusiones automáticas en texto
# \\\\\\\\\\\

def generate_conclusions(
    team_probs: pd.DataFrame,
    finals_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    third_stats: pd.DataFrame,
    variance_df: pd.DataFrame,
) -> str:
    """Genera el archivo de conclusiones con todos los análisis."""
    lines = [
        "=" * 65,
        "  CONCLUSIONES DEL SIMULADOR MONTE CARLO — MUNDIAL 2026",
        "=" * 65, "",
    ]

    # 1. Favorito al título
    top = team_probs.iloc[0]
    lines.append(f"1. GRAN FAVORITO AL TÍTULO")
    lines.append(f"   {top['team']} con {top['champion_pct']}% de probabilidad.")
    lines.append(f"   Rating compuesto: {top['overall_rating']:.1f}")
    lines.append("")

    # 2. Más veces en la final
    top_final = team_probs.sort_values("reach_final_pct", ascending=False).iloc[0]
    lines.append(f"2. EQUIPO MÁS VECES EN LA FINAL")
    lines.append(f"   {top_final['team']} llega a la final en el {top_final['reach_final_pct']}% de simulaciones.")
    lines.append("")

    # 3. Equipo más consistente (cuartos+)
    top_cons = variance_df.sort_values("consistency_index", ascending=False).iloc[0]
    lines.append(f"3. EQUIPO MÁS CONSISTENTE (llega lejos sin grandes altibajos)")
    lines.append(f"   {top_cons['team']} — índice de consistencia: {top_cons['consistency_index']:.2f}")
    lines.append(f"   Semis: {top_cons['reach_semis_pct']}% | Eliminado en grupos: {top_cons['group_exit_pct']}%")
    lines.append("")

    # 4. Mayor varianza (sorpresa potencial)
    top_var = variance_df.sort_values("variance_index", ascending=False).iloc[0]
    lines.append(f"4. MAYOR VARIANZA — EQUIPO MÁS IMPREDECIBLE")
    lines.append(f"   {top_var['team']} puede ganarlo todo o caer en grupos.")
    lines.append(f"   Campeón: {top_var['champion_pct']}% | Eliminado grupos: {top_var['group_exit_pct']}%")
    lines.append("")

    # 5. Punto de terceros
    lines.append(f"5. MEDIA DE PUNTOS PARA CLASIFICARSE COMO TERCER")
    summary_rows = third_stats[third_stats["categoria"].str.startswith("RESUMEN")]
    if not summary_rows.empty:
        for _, r in summary_rows.iterrows():
            if pd.isna(r["frecuencia_pct"]):
                label = r["categoria"].replace("RESUMEN - ", "")
                lines.append(f"   {label}: {r['puntos']}")
    lines.append("")

    # 6. Grupo más difícil
    hard = group_summary.iloc[0]
    easy = group_summary.iloc[-1]
    lines.append(f"6. GRUPOS")
    lines.append(f"   Más difícil: Grupo {hard['group']} (rating medio {hard['average_rating']})")
    lines.append(f"   → {hard['teams']}")
    lines.append(f"   Más accesible: Grupo {easy['group']} (rating medio {easy['average_rating']})")
    lines.append(f"   → {easy['teams']}")
    lines.append("")

    # 7. Final más probable
    if not finals_df.empty:
        tf = finals_df.iloc[0]
        lines.append(f"7. FINAL MÁS PROBABLE")
        lines.append(f"   {tf['final']} — ocurre en el {tf['probability_pct']}% de simulaciones.")
    lines.append("")

    # 8. Sorpresas: rating < 80 pero llegan a cuartos frecuentemente
    lines.append(f"8. POSIBLES SORPRESAS (rating < 80, llegan a cuartos en >5% de simulaciones)")
    surprises = team_probs[
        (team_probs["overall_rating"] < 80) &
        (team_probs["reach_quarters_pct"] > 5)
    ].head(4)
    if surprises.empty:
        lines.append("   Ninguna selección de bajo rating llega frecuentemente a cuartos.")
    else:
        for _, s in surprises.iterrows():
            lines.append(f"   {s['team']}: {s['reach_quarters_pct']}% cuartos | rating {s['overall_rating']:.1f}")
    lines.append("")

    # 9. Clasificación de grupos por confederación
    lines.append(f"9. PASE DE GRUPOS POR CONFEDERACIÓN (media)")
    for conf, cdf in team_probs.groupby("confederation"):
        avg = cdf["pass_group_stage_pct"].mean()
        lines.append(f"   {conf:<10} → {avg:.1f}% de media")
    lines.append("")

    lines += [
        "-" * 65,
        "NOTA: Resultados probabilísticos, no predicciones.",
        "Modelo: Dixon-Coles + factores contextuales del Mundial.",
        "Modificar ratings.csv cambia significativamente los resultados.",
        "-" * 65,
    ]

    return "\n".join(lines)
