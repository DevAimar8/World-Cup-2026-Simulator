"""
analysis.py
-----------
Transforma los contadores Monte Carlo en tablas analíticas y conclusiones.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def pct(counter, team, n): return round(counter[team] / n * 100, 2)


def generate_probabilities(C, n, df):
    rows = []
    for _, r in df.iterrows():
        t = r["team"]
        rows.append({
            "team": t, "group": r["group"], "confederation": r["confederation"],
            "overall_rating":    round(float(r["overall_rating"]), 1),
            "power_score":       round(float(r.get("power_score", 50)), 1),
            "elo_rating":        int(r.get("elo_rating", 0)),
            "attack_coef":       round(float(r["attack_coef"]), 3),
            "defense_coef":      round(float(r["defense_coef"]), 3),
            "pass_group_stage_pct": round((1 - C["group_exit"][t] / n) * 100, 2),
            "reach_round32_pct":    pct(C["r32"],            t, n),
            "reach_round16_pct":    pct(C["r16"],            t, n),
            "reach_quarters_pct":   pct(C["quarterfinalist"],t, n),
            "reach_semis_pct":      pct(C["semifinalist"],   t, n),
            "reach_final_pct":      pct(C["finalist"],       t, n),
            "champion_pct":         pct(C["champion"],       t, n),
            "group_exit_pct":       pct(C["group_exit"],     t, n),
        })
    return pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)


def generate_finals_table(C, n):
    rows = [{"final": p, "count": c, "probability_pct": round(c/n*100, 2)}
            for p, c in C["finals_pairs"].items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(25).reset_index(drop=True)


def generate_group_summary(df):
    rows = []
    for g, gdf in df.groupby("group"):
        rows.append({
            "group": g,
            "teams": " / ".join(gdf["team"].tolist()),
            "average_rating": round(gdf["overall_rating"].mean(), 1),
            "average_power_score": round(gdf["power_score"].mean(), 1) if "power_score" in gdf.columns else 0,
            "max_rating": round(gdf["overall_rating"].max(), 1),
            "min_rating": round(gdf["overall_rating"].min(), 1),
        })
    return pd.DataFrame(rows).sort_values("average_rating", ascending=False).reset_index(drop=True)


def generate_third_place_stats(C):
    cls_arr  = np.array(C["third_pts_cls"])  if C["third_pts_cls"]  else np.array([0])
    elim_arr = np.array(C["third_pts_elim"]) if C["third_pts_elim"] else np.array([0])
    rows = [
        {"categoria": "RESUMEN - Clasificados",       "puntos": round(float(cls_arr.mean()), 2), "frecuencia_pct": float("nan")},
        {"categoria": "RESUMEN - Mínimo clasificado", "puntos": int(cls_arr.min()),               "frecuencia_pct": float("nan")},
        {"categoria": "RESUMEN - Máximo clasificado", "puntos": int(cls_arr.max()),               "frecuencia_pct": float("nan")},
        {"categoria": "RESUMEN - Máximo eliminado",   "puntos": int(elim_arr.max()),              "frecuencia_pct": float("nan")},
    ]
    for v in sorted(set(cls_arr.tolist())):
        rows.append({"categoria": "Clasificado (top 8)", "puntos": int(v),
                     "frecuencia_pct": round((cls_arr == v).mean() * 100, 2)})
    for v in sorted(set(elim_arr.tolist())):
        rows.append({"categoria": "Eliminado", "puntos": int(v),
                     "frecuencia_pct": round((elim_arr == v).mean() * 100, 2)})
    return pd.DataFrame(rows)


def generate_path_to_title(C, n):
    rnd_map = {"r32": "Dieciseisavos", "r16": "Octavos", "qf": "Cuartos", "sf": "Semifinal"}
    rows = []
    for champ, titles in sorted(C["champion"].items(), key=lambda x: -x[1]):
        if titles == 0: continue
        row = {"team": champ, "titles": titles, "champion_pct": round(titles/n*100, 2)}
        for rnd, lbl in rnd_map.items():
            opps = {k: v for k, v in C[f"{rnd}_opp"].items() if k.startswith(champ + " vs ")}
            if opps:
                best = max(opps, key=opps.get)
                row[f"{lbl}_rival"]    = best.replace(champ + " vs ", "")
                row[f"{lbl}_freq_pct"] = round(opps[best] / titles * 100, 1)
            else:
                row[f"{lbl}_rival"] = "-"; row[f"{lbl}_freq_pct"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)


def generate_variance_table(tp):
    df = tp.copy()
    df["consistency_index"] = (df["reach_semis_pct"] / (df["group_exit_pct"] + 1)).round(3)
    df["variance_index"]    = (df["champion_pct"] * df["group_exit_pct"]).round(3)
    return df[["team","group","overall_rating","power_score","champion_pct",
               "reach_semis_pct","group_exit_pct","consistency_index","variance_index"]
             ].sort_values("consistency_index", ascending=False).reset_index(drop=True)


def generate_conclusions(tp, fin, gs, t3, var):
    lines = ["=" * 65, "  CONCLUSIONES — MUNDIAL 2026 MONTE CARLO SIMULATOR", "=" * 65, ""]
    top  = tp.iloc[0]
    lines.append(f"1. GRAN FAVORITO: {top['team']} ({top['champion_pct']}%)")
    fin1 = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
    lines.append(f"2. MÁS VECES EN LA FINAL: {fin1['team']} ({fin1['reach_final_pct']}%)")
    if not var.empty:
        cons = var.sort_values("consistency_index", ascending=False).iloc[0]
        lines.append(f"3. MÁS CONSISTENTE: {cons['team']} (índice {cons['consistency_index']:.2f})")
    if not gs.empty:
        lines.append(f"4. GRUPO MÁS DURO: Grupo {gs.iloc[0]['group']} (rating {gs.iloc[0]['average_rating']})")
        lines.append(f"5. GRUPO MÁS FÁCIL: Grupo {gs.iloc[-1]['group']} (rating {gs.iloc[-1]['average_rating']})")
    if not fin.empty:
        lines.append(f"6. FINAL MÁS PROBABLE: {fin.iloc[0]['final']} ({fin.iloc[0]['probability_pct']}%)")
    summ = t3[t3["categoria"].str.startswith("RESUMEN")] if not t3.empty else pd.DataFrame()
    if not summ.empty:
        avg = summ[summ["categoria"] == "RESUMEN - Clasificados"]["puntos"].values
        lines.append(f"7. MEDIA PUNTOS TERCER CLASIFICADO: {avg[0]:.2f} pts")
    lines += ["", "-" * 65,
              "NOTA: Resultados probabilísticos. Modelo Dixon-Coles + Power Score.",
              "-" * 65]
    return "\n".join(lines)
