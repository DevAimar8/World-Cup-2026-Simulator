"""
monte_carlo.py
--------------
Núcleo del proyecto. Repite el torneo N veces y agrega resultados.
Genera todos los CSVs de outputs listos para el dashboard.
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED, OUTPUTS_DIR
from src.power_score import build_ratings_df
from src.tournament import simulate_all_groups, get_qualified, simulate_knockout_stage
from src.analysis import (
    generate_probabilities, generate_finals_table, generate_group_summary,
    generate_third_place_stats, generate_path_to_title,
    generate_variance_table, generate_conclusions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# \\\\\\\\\\\
# Inicialización de contadores
# \\\\\\\\\\\

def _init_counters() -> dict:
    C = {k: defaultdict(int) for k in [
        "champion","finalist","semifinalist","quarterfinalist",
        "r16","r32","group_exit","finals_pairs",
        "r32_opp","r16_opp","qf_opp","sf_opp",
    ]}
    C["third_pts_cls"]  = []
    C["third_pts_elim"] = []
    return C


# \\\\\\\\\\\
# Actualización de contadores tras cada simulación
# \\\\\\\\\\\

def _update(C: dict, result: dict, all_teams: set, t3_cls: list, t3_elim: list) -> None:
    ch = result["champion"]
    C["champion"][ch]          += 1
    C["finalist"][result["finalist"]] += 1
    for t in result["semis"]:    C["semifinalist"][t]   += 1
    for t in result["quarters"]: C["quarterfinalist"][t] += 1
    for t in result["r16"]:      C["r16"][t]            += 1
    for t in result["r32"]:      C["r32"][t]            += 1

    qualified_set = set(result["r32"])
    for t in all_teams - qualified_set:
        C["group_exit"][t] += 1

    C["finals_pairs"][" vs ".join(sorted([ch, result["finalist"]]))] += 1
    for rnd, opp in result["champion_path"].items():
        C[f"{rnd}_opp"][f"{ch} vs {opp}"] += 1

    C["third_pts_cls"].extend(t3_cls)
    C["third_pts_elim"].extend(t3_elim)


# \\\\\\\\\\\
# Bucle principal Monte Carlo
# \\\\\\\\\\\

def run_monte_carlo(
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    fixtures_df: pd.DataFrame = None,
    standings_df: pd.DataFrame = None,
) -> dict:
    """
    Ejecuta N simulaciones completas del Mundial.
    Si se pasan fixtures_df / standings_df, actualiza los ratings con datos en vivo.
    """
    np.random.seed(seed)

    # Construir ratings (con datos en vivo si están disponibles)
    ratings_df = build_ratings_df(fixtures_df=fixtures_df, standings_df=standings_df)
    all_teams  = set(ratings_df["team"].tolist())
    C          = _init_counters()

    log_every = max(1, n_simulations // 10)
    t0        = time.time()
    logger.info(f"Iniciando {n_simulations:,} simulaciones (seed={seed})...")

    for i in range(n_simulations):
        tables       = simulate_all_groups(ratings_df)
        qualified, t3_cls, t3_elim = get_qualified(tables)
        result       = simulate_knockout_stage(qualified, ratings_df)
        _update(C, result, all_teams, t3_cls, t3_elim)

        if verbose and (i + 1) % log_every == 0:
            pct = (i + 1) / n_simulations * 100
            logger.info(f"  {pct:.0f}% ({i+1:,} sims) — {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    logger.info(f"Completado en {elapsed:.1f}s")

    # \\\\\\\\\\\
    # Generar y guardar todos los outputs
    # \\\\\\\\\\\

    tp  = generate_probabilities(C, n_simulations, ratings_df)
    fin = generate_finals_table(C, n_simulations)
    gs  = generate_group_summary(ratings_df)
    t3  = generate_third_place_stats(C)
    pth = generate_path_to_title(C, n_simulations)
    var = generate_variance_table(tp)
    con = generate_conclusions(tp, fin, gs, t3, var)

    tp.to_csv(OUTPUTS_DIR / "team_probabilities.csv",  index=False)
    fin.to_csv(OUTPUTS_DIR / "finals.csv",             index=False)
    gs.to_csv(OUTPUTS_DIR  / "group_summary.csv",      index=False)
    t3.to_csv(OUTPUTS_DIR  / "third_place_stats.csv",  index=False)
    pth.to_csv(OUTPUTS_DIR / "path_to_title.csv",      index=False)
    var.to_csv(OUTPUTS_DIR / "variance_table.csv",     index=False)

    pd.DataFrame([
        {"metric": "n_simulations", "value": n_simulations},
        {"metric": "seed",          "value": seed},
        {"metric": "elapsed_sec",   "value": round(elapsed, 1)},
    ]).to_csv(OUTPUTS_DIR / "simulation_log.csv", index=False)

    open(OUTPUTS_DIR / "conclusions.txt", "w", encoding="utf-8").write(con)

    logger.info(f"Outputs guardados en {OUTPUTS_DIR}")
    return {"team_probabilities": tp, "finals": fin, "group_summary": gs,
            "third_place_stats": t3, "path_to_title": pth, "variance_table": var,
            "ratings": ratings_df, "n_simulations": n_simulations}


# \\\\\\\\\\\
# CLI
# \\\\\\\\\\\

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup 2026 — Monte Carlo Simulator")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()
    run_monte_carlo(n_simulations=args.simulations, seed=args.seed, verbose=not args.quiet)
