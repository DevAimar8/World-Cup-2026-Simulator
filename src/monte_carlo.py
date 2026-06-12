"""
monte_carlo.py
--------------
Núcleo del simulador. Repite el torneo N veces y acumula todos los
contadores necesarios para los análisis nuevos:

  - Probabilidades por fase (campeón, final, semis, cuartos, R16, R32, grupos)
  - Media de puntos de los clasificados terceros
  - Camino más probable al título (rivales por ronda)
  - Varianza por equipo (coeficiente de variación de victorias)
  - Simulación de escenarios (equipo forzado fuera)
"""

import argparse
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED, OUTPUTS_DIR
from src.data_loader import load_tournament_data
from src.group_stage import simulate_all_groups, get_qualified_teams
from src.knockout_stage import simulate_knockout_stage
from src.analysis import (
    generate_probabilities,
    generate_finals_table,
    generate_group_summary,
    generate_third_place_stats,
    generate_path_to_title,
    generate_variance_table,
    generate_conclusions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# \\\\\\\\\\\
# Inicialización de todos los contadores
# \\\\\\\\\\\

def _init_counters() -> dict:
    """Crea los contadores vacíos para todas las métricas."""
    return {
        "champion":          defaultdict(int),
        "finalist":          defaultdict(int),
        "semifinalist":      defaultdict(int),
        "quarterfinalist":   defaultdict(int),
        "round_of_16":       defaultdict(int),
        "round_of_32":       defaultdict(int),
        "group_exit":        defaultdict(int),
        "finals_pairs":      defaultdict(int),

        # Camino al título: para cada ronda, con quién se cruzó el campeón
        "champion_r32_opp":  defaultdict(int),
        "champion_r16_opp":  defaultdict(int),
        "champion_qf_opp":   defaultdict(int),
        "champion_sf_opp":   defaultdict(int),

        # Puntos acumulados por los terceros clasificados
        "third_place_points": [],

        # Para varianza: lista de victorias absolutas por equipo
        "champion_wins_list": defaultdict(list),
    }


# \\\\\\\\\\\
# Actualización de contadores tras cada simulación
# \\\\\\\\\\\

def _update_counters(counters: dict, result: dict, sim_idx: int) -> None:
    """Acumula todos los resultados de una simulación."""
    champ = result["champion"]

    counters["champion"][champ]   += 1
    counters["finalist"][result["finalist"]] += 1

    for team in result["semifinalists"]:
        counters["semifinalist"][team] += 1
    for team in result["quarterfinalists"]:
        counters["quarterfinalist"][team] += 1
    for team in result["round_of_16"]:
        counters["round_of_16"][team] += 1
    for team in result["round_of_32"]:
        counters["round_of_32"][team] += 1
    for team in result["group_exits"]:
        counters["group_exit"][team] += 1

    # Final más repetida
    pair = " vs ".join(sorted([champ, result["finalist"]]))
    counters["finals_pairs"][pair] += 1

    # Camino del campeón por ronda
    if "champion_path" in result:
        path = result["champion_path"]
        for rnd, opp in path.items():
            counters[f"champion_{rnd}_opp"][f"{champ} vs {opp}"] += 1

    # Puntos de los mejores terceros en esta simulación
    t3 = result.get("third_place_points")
    if isinstance(t3, dict):
        counters["third_place_points"].append(t3)


# \\\\\\\\\\\
# Una simulación completa del torneo
# \\\\\\\\\\\

def run_single_simulation(tournament_df: pd.DataFrame) -> dict:
    """
    Simula el Mundial completo:
    fase de grupos → clasificados → eliminatorias → campeón.
    Devuelve resultado completo incluyendo puntos de terceros.
    """
    group_tables = simulate_all_groups(tournament_df)
    qualified, third_points = get_qualified_teams(group_tables, return_third_points=True)

    all_teams     = set(tournament_df["team"].tolist())
    qualified_set = set(
        qualified["first"] + qualified["second"] + qualified["third_best"]
    )
    group_exits = all_teams - qualified_set

    knockout_result = simulate_knockout_stage(qualified, tournament_df)
    knockout_result["group_exits"]         = group_exits
    knockout_result["third_place_points"]  = third_points
    return knockout_result


# \\\\\\\\\\\
# Bucle principal Monte Carlo
# \\\\\\\\\\\

def run_monte_carlo(
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    exclude_team: str = None,
) -> dict:
    """
    Ejecuta N simulaciones completas del Mundial.

    exclude_team: si se especifica, ese equipo se elimina antes de simular
                  (permite análisis de escenarios).
    """
    np.random.seed(seed)

    tournament_df = load_tournament_data()

    # \\\\\\\\\\\
    # Escenario: eliminar un equipo concreto del torneo
    # \\\\\\\\\\\
    if exclude_team and exclude_team in tournament_df["team"].values:
        logger.info(f"Escenario: simulando SIN {exclude_team}")
        tournament_df = tournament_df[tournament_df["team"] != exclude_team].copy()

    counters  = _init_counters()
    log_every = max(1, n_simulations // 10)
    start     = time.time()

    logger.info(f"Iniciando {n_simulations:,} simulaciones (seed={seed})...")

    for i in range(n_simulations):
        result = run_single_simulation(tournament_df)
        _update_counters(counters, result, i)

        if verbose and (i + 1) % log_every == 0:
            pct     = (i + 1) / n_simulations * 100
            elapsed = time.time() - start
            logger.info(f"  {pct:.0f}% ({i+1:,} sim) — {elapsed:.1f}s")

    elapsed_total = time.time() - start
    logger.info(f"Completado en {elapsed_total:.1f}s.")

    # \\\\\\\\\\\
    # Generar todos los outputs
    # \\\\\\\\\\\

    team_probs    = generate_probabilities(counters, n_simulations, tournament_df)
    finals_df     = generate_finals_table(counters, n_simulations)
    group_summary = generate_group_summary(tournament_df)
    third_stats   = generate_third_place_stats(counters)
    path_df       = generate_path_to_title(counters, n_simulations)
    variance_df   = generate_variance_table(team_probs)
    conclusions   = generate_conclusions(team_probs, finals_df, group_summary, third_stats, variance_df)

    # Guardar CSVs
    team_probs.to_csv(   OUTPUTS_DIR / "team_probabilities.csv", index=False)
    finals_df.to_csv(    OUTPUTS_DIR / "finals.csv",             index=False)
    group_summary.to_csv(OUTPUTS_DIR / "group_summary.csv",      index=False)
    third_stats.to_csv(  OUTPUTS_DIR / "third_place_stats.csv",  index=False)
    path_df.to_csv(      OUTPUTS_DIR / "path_to_title.csv",      index=False)
    variance_df.to_csv(  OUTPUTS_DIR / "variance_table.csv",     index=False)

    pd.DataFrame([
        {"metric": "n_simulations", "value": n_simulations},
        {"metric": "seed",          "value": seed},
        {"metric": "elapsed_sec",   "value": round(elapsed_total, 2)},
        {"metric": "exclude_team",  "value": exclude_team or "none"},
    ]).to_csv(OUTPUTS_DIR / "simulation_log.csv", index=False)

    with open(OUTPUTS_DIR / "conclusions.txt", "w", encoding="utf-8") as f:
        f.write(conclusions)

    logger.info(f"Outputs guardados en {OUTPUTS_DIR}")

    return {
        "counters":            counters,
        "team_probabilities":  team_probs,
        "finals":              finals_df,
        "group_summary":       group_summary,
        "third_place_stats":   third_stats,
        "path_to_title":       path_df,
        "variance_table":      variance_df,
        "n_simulations":       n_simulations,
    }


# \\\\\\\\\\\
# Punto de entrada CLI
# \\\\\\\\\\\

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Monte Carlo Simulator")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--exclude",     type=str, default=None,
                        help="Simular sin un equipo concreto (escenario)")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        verbose=not args.quiet,
        exclude_team=args.exclude,
    )
