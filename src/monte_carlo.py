"""
monte_carlo.py
--------------
Núcleo del proyecto. Repite el torneo N veces y acumula resultados.
Al final genera los CSV de salida en la carpeta outputs/.

Uso directo:
    python -m src.monte_carlo --simulations 10000 --seed 42
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
from src.analysis import generate_probabilities, generate_finals_table, generate_group_summary, generate_conclusions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# \\\\\\\\\\\
# Contadores de resultados acumulados por N simulaciones
# \\\\\\\\\\\

def _init_counters(teams: list[str]) -> dict:
    """Inicializa todos los contadores de resultados en cero para cada equipo."""
    return {
        "champion":        defaultdict(int),
        "finalist":        defaultdict(int),
        "semifinalist":    defaultdict(int),
        "quarterfinalist": defaultdict(int),
        "round_of_16":     defaultdict(int),
        "round_of_32":     defaultdict(int),
        "group_exit":      defaultdict(int),
        "finals_pairs":    defaultdict(int),   # pares de finalistas
    }


# \\\\\\\\\\\
# Ejecución de una simulación completa del torneo
# \\\\\\\\\\\

def run_single_simulation(tournament_df: pd.DataFrame) -> dict:
    """
    Simula el Mundial completo una sola vez:
    fase de grupos → clasificados → eliminatorias → campeón.

    Devuelve el resultado de simulate_knockout_stage más la lista
    de equipos eliminados en grupos.
    """
    group_tables = simulate_all_groups(tournament_df)
    qualified    = get_qualified_teams(group_tables)

    # Equipos que no pasan de grupos
    all_teams       = set(tournament_df["team"].tolist())
    qualified_set   = set(
        qualified["first"] + qualified["second"] + qualified["third_best"]
    )
    group_exits     = all_teams - qualified_set

    knockout_result = simulate_knockout_stage(qualified, tournament_df)
    knockout_result["group_exits"] = group_exits
    return knockout_result


# \\\\\\\\\\\
# Actualización de contadores tras cada simulación
# \\\\\\\\\\\

def _update_counters(counters: dict, result: dict) -> None:
    """Acumula los resultados de una simulación en los contadores globales."""
    counters["champion"][result["champion"]]    += 1
    counters["finalist"][result["finalist"]]    += 1

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

    # Par de finalistas (orden alfabético para evitar duplicados)
    pair = " vs ".join(sorted([result["champion"], result["finalist"]]))
    counters["finals_pairs"][pair] += 1


# \\\\\\\\\\\
# Bucle principal de Monte Carlo
# \\\\\\\\\\\

def run_monte_carlo(
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """
    Ejecuta N simulaciones completas del Mundial.
    Imprime progreso cada 10% y guarda los resultados en outputs/.

    Returns:
        Diccionario con counters, team_probabilities DataFrame y metadata.
    """
    np.random.seed(seed)

    tournament_df = load_tournament_data()
    all_teams     = tournament_df["team"].tolist()
    counters      = _init_counters(all_teams)

    start_time = time.time()
    log_every  = max(1, n_simulations // 10)

    logger.info(f"Iniciando {n_simulations:,} simulaciones (seed={seed})...")

    for i in range(n_simulations):
        result = run_single_simulation(tournament_df)
        _update_counters(counters, result)

        if verbose and (i + 1) % log_every == 0:
            pct = (i + 1) / n_simulations * 100
            elapsed = time.time() - start_time
            logger.info(f"  {pct:.0f}% completado ({i+1:,} sim) — {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    logger.info(f"Simulaciones completadas en {elapsed_total:.1f}s.")

    # \\\\\\\\\\\
    # Generación y guardado de resultados
    # \\\\\\\\\\\

    team_probs      = generate_probabilities(counters, n_simulations, tournament_df)
    finals_df       = generate_finals_table(counters, n_simulations)
    group_summary   = generate_group_summary(tournament_df)
    conclusions     = generate_conclusions(team_probs, finals_df, group_summary)

    # Guardar CSV principales
    team_probs.to_csv(OUTPUTS_DIR / "team_probabilities.csv", index=False)
    finals_df.to_csv(OUTPUTS_DIR  / "finals.csv",             index=False)
    group_summary.to_csv(OUTPUTS_DIR / "group_summary.csv",   index=False)

    # Guardar log de simulación
    log_df = pd.DataFrame([
        {"metric": "n_simulations", "value": n_simulations},
        {"metric": "seed",          "value": seed},
        {"metric": "elapsed_sec",   "value": round(elapsed_total, 2)},
    ])
    log_df.to_csv(OUTPUTS_DIR / "simulation_log.csv", index=False)

    # Guardar conclusiones en texto
    with open(OUTPUTS_DIR / "conclusions.txt", "w", encoding="utf-8") as f:
        f.write(conclusions)

    logger.info(f"Resultados guardados en: {OUTPUTS_DIR}")

    return {
        "counters":          counters,
        "team_probabilities": team_probs,
        "finals":            finals_df,
        "group_summary":     group_summary,
        "n_simulations":     n_simulations,
    }


# \\\\\\\\\\\
# Punto de entrada por línea de comandos
# \\\\\\\\\\\

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Monte Carlo Simulator")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help="Número de simulaciones (default: 10000)")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Semilla aleatoria (default: 42)")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suprimir logs de progreso")
    args = parser.parse_args()

    run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        verbose=not args.quiet,
    )
