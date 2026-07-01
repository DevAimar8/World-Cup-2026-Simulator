"""
monte_carlo.py
--------------
Núcleo del proyecto. Repite el torneo completo N veces (por defecto 10.000)
y acumula contadores para calcular probabilidades estables.

La simulación Monte Carlo es necesaria porque el fútbol tiene demasiada
varianza para confiar en una sola estimación. Repetir 10.000 veces permite
obtener distribuciones de probabilidad robustas con error ~0.3%.

Al terminar genera todos los CSV de outputs que usa el dashboard.

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
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


# ─── Inicialización de contadores ────────────────────────────────────────────
def _init_counters() -> dict:
    """
    Crea los contadores vacíos para todas las métricas que rastrea el simulador.

    Los contadores defaultdict(int) devuelven 0 automáticamente para
    cualquier equipo, evitando KeyErrors durante la acumulación.
    """
    C = {k: defaultdict(int) for k in [
        "champion",         # Cuántas veces ganó el Mundial
        "finalist",         # Cuántas veces llegó a la final
        "semifinalist",     # Cuántas veces llegó a semis
        "quarterfinalist",  # Cuántas veces llegó a cuartos
        "r16",              # Cuántas veces llegó a octavos
        "r32",              # Cuántas veces llegó a dieciseisavos (clasificó)
        "group_exit",       # Cuántas veces fue eliminado en grupos
        "finals_pairs",     # Pares de finalistas (para ranking de finales más probables)
        # Rival más frecuente del campeón en cada ronda
        "r32_opp", "r16_opp", "qf_opp", "sf_opp",
    ]}
    # Puntos de los terceros: clasificados y eliminados (para el análisis del umbral)
    C["third_pts_cls"]  = []
    C["third_pts_elim"] = []
    return C


# ─── Actualización de contadores tras cada simulación ────────────────────────
def _update(C: dict, result: dict, all_teams: set,
            t3_cls: list, t3_elim: list) -> None:
    """
    Acumula los resultados de una simulación en los contadores globales.
    Se llama una vez por cada iteración del bucle Monte Carlo.
    """
    ch = result["champion"]

    # Fases eliminatorias
    C["champion"][ch]                    += 1
    C["finalist"][result["finalist"]]    += 1
    for t in result["semis"]:    C["semifinalist"][t]    += 1
    for t in result["quarters"]: C["quarterfinalist"][t] += 1
    for t in result["r16"]:      C["r16"][t]             += 1
    for t in result["r32"]:      C["r32"][t]             += 1

    # Equipos eliminados en grupos = todos los que NO clasificaron a eliminatorias
    qualified_set = set(result["r32"])
    for t in all_teams - qualified_set:
        C["group_exit"][t] += 1

    # Par de finalistas (en orden alfabético para evitar duplicados)
    C["finals_pairs"][" vs ".join(sorted([ch, result["finalist"]]))] += 1

    # Camino del campeón: con quién jugó en cada ronda
    for rnd, opp in result["champion_path"].items():
        C[f"{rnd}_opp"][f"{ch} vs {opp}"] += 1

    # Puntos de los terceros para calcular el umbral de clasificación
    C["third_pts_cls"].extend(t3_cls)
    C["third_pts_elim"].extend(t3_elim)


# ─── Bucle principal Monte Carlo ─────────────────────────────────────────────
def run_monte_carlo(
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    fixtures_df: pd.DataFrame = None,
    standings_df: pd.DataFrame = None,
) -> dict:
    """
    Ejecuta N simulaciones completas del Mundial y genera todos los outputs.

    Flujo de cada simulación:
    1. Simular fase de grupos (12 grupos × 4 equipos = 72 partidos)
    2. Determinar los 32 clasificados (1º, 2º y 8 mejores 3º)
    3. Simular eliminatorias (32 → 16 → 8 → 4 → 2 → 1)
    4. Acumular resultados en los contadores

    Con 10.000 simulaciones el bucle completo tarda ~30-60s en local.
    En Streamlit Cloud tarda ~1-2 minutos.

    Args:
        n_simulations: Número de torneos a simular
        seed:          Semilla para reproducibilidad (misma semilla = mismos resultados)
        verbose:       Si True, imprime progreso cada 10%
        fixtures_df:   Partidos reales descargados de la API (opcional)
        standings_df:  Tabla de grupos real de la API (opcional)

    Returns:
        Diccionario con todos los DataFrames generados y metadatos
    """
    # Fijar la semilla para que los resultados sean reproducibles
    np.random.seed(seed)

    # Construir ratings (actualiza el Power Score si hay datos de API)
    ratings_df = build_ratings_df(fixtures_df=fixtures_df, standings_df=standings_df)
    all_teams  = set(ratings_df["team"].tolist())
    C          = _init_counters()

    log_every = max(1, n_simulations // 10)  # Imprimir progreso cada 10%
    t0        = time.time()
    logger.info(f"Iniciando {n_simulations:,} simulaciones (seed={seed})...")

    for i in range(n_simulations):
        # Una simulación completa del torneo
        tables               = simulate_all_groups(ratings_df)
        qualified, cls, elim = get_qualified(tables)
        # El qualified dict lleva el lookup precomputado internamente
        # simulate_knockout_stage lo reutiliza sin reconstruirlo
        qualified["_lookup"] = tables["_lookup"]
        result               = simulate_knockout_stage(qualified, ratings_df)
        _update(C, result, all_teams, cls, elim)

        # Imprimir progreso
        if verbose and (i + 1) % log_every == 0:
            pct = (i + 1) / n_simulations * 100
            logger.info(f"  {pct:.0f}% completado ({i+1:,} sims) — {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    logger.info(f"✅ {n_simulations:,} simulaciones completadas en {elapsed:.1f}s")

    # ─── Generar y guardar todos los outputs ──────────────────────────────────
    # Cada función de analysis.py convierte los contadores en un DataFrame limpio

    tp  = generate_probabilities(C, n_simulations, ratings_df)
    fin = generate_finals_table(C, n_simulations)
    gs  = generate_group_summary(ratings_df)
    t3  = generate_third_place_stats(C)
    pth = generate_path_to_title(C, n_simulations)
    var = generate_variance_table(tp)
    con = generate_conclusions(tp, fin, gs, t3, var)

    # Guardar CSV (el dashboard los lee directamente de disco)
    tp.to_csv(OUTPUTS_DIR  / "team_probabilities.csv", index=False)
    fin.to_csv(OUTPUTS_DIR / "finals.csv",             index=False)
    gs.to_csv(OUTPUTS_DIR  / "group_summary.csv",      index=False)
    t3.to_csv(OUTPUTS_DIR  / "third_place_stats.csv",  index=False)
    pth.to_csv(OUTPUTS_DIR / "path_to_title.csv",      index=False)
    var.to_csv(OUTPUTS_DIR / "variance_table.csv",     index=False)

    # Log de la simulación (para que el dashboard sepa cuántas sims hay)
    pd.DataFrame([
        {"metric": "n_simulations", "value": n_simulations},
        {"metric": "seed",          "value": seed},
        {"metric": "elapsed_sec",   "value": round(elapsed, 1)},
    ]).to_csv(OUTPUTS_DIR / "simulation_log.csv", index=False)

    # Conclusiones en texto plano
    open(OUTPUTS_DIR / "conclusions.txt", "w", encoding="utf-8").write(con)

    logger.info(f"Outputs guardados en {OUTPUTS_DIR}")

    return {
        "team_probabilities": tp, "finals": fin, "group_summary": gs,
        "third_place_stats": t3, "path_to_title": pth, "variance_table": var,
        "ratings": ratings_df,   "n_simulations": n_simulations,
    }


# ─── Punto de entrada CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Permite ejecutar directamente: python -m src.monte_carlo --simulations 5000
    parser = argparse.ArgumentParser(description="WC 2026 Monte Carlo Simulator — Aimar Esqueta")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help=f"Número de simulaciones (default: {DEFAULT_SIMULATIONS})")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suprimir logs de progreso")
    args = parser.parse_args()

    run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        verbose=not args.quiet,
    )
