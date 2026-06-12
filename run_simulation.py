"""
run_simulation.py
-----------------
Punto de entrada principal. Ejecuta la simulación Monte Carlo completa
con el modelo Dixon-Coles y genera todos los outputs.

Uso:
    python run_simulation.py
    python run_simulation.py --simulations 10000 --seed 42
    python run_simulation.py --simulations 5000 --exclude "Argentina"
"""

import argparse
import logging
from src.monte_carlo import run_monte_carlo
from src.visualizations import generate_all_charts
from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Monte Carlo Simulator")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--exclude",     type=str, default=None,
                        help="Simular sin un equipo concreto")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  WORLD CUP 2026 — MONTE CARLO SIMULATOR")
    logger.info("  Modelo: Dixon-Coles + factores contextuales")
    logger.info("=" * 60)

    results = run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        exclude_team=args.exclude,
    )

    logger.info("Generando gráficos...")
    generate_all_charts(
        results["team_probabilities"],
        results["group_summary"],
        results["third_place_stats"],
        results["variance_table"],
    )

    # Resumen TOP 5
    tp = results["team_probabilities"]
    logger.info("\nTOP 5 FAVORITOS AL TÍTULO:")
    for _, row in tp.head(5).iterrows():
        logger.info(f"  {row['team']:<20} Campeón: {row['champion_pct']:>5.2f}%  "
                    f"Semis: {row['reach_semis_pct']:>5.1f}%  "
                    f"Pasa grupos: {row['pass_group_stage_pct']:>5.1f}%")

    # Resumen terceros
    t3 = results["third_place_stats"]
    summary = t3[t3["categoria"].str.startswith("RESUMEN")] if not t3.empty else None
    if summary is not None and not summary.empty:
        logger.info("\nTERCEROS CLASIFICADOS:")
        for _, r in summary.iterrows():
            label = r["categoria"].replace("RESUMEN - ", "")
            logger.info(f"  {label}: {r['puntos']} puntos")

    logger.info("\n✅ Completado. Revisa la carpeta outputs/")
    logger.info("   Dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
