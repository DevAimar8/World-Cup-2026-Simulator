"""
run_simulation.py
-----------------
Punto de entrada principal del proyecto.
Ejecuta la simulación Monte Carlo con los parámetros por defecto
y genera todos los outputs en la carpeta outputs/.

Uso:
    python run_simulation.py
    python run_simulation.py --simulations 5000 --seed 99
"""

import argparse
import logging

from src.monte_carlo import run_monte_carlo
from src.visualizations import generate_all_charts
from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    # \\\\\\\\\\\
    # Argumentos de línea de comandos
    # \\\\\\\\\\\
    parser = argparse.ArgumentParser(description="World Cup Monte Carlo Simulator")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help=f"Número de simulaciones (default: {DEFAULT_SIMULATIONS})")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help=f"Semilla aleatoria (default: {RANDOM_SEED})")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("  WORLD CUP 2026 - MONTE CARLO SIMULATOR")
    logger.info("=" * 55)

    # \\\\\\\\\\\
    # Ejecución de simulaciones y guardado de resultados
    # \\\\\\\\\\\
    results = run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
    )

    # \\\\\\\\\\\
    # Generación de gráficos
    # \\\\\\\\\\\
    logger.info("Generando gráficos...")
    chart_paths = generate_all_charts(
        results["team_probabilities"],
        results["group_summary"],
    )
    for path in chart_paths:
        logger.info(f"  Gráfico guardado: {path.name}")

    # \\\\\\\\\\\
    # Resumen de los 5 favoritos al título
    # \\\\\\\\\\\
    top5 = results["team_probabilities"].head(5)
    logger.info("\nTOP 5 FAVORITOS AL TÍTULO:")
    for _, row in top5.iterrows():
        logger.info(f"  {row['team']:<20} {row['champion_probability']:>6.2f}%")

    logger.info("\n✅ Ejecución completa. Revisa la carpeta outputs/.")
    logger.info("   Para el dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
