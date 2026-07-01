"""
run_pipeline.py
---------------
Pipeline completo:
  1. Descarga datos en vivo de API-Football
  2. Calcula Power Score actualizado
  3. Ejecuta 10.000 simulaciones Monte Carlo
  4. Guarda todos los outputs

Uso:
    python scripts/run_pipeline.py                    # 10.000 sims
    python scripts/run_pipeline.py --simulations 1000 # más rápido para test
    python scripts/run_pipeline.py --no-api           # sin API (datos estáticos)
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monte_carlo import run_monte_carlo
from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="WC2026 Full Pipeline")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--no-api",      action="store_true", help="Usar datos estáticos sin API")
    args = parser.parse_args()

    fixtures_df  = None
    standings_df = None

    # \\\\\\\\\\\
    # Paso 1: Datos en vivo (si API key disponible)
    # \\\\\\\\\\\
    if not args.no_api:
        try:
            from src.api_client import fetch_fixtures, fetch_teams_and_groups
            logger.info("Descargando datos en vivo de API-Football...")
            fixtures_df  = fetch_fixtures()
            standings_df = fetch_teams_and_groups()
            logger.info("✅ Datos en vivo cargados.")
        except EnvironmentError as e:
            logger.warning(f"Sin API key: {e}\nUsando datos estáticos.")
        except Exception as e:
            logger.warning(f"Error en API: {e}\nUsando datos estáticos.")

    # \\\\\\\\\\\
    # Paso 2: Simulación Monte Carlo
    # \\\\\\\\\\\
    logger.info(f"Iniciando pipeline con {args.simulations:,} simulaciones...")
    results = run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        fixtures_df=fixtures_df,
        standings_df=standings_df,
    )

    # \\\\\\\\\\\
    # Resumen final
    # \\\\\\\\\\\
    tp = results["team_probabilities"]
    logger.info("\n" + "=" * 50)
    logger.info("  TOP 5 FAVORITOS AL TÍTULO")
    logger.info("=" * 50)
    for _, row in tp.head(5).iterrows():
        logger.info(f"  {row['team']:<20} {row['champion_pct']:>6.2f}%  |  Power Score: {row['power_score']:.1f}")
    logger.info(f"\n✅ Pipeline completado. Outputs en /outputs/")
    logger.info("   Dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
