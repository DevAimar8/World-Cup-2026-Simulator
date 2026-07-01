"""
run_pipeline.py
---------------
Punto de entrada principal del proyecto. Ejecuta el pipeline completo:

  1. Descarga datos en vivo de la API (fixtures, standings del Mundial)
  2. Calcula el Power Score actualizado con los resultados reales
  3. Ejecuta 10.000 simulaciones Monte Carlo con Dixon-Coles
  4. Guarda todos los CSVs de outputs para el dashboard

Uso:
    python scripts/run_pipeline.py                     # Con datos de API
    python scripts/run_pipeline.py --no-api            # Sin API (datos estáticos)
    python scripts/run_pipeline.py --simulations 1000  # Simulación rápida para test

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
"""

import sys
import argparse
import logging
from pathlib import Path

# Añadir la raíz del proyecto al path para que los imports funcionen
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monte_carlo import run_monte_carlo
from src.config import DEFAULT_SIMULATIONS, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="FIFA WC 2026 Prediction Model — Aimar Esqueta"
    )
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS,
                        help=f"Número de simulaciones Monte Carlo (default: {DEFAULT_SIMULATIONS})")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--no-api",      action="store_true",
                        help="Omitir la llamada a la API y usar datos estáticos")
    args = parser.parse_args()

    fixtures_df  = None
    standings_df = None

    # ─── Paso 1: Datos en vivo ────────────────────────────────────────────────
    if not args.no_api:
        try:
            from src.api_client import fetch_fixtures, fetch_standings, fetch_wc_league_id
            logger.info("Conectando con API-Football para datos en vivo...")

            # Buscar el ID del Mundial 2026 en la API
            wc_id = fetch_wc_league_id()

            if wc_id:
                # Descargar partidos y tabla de grupos
                fixtures_df  = fetch_fixtures(wc_id)
                standings_df = fetch_standings(wc_id)
                logger.info("✅ Datos en vivo cargados correctamente.")
            else:
                logger.warning("No se encontró el ID del Mundial. Usando datos estáticos.")

        except EnvironmentError as e:
            # Sin API key configurada → pasar a datos estáticos
            logger.warning(f"API key no configurada: {e}\nUsando datos estáticos.")
        except Exception as e:
            # Cualquier otro error de red → pasar a datos estáticos
            logger.warning(f"Error en la API: {e}\nUsando datos estáticos.")
    else:
        logger.info("Modo --no-api: usando datos base históricos.")

    # ─── Paso 2: Simulación Monte Carlo ───────────────────────────────────────
    logger.info(f"Iniciando pipeline con {args.simulations:,} simulaciones...")
    results = run_monte_carlo(
        n_simulations=args.simulations,
        seed=args.seed,
        fixtures_df=fixtures_df,
        standings_df=standings_df,
    )

    # ─── Resumen final ────────────────────────────────────────────────────────
    tp = results["team_probabilities"]
    logger.info("\n" + "=" * 55)
    logger.info("  TOP 5 FAVORITOS AL TÍTULO")
    logger.info("=" * 55)
    for _, row in tp.head(5).iterrows():
        logger.info(
            f"  {row['team']:<22} {row['champion_pct']:>6.2f}%  "
            f"Power Score: {row['power_score']:.1f}"
        )
    logger.info(f"\n✅ Pipeline completado. Outputs en /outputs/")
    logger.info("   Lanza el dashboard con: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
