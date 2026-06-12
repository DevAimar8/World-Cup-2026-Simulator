"""
run_data_fetch.py
-----------------
Descarga datos reales de selecciones usando soccerdata (ClubElo + SoFIFA)
y actualiza data/ratings.csv con los valores obtenidos.

Ejecutar antes de run_simulation.py para usar datos reales.

Uso:
    python run_data_fetch.py
    python run_data_fetch.py --force   # fuerza re-descarga aunque ya existan datos
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Descarga datos reales con soccerdata")
    parser.add_argument("--force", action="store_true",
                        help="Re-descarga aunque ratings.csv ya contenga datos reales")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("  DESCARGA DE DATOS CON SOCCERDATA")
    logger.info("=" * 55)
    logger.info("Fuentes: ClubElo (Elo ratings) + SoFIFA (overall/attack/defence)")
    logger.info("Los datos se cachan localmente en ~/soccerdata/")

    # \\\\\\\\\\\
    # Importación diferida para dar mejor mensaje de error si soccerdata no está
    # \\\\\\\\\\\
    try:
        from src.data_fetcher import fetch_and_update_ratings
    except ImportError as e:
        logger.error(f"Error de importación: {e}")
        logger.error("Asegúrate de haber ejecutado: pip install -r requirements.txt")
        return

    # \\\\\\\\\\\
    # Ejecución de la descarga y actualización de ratings
    # \\\\\\\\\\\
    try:
        ratings_df = fetch_and_update_ratings(force=args.force)
        n_real = (ratings_df["data_source"] != "manual_base").sum()
        n_total = len(ratings_df)

        logger.info(f"\nResultado:")
        logger.info(f"  Total selecciones: {n_total}")
        logger.info(f"  Con datos reales:  {n_real}")
        logger.info(f"  Con datos manuales:{n_total - n_real}")

        if n_real == 0:
            logger.warning(
                "\n⚠ No se obtuvieron datos reales. Posibles causas:\n"
                "  - Sin conexión a internet\n"
                "  - ClubElo o SoFIFA no disponibles temporalmente\n"
                "  - Los nombres de clubes del mapa no coinciden con la fuente\n"
                "El simulador usará los ratings manuales base, que son funcionales."
            )
        else:
            logger.info(f"\n✅ ratings.csv actualizado con datos reales de {n_real} selecciones.")

    except Exception as e:
        logger.error(f"Error durante la descarga: {e}")
        logger.info("El simulador puede ejecutarse igualmente con los ratings manuales.")


if __name__ == "__main__":
    main()
