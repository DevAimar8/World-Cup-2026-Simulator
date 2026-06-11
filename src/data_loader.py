"""
data_loader.py
--------------
Carga, valida y une los datos de teams.csv y ratings.csv.
Devuelve un DataFrame limpio que sirve como entrada al simulador.
"""

import logging
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# \\\\\\\\\\\
# Columnas mínimas requeridas en cada archivo
# \\\\\\\\\\\

REQUIRED_TEAMS_COLS   = {"group", "team", "confederation", "is_host"}
REQUIRED_RATINGS_COLS = {"team", "overall_rating", "attack_rating", "defense_rating", "recent_form"}


def _validate_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    """Lanza ValueError si faltan columnas obligatorias."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} le faltan columnas: {missing}")


# \\\\\\\\\\\
# Carga de equipos y grupos
# \\\\\\\\\\\

def load_teams() -> pd.DataFrame:
    """Carga teams.csv y devuelve DataFrame validado."""
    path = DATA_DIR / "teams.csv"
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_TEAMS_COLS, "teams.csv")
    df["is_host"] = df["is_host"].astype(int)
    logger.debug(f"Equipos cargados: {len(df)}")
    return df


# \\\\\\\\\\\
# Carga de ratings deportivos
# \\\\\\\\\\\

def load_ratings() -> pd.DataFrame:
    """Carga ratings.csv y devuelve DataFrame validado."""
    path = DATA_DIR / "ratings.csv"
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_RATINGS_COLS, "ratings.csv")

    # Asegurar que los ratings estén en rango 0–99
    for col in ["overall_rating", "attack_rating", "defense_rating", "recent_form"]:
        df[col] = df[col].clip(0, 99)

    logger.debug(f"Ratings cargados: {len(df)}")
    return df


# \\\\\\\\\\\
# Construcción del DataFrame maestro del torneo
# \\\\\\\\\\\

def load_tournament_data() -> pd.DataFrame:
    """
    Une teams y ratings en un único DataFrame por equipo.
    Añade columna 'form_factor' normalizada para el modelo.
    """
    teams   = load_teams()
    ratings = load_ratings()

    df = teams.merge(ratings, on="team", how="left")

    # Equipos sin rating reciben la media del torneo (fallback robusto)
    for col in ["overall_rating", "attack_rating", "defense_rating", "recent_form"]:
        median_val = df[col].median()
        missing    = df[col].isna().sum()
        if missing > 0:
            logger.warning(f"{missing} equipos sin '{col}', usando mediana ({median_val:.1f})")
            df[col] = df[col].fillna(median_val)

    # Factor de forma normalizado entre 0.85 y 1.15
    df["form_factor"] = 0.85 + (df["recent_form"] / 99) * 0.30

    logger.info(f"Datos del torneo listos: {len(df)} equipos en {df['group'].nunique()} grupos.")
    return df


# \\\\\\\\\\\
# Utilidad: obtener datos de un equipo concreto
# \\\\\\\\\\\

def get_team_data(df: pd.DataFrame, team_name: str) -> pd.Series:
    """Devuelve la fila de un equipo por nombre. Lanza KeyError si no existe."""
    result = df[df["team"] == team_name]
    if result.empty:
        raise KeyError(f"Equipo '{team_name}' no encontrado en los datos.")
    return result.iloc[0]
