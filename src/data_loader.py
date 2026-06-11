"""
data_loader.py
--------------
Carga, valida y une teams.csv + ratings.csv.
Calcula el rating compuesto ponderado (Elo + FIFA Ranking + SoFIFA + forma)
y los coeficientes Dixon-Coles de ataque y defensa.
"""

import logging
import pandas as pd
import numpy as np
from src.config import DATA_DIR, ELO_WEIGHT, FIFA_RANKING_WEIGHT, SOFIFA_WEIGHT, FORM_DATA_WEIGHT

logger = logging.getLogger(__name__)

REQUIRED_TEAMS_COLS   = {"group", "team", "confederation", "is_host"}
REQUIRED_RATINGS_COLS = {"team", "overall_rating", "attack_rating", "defense_rating", "recent_form"}


def _validate_columns(df, required, filename):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} le faltan columnas: {missing}")


# \\\\\\\\\\\
# Cálculo del rating compuesto ponderado por múltiples fuentes
# \\\\\\\\\\\

def _compute_composite_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina overall_rating, elo_rating y fifa_ranking en un rating
    compuesto único usando los pesos definidos en config.json.

    El FIFA Ranking se invierte (ranking 1 = máximo) y normaliza a 0-99.
    El Elo se normaliza desde el rango real (~1500-2100) a 0-99.
    """
    df = df.copy()

    # Normalizar Elo a escala 0-99
    elo_min, elo_max = 1500.0, 2100.0
    if "elo_rating" in df.columns and df["elo_rating"].notna().any():
        df["elo_norm"] = ((df["elo_rating"].fillna(1700) - elo_min)
                          / (elo_max - elo_min) * 99).clip(0, 99)
    else:
        df["elo_norm"] = df["overall_rating"]

    # Normalizar FIFA Ranking a escala 0-99 (ranking 1 = 99, ranking 200 = 0)
    if "fifa_ranking" in df.columns and df["fifa_ranking"].notna().any():
        df["fifa_norm"] = (1.0 - (df["fifa_ranking"].fillna(100) - 1) / 199.0) * 99
        df["fifa_norm"] = df["fifa_norm"].clip(0, 99)
    else:
        df["fifa_norm"] = df["overall_rating"]

    # Rating SoFIFA (overall_rating actúa como proxy si no hay datos reales)
    df["sofifa_norm"] = df["overall_rating"].clip(0, 99)

    # Forma reciente normalizada
    df["form_norm"] = df["recent_form"].clip(0, 99)

    # Rating compuesto ponderado
    df["composite_rating"] = (
        ELO_WEIGHT          * df["elo_norm"]    +
        FIFA_RANKING_WEIGHT * df["fifa_norm"]   +
        SOFIFA_WEIGHT       * df["sofifa_norm"] +
        FORM_DATA_WEIGHT    * df["form_norm"]
    ).clip(0, 99)

    return df


# \\\\\\\\\\\
# Validación y recalibración de coeficientes Dixon-Coles
# \\\\\\\\\\\

def _ensure_dixon_coefs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si attack_coef o defense_coef no están en el CSV, los estima
    a partir del attack_rating y defense_rating normalizados.
    Rango esperado: attack_coef ∈ [0.7, 1.6], defense_coef ∈ [0.7, 1.1]
    """
    df = df.copy()
    if "attack_coef" not in df.columns or df["attack_coef"].isna().all():
        df["attack_coef"]  = 0.7 + (df["attack_rating"]  / 99) * 0.9
    if "defense_coef" not in df.columns or df["defense_coef"].isna().all():
        # defense_coef bajo = buena defensa (dificulta goles al rival)
        df["defense_coef"] = 1.1 - (df["defense_rating"] / 99) * 0.4

    df["attack_coef"]  = df["attack_coef"].clip(0.6, 1.7)
    df["defense_coef"] = df["defense_coef"].clip(0.65, 1.15)
    return df


# \\\\\\\\\\\
# Carga principal del torneo
# \\\\\\\\\\\

def load_tournament_data() -> pd.DataFrame:
    """
    Une teams y ratings, calcula el rating compuesto ponderado,
    asegura los coeficientes Dixon-Coles y añade form_factor normalizado.
    """
    teams_df   = pd.read_csv(DATA_DIR / "teams.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")

    _validate_columns(teams_df,   REQUIRED_TEAMS_COLS,   "teams.csv")
    _validate_columns(ratings_df, REQUIRED_RATINGS_COLS, "ratings.csv")

    df = teams_df.merge(ratings_df, on="team", how="left")

    # Rellenar valores faltantes con la mediana del torneo
    for col in ["overall_rating", "attack_rating", "defense_rating", "recent_form"]:
        median_val = df[col].median()
        n_missing  = df[col].isna().sum()
        if n_missing:
            logger.warning(f"{n_missing} equipos sin '{col}', usando mediana ({median_val:.1f})")
            df[col] = df[col].fillna(median_val)

    df = _compute_composite_rating(df)
    df = _ensure_dixon_coefs(df)

    # Usar el rating compuesto como overall_rating efectivo del modelo
    df["overall_rating"] = df["composite_rating"]

    # Factor de forma normalizado [0.85, 1.15]
    df["form_factor"] = 0.85 + (df["recent_form"] / 99) * 0.30
    df["is_host"]     = df["is_host"].astype(int)

    logger.info(f"Torneo listo: {len(df)} equipos en {df['group'].nunique()} grupos.")
    return df


def get_team_data(df: pd.DataFrame, team_name: str) -> pd.Series:
    result = df[df["team"] == team_name]
    if result.empty:
        raise KeyError(f"Equipo '{team_name}' no encontrado.")
    return result.iloc[0]
