"""
power_score.py
--------------
Calcula el Power Score de cada selección y construye el rating compuesto
que usa el modelo Dixon-Coles para simular partidos.

El Power Score es un indicador propio que mide la fortaleza relativa
actual de cada selección combinando rendimiento reciente y datos del torneo.

Rating compuesto final = 35% ELO + 30% Power Score + 20% FIFA Ranking + 15% Forma

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_DIR, W_ELO, W_POWER, W_FIFA, W_FORM

logger = logging.getLogger(__name__)


# ─── Datos base de las 48 selecciones ────────────────────────────────────────
# Fuentes: eloratings.net + FIFA Rankings junio 2026
# Formato: team → (elo, fifa_ranking, attack_coef, defense_coef, confederation)
#
# attack_coef:  cuántos goles genera el equipo respecto a la media (>1 = ofensivo)
# defense_coef: cuántos goles concede respecto a la media (>1 = defensa porosa)
# Estos coeficientes se usan directamente en la fórmula Dixon-Coles
BASE_RATINGS = {
    "Argentina":          (1990, 1,   1.45, 0.67, "CONMEBOL"),
    "Spain":              (1985, 2,   1.43, 0.68, "UEFA"),
    "France":             (1980, 3,   1.41, 0.69, "UEFA"),
    "England":            (1965, 4,   1.36, 0.71, "UEFA"),
    "Brazil":             (1970, 5,   1.36, 0.71, "CONMEBOL"),
    "Portugal":           (1960, 6,   1.37, 0.73, "UEFA"),
    "Morocco":            (1930, 7,   1.30, 0.71, "CAF"),
    "Netherlands":        (1935, 8,   1.29, 0.73, "UEFA"),
    "Germany":            (1940, 9,   1.29, 0.73, "UEFA"),
    "Croatia":            (1900, 10,  1.22, 0.78, "UEFA"),
    "Belgium":            (1895, 11,  1.23, 0.80, "UEFA"),
    "Colombia":           (1895, 13,  1.24, 0.81, "CONMEBOL"),
    "Mexico":             (1885, 14,  1.21, 0.82, "CONCACAF"),
    "USA":                (1855, 16,  1.19, 0.83, "CONCACAF"),
    "Uruguay":            (1905, 17,  1.22, 0.79, "CONMEBOL"),
    "Switzerland":        (1850, 19,  1.18, 0.82, "UEFA"),
    "Iran":               (1830, 20,  1.16, 0.82, "AFC"),
    "Japan":              (1845, 21,  1.20, 0.82, "AFC"),
    "Turkey":             (1820, 22,  1.16, 0.83, "UEFA"),
    "Senegal":            (1835, 23,  1.18, 0.83, "CAF"),
    "Norway":             (1830, 24,  1.17, 0.83, "UEFA"),
    "Austria":            (1820, 25,  1.17, 0.84, "UEFA"),
    "Ecuador":            (1800, 29,  1.15, 0.85, "CONMEBOL"),
    "Australia":          (1800, 24,  1.14, 0.85, "AFC"),
    "Sweden":             (1810, 38,  1.14, 0.84, "UEFA"),
    "Scotland":           (1785, 31,  1.12, 0.86, "UEFA"),
    "South Korea":        (1830, 23,  1.17, 0.84, "AFC"),
    "Algeria":            (1775, 32,  1.12, 0.86, "CAF"),
    "Paraguay":           (1710, 47,  1.07, 0.90, "CONMEBOL"),
    "Bosnia Herzegovina": (1705, 65,  1.07, 0.90, "UEFA"),
    "Egypt":              (1755, 36,  1.10, 0.88, "CAF"),
    "Tunisia":            (1740, 44,  1.09, 0.87, "CAF"),
    "Ivory Coast":        (1780, 34,  1.13, 0.87, "CAF"),
    "Ghana":              (1735, 53,  1.09, 0.89, "CAF"),
    "Saudi Arabia":       (1715, 56,  1.07, 0.90, "AFC"),
    "Czech Republic":     (1750, 39,  1.10, 0.87, "UEFA"),
    "DR Congo":           (1690, 67,  1.04, 0.91, "CAF"),
    "Panama":             (1655, 79,  1.00, 0.91, "CONCACAF"),
    "Uzbekistan":         (1645, 68,  0.99, 0.93, "AFC"),
    "Jordan":             (1615, 84,  0.95, 0.95, "AFC"),
    "Iraq":               (1625, 63,  0.96, 0.95, "AFC"),
    "New Zealand":        (1610, 91,  0.92, 0.96, "OFC"),
    "South Africa":       (1635, 64,  0.97, 0.94, "CAF"),
    "Haiti":              (1570, 102, 0.88, 0.99, "CONCACAF"),
    "Curacao":            (1545, 100, 0.86, 1.01, "CONCACAF"),
    "Cape Verde":         (1615, 75,  0.95, 0.95, "CAF"),
    "Canada":             (1800, 47,  1.13, 0.85, "CONCACAF"),
    "Qatar":              (1610, 37,  0.93, 0.96, "AFC"),
}

# ─── Grupos oficiales del Mundial 2026 ───────────────────────────────────────
# Sorteo realizado en diciembre de 2025 en el Kennedy Center (Washington DC)
GROUPS = {
    "A": ["Mexico",      "South Korea",  "South Africa",       "Czech Republic"],
    "B": ["Canada",      "Bosnia Herzegovina", "Qatar",         "Switzerland"],
    "C": ["Brazil",      "Morocco",      "Haiti",              "Scotland"],
    "D": ["USA",         "Paraguay",     "Australia",          "Turkey"],
    "E": ["Germany",     "Ecuador",      "Ivory Coast",        "Curacao"],
    "F": ["Netherlands", "Japan",        "Sweden",             "Tunisia"],
    "G": ["Belgium",     "Egypt",        "Iran",               "New Zealand"],
    "H": ["Spain",       "Uruguay",      "Saudi Arabia",       "Cape Verde"],
    "I": ["France",      "Senegal",      "Norway",             "Iraq"],
    "J": ["Argentina",   "Austria",      "Algeria",            "Jordan"],
    "K": ["Portugal",    "Colombia",     "Uzbekistan",         "DR Congo"],
    "L": ["England",     "Croatia",      "Ghana",              "Panama"],
}

# Los tres países anfitriones reciben un pequeño bonus de localía
HOSTS = ["USA", "Mexico", "Canada"]


# ─── Normalización del ELO a escala 0-100 ────────────────────────────────────
def _normalize_elo(elo_series: pd.Series) -> pd.Series:
    """
    Convierte el ELO (rango ~1500-2100) a una escala de 0 a 100.
    ELO 2100+ → 100 puntos, ELO 1500 → 0 puntos.
    Se usa clip para que ningún valor salga del rango.
    """
    return ((elo_series - 1500) / 600 * 100).clip(0, 100)


# ─── Normalización del FIFA Ranking a escala 0-100 ───────────────────────────
def _normalize_fifa(fifa_series: pd.Series) -> pd.Series:
    """
    Invierte el ranking FIFA para convertirlo en una puntuación.
    Ranking #1 → 100 puntos, Ranking #200 → 0 puntos.
    El ranking es un número donde menor = mejor, así que lo invertimos.
    """
    return ((1 - (fifa_series - 1) / 199) * 100).clip(0, 100)


# ─── Cálculo de la forma reciente desde fixtures en vivo ─────────────────────
def _compute_recent_form(fixtures_df: pd.DataFrame, team: str, last: int = 10) -> float:
    """
    Calcula la forma reciente de un equipo basándose en sus últimos N partidos.

    Sistema de puntuación por partido:
      - Victoria:  3 puntos base
      - Empate:    1 punto base
      - Derrota:   0 puntos base
      - Bonus/malus de goles: +0.1 por gol a favor, -0.05 por gol en contra

    El resultado se normaliza a una escala 0-100 donde 50 es el valor neutral.
    Si no hay datos de la API, devuelve 50 (sin información = valor neutral).

    Args:
        fixtures_df: DataFrame con todos los partidos descargados de la API
        team:        Nombre del equipo a evaluar
        last:        Número de partidos a considerar (por defecto 10)
    """
    if fixtures_df is None or fixtures_df.empty:
        return 50.0  # Sin datos → valor neutral, no penalizamos ni bonificamos

    # Filtrar partidos del equipo que ya se hayan jugado (status = FT)
    played = fixtures_df[
        ((fixtures_df["home"] == team) | (fixtures_df["away"] == team)) &
        (fixtures_df["status"].isin(["FT", "AET", "PEN"]))
    ].tail(last)

    if played.empty:
        return 50.0

    total_score = 0
    max_score   = 0

    for _, row in played.iterrows():
        is_home = row["home"] == team
        # Asignamos goles según si jugó de local o visitante
        gf = (row["home_goals"] if is_home else row["away_goals"]) or 0
        ga = (row["away_goals"] if is_home else row["home_goals"]) or 0

        # Determinar resultado: victoria, empate o derrota
        if gf > ga:   pts = 3
        elif gf == ga: pts = 1
        else:          pts = 0

        # Puntuación del partido incluyendo el bonus de goles
        match_score  = pts + 0.1 * gf - 0.05 * ga
        total_score += max(match_score, 0)  # nunca negativos
        max_score   += 3.5  # máximo teórico por partido

    # Normalizar al rango 0-100
    return min((total_score / max_score) * 100, 100) if max_score > 0 else 50.0


# ─── Construcción del DataFrame de ratings ───────────────────────────────────
def build_ratings_df(
    fixtures_df: pd.DataFrame = None,
    standings_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame maestro con el rating compuesto de las 48 selecciones.

    Combina cuatro fuentes de datos con los pesos definidos en config.py:
      - ELO Rating normalizado        (35%)
      - Power Score propio            (30%)
      - FIFA Ranking normalizado      (20%)
      - Forma reciente                (15%)

    Si hay datos en vivo de la API (fixtures_df, standings_df), los usa para
    actualizar el Power Score con información del torneo actual.
    Si no hay datos de API (--no-api), usa valores base históricos.

    El DataFrame resultante es el input de todos los módulos de simulación.
    """
    rows = []
    # Crear mapa inverso: equipo → letra del grupo
    team_to_group = {t: g for g, teams in GROUPS.items() for t in teams}

    for team, (elo, fifa_rank, atk, dfc, conf) in BASE_RATINGS.items():

        # Normalizar las fuentes principales
        elo_norm  = float(np.clip((elo - 1500) / 600 * 100, 0, 100))
        fifa_norm = float(np.clip((1 - (fifa_rank - 1) / 199) * 100, 0, 100))

        # Forma reciente: desde la API si está disponible, si no valor neutral
        form_score = _compute_recent_form(fixtures_df, team) if fixtures_df is not None else 50.0

        # Rendimiento actual en el torneo (solo disponible cuando el Mundial ya empezó)
        tournament_perf = 50.0
        if standings_df is not None and not standings_df.empty and team in standings_df["team"].values:
            row_s  = standings_df[standings_df["team"] == team].iloc[0]
            played = row_s.get("played", 0)
            if played > 0:
                # Calculamos puntos por partido y ajustamos con diferencia de gol
                pts_per_game    = row_s.get("points", 0) / played
                gd              = row_s.get("goal_diff", 0)
                # Puntos por partido vale 80%, diferencia de gol vale hasta 20%
                tournament_perf = min((pts_per_game / 3 * 80) + min(gd * 2, 20), 100)

        # Power Score: combina forma reciente (60%) y rendimiento en el torneo (40%)
        # El 40% de torneo solo tiene efecto cuando hay partidos jugados
        power_score = 0.6 * form_score + 0.4 * tournament_perf

        # Rating compuesto final ponderado (suma de pesos = 1.0)
        composite = (
            W_ELO   * elo_norm    +
            W_POWER * power_score +
            W_FIFA  * fifa_norm   +
            W_FORM  * form_score
        )

        rows.append({
            "team":             team,
            "group":            team_to_group.get(team, "?"),
            "confederation":    conf,
            "is_host":          int(team in HOSTS),
            # Fuentes individuales (para transparencia en el dashboard)
            "elo_rating":       elo,
            "elo_norm":         round(elo_norm, 1),
            "fifa_ranking":     fifa_rank,
            "fifa_norm":        round(fifa_norm, 1),
            "form_score":       round(form_score, 1),
            "power_score":      round(power_score, 1),
            # Rating final que usa el motor de simulación
            "composite_rating": round(composite, 1),
            "overall_rating":   round(composite, 1),
            # Coeficientes Dixon-Coles calibrados manualmente
            "attack_coef":      atk,
            "defense_coef":     dfc,
            # Factor de forma para el cálculo de lambdas (rango 0.85-1.15)
            "form_factor":      round(0.85 + (form_score / 100) * 0.30, 3),
        })

    df = (pd.DataFrame(rows)
          .sort_values("composite_rating", ascending=False)
          .reset_index(drop=True))

    # Guardar en disco para que el dashboard lo pueda leer sin recalcular
    df.to_csv(DATA_DIR / "processed" / "ratings.csv", index=False)
    logger.info(f"✅ Ratings calculados para {len(df)} selecciones")
    return df
