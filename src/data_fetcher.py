"""
data_fetcher.py
---------------
Extrae ratings reales de selecciones usando soccerdata (ClubElo + SoFIFA).
Los datos se fusionan con los equipos del Mundial y se guardan en ratings.csv.

Fuentes usadas:
  - ClubElo  → rating Elo histórico de clubes (proxy para selecciones)
  - SoFIFA   → overall, attack, midfield, defence de equipos de clubes
               (se agregan por confederación/país para estimar la selección)

Nota: soccerdata cubre clubes, no selecciones directamente.
      Usamos la media de los mejores clubes de cada país como proxy del
      nivel de la selección nacional. Este es un enfoque razonable y
      transparente que puede mejorarse con datos FIFA Ranking en el futuro.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.config import DATA_DIR, normalize_team_name

logger = logging.getLogger(__name__)

# \\\\\\\\\\\
# Mapeo país → nombres de clubes top en ClubElo / SoFIFA
# Permite calcular el rating de una selección a partir de sus clubes
# \\\\\\\\\\\

COUNTRY_CLUBS_MAP: dict = {
    "France":       ["PSG", "Monaco", "Lyon", "Marseille"],
    "Spain":        ["Real Madrid", "Barcelona", "Atletico", "Sevilla"],
    "England":      ["Man City", "Liverpool", "Arsenal", "Chelsea"],
    "Portugal":     ["Sporting CP", "Porto", "Benfica"],
    "Germany":      ["Bayern", "Dortmund", "Leverkusen", "Leipzig"],
    "Brazil":       ["Flamengo", "Palmeiras", "Atletico MG"],
    "Argentina":    ["River Plate", "Boca Juniors", "Racing Club"],
    "Netherlands":  ["Ajax", "PSV", "Feyenoord"],
    "Belgium":      ["Club Brugge", "Anderlecht", "Genk"],
    "Italy":        ["Inter", "Juventus", "Milan", "Napoli"],
    "Uruguay":      ["Nacional", "Penarol"],
    "Colombia":     ["Atletico Nacional", "Millonarios"],
    "Mexico":       ["America", "Chivas", "Cruz Azul"],
    "USA":          ["LA Galaxy", "NYCFC", "Seattle Sounders"],
    "Japan":        ["Kashima", "Urawa", "Yokohama FM"],
    "Morocco":      ["Wydad", "Raja Casablanca"],
    "Croatia":      ["Dinamo Zagreb", "Hajduk Split"],
    "Denmark":      ["Copenhagen", "Midtjylland"],
    "Switzerland":  ["Young Boys", "Basel", "Zurich"],
    "Senegal":      ["Jaraaf", "Generation Foot"],
    "South Korea":  ["Jeonbuk", "Ulsan", "Pohang"],
    "Ecuador":      ["LDU Quito", "Barcelona SC"],
    "Canada":       ["Toronto FC", "Vancouver Whitecaps", "CF Montreal"],
    "Serbia":       ["Red Star", "Partizan"],
    "Poland":       ["Legia", "Lech Poznan"],
    "Australia":    ["Melbourne City", "Sydney FC"],
    "Wales":        ["Cardiff", "Swansea"],
    "Cameroon":     ["Canon Yaounde", "Coton Sport"],
    "Tunisia":      ["Esperance", "Club Africain"],
    "Nigeria":      ["Enyimba", "Kano Pillars"],
    "Ghana":        ["Hearts of Oak", "Asante Kotoko"],
    "Saudi Arabia": ["Al Hilal", "Al Nassr", "Al Ahli"],
    "Iran":         ["Persepolis", "Esteghlal"],
    "Egypt":        ["Al Ahly", "Zamalek"],
    "Costa Rica":   ["Saprissa", "Alajuelense"],
    "Qatar":        ["Al Sadd", "Al Duhail"],
    "Algeria":      ["Mouloudia", "CR Belouizdad"],
    "Ivory Coast":  ["ASEC Mimosas", "Africa Sports"],
    "Mali":         ["Stade Malien", "Real Bamako"],
    "Panama":       ["Tauro FC", "CAI"],
    "Honduras":     ["Olimpia", "Real Espana"],
    "Jamaica":      ["Harbour View", "Arnett Gardens"],
    "New Zealand":  ["Auckland City", "Wellington Phoenix"],
    "Indonesia":    ["Persija", "Persib"],
    "Thailand":     ["Buriram United", "Muangthong"],
    "Venezuela":    ["Caracas FC", "Deportivo Tachira"],
    "Bolivia":      ["Bolivar", "The Strongest"],
    "Paraguay":     ["Olimpia", "Cerro Porteno"],
}


# \\\\\\\\\\\
# Fetch desde ClubElo: ratings Elo actuales de clubes
# \\\\\\\\\\\

def fetch_clubelo_ratings() -> Optional[pd.DataFrame]:
    """
    Descarga los ratings Elo actuales de todos los clubes vía ClubElo API.
    Devuelve un DataFrame con columnas [team, elo] o None si falla.
    """
    try:
        import soccerdata as sd
        logger.info("Descargando ratings Elo desde ClubElo...")
        elo = sd.ClubElo()
        df = elo.read_by_date()
        # Normalizar índice a columna y renombrar
        df = df.reset_index()[["team", "elo"]].copy()
        df["team"] = df["team"].apply(normalize_team_name)
        logger.info(f"ClubElo: {len(df)} clubes descargados.")
        return df
    except Exception as e:
        logger.warning(f"No se pudo obtener ClubElo: {e}")
        return None


# \\\\\\\\\\\
# Fetch desde SoFIFA: ratings globales de equipos (overall, attack, defence)
# \\\\\\\\\\\

def fetch_sofifa_ratings(league: str = "ESP-La Liga") -> Optional[pd.DataFrame]:
    """
    Descarga ratings de equipos de SoFIFA para una liga concreta.
    Devuelve columnas [team, overall, attack, defence] o None si falla.
    """
    try:
        import soccerdata as sd
        logger.info(f"Descargando ratings SoFIFA para {league}...")
        sofifa = sd.SoFIFA(leagues=league, versions="latest")
        df = sofifa.read_team_ratings()
        df = df.reset_index()
        # Renombrar columnas al esquema interno del proyecto
        df = df.rename(columns={
            "overall":  "sofifa_overall",
            "attack":   "sofifa_attack",
            "defence":  "sofifa_defence",
        })
        df["team"] = df["team"].apply(normalize_team_name)
        logger.info(f"SoFIFA {league}: {len(df)} equipos descargados.")
        return df[["team", "sofifa_overall", "sofifa_attack", "sofifa_defence"]]
    except Exception as e:
        logger.warning(f"No se pudo obtener SoFIFA ({league}): {e}")
        return None


# \\\\\\\\\\\
# Estimación de rating de selección nacional a partir de sus clubes
# \\\\\\\\\\\

def estimate_national_rating(
    country: str,
    elo_df: Optional[pd.DataFrame],
    sofifa_df: Optional[pd.DataFrame],
) -> dict:
    """
    Estima el rating de una selección nacional usando la media Elo
    y SoFIFA de sus clubes más representativos.

    Devuelve un dict con overall_rating, attack_rating, defense_rating,
    elo_rating y data_source.
    """
    clubs = COUNTRY_CLUBS_MAP.get(country, [])

    elo_scores = []
    sofifa_overall = []
    sofifa_attack = []
    sofifa_defence = []

    # Extraer Elo de los clubes del país
    if elo_df is not None and clubs:
        mask = elo_df["team"].isin(clubs)
        elo_scores = elo_df.loc[mask, "elo"].tolist()

    # Extraer SoFIFA de los clubes del país
    if sofifa_df is not None and clubs:
        mask = sofifa_df["team"].isin(clubs)
        sofifa_overall  = sofifa_df.loc[mask, "sofifa_overall"].tolist()
        sofifa_attack   = sofifa_df.loc[mask, "sofifa_attack"].tolist()
        sofifa_defence  = sofifa_df.loc[mask, "sofifa_defence"].tolist()

    has_elo    = len(elo_scores) > 0
    has_sofifa = len(sofifa_overall) > 0

    if has_elo and has_sofifa:
        # Combinar Elo (normalizado a 100) + SoFIFA
        elo_norm       = np.mean(elo_scores) / 20          # 2000 Elo ≈ 100
        overall        = round(elo_norm * 0.5 + np.mean(sofifa_overall) * 0.5)
        attack_rating  = round(elo_norm * 0.4 + np.mean(sofifa_attack)  * 0.6)
        defense_rating = round(elo_norm * 0.4 + np.mean(sofifa_defence) * 0.6)
        data_source    = "clubelo+sofifa"
    elif has_elo:
        elo_norm       = np.mean(elo_scores) / 20
        overall        = round(elo_norm)
        attack_rating  = round(elo_norm * 0.95)
        defense_rating = round(elo_norm * 0.95)
        data_source    = "clubelo_only"
    elif has_sofifa:
        overall        = round(np.mean(sofifa_overall))
        attack_rating  = round(np.mean(sofifa_attack))
        defense_rating = round(np.mean(sofifa_defence))
        data_source    = "sofifa_only"
    else:
        # Sin datos reales: se mantiene el valor manual base
        return {"data_source": "manual_base"}

    return {
        "overall_rating":  min(overall, 99),
        "attack_rating":   min(attack_rating, 99),
        "defense_rating":  min(defense_rating, 99),
        "elo_rating":      round(np.mean(elo_scores)) if has_elo else None,
        "data_source":     data_source,
    }


# \\\\\\\\\\\
# Función principal: actualiza ratings.csv con datos reales
# \\\\\\\\\\\

def fetch_and_update_ratings(force: bool = False) -> pd.DataFrame:
    """
    Descarga datos de soccerdata, estima ratings de selecciones y
    actualiza data/ratings.csv con los valores reales.

    Si force=False y ratings.csv ya tiene datos de fuentes reales,
    no vuelve a descargar.

    Devuelve el DataFrame de ratings resultante.
    """
    ratings_path = DATA_DIR / "ratings.csv"
    teams_path   = DATA_DIR / "teams.csv"

    # Cargar base manual
    ratings_df = pd.read_csv(ratings_path)
    teams_df   = pd.read_csv(teams_path)

    # Si ya hay datos reales y no se fuerza la actualización, salir
    if not force and "clubelo" in ratings_df.get("data_source", pd.Series()).str.cat():
        logger.info("ratings.csv ya contiene datos reales. Usa force=True para actualizar.")
        return ratings_df

    # Descargar fuentes externas
    elo_df     = fetch_clubelo_ratings()
    sofifa_df  = fetch_sofifa_ratings("ESP-La Liga")   # Liga de mayor cobertura

    updated_rows = []
    for _, row in ratings_df.iterrows():
        country = row["team"]
        estimates = estimate_national_rating(country, elo_df, sofifa_df)

        if estimates.get("data_source") != "manual_base":
            # Actualizar con datos reales (conservar recent_form manual)
            new_row = row.to_dict()
            new_row.update(estimates)
            updated_rows.append(new_row)
            logger.info(f"  {country}: actualizado ({estimates['data_source']})")
        else:
            updated_rows.append(row.to_dict())
            logger.info(f"  {country}: sin datos externos, se mantiene base manual")

    result_df = pd.DataFrame(updated_rows)
    result_df.to_csv(ratings_path, index=False)
    logger.info(f"ratings.csv actualizado con {len(result_df)} selecciones.")
    return result_df
