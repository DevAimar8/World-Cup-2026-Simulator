"""
api_client.py
-------------
Cliente para Free API Live Football Data (RapidAPI).
Host: free-api-live-football-data.p.rapidapi.com

Endpoints usados en el proyecto:
  /football-get-all-leagues           → ligas disponibles
  /football-league-standings          → tabla de grupos / clasificación
  /football-get-matches-by-league     → partidos por liga
  /football-get-match-by-id          → detalle de un partido
  /football-team-statistics           → estadísticas de equipo

Documentación: https://rapidapi.com/Creativesdev/api/free-api-live-football-data
"""

import logging
import time
from typing import Optional

import requests
import pandas as pd

from src.config import RAPIDAPI_KEY, DATA_DIR

logger = logging.getLogger(__name__)

# \\\\\\\\\\\
# Configuración de la API
# \\\\\\\\\\\

HOST = "free-api-live-football-data.p.rapidapi.com"
BASE = f"https://{HOST}"

# ID de la FIFA World Cup 2026 en esta API
# Se busca con /football-get-all-leagues filtrando por "World Cup"
WC2026_LEAGUE_ID = 1  # se actualiza automáticamente con fetch_wc_league_id()


def _headers() -> dict:
    """Headers de autenticación para todas las llamadas."""
    if not RAPIDAPI_KEY:
        raise EnvironmentError(
            "RAPIDAPI_KEY no encontrada.\n"
            "Añade en tu .env:\n"
            "RAPIDAPI_KEY=7ffa36a2a9msh8da15c6f6cb1f41p1186c0jsnf167f8a19e7c"
        )
    return {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": HOST,
        "Content-Type":    "application/json",
    }


def _get(endpoint: str, params: dict = None, retries: int = 3) -> dict:
    """Llamada GET con reintentos automáticos."""
    url = f"{BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(),
                                params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Intento {attempt+1}/{retries} fallido ({endpoint}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise ConnectionError(f"API no responde tras {retries} intentos: {endpoint}")


# \\\\\\\\\\\
# Buscar el ID de la World Cup 2026 en esta API
# \\\\\\\\\\\

def fetch_wc_league_id() -> Optional[int]:
    """
    Busca el ID de la FIFA World Cup 2026.
    Guarda el resultado para no gastar llamadas en búsquedas repetidas.
    """
    try:
        data = _get("football-get-all-leagues")
        leagues = data.get("result", data.get("response", []))
        for league in leagues:
            name = str(league.get("name", "")).lower()
            season = str(league.get("season", ""))
            if "world cup" in name and "2026" in season:
                wc_id = league.get("league_id") or league.get("id")
                logger.info(f"✅ World Cup 2026 ID encontrado: {wc_id}")
                return wc_id
        logger.warning("No se encontró la World Cup 2026 — usando ID por defecto")
        return None
    except Exception as e:
        logger.error(f"Error buscando league ID: {e}")
        return None


# \\\\\\\\\\\
# Partidos del Mundial (todos: jugados + pendientes)
# \\\\\\\\\\\

def fetch_fixtures(league_id: int = None) -> pd.DataFrame:
    """
    Obtiene todos los partidos de la World Cup 2026.
    Devuelve DataFrame con columnas estandarizadas para el modelo.
    """
    lid = league_id or WC2026_LEAGUE_ID
    logger.info(f"Descargando partidos (league_id={lid})...")

    try:
        data = _get("football-get-matches-by-league", {"leagueId": lid})
        matches = data.get("result", data.get("response", []))
    except Exception as e:
        logger.error(f"Error en fixtures: {e}")
        return pd.DataFrame()

    rows = []
    for m in matches:
        # La API puede devolver estructuras distintas — normalizamos
        home = m.get("homeTeam", m.get("home_team", {}))
        away = m.get("awayTeam", m.get("away_team", {}))
        score = m.get("score", m.get("goals", {}))

        rows.append({
            "fixture_id":  m.get("id") or m.get("fixture_id"),
            "date":        m.get("date") or m.get("datetime"),
            "status":      m.get("status", "NS"),
            "round":       m.get("round") or m.get("stage"),
            "home":        home.get("name") if isinstance(home, dict) else home,
            "away":        away.get("name") if isinstance(away, dict) else away,
            "home_goals":  _safe_int(score.get("home") if isinstance(score, dict) else m.get("home_score")),
            "away_goals":  _safe_int(score.get("away") if isinstance(score, dict) else m.get("away_score")),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(DATA_DIR / "processed" / "fixtures_live.csv", index=False)
        played = df[df["status"].isin(["FT","AET","PEN"])].shape[0] if not df.empty else 0
        logger.info(f"✅ {len(df)} partidos descargados ({played} jugados)")
    return df


# \\\\\\\\\\\
# Tabla de posiciones / grupos
# \\\\\\\\\\\

def fetch_standings(league_id: int = None) -> pd.DataFrame:
    """
    Obtiene la clasificación de grupos del Mundial.
    Normaliza la respuesta al esquema interno del proyecto.
    """
    lid = league_id or WC2026_LEAGUE_ID
    logger.info(f"Descargando standings (league_id={lid})...")

    try:
        data = _get("football-league-standings", {"leagueId": lid})
        standings = data.get("result", data.get("response", []))
    except Exception as e:
        logger.error(f"Error en standings: {e}")
        return pd.DataFrame()

    rows = []
    for group_data in standings:
        # Puede ser una lista de grupos o directamente equipos
        group_name = group_data.get("group", group_data.get("name", ""))
        teams = group_data.get("standings", group_data.get("teams", [group_data]))
        if isinstance(teams, dict):
            teams = [teams]

        for team in teams:
            rows.append({
                "team":          team.get("team", {}).get("name", team.get("name", "")),
                "group":         group_name.replace("Group ", "").strip(),
                "played":        _safe_int(team.get("played") or team.get("games_played")),
                "won":           _safe_int(team.get("won") or team.get("wins")),
                "drawn":         _safe_int(team.get("drawn") or team.get("draws")),
                "lost":          _safe_int(team.get("lost") or team.get("losses")),
                "goals_for":     _safe_int(team.get("goals_for") or team.get("gf")),
                "goals_against": _safe_int(team.get("goals_against") or team.get("ga")),
                "goal_diff":     _safe_int(team.get("goals_diff") or team.get("gd")),
                "points":        _safe_int(team.get("points") or team.get("pts")),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(DATA_DIR / "processed" / "teams_live.csv", index=False)
        logger.info(f"✅ {len(df)} equipos en standings")
    return df


# \\\\\\\\\\\
# Estadísticas detalladas de un partido
# \\\\\\\\\\\

def fetch_match_stats(fixture_id: int) -> dict:
    """Estadísticas de un partido concreto: posesión, tiros, tarjetas..."""
    try:
        data = _get("football-get-match-by-id", {"matchId": fixture_id})
        return data.get("result", data.get("response", {}))
    except Exception as e:
        logger.error(f"Error en match stats ({fixture_id}): {e}")
        return {}


# \\\\\\\\\\\
# Actualización completa de datos
# \\\\\\\\\\\

def refresh_all_data() -> dict:
    """
    Descarga todos los datos del Mundial en una sola llamada.
    Devuelve resumen de lo obtenido.
    """
    logger.info("=" * 50)
    logger.info("ACTUALIZANDO DATOS DEL MUNDIAL 2026")
    logger.info(f"API: {HOST}")
    logger.info("=" * 50)

    results = {}

    # Buscar ID del Mundial
    wc_id = fetch_wc_league_id()
    results["wc_league_id"] = wc_id

    # Partidos
    try:
        fixtures_df = fetch_fixtures(wc_id)
        played = fixtures_df[fixtures_df["status"].isin(["FT","AET","PEN"])].shape[0] if not fixtures_df.empty else 0
        results["fixtures_total"]  = len(fixtures_df)
        results["fixtures_played"] = played
    except Exception as e:
        logger.error(f"Error fixtures: {e}")
        results["fixtures_total"] = results["fixtures_played"] = 0

    # Standings
    try:
        standings_df = fetch_standings(wc_id)
        results["teams_in_standings"] = len(standings_df)
    except Exception as e:
        logger.error(f"Error standings: {e}")
        results["teams_in_standings"] = 0

    logger.info(f"Resumen: {results}")
    return results


# \\\\\\\\\\\
# Utilidad
# \\\\\\\\\\\

def _safe_int(val) -> Optional[int]:
    """Convierte a int de forma segura. Devuelve None si no es posible."""
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None
