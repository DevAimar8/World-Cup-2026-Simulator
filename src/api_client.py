"""
api_client.py
-------------
Cliente para la API "Free API Live Football Data" (RapidAPI).
Se encarga de toda la comunicación con la fuente de datos externa:
fixtures del Mundial, standings de grupos y estadísticas de partidos.

La API devuelve datos en vivo que se usan para actualizar el Power Score
y los ratings de cada selección antes de cada simulación.

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
"""

import logging
import time
from typing import Optional

import requests
import pandas as pd

from src.config import RAPIDAPI_KEY, RAPIDAPI_HOST, RAPIDAPI_BASE, DATA_DIR

logger = logging.getLogger(__name__)


# ─── Autenticación ────────────────────────────────────────────────────────────
def _headers() -> dict:
    """
    Construye los headers de autenticación que RapidAPI requiere en cada llamada.
    La key se lee del archivo .env para no exponerla en el código.
    """
    if not RAPIDAPI_KEY:
        raise EnvironmentError(
            "RAPIDAPI_KEY no encontrada.\n"
            "Crea un archivo .env en la raíz del proyecto con:\n"
            "RAPIDAPI_KEY=tu_key_aqui\n"
            "Consíguela gratis en: https://rapidapi.com/Creativesdev/api/free-api-live-football-data"
        )
    return {
        "x-rapidapi-key":  RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type":    "application/json",
    }


# ─── Llamada base con reintentos ──────────────────────────────────────────────
def _get(endpoint: str, params: dict = None, retries: int = 3) -> dict:
    """
    Ejecuta una petición GET a la API con reintentos automáticos en caso de error.
    Espera exponencialmente entre reintentos para no saturar la API.

    Args:
        endpoint: Ruta del endpoint (sin la base URL)
        params:   Parámetros de query string
        retries:  Número máximo de intentos

    Returns:
        Diccionario con la respuesta JSON de la API
    """
    url = f"{RAPIDAPI_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(),
                                params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Intento {attempt+1}/{retries} fallido ({endpoint}): {e}")
            if attempt < retries - 1:
                # Espera 1s, 2s, 4s... entre reintentos
                time.sleep(2 ** attempt)
    raise ConnectionError(f"API no responde tras {retries} intentos: {endpoint}")


# ─── Buscar ID del Mundial 2026 ───────────────────────────────────────────────
def fetch_wc_league_id() -> Optional[int]:
    """
    Busca el ID de la FIFA World Cup 2026 en la API.

    Cada liga tiene un ID único en la API. Necesitamos el de la WC2026
    para filtrar fixtures y standings correctamente.
    Si no lo encuentra, el pipeline usa datos estáticos.
    """
    try:
        data     = _get("football-get-all-leagues")
        leagues  = data.get("result", data.get("response", []))

        for league in leagues:
            name   = str(league.get("name", "")).lower()
            season = str(league.get("season", ""))
            # Buscamos la que contenga "world cup" y temporada 2026
            if "world cup" in name and "2026" in season:
                wc_id = league.get("league_id") or league.get("id")
                logger.info(f"✅ World Cup 2026 encontrada — ID: {wc_id}")
                return wc_id

        logger.warning("World Cup 2026 no encontrada en la API. Usando datos estáticos.")
        return None
    except Exception as e:
        logger.error(f"Error buscando ID del Mundial: {e}")
        return None


# ─── Partidos del Mundial ─────────────────────────────────────────────────────
def fetch_fixtures(league_id: int = None) -> pd.DataFrame:
    """
    Descarga todos los partidos de la World Cup 2026: jugados y pendientes.

    El estado del partido (status) indica si ya se jugó:
      - "FT"  → Final Time (terminado)
      - "NS"  → Not Started (pendiente)
      - "1H"  → Primer tiempo en juego
      - "HT"  → Descanso
      - "2H"  → Segundo tiempo en juego

    Los partidos con status FT/AET/PEN se usan para calcular la forma reciente.
    Los pendientes se usan para construir el bracket de eliminatorias.
    """
    logger.info(f"Descargando partidos del Mundial (league_id={league_id})...")

    try:
        data    = _get("football-get-matches-by-league", {"leagueId": league_id})
        matches = data.get("result", data.get("response", []))
    except Exception as e:
        logger.error(f"Error descargando fixtures: {e}")
        return pd.DataFrame()

    rows = []
    for m in matches:
        # La API puede usar estructuras ligeramente distintas según la versión
        # Normalizamos extrayendo los campos con múltiples nombres posibles
        home  = m.get("homeTeam", m.get("home_team", {}))
        away  = m.get("awayTeam", m.get("away_team", {}))
        score = m.get("score",    m.get("goals",     {}))

        rows.append({
            "fixture_id":  m.get("id") or m.get("fixture_id"),
            "date":        m.get("date") or m.get("datetime"),
            "status":      m.get("status", "NS"),
            "round":       m.get("round") or m.get("stage"),
            "home":        home.get("name") if isinstance(home, dict) else home,
            "away":        away.get("name") if isinstance(away, dict) else away,
            # Los goles pueden ser None si el partido aún no se jugó
            "home_goals":  _safe_int(score.get("home") if isinstance(score, dict) else m.get("home_score")),
            "away_goals":  _safe_int(score.get("away") if isinstance(score, dict) else m.get("away_score")),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Guardamos en disco para no repetir la llamada si el servidor se reinicia
        df.to_csv(DATA_DIR / "processed" / "fixtures_live.csv", index=False)
        played = df[df["status"].isin(["FT", "AET", "PEN"])].shape[0]
        logger.info(f"✅ {len(df)} partidos descargados · {played} ya jugados")
    return df


# ─── Tabla de grupos / Standings ─────────────────────────────────────────────
def fetch_standings(league_id: int = None) -> pd.DataFrame:
    """
    Obtiene la clasificación actual de grupos del Mundial.

    Los standings son clave para dos cosas:
    1. Calcular el rendimiento actual en el torneo (parte del Power Score)
    2. Identificar qué equipos ya están clasificados a eliminatorias

    La respuesta está organizada por grupos (A, B, C...) y dentro de cada
    grupo aparecen los 4 equipos ordenados por puntos.
    """
    logger.info(f"Descargando standings (league_id={league_id})...")

    try:
        data      = _get("football-league-standings", {"leagueId": league_id})
        standings = data.get("result", data.get("response", []))
    except Exception as e:
        logger.error(f"Error descargando standings: {e}")
        return pd.DataFrame()

    rows = []
    for group_data in standings:
        # Cada elemento puede ser un grupo completo o un equipo individual
        group_name = group_data.get("group", group_data.get("name", ""))
        teams      = group_data.get("standings", group_data.get("teams", [group_data]))
        if isinstance(teams, dict):
            teams = [teams]

        for team in teams:
            rows.append({
                "team":          team.get("team", {}).get("name", team.get("name", "")),
                "group":         group_name.replace("Group ", "").strip(),
                "played":        _safe_int(team.get("played")        or team.get("games_played")),
                "won":           _safe_int(team.get("won")           or team.get("wins")),
                "drawn":         _safe_int(team.get("drawn")         or team.get("draws")),
                "lost":          _safe_int(team.get("lost")          or team.get("losses")),
                "goals_for":     _safe_int(team.get("goals_for")     or team.get("gf")),
                "goals_against": _safe_int(team.get("goals_against") or team.get("ga")),
                "goal_diff":     _safe_int(team.get("goals_diff")    or team.get("gd")),
                "points":        _safe_int(team.get("points")        or team.get("pts")),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(DATA_DIR / "processed" / "teams_live.csv", index=False)
        logger.info(f"✅ {len(df)} equipos en la tabla de grupos")
    return df


# ─── Estadísticas de un partido concreto ─────────────────────────────────────
def fetch_match_stats(fixture_id: int) -> dict:
    """
    Obtiene las estadísticas detalladas de un partido ya jugado:
    posesión, tiros a puerta, tarjetas, corners, etc.

    Estas estadísticas son opcionales — se usan para enriquecer
    el análisis pero el modelo funciona sin ellas.
    """
    try:
        data = _get("football-get-match-by-id", {"matchId": fixture_id})
        return data.get("result", data.get("response", {}))
    except Exception as e:
        logger.error(f"Error en estadísticas del partido {fixture_id}: {e}")
        return {}


# ─── Actualización completa (punto de entrada) ───────────────────────────────
def refresh_all_data() -> dict:
    """
    Descarga todos los datos del Mundial en secuencia.
    Llamado automáticamente por el pipeline antes de cada simulación.

    Returns:
        Diccionario con resumen de lo descargado (para logs y debug)
    """
    logger.info("=" * 55)
    logger.info("  ACTUALIZANDO DATOS DEL MUNDIAL 2026")
    logger.info(f"  Fuente: {RAPIDAPI_HOST}")
    logger.info("=" * 55)

    results = {}

    # Paso 1: encontrar el ID de la liga
    wc_id = fetch_wc_league_id()
    results["wc_league_id"] = wc_id

    # Paso 2: descargar partidos
    try:
        fixtures_df = fetch_fixtures(wc_id)
        played = fixtures_df[fixtures_df["status"].isin(["FT","AET","PEN"])].shape[0] if not fixtures_df.empty else 0
        results["fixtures_total"]  = len(fixtures_df)
        results["fixtures_played"] = played
    except Exception as e:
        logger.error(f"Error en fixtures: {e}")
        results["fixtures_total"] = results["fixtures_played"] = 0

    # Paso 3: descargar tabla de grupos
    try:
        standings_df = fetch_standings(wc_id)
        results["teams_in_standings"] = len(standings_df)
    except Exception as e:
        logger.error(f"Error en standings: {e}")
        results["teams_in_standings"] = 0

    logger.info(f"Datos descargados: {results}")
    return results


# ─── Utilidad interna ─────────────────────────────────────────────────────────
def _safe_int(val) -> Optional[int]:
    """
    Convierte un valor a entero de forma segura.
    La API a veces devuelve strings, None o valores vacíos para goles
    de partidos no jugados — esta función los maneja sin lanzar excepciones.
    """
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None
