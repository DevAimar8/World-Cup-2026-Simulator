"""
config.py
---------
Configuración central del proyecto. Lee variables de entorno del archivo .env
y expone constantes usadas por todos los módulos.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# \\\\\\\\\\\
# Rutas base
# \\\\\\\\\\\

ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# \\\\\\\\\\\
# Variables de entorno
# \\\\\\\\\\\

load_dotenv(ROOT_DIR / ".env")

RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
WC_2026_ID:   int = int(os.getenv("WC_2026_ID", "1096"))

# \\\\\\\\\\\
# Parámetros del modelo matemático
# \\\\\\\\\\\

# Dixon-Coles
BASE_GOALS:      float = 1.15
DIXON_COLES_RHO: float = 0.13
PENALTY_ALPHA:   float = 2.5

# Rating compuesto — pesos de cada fuente
W_ELO:    float = 0.35   # ELO Rating histórico
W_POWER:  float = 0.30   # Power Score (rendimiento reciente)
W_FIFA:   float = 0.20   # FIFA Ranking oficial
W_FORM:   float = 0.15   # Forma últimos 10 partidos

# Factores contextuales
HOME_ADV:        float = 1.04   # Ventaja anfitrión en Mundial
FORM_WEIGHT:     float = 0.08
RATING_SCALE:    float = 200

# Fuerza histórica por confederación
CONFEDERATION_STRENGTH: dict = {
    "UEFA":     1.00,
    "CONMEBOL": 0.97,
    "CONCACAF": 0.78,
    "CAF":      0.76,
    "AFC":      0.75,
    "OFC":      0.62,
}

# \\\\\\\\\\\
# Parámetros de simulación
# \\\\\\\\\\\

DEFAULT_SIMULATIONS: int = 10000
RANDOM_SEED:         int = 42

# \\\\\\\\\\\
# API-Football endpoints
# \\\\\\\\\\\

RAPIDAPI_HOST: str = "api-football-v1.p.rapidapi.com"
RAPIDAPI_BASE: str = f"https://{RAPIDAPI_HOST}/v3"

ENDPOINTS: dict = {
    "teams":    f"{RAPIDAPI_BASE}/teams",
    "fixtures": f"{RAPIDAPI_BASE}/fixtures",
    "standings":f"{RAPIDAPI_BASE}/standings",
    "players":  f"{RAPIDAPI_BASE}/players",
    "odds":     f"{RAPIDAPI_BASE}/odds",
}
