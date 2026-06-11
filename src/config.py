import json
from pathlib import Path

# \\\\\\\\\\\
# Rutas base del proyecto
# \\\\\\\\\\\

ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# \\\\\\\\\\\
# Carga de parámetros desde config.json
# \\\\\\\\\\\

def load_config() -> dict:
    with open(DATA_DIR / "config.json", "r") as f:
        return json.load(f)

_cfg = load_config()

# \\\\\\\\\\\
# Parámetros globales del modelo
# \\\\\\\\\\\

DEFAULT_SIMULATIONS:    int   = _cfg["default_simulations"]
RANDOM_SEED:            int   = _cfg["random_seed"]
BASE_GOALS:             float = _cfg["base_goals_per_team"]
HOME_ADVANTAGE:         float = _cfg["home_advantage_multiplier"]
FORM_WEIGHT:            float = _cfg["form_weight"]
RATING_SCALE:           float = _cfg["rating_scale"]
USE_HOME_ADVANTAGE:     bool  = _cfg["use_home_advantage"]
USE_RECENT_FORM:        bool  = _cfg["use_recent_form"]
THIRD_PLACE_QUALIFIERS: int   = _cfg["third_place_qualifiers"]
GROUPS_COUNT:           int   = _cfg["groups_count"]
TEAMS_PER_GROUP:        int   = _cfg["teams_per_group"]

# \\\\\\\\\\\
# Parámetros del modelo Dixon-Coles
# \\\\\\\\\\\

DIXON_COLES_RHO: float = _cfg["dixon_coles_rho"]

# \\\\\\\\\\\
# Pesos de las fuentes de datos para el rating compuesto
# \\\\\\\\\\\

FIFA_RANKING_WEIGHT: float = _cfg["fifa_ranking_weight"]
ELO_WEIGHT:          float = _cfg["elo_weight"]
SOFIFA_WEIGHT:       float = _cfg["sofifa_weight"]
FORM_DATA_WEIGHT:    float = _cfg["form_data_weight"]

# \\\\\\\\\\\
# Fuerza histórica por confederación (factor contextual del Mundial)
# \\\\\\\\\\\

CONFEDERATION_STRENGTH: dict = _cfg["confederation_strength"]

# \\\\\\\\\\\
# Exponente para penaltis (mayor = más ventaja al equipo mejor)
# \\\\\\\\\\\

PENALTY_ALPHA: float = _cfg["penalty_alpha"]

# \\\\\\\\\\\
# Mapa de normalización de nombres entre fuentes externas y teams.csv
# \\\\\\\\\\\

TEAM_NAME_MAP: dict = {
    "Korea Republic":   "South Korea",
    "Côte d'Ivoire":    "Ivory Coast",
    "IR Iran":          "Iran",
    "United States":    "USA",
    "Man City":         "Manchester City",
    "Inter":            "Inter Milan",
    "Atletico":         "Atletico Madrid",
}

def normalize_team_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)
