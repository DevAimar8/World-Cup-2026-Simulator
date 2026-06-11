import json
from pathlib import Path

# \\\\\\\\\\\
# Rutas base del proyecto
# \\\\\\\\\\\

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# \\\\\\\\\\\
# Carga de parámetros desde config.json
# \\\\\\\\\\\

def load_config() -> dict:
    """Carga y devuelve el diccionario de configuración del modelo."""
    config_path = DATA_DIR / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)

# \\\\\\\\\\\
# Parámetros globales accesibles directamente
# \\\\\\\\\\\

_cfg = load_config()

DEFAULT_SIMULATIONS: int       = _cfg["default_simulations"]
RANDOM_SEED: int               = _cfg["random_seed"]
BASE_GOALS: float              = _cfg["base_goals_per_team"]
HOME_ADVANTAGE: float          = _cfg["home_advantage_multiplier"]
FORM_WEIGHT: float             = _cfg["form_weight"]
RATING_SCALE: float            = _cfg["rating_scale"]
USE_HOME_ADVANTAGE: bool       = _cfg["use_home_advantage"]
USE_RECENT_FORM: bool          = _cfg["use_recent_form"]
THIRD_PLACE_QUALIFIERS: int    = _cfg["third_place_qualifiers"]
GROUPS_COUNT: int              = _cfg["groups_count"]
TEAMS_PER_GROUP: int           = _cfg["teams_per_group"]

# \\\\\\\\\\\
# Mapeo de nombres: soccerdata → teams.csv
# Permite alinear nombres de ClubElo/SoFIFA con los usados en el proyecto
# \\\\\\\\\\\

TEAM_NAME_MAP: dict = {
    # ClubElo usa nombres abreviados o en inglés
    "Man City":         "Manchester City",
    "Man Utd":          "Manchester United",
    "Inter":            "Inter Milan",
    "Atletico":         "Atletico Madrid",
    "Leverkusen":       "Bayer Leverkusen",
    # Nombres de selecciones nacionales
    "Korea Republic":   "South Korea",
    "Côte d'Ivoire":    "Ivory Coast",
    "IR Iran":          "Iran",
    "USA":              "USA",
    "United States":    "USA",
}

def normalize_team_name(name: str) -> str:
    """Normaliza un nombre de equipo usando el mapa de equivalencias."""
    return TEAM_NAME_MAP.get(name, name)
