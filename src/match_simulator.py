"""
match_simulator.py
------------------
Motor matemático del simulador basado en el modelo Dixon-Coles (1997)
con factores contextuales del Mundial.

El modelo Dixon-Coles corrige la distribución de Poisson estándar para
resultados bajos (0-0, 1-0, 0-1, 1-1), que son sistemáticamente
mal estimados por Poisson puro. Es el modelo de referencia en
análisis cuantitativo de fútbol y apuestas deportivas.

Fórmula base:
    lambda_A = BASE * alpha_A * beta_B * gamma_home * delta_form * epsilon_conf
    lambda_B = BASE * alpha_B * beta_A * gamma_home * delta_form * epsilon_conf

    P(goals_A=j, goals_B=k) = tau(j,k) * Poisson(j|lambda_A) * Poisson(k|lambda_B)

donde tau es la corrección de Dixon-Coles para resultados bajos.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from src.config import (
    BASE_GOALS, HOME_ADVANTAGE, FORM_WEIGHT, RATING_SCALE,
    USE_HOME_ADVANTAGE, USE_RECENT_FORM,
    DIXON_COLES_RHO, CONFEDERATION_STRENGTH, PENALTY_ALPHA,
)

# \\\\\\\\\\\
# Corrección tau de Dixon-Coles para resultados bajos
# Ajusta la probabilidad de 0-0, 1-0, 0-1, 1-1
# \\\\\\\\\\\

def _tau(j: int, k: int, lam_a: float, lam_b: float, rho: float) -> float:
    """
    Factor de corrección de Dixon-Coles para resultados con goles bajos.
    rho > 0 aumenta la prob de 0-0 y 1-1, reduce 1-0 y 0-1.
    """
    if j == 0 and k == 0:
        return 1.0 - lam_a * lam_b * rho
    elif j == 1 and k == 0:
        return 1.0 + lam_b * rho
    elif j == 0 and k == 1:
        return 1.0 + lam_a * rho
    elif j == 1 and k == 1:
        return 1.0 - rho
    else:
        return 1.0


# \\\\\\\\\\\
# Factor de rating global: diferencia de nivel entre selecciones
# \\\\\\\\\\\

def _rating_factor(rating_a: float, rating_b: float) -> float:
    """Convierte diferencia de rating en multiplicador de xG."""
    diff   = rating_a - rating_b
    factor = 1.0 + (diff / RATING_SCALE)
    return float(np.clip(factor, 0.45, 2.2))


# \\\\\\\\\\\
# Factor de ventaja de localía (anfitriones del Mundial)
# \\\\\\\\\\\

def _home_factor(is_host: int) -> float:
    """Bonus de localía para los países anfitriones (USA, México, Canadá)."""
    if not USE_HOME_ADVANTAGE:
        return 1.0
    return HOME_ADVANTAGE if int(is_host) else 1.0


# \\\\\\\\\\\
# Factor de forma reciente normalizado
# \\\\\\\\\\\

def _form_factor(form_factor: float) -> float:
    """Aplica el peso de la forma reciente al xG."""
    if not USE_RECENT_FORM:
        return 1.0
    return 1.0 + FORM_WEIGHT * (float(form_factor) - 1.0)


# \\\\\\\\\\\
# Factor de confederación: penaliza/bonifica según nivel medio histórico
# \\\\\\\\\\\

def _conf_factor(confederation: str) -> float:
    """
    Ajuste por confederación. UEFA y CONMEBOL por encima de la media,
    OFC por debajo. Refleja diferencias históricas de nivel global.
    """
    return CONFEDERATION_STRENGTH.get(confederation, 0.80)


# \\\\\\\\\\\
# Cálculo de lambdas (xG) para ambos equipos en un partido
# \\\\\\\\\\\

def compute_lambdas(team_a: pd.Series, team_b: pd.Series) -> tuple[float, float]:
    """
    Calcula los goles esperados (lambda) para cada equipo usando
    los coeficientes de ataque/defensa individuales más los factores
    contextuales del partido.

    attack_coef y defense_coef se calibran en ratings.csv.
    attack_coef > 1 → equipo ofensivo
    defense_coef < 1 → defensa sólida (dificulta goles al rival)
    """
    # Coeficientes individuales del equipo (Dixon-Coles)
    alpha_a = float(team_a.get("attack_coef",  1.0))
    beta_a  = float(team_a.get("defense_coef", 1.0))
    alpha_b = float(team_b.get("attack_coef",  1.0))
    beta_b  = float(team_b.get("defense_coef", 1.0))

    # Factores contextuales
    rf_a  = _rating_factor(team_a["overall_rating"], team_b["overall_rating"])
    rf_b  = _rating_factor(team_b["overall_rating"], team_a["overall_rating"])
    hf_a  = _home_factor(team_a["is_host"])
    hf_b  = _home_factor(team_b["is_host"])
    ff_a  = _form_factor(team_a["form_factor"])
    ff_b  = _form_factor(team_b["form_factor"])
    cf_a  = _conf_factor(team_a["confederation"])
    cf_b  = _conf_factor(team_b["confederation"])

    # Lambda Dixon-Coles completo
    lam_a = BASE_GOALS * alpha_a * beta_b * rf_a * hf_a * ff_a * cf_a
    lam_b = BASE_GOALS * alpha_b * beta_a * rf_b * hf_b * ff_b * cf_b

    # Mínimo técnico para evitar lambda=0
    return max(lam_a, 0.08), max(lam_b, 0.08)


# \\\\\\\\\\\
# Simulación de un partido de fase de grupos con corrección Dixon-Coles
# \\\\\\\\\\\

def simulate_group_match(team_a: pd.Series, team_b: pd.Series) -> tuple[int, int]:
    """
    Simula un partido de fase de grupos usando el modelo Dixon-Coles.

    1. Genera distribución bivariante hasta MAX_GOALS goles por equipo.
    2. Aplica la corrección tau en resultados bajos.
    3. Muestrea un resultado de la distribución ajustada.

    Returns: (goals_a, goals_b)
    """
    MAX_GOALS = 8
    lam_a, lam_b = compute_lambdas(team_a, team_b)
    rho = DIXON_COLES_RHO

    # Matriz de probabilidades P(j, k) para j,k en [0, MAX_GOALS]
    probs = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for j in range(MAX_GOALS + 1):
        for k in range(MAX_GOALS + 1):
            p = (poisson.pmf(j, lam_a)
                 * poisson.pmf(k, lam_b)
                 * _tau(j, k, lam_a, lam_b, rho))
            probs[j, k] = max(p, 0.0)

    # Normalizar (tau puede desplazar ligeramente la suma de 1)
    total = probs.sum()
    if total <= 0:
        return int(np.random.poisson(lam_a)), int(np.random.poisson(lam_b))
    probs /= total

    # Muestreo del resultado
    flat  = probs.ravel()
    idx   = np.random.choice(len(flat), p=flat)
    goals_a, goals_b = divmod(idx, MAX_GOALS + 1)
    return int(goals_a), int(goals_b)


# \\\\\\\\\\\
# Simulación de un partido de eliminatoria (debe haber ganador)
# \\\\\\\\\\\

def simulate_knockout_match(team_a: pd.Series, team_b: pd.Series) -> str:
    """
    Simula un partido de eliminatoria con el modelo Dixon-Coles.
    Si hay empate en 90 min → penaltis con probabilidad proporcional
    al parámetro PENALTY_ALPHA (favorece ligeramente al equipo mejor).

    Returns: nombre del equipo ganador.
    """
    goals_a, goals_b = simulate_group_match(team_a, team_b)

    if goals_a > goals_b:
        return str(team_a["team"])
    elif goals_b > goals_a:
        return str(team_b["team"])
    else:
        # Penaltis: probabilidad proporcional a overall_rating ^ alpha
        strength_a = float(team_a["overall_rating"]) ** PENALTY_ALPHA
        strength_b = float(team_b["overall_rating"]) ** PENALTY_ALPHA
        prob_a     = strength_a / (strength_a + strength_b)
        return str(team_a["team"]) if np.random.random() < prob_a else str(team_b["team"])
