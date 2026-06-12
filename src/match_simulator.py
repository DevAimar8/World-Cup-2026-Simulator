"""
match_simulator.py
------------------
Motor matemático basado en Dixon-Coles (1997) con factores contextuales.

CAMBIOS v3:
- rating_scale reducido a 200: diferencias de rating impactan más
- home_advantage reducido a 1.04: localía da ventaja pequeña, no decisiva
- confederación con mayor peso: UEFA/CONMEBOL muy por encima de CAF/AFC
- attack_coef/defense_coef calibrados con escala Elo real (no normalizada)
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
# Corrección tau Dixon-Coles para resultados bajos
# \\\\\\\\\\\

def _tau(j: int, k: int, la: float, lb: float, rho: float) -> float:
    if   j==0 and k==0: return max(1.0 - la*lb*rho, 0.01)
    elif j==1 and k==0: return 1.0 + lb*rho
    elif j==0 and k==1: return 1.0 + la*rho
    elif j==1 and k==1: return 1.0 - rho
    return 1.0


# \\\\\\\\\\\
# Factor de diferencia de rating — escala exponencial para mayor separación
# \\\\\\\\\\\

def _rating_factor(ra: float, rb: float) -> float:
    """
    Usa escala exponencial en lugar de lineal.
    Con rating_scale=200, una diferencia de 15 puntos (Argentina vs USA)
    da factor ~1.08 de forma que el xG es claramente mayor.
    Con diferencia de 40 puntos (España vs Cabo Verde) da ~1.22.
    """
    diff = (ra - rb) / RATING_SCALE
    return float(np.clip(np.exp(diff * 0.7), 0.40, 2.50))


# \\\\\\\\\\\
# Factor de confederación — diferencia real de nivel histórico
# \\\\\\\\\\\

def _conf_ratio(conf_a: str, conf_b: str) -> float:
    """
    Ratio de fuerza entre confederaciones.
    Si UEFA juega contra CAF: ratio = 1.00/0.76 = 1.32 → ventaja real.
    """
    strength_a = CONFEDERATION_STRENGTH.get(conf_a, 0.75)
    strength_b = CONFEDERATION_STRENGTH.get(conf_b, 0.75)
    ratio = strength_a / strength_b
    return float(np.clip(ratio, 0.55, 1.55))


# \\\\\\\\\\\
# Factor de localía (reducido — solo anfitriones del torneo)
# \\\\\\\\\\\

def _home_factor(is_host: int) -> float:
    if not USE_HOME_ADVANTAGE: return 1.0
    return HOME_ADVANTAGE if int(is_host) else 1.0


# \\\\\\\\\\\
# Factor de forma reciente (peso reducido)
# \\\\\\\\\\\

def _form_factor(form_factor: float) -> float:
    if not USE_RECENT_FORM: return 1.0
    return 1.0 + FORM_WEIGHT * (float(form_factor) - 1.0)


# \\\\\\\\\\\
# Cálculo de lambdas (xG esperados) para cada equipo
# \\\\\\\\\\\

def compute_lambdas(team_a: pd.Series, team_b: pd.Series) -> tuple[float, float]:
    """
    Lambda de Dixon-Coles para cada equipo.

    Los factores se aplican de forma asimétrica y multiplicativa:
      - rating_factor: diferencia global de nivel
      - attack_coef del equipo que ataca
      - defense_coef del equipo que defiende (bajo = buena defensa)
      - conf_ratio: diferencia estructural de confederación
      - home_factor y form_factor: ajustes menores
    """
    alpha_a = float(team_a.get("attack_coef",  1.0))
    beta_a  = float(team_a.get("defense_coef", 1.0))
    alpha_b = float(team_b.get("attack_coef",  1.0))
    beta_b  = float(team_b.get("defense_coef", 1.0))

    ra = float(team_a["overall_rating"])
    rb = float(team_b["overall_rating"])

    rf_a = _rating_factor(ra, rb)
    rf_b = _rating_factor(rb, ra)

    conf_a = str(team_a.get("confederation", "UEFA"))
    conf_b = str(team_b.get("confederation", "UEFA"))
    cf_ab  = _conf_ratio(conf_a, conf_b)   # >1 si A es de confederación más fuerte
    cf_ba  = _conf_ratio(conf_b, conf_a)

    hf_a = _home_factor(int(team_a.get("is_host", 0)))
    hf_b = _home_factor(int(team_b.get("is_host", 0)))

    ff_a = _form_factor(float(team_a.get("form_factor", 1.0)))
    ff_b = _form_factor(float(team_b.get("form_factor", 1.0)))

    lam_a = BASE_GOALS * alpha_a * beta_b * rf_a * cf_ab * hf_a * ff_a
    lam_b = BASE_GOALS * alpha_b * beta_a * rf_b * cf_ba * hf_b * ff_b

    return max(lam_a, 0.08), max(lam_b, 0.08)


# \\\\\\\\\\\
# Simulación de partido con modelo Dixon-Coles completo
# \\\\\\\\\\\

def simulate_group_match(team_a: pd.Series, team_b: pd.Series) -> tuple[int, int]:
    """
    Genera el resultado usando la distribución bivariante Dixon-Coles.
    Muestrea de la matriz de probabilidades P(j,k) hasta MAX_GOALS.
    """
    MAX_GOALS = 8
    lam_a, lam_b = compute_lambdas(team_a, team_b)
    rho = DIXON_COLES_RHO

    probs = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for j in range(MAX_GOALS + 1):
        pj = poisson.pmf(j, lam_a)
        for k in range(MAX_GOALS + 1):
            pk  = poisson.pmf(k, lam_b)
            tau = _tau(j, k, lam_a, lam_b, rho)
            probs[j, k] = max(pj * pk * tau, 0.0)

    total = probs.sum()
    if total <= 0:
        return int(np.random.poisson(lam_a)), int(np.random.poisson(lam_b))

    probs /= total
    idx = np.random.choice((MAX_GOALS + 1) ** 2, p=probs.ravel())
    return divmod(idx, MAX_GOALS + 1)


# \\\\\\\\\\\
# Partido eliminatorio — siempre hay ganador
# \\\\\\\\\\\

def simulate_knockout_match(team_a: pd.Series, team_b: pd.Series) -> str:
    """
    Empate → penaltis con probabilidad proporcional a rating^PENALTY_ALPHA.
    Con alpha=2.5, el mejor equipo tiene ventaja clara pero no garantizada.
    """
    ga, gb = simulate_group_match(team_a, team_b)
    if ga > gb: return str(team_a["team"])
    if gb > ga: return str(team_b["team"])

    ra = float(team_a["overall_rating"]) ** PENALTY_ALPHA
    rb = float(team_b["overall_rating"]) ** PENALTY_ALPHA
    prob_a = ra / (ra + rb)
    return str(team_a["team"]) if np.random.random() < prob_a else str(team_b["team"])
