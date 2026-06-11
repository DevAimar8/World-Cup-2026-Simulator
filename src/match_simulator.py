"""
match_simulator.py
------------------
Motor matemático del simulador. Calcula goles esperados (xG) para cada
selección y genera resultados usando la distribución de Poisson.

Fórmula base:
    xG_A = BASE_GOALS * f_rating(A,B) * f_attack(A) * f_defence(B) * f_home(A) * f_form(A)
    goals_A ~ Poisson(xG_A)
"""

import numpy as np
import pandas as pd
from src.config import BASE_GOALS, HOME_ADVANTAGE, FORM_WEIGHT, RATING_SCALE, USE_HOME_ADVANTAGE, USE_RECENT_FORM

# \\\\\\\\\\\
# Factor de ajuste por diferencia de rating entre equipos
# \\\\\\\\\\\

def _rating_factor(rating_a: float, rating_b: float) -> float:
    """
    Escala la ventaja de rating como ratio centrado en 1.
    rating_scale controla cuánto impacta la diferencia.
    Ejemplo: A=90, B=70 → factor A ligeramente > 1.
    """
    diff    = rating_a - rating_b
    factor  = 1.0 + (diff / RATING_SCALE)
    return max(0.5, min(factor, 2.0))     # clamp para evitar extremos


# \\\\\\\\\\\
# Factor de ataque normalizado sobre el rating máximo posible
# \\\\\\\\\\\

def _attack_factor(attack_rating: float) -> float:
    """Normaliza el rating de ataque a un multiplicador entre 0.7 y 1.3."""
    return 0.7 + (attack_rating / 99) * 0.6


# \\\\\\\\\\\
# Factor de debilidad defensiva del rival
# \\\\\\\\\\\

def _defence_weakness(defence_rating: float) -> float:
    """
    Un rival con defensa muy alta supone factor < 1 (más difícil marcar).
    Un rival con defensa baja supone factor > 1 (más fácil marcar).
    """
    return 1.3 - (defence_rating / 99) * 0.6


# \\\\\\\\\\\
# Factor de ventaja de localía
# \\\\\\\\\\\

def _home_factor(is_host: int) -> float:
    """Devuelve HOME_ADVANTAGE si el equipo es anfitrión, 1.0 en caso contrario."""
    if not USE_HOME_ADVANTAGE:
        return 1.0
    return HOME_ADVANTAGE if is_host else 1.0


# \\\\\\\\\\\
# Factor de forma reciente
# \\\\\\\\\\\

def _form_factor(form_factor: float) -> float:
    """
    form_factor ya viene normalizado en [0.85, 1.15] desde data_loader.
    FORM_WEIGHT controla el peso de la forma en el cálculo final.
    """
    if not USE_RECENT_FORM:
        return 1.0
    return 1.0 + FORM_WEIGHT * (form_factor - 1.0)


# \\\\\\\\\\\
# Cálculo de goles esperados para un equipo en un partido
# \\\\\\\\\\\

def expected_goals(
    rating_a: float,
    attack_a: float,
    defence_b: float,
    form_factor_a: float,
    is_host_a: int,
    rating_b: float,
) -> float:
    """
    Devuelve los goles esperados (xG) de un equipo en un partido concreto.
    Combina: rating global, ataque, defensa rival, localía y forma.
    """
    xg = (
        BASE_GOALS
        * _rating_factor(rating_a, rating_b)
        * _attack_factor(attack_a)
        * _defence_weakness(defence_b)
        * _home_factor(is_host_a)
        * _form_factor(form_factor_a)
    )
    return max(xg, 0.1)     # mínimo 0.1 para evitar xG = 0


# \\\\\\\\\\\
# Simulación de un partido de fase de grupos (puede terminar en empate)
# \\\\\\\\\\\

def simulate_group_match(team_a: pd.Series, team_b: pd.Series) -> tuple[int, int]:
    """
    Simula un partido de fase de grupos.
    Los goles se generan con distribución de Poisson.
    El empate es resultado válido.

    Returns:
        (goals_a, goals_b): goles de cada equipo.
    """
    xg_a = expected_goals(
        rating_a     = team_a["overall_rating"],
        attack_a     = team_a["attack_rating"],
        defence_b    = team_b["defense_rating"],
        form_factor_a= team_a["form_factor"],
        is_host_a    = team_a["is_host"],
        rating_b     = team_b["overall_rating"],
    )
    xg_b = expected_goals(
        rating_a     = team_b["overall_rating"],
        attack_a     = team_b["attack_rating"],
        defence_b    = team_a["defense_rating"],
        form_factor_a= team_b["form_factor"],
        is_host_a    = team_b["is_host"],
        rating_b     = team_a["overall_rating"],
    )

    goals_a = int(np.random.poisson(xg_a))
    goals_b = int(np.random.poisson(xg_b))
    return goals_a, goals_b


# \\\\\\\\\\\
# Simulación de un partido de eliminatoria (debe haber ganador)
# \\\\\\\\\\\

def simulate_knockout_match(team_a: pd.Series, team_b: pd.Series) -> str:
    """
    Simula un partido de eliminatoria.
    Si hay empate tras los 90 minutos, se resuelve por penaltis (50/50).

    Returns:
        Nombre del equipo ganador.
    """
    goals_a, goals_b = simulate_group_match(team_a, team_b)

    if goals_a > goals_b:
        return team_a["team"]
    elif goals_b > goals_a:
        return team_b["team"]
    else:
        # Penaltis: probabilidad ajustada ligeramente por overall_rating
        strength_a = team_a["overall_rating"]
        strength_b = team_b["overall_rating"]
        total      = strength_a + strength_b
        prob_a     = strength_a / total
        return team_a["team"] if np.random.random() < prob_a else team_b["team"]
