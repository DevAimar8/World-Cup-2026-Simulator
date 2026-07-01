"""
match_simulator.py
------------------
Motor matemático de simulación de partidos basado en Dixon-Coles (1997).

Para cada partido calcula:
  λ_A = goles esperados del equipo A
  λ_B = goles esperados del equipo B

Luego genera la distribución bivariante de resultados P(j,k) aplicando
la corrección tau de Dixon-Coles en marcadores bajos.

También calcula las probabilidades 1X2 para el dashboard de partidos.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import (
    BASE_GOALS, DIXON_COLES_RHO, PENALTY_ALPHA,
    HOME_ADV, FORM_WEIGHT, RATING_SCALE, CONFEDERATION_STRENGTH,
)

# \\\\\\\\\\\
# Corrección tau de Dixon-Coles para marcadores bajos
# \\\\\\\\\\\

def _tau(j: int, k: int, la: float, lb: float, rho: float) -> float:
    """
    Corrige la probabilidad de resultados con pocos goles.
    Sin esta corrección, Poisson subestima el 0-0 y sobreestima el 1-0.
    """
    if   j == 0 and k == 0: return max(1.0 - la * lb * rho, 0.01)
    elif j == 1 and k == 0: return 1.0 + lb * rho
    elif j == 0 and k == 1: return 1.0 + la * rho
    elif j == 1 and k == 1: return 1.0 - rho
    return 1.0


# \\\\\\\\\\\
# Cálculo de lambdas (goles esperados) para cada equipo
# \\\\\\\\\\\

def compute_lambdas(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """
    Calcula los goles esperados de cada equipo usando:
    - Power Score / composite_rating (diferencia de nivel)
    - Coeficientes de ataque y defensa individuales
    - Factor de confederación
    - Ventaja de localía (anfitriones del Mundial)
    - Forma reciente
    """
    ra, rb = float(a["overall_rating"]), float(b["overall_rating"])

    # Factor de rating: escala exponencial — diferencias grandes impactan más
    rf_a = float(np.clip(np.exp((ra - rb) / RATING_SCALE * 0.7), 0.40, 2.50))
    rf_b = float(np.clip(np.exp((rb - ra) / RATING_SCALE * 0.7), 0.40, 2.50))

    # Factor de confederación
    ca, cb   = CONFEDERATION_STRENGTH.get(a["confederation"], 0.75), CONFEDERATION_STRENGTH.get(b["confederation"], 0.75)
    cf_ab    = float(np.clip(ca / cb, 0.55, 1.55))
    cf_ba    = float(np.clip(cb / ca, 0.55, 1.55))

    # Localía y forma
    hf_a = HOME_ADV if int(a.get("is_host", 0)) else 1.0
    hf_b = HOME_ADV if int(b.get("is_host", 0)) else 1.0
    ff_a = 1.0 + FORM_WEIGHT * (float(a.get("form_factor", 1.0)) - 1.0)
    ff_b = 1.0 + FORM_WEIGHT * (float(b.get("form_factor", 1.0)) - 1.0)

    la = max(BASE_GOALS * float(a["attack_coef"]) * float(b["defense_coef"]) * rf_a * cf_ab * hf_a * ff_a, 0.08)
    lb = max(BASE_GOALS * float(b["attack_coef"]) * float(a["defense_coef"]) * rf_b * cf_ba * hf_b * ff_b, 0.08)
    return la, lb


# \\\\\\\\\\\
# Distribución completa de resultados para el dashboard de partidos
# \\\\\\\\\\\

def match_distribution(a: pd.Series, b: pd.Series, max_goals: int = 6) -> pd.DataFrame:
    """
    Calcula la probabilidad de cada marcador posible (j-k) hasta max_goals.
    Devuelve un DataFrame con columnas: home_goals, away_goals, probability.

    Usado en el simulador de partidos del dashboard.
    """
    la, lb = compute_lambdas(a, b)
    rows   = []
    for j in range(max_goals + 1):
        pj = poisson.pmf(j, la)
        for k in range(max_goals + 1):
            prob = max(pj * poisson.pmf(k, lb) * _tau(j, k, la, lb, DIXON_COLES_RHO), 0.0)
            rows.append({"home_goals": j, "away_goals": k, "probability": prob})

    df    = pd.DataFrame(rows)
    total = df["probability"].sum()
    if total > 0:
        df["probability"] /= total
    return df.sort_values("probability", ascending=False)


# \\\\\\\\\\\
# Probabilidades 1X2 — victoria local, empate, victoria visitante
# \\\\\\\\\\\

def win_draw_loss_probs(a: pd.Series, b: pd.Series) -> dict:
    """
    Calcula las probabilidades 1X2 del partido y los goles esperados.
    Devuelve dict con: p_home, p_draw, p_away, xg_home, xg_away, most_likely_score.
    """
    dist = match_distribution(a, b)
    la, lb = compute_lambdas(a, b)

    p_home = dist[dist["home_goals"] > dist["away_goals"]]["probability"].sum()
    p_draw = dist[dist["home_goals"] == dist["away_goals"]]["probability"].sum()
    p_away = dist[dist["home_goals"] < dist["away_goals"]]["probability"].sum()

    top_score = dist.iloc[0]

    return {
        "p_home":          round(p_home * 100, 1),
        "p_draw":          round(p_draw * 100, 1),
        "p_away":          round(p_away * 100, 1),
        "xg_home":         round(la, 2),
        "xg_away":         round(lb, 2),
        "most_likely_score": f"{int(top_score['home_goals'])}-{int(top_score['away_goals'])}",
        "most_likely_prob":  round(top_score["probability"] * 100, 1),
    }


# \\\\\\\\\\\
# Simulación de un partido (fase de grupos — empate válido)
# \\\\\\\\\\\

def simulate_group_match(a: pd.Series, b: pd.Series) -> tuple[int, int]:
    """Simula un partido de fase de grupos. Muestrea de la distribución Dixon-Coles."""
    la, lb  = compute_lambdas(a, b)
    MAX     = 8
    probs   = np.zeros((MAX + 1, MAX + 1))
    for j in range(MAX + 1):
        pj = poisson.pmf(j, la)
        for k in range(MAX + 1):
            probs[j, k] = max(pj * poisson.pmf(k, lb) * _tau(j, k, la, lb, DIXON_COLES_RHO), 0.0)
    total = probs.sum()
    if total <= 0:
        return int(np.random.poisson(la)), int(np.random.poisson(lb))
    probs /= total
    idx = np.random.choice((MAX + 1) ** 2, p=probs.ravel())
    return divmod(idx, MAX + 1)


# \\\\\\\\\\\
# Simulación de partido eliminatorio — siempre hay ganador
# \\\\\\\\\\\

def simulate_knockout_match(a: pd.Series, b: pd.Series) -> str:
    """
    Simula un partido de eliminatoria.
    Empate en 90' → penaltis con probabilidad ponderada por rating^PENALTY_ALPHA.
    El mejor equipo tiene ventaja en penaltis, pero no certeza.
    """
    ga, gb = simulate_group_match(a, b)
    if ga > gb: return str(a["team"])
    if gb > ga: return str(b["team"])
    # Penaltis
    ra = float(a["overall_rating"]) ** PENALTY_ALPHA
    rb = float(b["overall_rating"]) ** PENALTY_ALPHA
    return str(a["team"]) if np.random.random() < ra / (ra + rb) else str(b["team"])
