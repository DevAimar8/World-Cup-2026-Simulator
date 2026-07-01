"""
match_simulator.py
------------------
Motor matemático de simulación de partidos — versión vectorizada.

Mejora clave respecto a la versión anterior:
  ANTES: doble bucle Python para calcular la matriz 9×9 → 3.8ms/partido
  AHORA: np.outer vectorizado → 0.13ms/partido → 30x más rápido

El modelo matemático es idéntico (Dixon-Coles 1997), solo cambia
la implementación interna para evitar bucles Python lentos.

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import (
    BASE_GOALS, DIXON_COLES_RHO, PENALTY_ALPHA,
    HOME_ADV, FORM_WEIGHT, RATING_SCALE, CONFEDERATION_STRENGTH,
)

# ─── Arrays pre-computados para toda la sesión ───────────────────────────────
# Evita recrearlos en cada llamada — pequeña optimización pero suma en 10K sims
_J = np.arange(9)  # índices de goles 0..8 para el equipo A
_K = np.arange(9)  # índices de goles 0..8 para el equipo B


# ─── Cálculo de lambdas (goles esperados) ────────────────────────────────────
def compute_lambdas(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """
    Calcula los goles esperados (λ) de cada equipo.
    Misma lógica que antes, sin cambios en el modelo matemático.
    """
    ra, rb = float(a["overall_rating"]), float(b["overall_rating"])

    # Escala exponencial: diferencias grandes de rating impactan más
    rf_a = float(np.clip(np.exp((ra - rb) / RATING_SCALE * 0.7), 0.40, 2.50))
    rf_b = float(np.clip(np.exp((rb - ra) / RATING_SCALE * 0.7), 0.40, 2.50))

    # Ratio de fuerzas por confederación
    ca = CONFEDERATION_STRENGTH.get(a.get("confederation", "UEFA"), 0.75)
    cb = CONFEDERATION_STRENGTH.get(b.get("confederation", "UEFA"), 0.75)
    cf_ab = float(np.clip(ca / cb, 0.55, 1.55))
    cf_ba = float(np.clip(cb / ca, 0.55, 1.55))

    hf_a = HOME_ADV if int(a.get("is_host", 0)) else 1.0
    hf_b = HOME_ADV if int(b.get("is_host", 0)) else 1.0
    ff_a = 1.0 + FORM_WEIGHT * (float(a.get("form_factor", 1.0)) - 1.0)
    ff_b = 1.0 + FORM_WEIGHT * (float(b.get("form_factor", 1.0)) - 1.0)

    la = max(BASE_GOALS * float(a["attack_coef"]) * float(b["defense_coef"]) * rf_a * cf_ab * hf_a * ff_a, 0.08)
    lb = max(BASE_GOALS * float(b["attack_coef"]) * float(a["defense_coef"]) * rf_b * cf_ba * hf_b * ff_b, 0.08)
    return la, lb


# ─── Matriz de probabilidades vectorizada ────────────────────────────────────
def _prob_matrix(la: float, lb: float) -> np.ndarray:
    """
    Calcula la matriz 9×9 de probabilidades P(j,k) usando NumPy vectorizado.

    En lugar de un doble bucle Python (lento), usamos np.outer que calcula
    el producto exterior en C puro: 30x más rápido.

    Después aplicamos la corrección Dixon-Coles solo en los 4 marcadores
    especiales (0-0, 1-0, 0-1, 1-1) con acceso directo por índice.
    """
    # Distribución de Poisson vectorizada para todos los valores de golesa la vez
    pj = poisson.pmf(_J, la)   # P(0 goles), P(1 gol), ..., P(8 goles) para A
    pk = poisson.pmf(_K, lb)   # Idem para B

    # Producto exterior: P(j,k) = P(j) × P(k) para todos los pares → matriz 9×9
    probs = np.outer(pj, pk)

    # Corrección Dixon-Coles en los 4 marcadores especiales
    # Los demás marcadores (≥2 goles total) no necesitan corrección
    rho = DIXON_COLES_RHO
    probs[0, 0] = max(probs[0, 0] * (1 - la * lb * rho), 0.0)
    probs[1, 0] *= (1 + lb * rho)
    probs[0, 1] *= (1 + la * rho)
    probs[1, 1] *= (1 - rho)

    # Eliminar valores negativos (puede ocurrir con rho muy alto) y normalizar
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


# ─── Distribución completa de marcadores (para el dashboard) ─────────────────
def match_distribution(a: pd.Series, b: pd.Series, max_goals: int = 6) -> pd.DataFrame:
    """
    Devuelve la distribución de marcadores en formato DataFrame para el dashboard.
    Usa la misma matriz vectorizada, restringida a max_goals para legibilidad.
    """
    la, lb  = compute_lambdas(a, b)
    probs   = _prob_matrix(la, lb)

    rows = []
    for j in range(max_goals + 1):
        for k in range(max_goals + 1):
            rows.append({"home_goals": j, "away_goals": k, "probability": probs[j, k]})

    df    = pd.DataFrame(rows)
    # Re-normalizar al subconjunto de marcadores mostrados
    total = df["probability"].sum()
    if total > 0:
        df["probability"] /= total
    return df.sort_values("probability", ascending=False)


# ─── Probabilidades 1X2 (para el simulador de partidos) ──────────────────────
def win_draw_loss_probs(a: pd.Series, b: pd.Series) -> dict:
    """Calcula victoria local, empate y victoria visitante con xG y marcador más probable."""
    la, lb  = compute_lambdas(a, b)
    probs   = _prob_matrix(la, lb)

    # Máscara triangular superior/diagonal/inferior para V/E/D
    j_idx, k_idx = np.meshgrid(_J, _K, indexing="ij")
    p_home = probs[j_idx > k_idx].sum()
    p_draw = probs[j_idx == k_idx].sum()
    p_away = probs[j_idx < k_idx].sum()

    best_flat = probs.argmax()
    gj, gk    = divmod(int(best_flat), 9)

    return {
        "p_home":            round(float(p_home) * 100, 1),
        "p_draw":            round(float(p_draw) * 100, 1),
        "p_away":            round(float(p_away) * 100, 1),
        "xg_home":           round(la, 2),
        "xg_away":           round(lb, 2),
        "most_likely_score": f"{gj}-{gk}",
        "most_likely_prob":  round(float(probs[gj, gk]) * 100, 1),
    }


# ─── Simulación de partido (fase de grupos) ───────────────────────────────────
def simulate_group_match(a: pd.Series, b: pd.Series) -> tuple[int, int]:
    """
    Simula un partido de fase de grupos. El empate es válido.
    Muestrea de la distribución vectorizada.
    """
    la, lb = compute_lambdas(a, b)
    probs  = _prob_matrix(la, lb)
    idx    = np.random.choice(81, p=probs.ravel())
    return divmod(idx, 9)


# ─── Simulación de partido eliminatorio ──────────────────────────────────────
def simulate_knockout_match(a: pd.Series, b: pd.Series) -> str:
    """
    Simula un partido de eliminatoria. Empate → penaltis.
    El mejor equipo tiene ventaja en penaltis (rating^PENALTY_ALPHA).
    """
    ga, gb = simulate_group_match(a, b)
    if ga > gb: return str(a["team"])
    if gb > ga: return str(b["team"])
    ra = float(a["overall_rating"]) ** PENALTY_ALPHA
    rb = float(b["overall_rating"]) ** PENALTY_ALPHA
    return str(a["team"]) if np.random.random() < ra / (ra + rb) else str(b["team"])
