"""
config.py
---------
Configuración central del proyecto. Define rutas, parámetros del modelo
matemático, pesos de las fuentes de datos y constantes globales.

Todos los módulos importan desde aquí — si quieres ajustar el modelo,
este es el único archivo que necesitas tocar.

Autor: Aimar Esqueta
Proyecto: FIFA World Cup 2026 Prediction Model
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Rutas base del proyecto ──────────────────────────────────────────────────
# ROOT_DIR apunta a la carpeta raíz independientemente de desde dónde se ejecute
ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
OUTPUTS_DIR  = ROOT_DIR / "outputs"

# Crear carpetas si no existen (útil en primer arranque o Streamlit Cloud)
OUTPUTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# ─── Variables de entorno ─────────────────────────────────────────────────────
# Lee el archivo .env de la raíz del proyecto
# En Streamlit Cloud se configuran como "Secrets" en la UI
load_dotenv(ROOT_DIR / ".env")

RAPIDAPI_KEY:  str = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST: str = "free-api-live-football-data.p.rapidapi.com"
RAPIDAPI_BASE: str = f"https://{RAPIDAPI_HOST}"

# ─── Parámetros del modelo Dixon-Coles ───────────────────────────────────────
# Estos valores están calibrados con datos históricos de mundiales
# Puedes ajustarlos para experimentar con el modelo

BASE_GOALS      = 1.15   # Media de goles por equipo en mundiales históricos
DIXON_COLES_RHO = 0.13   # Corrección para resultados bajos (0-0, 1-0, etc.)
                          # Valor positivo → aumenta probabilidad de 0-0 y 1-1
PENALTY_ALPHA   = 2.5    # Exponente para penaltis — mayor valor = más ventaja
                          # al equipo con mejor rating en tanda de penaltis

# ─── Factores contextuales del partido ───────────────────────────────────────
HOME_ADV      = 1.04   # Multiplicador de ventaja para equipos anfitriones
                        # 1.04 = 4% de ventaja — menor que en ligas normales
                        # porque el Mundial se juega en sedes neutrales en su mayoría
FORM_WEIGHT   = 0.08   # Cuánto pesa la forma reciente en el cálculo de xG
RATING_SCALE  = 200    # Escala de la diferencia de rating en la fórmula exponencial
                        # Menor valor = las diferencias de rating impactan más

# Factor de fuerza histórica por confederación
# Refleja las diferencias estructurales de nivel entre confederaciones
# basándose en resultados históricos de mundiales
CONFEDERATION_STRENGTH = {
    "UEFA":     1.00,   # Europa — referencia base
    "CONMEBOL": 0.97,   # Sudamérica — muy cerca del nivel europeo
    "CONCACAF": 0.78,   # Norteamérica y Caribe
    "CAF":      0.76,   # África
    "AFC":      0.75,   # Asia
    "OFC":      0.62,   # Oceanía — nivel más bajo históricamente
}

# ─── Pesos del rating compuesto ──────────────────────────────────────────────
# El rating final de cada selección combina 4 fuentes con estos pesos
# La suma debe ser siempre 1.0
W_ELO   = 0.35   # ELO Rating histórico — la fuente más fiable a largo plazo
W_POWER = 0.30   # Power Score propio — rendimiento reciente ponderado
W_FIFA  = 0.20   # FIFA Ranking oficial — referencia institucional
W_FORM  = 0.15   # Forma últimos 10 partidos — captura el momento actual

# ─── Parámetros de simulación Monte Carlo ────────────────────────────────────
DEFAULT_SIMULATIONS = 10000   # Número de torneos simulados por defecto
                               # Con 10.000 el error estándar es ~0.3% — suficiente
RANDOM_SEED         = 42      # Semilla para reproducibilidad de resultados
