# ⚽ FIFA World Cup 2026 — Prediction Model

Modelo predictivo del Mundial FIFA 2026 construido con Python, Power Score propio, ELO Rating, API en vivo y simulación Monte Carlo.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Dixon--Coles-gold)](https://en.wikipedia.org/wiki/Dixon%E2%80%93Coles_model)
[![API](https://img.shields.io/badge/Data-API--Football-green)](https://www.api-football.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🎯 Objetivo del proyecto

Construir un sistema de predicción deportiva que:

- Analice el rendimiento reciente de las 48 selecciones con datos reales
- Calcule un **Power Score** propio por equipo combinando múltiples fuentes
- Simule el torneo completo **10.000 veces** con el modelo **Dixon-Coles**
- Proyecte probabilidades reales de campeón, finalista y eliminación por fase
- Se actualice **automáticamente** con los resultados reales de la API
- Permita simular cualquier partido en tiempo real con predicción de marcador

---

## 📊 Dashboard

El dashboard fue desarrollado en **Streamlit** con diseño oscuro personalizado.

### Páginas principales

| Página | Descripción |
|---|---|
| 🏠 **Dashboard** | Visión global: grupos, probabilidades de campeón y semifinales |
| 🗂 **Grupos** | Desglose completo de los 12 grupos con Power Score y probabilidades |
| 🌳 **Bracket** | Camino más probable al título por equipo + finales más repetidas |
| ⚽ **Simulador** | Predicción en tiempo real de cualquier partido: 1X2, xG, marcador más probable |
| ⭐ **Equipos** | Cards de las 48 selecciones con ranking, estadísticas y clasificación proyectada |
| 📊 **Stats** | Tabla completa de probabilidades + análisis de varianza + dificultad de grupos |

---

## 🔬 Metodología

### 1. Datos históricos — API-Football

Conexión en vivo a **API-Football (RapidAPI)** para obtener:

- Equipos, grupos y posiciones en tiempo real
- Resultados de todos los partidos jugados
- Estadísticas detalladas por partido (posesión, tiros, tarjetas, xG)
- Historial de enfrentamientos directos (head-to-head)
- Cuotas de casas de apuestas como señal de mercado

### 2. Power Score — Indicador propio

Indicador diseñado para medir la fortaleza relativa actual de cada selección:

```
Power Score = 0.6 × Forma reciente + 0.4 × Rendimiento en torneo
```

Donde la forma reciente pondera victorias (×3), empates (×1), derrotas (×0) y diferencia de gol de los últimos 10 partidos.

### 3. Rating compuesto — 4 fuentes ponderadas

```
Rating compuesto = 0.35 × ELO_norm
                 + 0.30 × Power Score
                 + 0.20 × FIFA Ranking_norm
                 + 0.15 × Forma reciente
```

| Fuente | Peso | Descripción |
|---|---|---|
| ELO Rating | 35% | Rating histórico de selecciones (eloratings.net) |
| Power Score | 30% | Rendimiento reciente propio |
| FIFA Ranking | 20% | Ranking oficial FIFA junio 2026 |
| Forma reciente | 15% | Últimos 10 partidos ponderados |

### 4. Modelo Dixon-Coles (1997)

Motor de predicción de partidos. Mejora la distribución de Poisson estándar corrigiendo la probabilidad de marcadores bajos (0-0, 1-0, 0-1, 1-1), que Poisson puro subestima sistemáticamente.

```
λ_A = BASE × α_A × β_B × factor_rating × factor_confederación × factor_localía × factor_forma
λ_B = BASE × α_B × β_A × factor_rating × factor_confederación × factor_localía × factor_forma

P(j, k) = τ(j, k, ρ) × Poisson(j | λ_A) × Poisson(k | λ_B)
```

**Corrección τ de Dixon-Coles:**

| Marcador | τ |
|---|---|
| 0-0 | `1 - λ_A × λ_B × ρ` |
| 1-0 | `1 + λ_B × ρ` |
| 0-1 | `1 + λ_A × ρ` |
| 1-1 | `1 - ρ` |
| Resto | `1.0` |

`ρ = 0.13` (calibrado en datos históricos de mundiales)

### 5. Factor de confederación

No todas las confederaciones tienen el mismo nivel histórico:

| Confederación | Factor |
|---|---|
| UEFA | 1.00 |
| CONMEBOL | 0.97 |
| CONCACAF | 0.78 |
| CAF | 0.76 |
| AFC | 0.75 |
| OFC | 0.62 |

### 6. Simulación Monte Carlo — 10.000 torneos

El torneo completo se repite 10.000 veces. En cada simulación:

1. Se simulan los 6 partidos de cada uno de los 12 grupos
2. Se clasifican los 24 primeros/segundos + 8 mejores terceros
3. Se simula el bracket de eliminatorias (32 → 16 → 8 → 4 → 2 → 1)
4. Los empates en eliminatoria se resuelven por penaltis con probabilidad `rating^2.5`
5. Se acumulan contadores de campeón, finalista, semifinalistas, etc.

Resultado: distribuciones de probabilidad estables para cada selección en cada fase.

---

## 📁 Estructura del repositorio

```
world-cup-2026-prediction-model/
│
├── README.md
├── requirements.txt
├── .env.example              ← plantilla de variables de entorno
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py             ← configuración central + parámetros del modelo
│   ├── api_client.py         ← cliente API-Football (fixtures, standings, h2h, odds)
│   ├── power_score.py        ← cálculo del Power Score y rating compuesto
│   ├── match_simulator.py    ← motor Dixon-Coles (lambdas, distribución, 1X2)
│   ├── tournament.py         ← fase de grupos + eliminatorias + bracket
│   ├── monte_carlo.py        ← 10.000 simulaciones + generación de outputs
│   └── analysis.py           ← probabilidades, tablas, conclusiones automáticas
│
├── app/
│   └── streamlit_app.py      ← dashboard completo (6 páginas)
│
├── scripts/
│   └── run_pipeline.py       ← pipeline completo: API → Power Score → Monte Carlo
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_power_score_analysis.ipynb
│   ├── 03_dixon_coles_calibration.ipynb
│   ├── 04_api_connection.ipynb
│   └── 05_results_analysis.ipynb
│
├── data/
│   ├── raw/                  ← datos crudos de API (no incluidos en repo)
│   └── processed/            ← datos procesados: ratings.csv, fixtures.csv
│
└── outputs/                  ← CSVs y gráficos generados por el modelo
    ├── team_probabilities.csv
    ├── finals.csv
    ├── group_summary.csv
    ├── third_place_stats.csv
    ├── path_to_title.csv
    └── variance_table.csv
```

---

## 🚀 Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/world-cup-2026-prediction-model.git
cd world-cup-2026-prediction-model
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar la API key

```bash
cp .env.example .env
# Edita .env y añade tu RAPIDAPI_KEY
# Consíguela gratis en: https://rapidapi.com/api-sports/api/api-football
```

### 4. Ejecutar el pipeline completo

```bash
# Con datos en vivo de la API
python scripts/run_pipeline.py --simulations 10000

# Sin API (datos estáticos — no requiere key)
python scripts/run_pipeline.py --no-api

# Simulación rápida para test
python scripts/run_pipeline.py --simulations 1000 --no-api
```

### 5. Abrir el dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Resultados tras 10.000 simulaciones

```
FAVORITOS AL TÍTULO:
  España          16.83%  |  Power Score: 88.2
  Argentina       11.25%  |  Power Score: 91.0
  Brasil           8.75%  |  Power Score: 86.4
  Países Bajos     9.48%  |  Power Score: 82.1
  Alemania         9.39%  |  Power Score: 83.0

GRUPOS MÁS DIFÍCILES:
  Grupo F — Países Bajos, Japón, Suecia, Túnez   (rating 70.1)
  Grupo I — Francia, Senegal, Noruega, Irak       (rating 68.7)
  Grupo L — Inglaterra, Croacia, Ghana, Panamá    (rating 67.1)

UMBRAL PARA CLASIFICARSE COMO MEJOR TERCERO:
  Media: 3.64 puntos · Mínimo: 2 pts · Máximo: 6 pts
```

---

## 🔐 Seguridad

Las claves de API y credenciales **no se incluyen** en este repositorio.

En el archivo `.env` (no subido a Git), reemplaza:

```
RAPIDAPI_KEY=YOUR_RAPIDAPI_KEY_HERE
```

---

## 🛠️ Tecnologías

| Tecnología | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| Pandas + NumPy | Manipulación de datos |
| SciPy | Distribución de Poisson |
| Streamlit | Dashboard interactivo |
| API-Football (RapidAPI) | Datos en vivo del Mundial |
| Dixon-Coles (1997) | Motor de predicción de partidos |
| Monte Carlo | Simulación del torneo completo |
| Matplotlib | Visualizaciones |

---

## 📋 Objetivo de portfolio

Este proyecto demuestra habilidades en:

- **Data Engineering** — Pipeline de ingesta, transformación y almacenamiento de datos
- **Data Science** — Modelado probabilístico con Dixon-Coles y Monte Carlo
- **Análisis deportivo** — Power Score, ELO Rating, análisis de rendimiento
- **Visualización** — Dashboard interactivo con Streamlit
- **Integración de APIs** — Conexión en vivo con API-Football
- **Ingeniería de software** — Código modular, documentado y reproducible

---

## Autor

**DevAimar8**
Data · Insights · Impact

---

*Los resultados son probabilísticos, no predicciones. Basados en datos FIFA Rankings junio 2026.*
