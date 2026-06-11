# World Cup Monte Carlo Simulator

Simulador matemático del Mundial 2026 basado en Monte Carlo.

Simula el torneo miles de veces para estimar probabilidades reales usando datos reales de **SoccerData** (ClubElo + SoFIFA).

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## Ejecución

```bash
# Descargar datos reales (requiere conexión)
python run_data_fetch.py

# Simulación completa (10.000 por defecto)
python run_simulation.py

# Con parámetros
python -m src.monte_carlo --simulations 10000 --seed 42

# Dashboard interactivo
streamlit run app/streamlit_app.py
```

---

## Estructura

```
world_cup_simulator/
├── data/
│   ├── teams.csv          → selecciones y grupos
│   ├── ratings.csv        → ratings generados por data_fetcher
│   └── config.json        → parámetros del modelo
├── src/
│   ├── config.py          → configuración global
│   ├── data_loader.py     → carga y validación de datos
│   ├── data_fetcher.py    → extracción con soccerdata
│   ├── match_simulator.py → motor matemático Poisson
│   ├── group_stage.py     → fase de grupos
│   ├── knockout_stage.py  → eliminatorias
│   ├── monte_carlo.py     → núcleo del simulador
│   ├── analysis.py        → análisis y conclusiones
│   └── visualizations.py → gráficos matplotlib
├── app/
│   └── streamlit_app.py   → dashboard visual
├── outputs/               → resultados generados
├── notebooks/             → análisis exploratorio
└── docs/
    └── model_explanation.md
```

---

## Modelo matemático

```
xG_equipo = base_goals * factor_rating * factor_ataque * factor_defensa_rival * factor_localía
goles ~ Poisson(xG_equipo)
```

Formato del torneo: 48 equipos → 12 grupos → 32 clasificados → eliminatorias hasta la final.
