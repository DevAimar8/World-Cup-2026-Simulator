"""
01_example_analysis.py
----------------------
Análisis exploratorio del simulador.
Ejecuta una simulación pequeña y muestra los resultados principales.

Uso:
    python notebooks/01_example_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from src.monte_carlo import run_monte_carlo
from src.data_loader import load_tournament_data
from src.analysis import generate_group_summary

# \\\\\\\\\\\
# 1. Datos del torneo: distribución de ratings
# \\\\\\\\\\\

print("\n=== DATOS DEL TORNEO ===")
df = load_tournament_data()
print(f"Total equipos: {len(df)}")
print(f"Grupos: {sorted(df['group'].unique())}")
print(f"\nRating medio por confederación:")
print(df.groupby("confederation")["overall_rating"].mean().sort_values(ascending=False).round(1))

# \\\\\\\\\\\
# 2. Simulación rápida de 1.000 torneos para explorar
# \\\\\\\\\\\

print("\n=== SIMULACIÓN RÁPIDA (1.000 iteraciones) ===")
results = run_monte_carlo(n_simulations=1000, seed=42, verbose=False)
probs   = results["team_probabilities"]

# \\\\\\\\\\\
# 3. Top 10 favoritos al título
# \\\\\\\\\\\

print("\nTop 10 favoritos al título:")
top10 = probs.head(10)[["team", "group", "overall_rating", "champion_probability", "group_exit_probability"]]
print(top10.to_string(index=False))

# \\\\\\\\\\\
# 4. Equipos con más riesgo de caer en grupos (overall > 80 pero alta salida)
# \\\\\\\\\\\

print("\nFavoritos con riesgo de caer en grupos:")
risky = probs[(probs["overall_rating"] > 80) & (probs["group_exit_probability"] > 10)]
print(risky[["team", "overall_rating", "champion_probability", "group_exit_probability"]].to_string(index=False))

# \\\\\\\\\\\
# 5. Resumen de dificultad de grupos
# \\\\\\\\\\\

print("\nDificultad de grupos (de mayor a menor):")
grp_sum = generate_group_summary(df)
print(grp_sum[["group", "average_rating", "max_rating", "teams"]].to_string(index=False))

# \\\\\\\\\\\
# 6. Finales más repetidas
# \\\\\\\\\\\

print("\nTop 5 finales más frecuentes:")
print(results["finals"].head(5)[["final", "probability_pct"]].to_string(index=False))

print("\n✅ Análisis completado. Outputs en la carpeta outputs/")
