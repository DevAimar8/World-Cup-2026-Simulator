# Explicación del modelo matemático

## 1. Goles esperados (xG)

Para cada partido, el modelo calcula los goles esperados de cada selección:

```
xG_A = BASE_GOALS
       × factor_rating(A, B)
       × factor_ataque(A)
       × factor_defensa_rival(B)
       × factor_localía(A)
       × factor_forma(A)
```

### Parámetros base (`data/config.json`)

| Parámetro | Valor | Descripción |
|---|---|---|
| `base_goals_per_team` | 1.32 | Media de goles por equipo en mundiales |
| `home_advantage_multiplier` | 1.08 | Bonus para anfitriones |
| `form_weight` | 0.12 | Peso de la forma reciente |
| `rating_scale` | 520 | Escala de impacto del rating |

### Factores del modelo

**factor_rating**: Ratio entre ratings globales, centrado en 1.
```
factor = 1.0 + (rating_A - rating_B) / rating_scale
```

**factor_ataque**: Normalizado entre 0.7 y 1.3 según attack_rating.

**factor_defensa_rival**: Inverso de la defensa del rival. Más defensa rival → factor más bajo.

**factor_localía**: HOME_ADVANTAGE si is_host=1, 1.0 si no.

**factor_forma**: Entre 0.85 y 1.15 según recent_form.

---

## 2. Generación de resultados

Una vez calculados xG_A y xG_B, los goles se generan con distribución de Poisson:

```
goals_A ~ Poisson(xG_A)
goals_B ~ Poisson(xG_B)
```

La distribución de Poisson es la distribución más aceptada en modelado de goles de fútbol, ya que los goles son eventos discretos, independientes entre sí (aproximación) y relativamente raros.

---

## 3. Fase de grupos

Cada grupo juega todos contra todos (6 partidos en grupos de 4). El empate es resultado válido.

**Criterios de clasificación:**
1. Puntos (V=3, E=1, D=0)
2. Diferencia de goles
3. Goles a favor
4. Aleatorio (desempate final para evitar sesgos)

---

## 4. Eliminatorias

En eliminatorias no hay empate. Si después de simular los 90 minutos hay empate, se resuelven por penaltis con probabilidad proporcional al overall_rating:

```
prob_A_penaltis = overall_rating_A / (overall_rating_A + overall_rating_B)
```

---

## 5. Monte Carlo

El torneo completo se repite N veces (por defecto 10.000). Tras N repeticiones:

```
P(campeón, equipo_X) = victorias_X / N × 100
```

Con 10.000 simulaciones, el error estándar de una probabilidad del 10% es ≈ 0.3%, lo que es suficientemente preciso para análisis cualitativo.

---

## 6. Fuentes de datos

| Fuente | Uso | Integración |
|---|---|---|
| **ClubElo** | Elo rating de clubes → proxy del nivel del país | `soccerdata.ClubElo()` |
| **SoFIFA** | Overall, attack, defence de equipos | `soccerdata.SoFIFA()` |
| **Manual** | Ratings base para selecciones sin datos de soccerdata | `data/ratings.csv` |

Los datos de soccerdata cubren clubes, no selecciones directamente. El proyecto usa la media de los mejores clubes de cada país como estimación del nivel de la selección nacional.

---

## 7. Limitaciones

- Rating basado en clubes, no convocatorias reales
- Bracket simplificado (no el oficial del Mundial)
- Sin simulación de lesiones ni sanciones
- Sin fatiga acumulada
- Penaltis simplificados (sin modelado de porteros)
- Forma reciente manual (no actualizada automáticamente)
