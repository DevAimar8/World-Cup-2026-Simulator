# Explicación del modelo matemático

## 1. Modelo Dixon-Coles (1997)

El modelo base es **Dixon-Coles**, el estándar de referencia en análisis cuantitativo de fútbol. Mejora la distribución de Poisson pura corrigiendo la probabilidad de resultados bajos (0-0, 1-0, 0-1, 1-1), que Poisson subestima sistemáticamente.

### Fórmula completa

```
lambda_A = BASE * alpha_A * beta_B * rf * hf * ff * cf
lambda_B = BASE * alpha_B * beta_A * rf * hf * ff * cf

P(j, k) = tau(j, k, rho) * Poisson(j | lambda_A) * Poisson(k | lambda_B)
```

### Corrección tau de Dixon-Coles

| Resultado | tau |
|---|---|
| 0 - 0 | 1 - lambda_A × lambda_B × rho |
| 1 - 0 | 1 + lambda_B × rho |
| 0 - 1 | 1 + lambda_A × rho |
| 1 - 1 | 1 - rho |
| resto  | 1.0 (sin corrección) |

rho = 0.13 (calibrado empíricamente en datos de mundiales)

---

## 2. Factores contextuales del Mundial

| Factor | Variable | Descripción |
|---|---|---|
| `alpha` | `attack_coef` | Fuerza ofensiva individual del equipo |
| `beta` | `defense_coef` | Debilidad defensiva del equipo (beta alto = defensa porosa) |
| `rf` | rating diferencial | Ventaja por diferencia de rating global |
| `hf` | `is_host` | ×1.10 para USA, México, Canadá |
| `ff` | `recent_form` | ±15% según forma reciente |
| `cf` | `confederation` | Factor histórico UEFA=1.00 … OFC=0.68 |

---

## 3. Rating compuesto ponderado (4 fuentes)

```
composite_rating = 0.40 × Elo_norm
                 + 0.25 × FIFA_Ranking_norm
                 + 0.20 × SoFIFA_overall_norm
                 + 0.15 × recent_form_norm
```

Elo normalizado desde [1500, 2100] → [0, 99]
FIFA Ranking invertido: ranking 1 = 99, ranking 200 = 0

---

## 4. Penaltis

```
P(gana_A) = rating_A^1.8 / (rating_A^1.8 + rating_B^1.8)
```

El exponente 1.8 hace que la ventaja de rating sea significativa pero no determinista.

---

## 5. Nuevos outputs

| Output | Descripción |
|---|---|
| `third_place_stats.csv` | Distribución de puntos de los terceros clasificados y eliminados |
| `path_to_title.csv` | Rival más frecuente por ronda para cada campeón potencial |
| `variance_table.csv` | Índice de consistencia y varianza por equipo |
| `phase_heatmap.png` | Heatmap de probabilidades por fase |
| `variance_scatter.png` | Scatter potencial vs inconsistencia |
| `third_place_points.png` | Histograma de puntos de terceros |

---

## 6. Formato del torneo (Mundial 2026)

```
48 equipos → 12 grupos × 4 equipos
Clasifican: 1° y 2° de cada grupo (24) + 8 mejores terceros = 32
Eliminatorias: Dieciseisavos → Octavos → Cuartos → Semis → Final
```
