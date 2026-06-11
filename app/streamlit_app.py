"""
streamlit_app.py
----------------
Dashboard del simulador Monte Carlo — Mundial 2026.
Tabs: Probabilidades | Fases | Finales | Grupos | Terceros |
      Varianza | Camino al título | Escenarios | Gráficos | Conclusiones
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.monte_carlo import run_monte_carlo
from src.data_loader import load_tournament_data
from src.analysis import generate_group_summary
from src.visualizations import generate_all_charts
from src.config import OUTPUTS_DIR

st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-box { background:#161B22; border:1px solid #21262D;
              border-radius:8px; padding:14px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# \\\\\\\\\\\
# Sidebar: configuración
# \\\\\\\\\\\

with st.sidebar:
    st.title("⚽ Configuración")
    st.markdown("---")
    n_sims = st.slider("Simulaciones", 100, 10000, 1000, 100)
    seed   = st.number_input("Semilla aleatoria", 0, 9999, 42)

    st.markdown("---")
    st.markdown("**Escenario: simular sin un equipo**")
    try:
        td       = load_tournament_data()
        all_teams = ["(ninguno)"] + sorted(td["team"].tolist())
    except Exception:
        all_teams = ["(ninguno)"]
    exclude = st.selectbox("Excluir equipo", all_teams)
    exclude_team = None if exclude == "(ninguno)" else exclude

    st.markdown("---")
    run_btn = st.button("▶ Ejecutar simulación", type="primary", use_container_width=True)
    st.caption("Modelo: Dixon-Coles + factores contextuales")
    st.caption("Datos: ClubElo · SoFIFA · FIFA Ranking")


st.title("🌍 World Cup 2026 — Monte Carlo Simulator")
st.markdown("Simulador probabilístico basado en el modelo **Dixon-Coles** con datos reales.")
st.markdown("---")


# \\\\\\\\\\\
# Carga o ejecución de resultados
# \\\\\\\\\\\

@st.cache_data
def load_saved():
    p = lambda f: OUTPUTS_DIR / f
    if not (p("team_probabilities.csv")).exists():
        return None
    return {
        "team_probabilities": pd.read_csv(p("team_probabilities.csv")),
        "finals":             pd.read_csv(p("finals.csv")),
        "group_summary":      pd.read_csv(p("group_summary.csv")),
        "third_place_stats":  pd.read_csv(p("third_place_stats.csv")) if p("third_place_stats.csv").exists() else pd.DataFrame(),
        "path_to_title":      pd.read_csv(p("path_to_title.csv"))     if p("path_to_title.csv").exists()     else pd.DataFrame(),
        "variance_table":     pd.read_csv(p("variance_table.csv"))    if p("variance_table.csv").exists()    else pd.DataFrame(),
    }

if run_btn:
    with st.spinner(f"Ejecutando {n_sims:,} simulaciones (Dixon-Coles)..."):
        res = run_monte_carlo(n_simulations=n_sims, seed=seed,
                              verbose=False, exclude_team=exclude_team)
        generate_all_charts(res["team_probabilities"], res["group_summary"],
                            res["third_place_stats"], res["variance_table"])
    label = f"✅ {n_sims:,} simulaciones completadas"
    if exclude_team:
        label += f" (sin {exclude_team})"
    st.success(label)
    st.cache_data.clear()
    data = {k: res[k] for k in ["team_probabilities","finals","group_summary",
                                  "third_place_stats","path_to_title","variance_table"]}
else:
    data = load_saved()

if data is None:
    st.info("👈 Pulsa **Ejecutar simulación** para comenzar.")
    try:
        df = load_tournament_data()
        st.subheader("Equipos del torneo")
        st.dataframe(df[["group","team","confederation","overall_rating","is_host"]],
                     use_container_width=True, height=380)
    except Exception:
        pass
    st.stop()

tp      = data["team_probabilities"]
finals  = data["finals"]
gs      = data["group_summary"]
thirds  = data["third_place_stats"]
path_df = data["path_to_title"]
var_df  = data["variance_table"]

# \\\\\\\\\\\
# Métricas resumen
# \\\\\\\\\\\

top   = tp.iloc[0]
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("🏆 Gran favorito", top["team"], f"{top['champion_pct']}%")
with c2:
    tf = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
    st.metric("🥈 Más veces finalista", tf["team"], f"{tf['reach_final_pct']}%")
with c3:
    if not finals.empty:
        st.metric("🎯 Final más probable", finals.iloc[0]["final"], f"{finals.iloc[0]['probability_pct']}%")
with c4:
    if not gs.empty:
        hg = gs.iloc[0]
        st.metric("💀 Grupo más duro", f"Grupo {hg['group']}", f"Rating {hg['average_rating']}")

st.markdown("---")

# \\\\\\\\\\\
# Tabs del dashboard
# \\\\\\\\\\\

tabs = st.tabs([
    "📊 Probabilidades", "📈 Por fase", "🎯 Finales",
    "🗂 Grupos", "3️⃣ Terceros", "📉 Varianza",
    "🛣 Camino al título", "🔀 Escenarios", "🖼 Gráficos", "📝 Conclusiones"
])


# --- Tab 1: Probabilidades generales ---
with tabs[0]:
    st.subheader("Probabilidades generales por selección")
    c1, c2 = st.columns(2)
    with c1:
        conf_f = st.multiselect("Confederación", sorted(tp["confederation"].unique()), key="conf1")
    with c2:
        grp_f  = st.multiselect("Grupo", sorted(tp["group"].unique()), key="grp1")

    filt = tp.copy()
    if conf_f: filt = filt[filt["confederation"].isin(conf_f)]
    if grp_f:  filt = filt[filt["group"].isin(grp_f)]

    cols_map = {
        "team": "Selección", "group": "Grupo", "confederation": "Conf.",
        "overall_rating": "Rating",
        "champion_pct": "Campeón %", "reach_final_pct": "Final %",
        "reach_semis_pct": "Semis %", "reach_quarters_pct": "Cuartos %",
        "pass_group_stage_pct": "Pasa grupos %", "group_exit_pct": "Eliminado grupos %",
    }
    disp = filt[list(cols_map)].rename(columns=cols_map)
    st.dataframe(disp.style.background_gradient(subset=["Campeón %"], cmap="Greens"),
                 use_container_width=True, height=480)
    st.download_button("⬇ Descargar CSV", tp.to_csv(index=False).encode(),
                       "team_probabilities.csv", "text/csv")


# --- Tab 2: Probabilidades por fase ---
with tabs[1]:
    st.subheader("Probabilidad de alcanzar cada fase del torneo")

    team_sel = st.selectbox("Seleccionar equipo", tp["team"].tolist(), key="team_phase")
    row = tp[tp["team"] == team_sel].iloc[0]

    phases = {
        "Pasa grupos": row["pass_group_stage_pct"],
        "Dieciseisavos": row["reach_round32_pct"],
        "Octavos":        row["reach_round16_pct"],
        "Cuartos":        row["reach_quarters_pct"],
        "Semis":          row["reach_semis_pct"],
        "Final":          row["reach_final_pct"],
        "Campeón":        row["champion_pct"],
    }
    phase_df = pd.DataFrame({"Fase": list(phases.keys()), "Probabilidad (%)": list(phases.values())})

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(phase_df)))
    bars = ax.bar(phase_df["Fase"], phase_df["Probabilidad (%)"],
                  color=colors, edgecolor="none")
    for bar, val in zip(bars, phase_df["Probabilidad (%)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#E6EDF3")
    ax.set_ylabel("Probabilidad (%)", color="#E6EDF3")
    ax.tick_params(colors="#E6EDF3"); ax.spines[:].set_visible(False)
    ax.grid(axis="y", alpha=0.2, color="#21262D")
    ax.set_title(f"Probabilidades de {team_sel} por fase", color="#E6EDF3", pad=10)
    st.pyplot(fig); plt.close(fig)

    st.dataframe(phase_df, use_container_width=True)


# --- Tab 3: Finales ---
with tabs[2]:
    st.subheader("Finales más repetidas en las simulaciones")
    if finals.empty:
        st.info("Sin datos.")
    else:
        top10 = finals.head(10).sort_values("probability_pct")
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        ax.barh(top10["final"], top10["probability_pct"], color="#DA3633", edgecolor="none")
        ax.set_xlabel("Frecuencia (%)", color="#E6EDF3")
        ax.tick_params(colors="#E6EDF3"); ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.2, color="#21262D")
        ax.set_title("Top 10 finales más frecuentes", color="#E6EDF3", pad=10)
        st.pyplot(fig); plt.close(fig)
        st.dataframe(finals, use_container_width=True)


# --- Tab 4: Grupos ---
with tabs[3]:
    st.subheader("Dificultad de grupos")
    if not gs.empty:
        st.dataframe(gs.style.background_gradient(subset=["average_rating"], cmap="RdYlGn_r"),
                     use_container_width=True)
    st.markdown("#### Desglose por grupo")
    try:
        td2 = load_tournament_data()
        for g, gdf in td2.groupby("group"):
            with st.expander(f"Grupo {g}"):
                st.dataframe(gdf[["team","confederation","overall_rating",
                                   "attack_coef","defense_coef","is_host"]],
                             use_container_width=True)
    except Exception:
        pass


# --- Tab 5: Puntos de los terceros ---
with tabs[4]:
    st.subheader("¿Con cuántos puntos se clasifican los terceros de grupo?")

    if thirds.empty:
        st.info("Ejecuta la simulación para ver esta información.")
    else:
        summary = thirds[thirds["categoria"].str.startswith("RESUMEN")]
        detail  = thirds[~thirds["categoria"].str.startswith("RESUMEN")]

        if not summary.empty:
            st.markdown("#### Estadísticas de los mejores terceros clasificados")
            for _, r in summary.iterrows():
                label = r["categoria"].replace("RESUMEN - ", "")
                val   = r["puntos"]
                st.metric(label, f"{val} puntos")

        st.markdown("#### Distribución de puntos")
        img = OUTPUTS_DIR / "third_place_points.png"
        if img.exists():
            st.image(str(img), use_column_width=True)

        st.markdown("#### Datos completos")
        if not detail.empty:
            pivot = detail.pivot_table(
                index="puntos", columns="categoria",
                values="frecuencia_pct", aggfunc="sum"
            ).fillna(0)
            st.dataframe(pivot, use_container_width=True)

        st.info(
            "**Lectura:** un tercer clasificado necesita típicamente **4-6 puntos** "
            "para entrar entre los 8 mejores. Con 3 puntos hay opciones pero no garantía. "
            "Con 7+ puntos la clasificación es casi segura."
        )


# --- Tab 6: Varianza ---
with tabs[5]:
    st.subheader("Varianza y consistencia por equipo")
    st.markdown("""
    - **Índice de consistencia alto** → el equipo llega lejos en casi todas las simulaciones.
    - **Índice de varianza alto** → alto potencial de campeón pero también de caer en grupos.
    """)

    img_scatter = OUTPUTS_DIR / "variance_scatter.png"
    if img_scatter.exists():
        st.image(str(img_scatter), use_column_width=True)

    if not var_df.empty:
        st.dataframe(
            var_df.style.background_gradient(subset=["consistency_index"], cmap="Greens")
                        .background_gradient(subset=["variance_index"], cmap="Reds"),
            use_container_width=True
        )


# --- Tab 7: Camino al título ---
with tabs[6]:
    st.subheader("Camino más probable al título")
    st.markdown("Muestra el rival más frecuente de cada equipo en cada ronda cuando ese equipo gana.")

    if path_df.empty:
        st.info("Sin datos. Ejecuta la simulación.")
    else:
        n_show = st.slider("Equipos a mostrar", 5, min(20, len(path_df)), 10, key="path_slider")
        show = path_df.head(n_show)
        st.dataframe(show, use_container_width=True)

        # Vista detallada de un equipo
        team_path = st.selectbox("Ver camino de", path_df["team"].tolist(), key="path_team")
        row_p = path_df[path_df["team"] == team_path]
        if not row_p.empty:
            row_p = row_p.iloc[0]
            st.markdown(f"#### Camino típico de {team_path} al título ({row_p['champion_pct']}% de ser campeón)")
            rondas = ["Dieciseisavos", "Octavos", "Cuartos", "Semifinal"]
            cols = st.columns(4)
            for i, rnd in enumerate(rondas):
                rival_col = f"{rnd}_rival"
                freq_col  = f"{rnd}_freq_pct"
                if rival_col in row_p.index:
                    with cols[i]:
                        st.metric(rnd, row_p[rival_col], f"{row_p[freq_col]}% de las veces")


# --- Tab 8: Escenarios ---
with tabs[7]:
    st.subheader("Análisis de escenarios")
    st.markdown(
        "Usa el panel izquierdo para **excluir un equipo** del torneo y comparar "
        "cómo cambian las probabilidades del resto. Útil para analizar el impacto "
        "de una baja importante o comparar mundiales alternativos."
    )

    if exclude_team:
        st.info(f"La última simulación se ejecutó **sin {exclude_team}**. "
                "Los resultados de las demás pestañas reflejan ese escenario.")
    else:
        st.info("Selecciona un equipo en el sidebar y ejecuta la simulación para ver el escenario.")

    st.markdown("#### Top 10 favoritos al título")
    if not tp.empty:
        top10 = tp.head(10)[["team", "group", "champion_pct", "reach_semis_pct", "group_exit_pct"]]
        st.dataframe(top10, use_container_width=True)


# --- Tab 9: Gráficos ---
with tabs[8]:
    st.subheader("Visualizaciones")
    imgs = [
        ("champion_probabilities.png", "Probabilidad de ser Campeón"),
        ("semifinal_probabilities.png", "Probabilidad de llegar a Semifinales"),
        ("phase_heatmap.png", "Heatmap de probabilidades por fase"),
        ("variance_scatter.png", "Varianza — Potencial vs Inconsistencia"),
        ("group_difficulty.png", "Dificultad de grupos"),
        ("third_place_points.png", "Puntos de los terceros clasificados"),
    ]
    for fname, caption in imgs:
        p = OUTPUTS_DIR / fname
        if p.exists():
            st.image(str(p), caption=caption, use_column_width=True)
            st.markdown("")

    if st.button("🔄 Regenerar gráficos"):
        generate_all_charts(tp, gs, thirds, var_df)
        st.success("Regenerados."); st.rerun()


# --- Tab 10: Conclusiones ---
with tabs[9]:
    st.subheader("Conclusiones automáticas")
    path_c = OUTPUTS_DIR / "conclusions.txt"
    if path_c.exists():
        st.code(open(path_c, encoding="utf-8").read(), language=None)
    else:
        st.info("Ejecuta la simulación para generar las conclusiones.")
