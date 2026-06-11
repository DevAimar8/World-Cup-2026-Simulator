"""
streamlit_app.py
----------------
Dashboard visual del simulador Monte Carlo del Mundial 2026.
Permite configurar, ejecutar simulaciones y explorar resultados interactivamente.

Ejecución:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Asegurar que el proyecto raíz está en el path
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

# \\\\\\\\\\\
# Configuración de la página de Streamlit
# \\\\\\\\\\\

st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# \\\\\\\\\\\
# Estilos CSS personalizados
# \\\\\\\\\\\

st.markdown("""
<style>
    .metric-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #238636;
    }
    .section-header {
        color: #58A6FF;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# \\\\\\\\\\\
# Sidebar: configuración del simulador
# \\\\\\\\\\\

with st.sidebar:
    st.title("⚽ Configuración")
    st.markdown("---")

    n_simulations = st.slider(
        "Número de simulaciones",
        min_value=100, max_value=10000, value=1000, step=100,
        help="Más simulaciones = resultados más precisos pero más lentos."
    )

    seed = st.number_input(
        "Semilla aleatoria",
        min_value=0, max_value=9999, value=42,
        help="Misma semilla = mismos resultados reproducibles."
    )

    st.markdown("---")
    run_button = st.button("▶ Ejecutar Simulación", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("📊 Datos: soccerdata (ClubElo + SoFIFA)")
    st.caption("🧮 Modelo: Distribución de Poisson")
    st.caption("🌍 Formato: 48 equipos · 12 grupos")


# \\\\\\\\\\\
# Título principal
# \\\\\\\\\\\

st.title("🌍 World Cup 2026 — Monte Carlo Simulator")
st.markdown("Simula el torneo miles de veces para estimar probabilidades reales de cada selección.")
st.markdown("---")


# \\\\\\\\\\\
# Carga de resultados existentes o ejecución de nueva simulación
# \\\\\\\\\\\

@st.cache_data
def load_existing_results():
    """Carga resultados guardados si existen."""
    probs_path = OUTPUTS_DIR / "team_probabilities.csv"
    finals_path = OUTPUTS_DIR / "finals.csv"
    groups_path = OUTPUTS_DIR / "group_summary.csv"
    log_path    = OUTPUTS_DIR / "simulation_log.csv"

    if probs_path.exists():
        return {
            "team_probabilities": pd.read_csv(probs_path),
            "finals":             pd.read_csv(finals_path),
            "group_summary":      pd.read_csv(groups_path),
            "log":                pd.read_csv(log_path) if log_path.exists() else None,
        }
    return None


if run_button:
    with st.spinner(f"Ejecutando {n_simulations:,} simulaciones..."):
        results = run_monte_carlo(n_simulations=n_simulations, seed=seed, verbose=False)
        generate_all_charts(results["team_probabilities"], results["group_summary"])
    st.success(f"✅ {n_simulations:,} simulaciones completadas.")
    st.cache_data.clear()
    results_data = {
        "team_probabilities": results["team_probabilities"],
        "finals":             results["finals"],
        "group_summary":      results["group_summary"],
        "log":                None,
    }
else:
    results_data = load_existing_results()


# \\\\\\\\\\\
# Pantalla de bienvenida si no hay resultados
# \\\\\\\\\\\

if results_data is None:
    st.info("👈 Configura los parámetros en el panel izquierdo y pulsa **Ejecutar Simulación**.")

    # Vista previa de los equipos
    st.subheader("📋 Equipos del Mundial 2026")
    try:
        df = load_tournament_data()
        st.dataframe(df[["group", "team", "confederation", "overall_rating", "is_host"]],
                     use_container_width=True, height=400)
    except Exception:
        st.warning("Carga los datos ejecutando primero `python run_data_fetch.py`.")
    st.stop()


# \\\\\\\\\\\
# Métricas resumen en la parte superior
# \\\\\\\\\\\

team_probs  = results_data["team_probabilities"]
finals_df   = results_data["finals"]
group_sum   = results_data["group_summary"]

top_team    = team_probs.iloc[0]
top_final   = finals_df.iloc[0] if not finals_df.empty else None
hardest_grp = group_sum.iloc[0] if not group_sum.empty else None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏆 Gran favorito", top_team["team"], f"{top_team['champion_probability']}%")
with col2:
    finalist_leader = team_probs.sort_values("finalist_probability", ascending=False).iloc[0]
    st.metric("🥈 Más veces finalista", finalist_leader["team"], f"{finalist_leader['finalist_probability']}%")
with col3:
    if top_final:
        st.metric("🎯 Final más probable", top_final["final"].replace(" vs ", " vs\n"), f"{top_final['probability_pct']}%")
with col4:
    if hardest_grp is not None:
        st.metric("💀 Grupo más duro", f"Grupo {hardest_grp['group']}", f"Rating {hardest_grp['average_rating']}")

st.markdown("---")


# \\\\\\\\\\\
# Tabs principales del dashboard
# \\\\\\\\\\\

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Probabilidades", "🎯 Finales", "🗂️ Grupos", "📈 Gráficos", "📝 Conclusiones"
])


# --- Tab 1: Tabla de probabilidades ---
with tab1:
    st.subheader("Probabilidades por selección")

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        conf_filter = st.multiselect(
            "Filtrar por confederación",
            options=sorted(team_probs["confederation"].unique()),
            default=[],
        )
    with col_f2:
        group_filter = st.multiselect(
            "Filtrar por grupo",
            options=sorted(team_probs["group"].unique()),
            default=[],
        )

    filtered = team_probs.copy()
    if conf_filter:
        filtered = filtered[filtered["confederation"].isin(conf_filter)]
    if group_filter:
        filtered = filtered[filtered["group"].isin(group_filter)]

    # Columnas visibles y renombradas
    display_cols = {
        "team":                     "Selección",
        "group":                    "Grupo",
        "overall_rating":           "Rating",
        "champion_probability":     "Campeón %",
        "finalist_probability":     "Final %",
        "semifinalist_probability": "Semis %",
        "quarterfinalist_prob":     "Cuartos %",
        "group_exit_probability":   "Eliminado grupos %",
    }
    display_df = filtered[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        display_df.style.background_gradient(subset=["Campeón %"], cmap="Greens"),
        use_container_width=True,
        height=500,
    )

    # Descarga de CSV
    csv = team_probs.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Descargar CSV completo", csv, "team_probabilities.csv", "text/csv")


# --- Tab 2: Finales más probables ---
with tab2:
    st.subheader("Finales más repetidas en las simulaciones")

    if finals_df.empty:
        st.info("Sin datos de finales disponibles.")
    else:
        # Gráfico de las top 10 finales
        top10 = finals_df.head(10).sort_values("probability_pct")

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        bars = ax.barh(top10["final"], top10["probability_pct"], color="#DA3633", edgecolor="none")
        for bar, val in zip(bars, top10["probability_pct"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha="left", fontsize=8, color="#E6EDF3")
        ax.set_xlabel("Frecuencia (%)", color="#E6EDF3")
        ax.tick_params(colors="#E6EDF3")
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.2, color="#21262D")
        ax.set_title("Top 10 Finales más frecuentes", color="#E6EDF3", pad=10)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(finals_df, use_container_width=True)


# --- Tab 3: Grupos ---
with tab3:
    st.subheader("Resumen y dificultad de grupos")

    if group_sum.empty:
        st.info("Sin datos de grupos disponibles.")
    else:
        st.dataframe(
            group_sum.style.background_gradient(subset=["average_rating"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        # Desglose por grupo con equipos
        st.markdown("#### Equipos por grupo")
        try:
            td = load_tournament_data()
            for grp, grp_df in td.groupby("group"):
                with st.expander(f"Grupo {grp}"):
                    st.dataframe(
                        grp_df[["team", "confederation", "overall_rating", "attack_rating", "defense_rating", "is_host"]],
                        use_container_width=True,
                    )
        except Exception:
            pass


# --- Tab 4: Gráficos ---
with tab4:
    st.subheader("Visualizaciones del simulador")

    champ_img = OUTPUTS_DIR / "champion_probabilities.png"
    semi_img  = OUTPUTS_DIR / "semifinal_probabilities.png"
    grp_img   = OUTPUTS_DIR / "group_difficulty.png"

    if champ_img.exists():
        st.image(str(champ_img), caption="Probabilidad de ser Campeón", use_column_width=True)
    if semi_img.exists():
        st.image(str(semi_img), caption="Probabilidad de llegar a Semifinales", use_column_width=True)
    if grp_img.exists():
        st.image(str(grp_img), caption="Dificultad media por grupo", use_column_width=True)

    if not champ_img.exists():
        st.info("Ejecuta la simulación para generar los gráficos.")

    # Botón para regenerar gráficos sin re-simular
    if st.button("🔄 Regenerar gráficos"):
        generate_all_charts(team_probs, group_sum)
        st.success("Gráficos regenerados.")
        st.rerun()


# --- Tab 5: Conclusiones ---
with tab5:
    st.subheader("Conclusiones automáticas")

    conclusions_path = OUTPUTS_DIR / "conclusions.txt"
    if conclusions_path.exists():
        with open(conclusions_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.code(content, language=None)
    else:
        st.info("Ejecuta la simulación para generar las conclusiones.")
