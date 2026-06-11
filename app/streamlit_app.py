"""
streamlit_app.py — World Cup 2026 Monte Carlo Simulator
UI completamente rediseñada: limpia, visual, sin confederaciones.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.monte_carlo import run_monte_carlo
from src.data_loader import load_tournament_data
from src.analysis import generate_group_summary
from src.visualizations import generate_all_charts
from src.config import OUTPUTS_DIR

# \\\\\\\\\\\
# Configuración de página
# \\\\\\\\\\\
st.set_page_config(page_title="World Cup 2026 Simulator", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.stButton > button { width: 100%; background: #238636; color: white;
                     border: none; border-radius: 6px; padding: 10px; font-size: 15px; }
.stButton > button:hover { background: #2ea043; }
.metric-card { background: #161b22; border: 1px solid #30363d;
               border-radius: 10px; padding: 18px 14px; text-align: center; }
.metric-title { color: #8b949e; font-size: 12px; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { color: #e6edf3; font-size: 26px; font-weight: 700; }
.metric-sub   { color: #58a6ff; font-size: 13px; margin-top: 4px; }
.bracket-match { background:#161b22; border:1px solid #30363d; border-radius:6px;
                 padding:6px 10px; margin:3px 0; font-size:13px; }
.bracket-winner { border-left: 3px solid #238636 !important; }
</style>
""", unsafe_allow_html=True)


# \\\\\\\\\\\
# Sidebar
# \\\\\\\\\\\
with st.sidebar:
    st.markdown("## ⚽ Simulador")
    st.markdown("---")
    n_sims = st.slider("Simulaciones", 500, 10000, 2000, 500)
    seed   = st.number_input("Semilla", 0, 9999, 42)
    st.markdown("---")
    st.markdown("**Escenario**")
    try:
        td_side   = load_tournament_data()
        team_opts = ["(torneo completo)"] + sorted(td_side["team"].tolist())
    except Exception:
        team_opts = ["(torneo completo)"]
    exclude = st.selectbox("Simular sin equipo", team_opts)
    exclude_team = None if exclude == "(torneo completo)" else exclude
    st.markdown("---")
    run_btn = st.button("▶ Ejecutar simulación")
    st.markdown("")
    st.markdown("<small style='color:#8b949e'>Modelo Dixon-Coles · Datos FIFA jun 2026</small>", unsafe_allow_html=True)


# \\\\\\\\\\\
# Carga o ejecución
# \\\\\\\\\\\
@st.cache_data
def load_saved():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists(): return None
    tp = pd.read_csv(p("team_probabilities.csv"))
    tp = tp.rename(columns={
        "champion_probability":"champion_pct","finalist_probability":"reach_final_pct",
        "semifinalist_probability":"reach_semis_pct","quarterfinalist_prob":"reach_quarters_pct",
        "round_of_16_probability":"reach_round16_pct","round_of_32_probability":"reach_round32_pct",
        "group_exit_probability":"group_exit_pct",
    })
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct","reach_quarters_pct","attack_coef","defense_coef"]:
        if col not in tp.columns:
            tp[col] = (100-tp["group_exit_pct"]).round(2) if col=="pass_group_stage_pct" and "group_exit_pct" in tp.columns else 0.0
    return {
        "team_probabilities": tp,
        "finals":             pd.read_csv(p("finals.csv")),
        "group_summary":      pd.read_csv(p("group_summary.csv")),
        "third_place_stats":  pd.read_csv(p("third_place_stats.csv")) if p("third_place_stats.csv").exists() else pd.DataFrame(),
        "path_to_title":      pd.read_csv(p("path_to_title.csv"))     if p("path_to_title.csv").exists()     else pd.DataFrame(),
        "variance_table":     pd.read_csv(p("variance_table.csv"))    if p("variance_table.csv").exists()    else pd.DataFrame(),
    }

if run_btn:
    with st.spinner(f"Ejecutando {n_sims:,} simulaciones…"):
        res = run_monte_carlo(n_simulations=n_sims, seed=seed, verbose=False, exclude_team=exclude_team)
        generate_all_charts(res["team_probabilities"], res["group_summary"],
                            res["third_place_stats"], res["variance_table"])
    st.cache_data.clear()
    label = f"✅ {n_sims:,} simulaciones completadas"
    if exclude_team: label += f" (sin {exclude_team})"
    st.success(label)
    data = {k: res[k] for k in ["team_probabilities","finals","group_summary","third_place_stats","path_to_title","variance_table"]}
else:
    data = load_saved()

if data is None:
    st.title("🌍 World Cup 2026 — Monte Carlo Simulator")
    st.info("👈 Configura y pulsa **Ejecutar simulación** para comenzar.")
    try:
        td = load_tournament_data()
        st.subheader("Equipos del Mundial 2026")
        for g, gdf in td.groupby("group"):
            with st.expander(f"Grupo {g}"):
                st.dataframe(gdf[["team","overall_rating"]].sort_values("overall_rating",ascending=False), use_container_width=True)
    except Exception: pass
    st.stop()

tp      = data["team_probabilities"]
finals  = data["finals"]
gs      = data["group_summary"]
thirds  = data["third_place_stats"]
path_df = data["path_to_title"]
var_df  = data["variance_table"]


# \\\\\\\\\\\
# Header y métricas top
# \\\\\\\\\\\
st.title("🌍 World Cup 2026 — Monte Carlo Simulator")

top  = tp.iloc[0]
fin1 = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
topf = finals.iloc[0] if not finals.empty else None
hardg = gs.iloc[0]   if not gs.empty else None

c1, c2, c3, c4 = st.columns(4)
def mcard(col, icon, title, val, sub=""):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

mcard(c1, "🏆", "Gran favorito",    top["team"],  f"{top['champion_pct']:.1f}% campeón")
mcard(c2, "🥈", "Más veces finalista", fin1["team"], f"{fin1['reach_final_pct']:.1f}% llega a la final")
mcard(c3, "🎯", "Final más probable", topf["final"].replace(" vs ","\nvs ") if topf is not None else "—",
      f"{topf['probability_pct']:.1f}%" if topf is not None else "")
mcard(c4, "💀", "Grupo más difícil", f"Grupo {hardg['group']}" if hardg is not None else "—",
      f"Rating medio {hardg['average_rating']}" if hardg is not None else "")

st.markdown("<br>", unsafe_allow_html=True)


# \\\\\\\\\\\
# Tabs
# \\\\\\\\\\\
tabs = st.tabs(["🏆 Favoritos", "🗂 Grupos", "📊 Por fase", "🎯 Finales",
                "3️⃣ Media puntos terceros", "🛣 Camino al título",
                "📉 Varianza", "🖼 Gráficos"])


# ─────────────────────────────────────────────
# TAB 1: FAVORITOS (tabla limpia sin confederaciones)
# ─────────────────────────────────────────────
with tabs[0]:
    st.subheader("Probabilidades de campeonato")

    col_f1, col_f2 = st.columns([1,1])
    with col_f1:
        grp_f = st.multiselect("Filtrar por grupo", sorted(tp["group"].unique()), key="grp_fav")
    with col_f2:
        top_n = st.slider("Mostrar top N equipos", 10, 48, 20, key="topn_fav")

    filt = tp.copy()
    if grp_f: filt = filt[filt["group"].isin(grp_f)]
    filt = filt.head(top_n)

    # Gráfico horizontal limpio
    fig, ax = plt.subplots(figsize=(10, max(4, len(filt)*0.4)))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
    colors = ["#238636" if i==0 else "#1f6feb" if i<3 else "#388bfd" if i<8 else "#8b949e"
              for i in range(len(filt))]
    bars = ax.barh(filt["team"], filt["champion_pct"], color=colors[::-1], edgecolor="none")
    for bar, val in zip(bars, filt["champion_pct"].values[::-1]):
        if val > 0:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", ha="left", fontsize=8.5, color="#e6edf3")
    ax.set_xlabel("Probabilidad de ser campeón (%)", color="#8b949e", fontsize=10)
    ax.tick_params(colors="#e6edf3", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.15, color="#30363d")
    ax.set_xlim(0, filt["champion_pct"].max()*1.3)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    # Tabla limpia
    display = filt[["team","group","overall_rating","champion_pct","reach_final_pct",
                     "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct"]].copy()
    display.columns = ["Selección","Grupo","Rating","Campeón %","Final %",
                       "Semis %","Cuartos %","Pasa grupos %","Eliminado grupos %"]
    display = display.reset_index(drop=True)
    display.index += 1
    st.dataframe(display.style
        .background_gradient(subset=["Campeón %"], cmap="Greens")
        .background_gradient(subset=["Eliminado grupos %"], cmap="Reds_r")
        .format({"Rating":"{:.1f}","Campeón %":"{:.1f}","Final %":"{:.1f}",
                 "Semis %":"{:.1f}","Cuartos %":"{:.1f}",
                 "Pasa grupos %":"{:.1f}","Eliminado grupos %":"{:.1f}"}),
        use_container_width=True, height=500)

    st.download_button("⬇ Descargar CSV", tp.to_csv(index=False).encode(), "probabilidades.csv", "text/csv")


# ─────────────────────────────────────────────
# TAB 2: GRUPOS DESGLOSADOS
# ─────────────────────────────────────────────
with tabs[1]:
    st.subheader("Grupos del Mundial 2026")

    try:
        td = load_tournament_data()
    except Exception:
        td = None

    if td is not None:
        # Dificultad general
        st.markdown("#### Dificultad de grupos por rating medio")
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        fig2.patch.set_facecolor("#0d1117"); ax2.set_facecolor("#161b22")
        gs_sorted = gs.sort_values("average_rating", ascending=False)
        q75 = gs_sorted["average_rating"].quantile(0.75)
        cols_g = ["#da3633" if r>=q75 else "#388bfd" for r in gs_sorted["average_rating"]]
        bars2 = ax2.bar([f"Grupo {g}" for g in gs_sorted["group"]],
                        gs_sorted["average_rating"], color=cols_g, edgecolor="none", width=0.6)
        for bar, val in zip(bars2, gs_sorted["average_rating"]):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{val:.1f}", ha="center", fontsize=9, color="#e6edf3")
        ax2.set_ylim(gs_sorted["average_rating"].min()*0.93, gs_sorted["average_rating"].max()*1.06)
        ax2.tick_params(colors="#8b949e"); ax2.spines[:].set_visible(False)
        ax2.grid(axis="y", alpha=0.15, color="#30363d")
        red_p = mpatches.Patch(color="#da3633", label="Grupos más duros")
        blue_p = mpatches.Patch(color="#388bfd", label="Grupos accesibles")
        ax2.legend(handles=[red_p, blue_p], facecolor="#161b22", labelcolor="#e6edf3", fontsize=9)
        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2); plt.close(fig2)

        st.markdown("#### Probabilidades por grupo (% de clasificarse 1°, 2° y como mejor 3°)")
        group_list = sorted(td["group"].unique())
        cols_per_row = 3
        for row_start in range(0, len(group_list), cols_per_row):
            row_groups = group_list[row_start:row_start+cols_per_row]
            cols_g2 = st.columns(len(row_groups))
            for col_g, grp in zip(cols_g2, row_groups):
                with col_g:
                    st.markdown(f"**Grupo {grp}**")
                    teams_in_grp = td[td["group"]==grp]["team"].tolist()
                    grp_probs = tp[tp["group"]==grp][["team","overall_rating","champion_pct",
                                                       "pass_group_stage_pct","group_exit_pct"]].copy()
                    grp_probs = grp_probs.sort_values("overall_rating", ascending=False)
                    grp_probs.columns = ["Equipo","Rating","Campeón %","Pasa %","Elim %"]
                    grp_probs = grp_probs.reset_index(drop=True)
                    grp_probs.index +=1
                    st.dataframe(grp_probs.style.format(
                        {"Rating":"{:.1f}","Campeón %":"{:.1f}","Pasa %":"{:.1f}","Elim %":"{:.1f}"}),
                        use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3: PROBABILIDAD POR FASE — selección concreta
# ─────────────────────────────────────────────
with tabs[2]:
    st.subheader("Probabilidad de alcanzar cada fase")

    team_sel = st.selectbox("Selección", tp["team"].tolist(), key="sel_phase")
    row = tp[tp["team"]==team_sel].iloc[0]

    phases = {
        "Pasa grupos":   row["pass_group_stage_pct"],
        "Dieciseisavos": row.get("reach_round32_pct", 0),
        "Octavos":       row.get("reach_round16_pct", 0),
        "Cuartos":       row["reach_quarters_pct"],
        "Semis":         row["reach_semis_pct"],
        "Final":         row["reach_final_pct"],
        "Campeón":       row["champion_pct"],
    }

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    fig3.patch.set_facecolor("#0d1117"); ax3.set_facecolor("#161b22")
    phase_vals = list(phases.values())
    phase_keys = list(phases.keys())
    gradient = plt.cm.YlOrRd(np.linspace(0.25, 0.9, len(phase_vals)))
    bars3 = ax3.bar(phase_keys, phase_vals, color=gradient, edgecolor="none", width=0.6)
    for bar, val in zip(bars3, phase_vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#e6edf3")
    ax3.set_ylabel("%", color="#8b949e"); ax3.tick_params(colors="#8b949e")
    ax3.spines[:].set_visible(False)
    ax3.grid(axis="y", alpha=0.15, color="#30363d")
    ax3.set_title(f"Probabilidades de {team_sel} por fase", color="#e6edf3", pad=12, fontsize=12)
    fig3.tight_layout(pad=1.5)
    st.pyplot(fig3); plt.close(fig3)

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Rating compuesto", f"{row['overall_rating']:.1f}")
    c_b.metric("Grupo", row["group"])
    c_c.metric("Probabilidad campeón", f"{row['champion_pct']:.2f}%")


# ─────────────────────────────────────────────
# TAB 4: FINALES + BRACKET MÁS COMÚN
# ─────────────────────────────────────────────
with tabs[3]:
    st.subheader("Finales más repetidas y bracket más probable")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Top 15 finales")
        if not finals.empty:
            top15 = finals.head(15).sort_values("probability_pct")
            fig4, ax4 = plt.subplots(figsize=(7, 6))
            fig4.patch.set_facecolor("#0d1117"); ax4.set_facecolor("#161b22")
            ax4.barh(top15["final"], top15["probability_pct"], color="#da3633", edgecolor="none", height=0.6)
            for _, row_f in top15.iterrows():
                ax4.text(row_f["probability_pct"]+0.03, top15.index.get_loc(row_f.name),
                         f"{row_f['probability_pct']:.2f}%", va="center", fontsize=8, color="#e6edf3")
            ax4.set_xlabel("Frecuencia (%)", color="#8b949e")
            ax4.tick_params(colors="#8b949e", labelsize=8)
            ax4.spines[:].set_visible(False)
            ax4.grid(axis="x", alpha=0.15, color="#30363d")
            fig4.tight_layout(pad=1.5)
            st.pyplot(fig4); plt.close(fig4)

    with col_right:
        st.markdown("#### Bracket más probable")
        if not path_df.empty:
            top3 = path_df.head(3)
            round_cols = [c for c in path_df.columns if "_rival" in c]
            round_labels = [c.replace("_rival","") for c in round_cols]

            for _, prow in top3.iterrows():
                st.markdown(f"**{prow['team']}** ({prow['champion_pct']:.1f}% campeón)")
                bracket_html = ""
                for rc in round_cols:
                    rnd = rc.replace("_rival","")
                    rival = prow.get(rc, "—")
                    freq_col = rc.replace("_rival","_freq_pct")
                    freq = prow.get(freq_col, 0)
                    bracket_html += f"""<div class="bracket-match bracket-winner">
                        <span style="color:#8b949e;font-size:11px">{rnd}</span>
                        <span style="color:#e6edf3;margin-left:8px">vs {rival}</span>
                        <span style="color:#58a6ff;float:right;font-size:11px">{freq:.0f}%</span>
                    </div>"""
                bracket_html += f"""<div class="bracket-match" style="background:#0d1117;border-color:#238636">
                    <span style="color:#238636;font-weight:700">🏆 CAMPEÓN</span>
                    <span style="color:#238636;float:right">{prow['champion_pct']:.1f}%</span>
                </div>"""
                st.markdown(bracket_html, unsafe_allow_html=True)
                st.markdown("")


# ─────────────────────────────────────────────
# TAB 5: MEDIA DE PUNTOS TERCEROS
# ─────────────────────────────────────────────
with tabs[4]:
    st.subheader("¿Con cuántos puntos te clasificas siendo tercero?")

    if thirds.empty:
        st.info("Ejecuta la simulación para ver esta información.")
    else:
        summary = thirds[thirds["categoria"].str.startswith("RESUMEN")]
        detail  = thirds[~thirds["categoria"].str.startswith("RESUMEN")]

        if not summary.empty:
            s_cols = st.columns(len(summary))
            for i, (_, sr) in enumerate(summary.iterrows()):
                label = sr["categoria"].replace("RESUMEN - ","")
                with s_cols[i]:
                    st.metric(label, f"{sr['puntos']:.2f} pts" if "Clasificados"==label else f"{int(sr['puntos'])} pts")

        st.markdown("---")
        st.markdown("#### Distribución de puntos: clasificados vs eliminados")

        img_path = OUTPUTS_DIR / "third_place_points.png"
        if img_path.exists():
            st.image(str(img_path), use_column_width=True)
        elif not detail.empty:
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            fig5.patch.set_facecolor("#0d1117"); ax5.set_facecolor("#161b22")
            for cat, color in [("Clasificado (top 8)","#238636"),("Eliminado","#da3633")]:
                sub = detail[detail["categoria"]==cat]
                if not sub.empty:
                    ax5.bar(sub["puntos"].astype(str), sub["frecuencia_pct"],
                            label=cat, color=color, alpha=0.85, edgecolor="none")
            ax5.legend(facecolor="#161b22", labelcolor="#e6edf3")
            ax5.tick_params(colors="#8b949e"); ax5.spines[:].set_visible(False)
            ax5.grid(axis="y", alpha=0.15, color="#30363d")
            fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)

        st.info("📌 Con **4 o más puntos** la clasificación como mejor tercero es muy probable. "
                "Con **3 puntos** depende de los otros grupos. Con **2 o menos** prácticamente imposible.")


# ─────────────────────────────────────────────
# TAB 6: CAMINO AL TÍTULO
# ─────────────────────────────────────────────
with tabs[5]:
    st.subheader("Camino más probable al título")
    st.caption("Muestra el rival más frecuente en cada ronda cuando ese equipo gana el torneo.")

    if path_df.empty:
        st.info("Sin datos.")
    else:
        n_show = st.slider("Equipos a mostrar", 3, min(15, len(path_df)), 8, key="path_n")
        show   = path_df.head(n_show)

        round_rivals = [c for c in show.columns if "_rival" in c]
        round_freqs  = [c for c in show.columns if "_freq_pct" in c]
        round_names  = [c.replace("_rival","") for c in round_rivals]

        for _, prow in show.iterrows():
            with st.expander(f"🏆 {prow['team']} — {prow['champion_pct']:.1f}% de ser campeón"):
                path_cols = st.columns(len(round_rivals)+1)
                for i, (rc, rn) in enumerate(zip(round_rivals, round_names)):
                    fc = rc.replace("_rival","_freq_pct")
                    rival = prow.get(rc,"—")
                    freq  = prow.get(fc, 0)
                    path_cols[i].metric(rn, rival, f"{freq:.0f}% de las veces")
                path_cols[-1].metric("🏆 Final", "CAMPEÓN", f"{prow['champion_pct']:.1f}%")


# ─────────────────────────────────────────────
# TAB 7: VARIANZA
# ─────────────────────────────────────────────
with tabs[6]:
    st.subheader("Varianza y consistencia por equipo")

    c_info1, c_info2 = st.columns(2)
    with c_info1:
        st.info("🟢 **Consistencia alta** → llega lejos en casi todas las simulaciones, pocas sorpresas.")
    with c_info2:
        st.info("🔴 **Varianza alta** → potencial de campeón pero también puede caer pronto.")

    img_var = OUTPUTS_DIR / "variance_scatter.png"
    if img_var.exists():
        st.image(str(img_var), use_column_width=True)

    if not var_df.empty:
        st.markdown("#### Tabla de consistencia")
        disp_var = var_df[["team","group","overall_rating","champion_pct",
                            "reach_semis_pct","group_exit_pct","consistency_index","variance_index"]].copy()
        disp_var.columns = ["Selección","Grupo","Rating","Campeón %","Semis %",
                             "Elim grupos %","Consistencia","Varianza"]
        disp_var = disp_var.reset_index(drop=True); disp_var.index+=1
        st.dataframe(disp_var.style
            .background_gradient(subset=["Consistencia"], cmap="Greens")
            .background_gradient(subset=["Varianza"], cmap="Reds")
            .format({"Rating":"{:.1f}","Campeón %":"{:.1f}","Semis %":"{:.1f}",
                     "Elim grupos %":"{:.1f}","Consistencia":"{:.2f}","Varianza":"{:.2f}"}),
            use_container_width=True)


# ─────────────────────────────────────────────
# TAB 8: GRÁFICOS
# ─────────────────────────────────────────────
with tabs[7]:
    st.subheader("Visualizaciones")

    imgs = [
        ("champion_probabilities.png",  "Probabilidad de ser campeón"),
        ("phase_heatmap.png",           "Heatmap de probabilidades por fase"),
        ("semifinal_probabilities.png", "Probabilidad de llegar a semifinales"),
        ("variance_scatter.png",        "Varianza: potencial vs inconsistencia"),
        ("group_difficulty.png",        "Dificultad de grupos"),
        ("third_place_points.png",      "Puntos de los terceros clasificados"),
    ]
    shown = 0
    for fname, caption in imgs:
        p = OUTPUTS_DIR / fname
        if p.exists():
            st.image(str(p), caption=caption, use_column_width=True)
            st.markdown("")
            shown += 1

    if shown == 0:
        st.info("Ejecuta la simulación para generar los gráficos.")
    else:
        if st.button("🔄 Regenerar gráficos"):
            generate_all_charts(tp, gs, thirds, var_df)
            st.success("Regenerados."); st.rerun()
