"""
streamlit_app.py — World Cup 2026 Monte Carlo Simulator
UI: Apex Stadium design system (navy + gold + glassmorphism)
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

st.set_page_config(
    page_title="FIFA WORLD CUP 2026 | Simulator",
    page_icon="🏆", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── APEX STADIUM CSS ───────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900&family=Inter:wght@400;600&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:        #05122d;
  --surface:   #0d1b36;
  --surface2:  #121f3a;
  --surface3:  #1d2945;
  --gold:      #e9c400;
  --gold-lt:   #ffd700;
  --green:     #00E676;
  --red:       #FF3D00;
  --blue:      #0A84FF;
  --text:      #d9e2ff;
  --muted:     #d0c6ab;
  --border:    rgba(255,255,255,0.12);
}

/* Reset Streamlit chrome */
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] {
  background: #010d28 !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: var(--gold) !important; }
.stButton > button {
  background: var(--gold) !important;
  color: #1a1400 !important;
  font-family: 'Montserrat', sans-serif !important;
  font-weight: 800 !important;
  font-size: 13px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 24px !important;
  width: 100% !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  box-shadow: 0 0 20px rgba(233,196,0,0.45) !important;
  transform: scale(1.02) !important;
}
[data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div,
.stSlider { color: var(--text) !important; }

/* Glass card */
.gc {
  background: rgba(255,255,255,0.045);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 16px;
}
.gc-gold  { border-top: 3px solid var(--gold); }
.gc-blue  { border-top: 3px solid var(--blue); }
.gc-green { border-top: 3px solid var(--green); }
.gc-red   { border-top: 3px solid var(--red); }

/* Typography */
.label-caps {
  font-family: 'Montserrat', sans-serif;
  font-size: 11px; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted);
}
.headline { font-family: 'Montserrat', sans-serif; font-weight: 800; color: #fff; }
.display  { font-family: 'Montserrat', sans-serif; font-weight: 900; color: #fff; font-size: 42px; line-height: 1.1; letter-spacing: -0.02em; }
.gold-text { color: var(--gold) !important; }
.green-text { color: var(--green) !important; }
.red-text   { color: var(--red) !important; }
.muted-text { color: var(--muted) !important; }

/* Metric card */
.metric-box {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 16px;
  text-align: center;
}
.metric-box .label { font-family:'Montserrat',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.metric-box .val   { font-family:'Montserrat',sans-serif; font-size:28px; font-weight:900; color:#fff; line-height:1; }
.metric-box .sub   { font-family:'Inter',sans-serif; font-size:12px; color:var(--gold); margin-top:5px; }

/* Live badge */
.live-badge {
  display:inline-flex; align-items:center; gap:6px;
  background: rgba(255,61,0,0.15); color:var(--red);
  border:1px solid rgba(255,61,0,0.3); border-radius:9999px;
  padding:4px 12px; font-family:'Montserrat',sans-serif;
  font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
}
.live-dot { width:7px; height:7px; background:var(--red); border-radius:50%; display:inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Prob bar */
.prob-row { margin-bottom:12px; }
.prob-row .name  { font-family:'Inter',sans-serif; font-size:13px; color:var(--text); }
.prob-row .pct   { font-family:'Montserrat',sans-serif; font-size:13px; font-weight:700; color:var(--green); }
.prob-bar-bg     { background:rgba(255,255,255,0.08); height:5px; border-radius:9999px; margin-top:4px; overflow:hidden; }
.prob-bar-fill   { height:100%; border-radius:9999px; }

/* Group table */
.grp-table { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:13px; }
.grp-table th { color:var(--muted); font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; padding:0 4px 8px; border-bottom:1px solid var(--border); }
.grp-table td { padding:10px 4px; border-bottom:1px solid rgba(255,255,255,0.05); color:var(--text); }
.grp-table tr:last-child td { border-bottom:none; }
.grp-table .pts { font-weight:700; }
.qualified-1 td:first-child { border-left:3px solid var(--gold); padding-left:8px; }
.qualified-2 td:first-child { border-left:3px solid var(--blue); padding-left:8px; }
.flag-emoji { font-size:18px; }

/* Bracket */
.bracket-wrap { font-family:'Inter',sans-serif; }
.bracket-rnd-title { font-family:'Montserrat',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--gold); margin-bottom:10px; text-align:center; }
.b-match {
  background:rgba(255,255,255,0.04); border:1px solid var(--border);
  border-radius:8px; padding:8px 12px; margin-bottom:6px;
}
.b-match .team { font-size:12px; color:var(--text); padding:3px 0; display:flex; justify-content:space-between; align-items:center; }
.b-match .winner { color:#fff; font-weight:700; }
.b-match .pct { font-size:11px; color:var(--muted); }
.b-champion {
  background: linear-gradient(135deg, rgba(233,196,0,0.2), rgba(233,196,0,0.05));
  border:1px solid rgba(233,196,0,0.4); border-radius:12px;
  padding:14px 18px; text-align:center;
  box-shadow: 0 0 24px rgba(233,196,0,0.2);
}
.b-champion .trophy { font-size:28px; margin-bottom:4px; }
.b-champion .name { font-family:'Montserrat',sans-serif; font-size:18px; font-weight:900; color:var(--gold); }
.b-champion .sub { font-size:11px; color:var(--muted); margin-top:2px; }

/* Tabs override */
[data-baseweb="tab-list"] { background: transparent !important; gap: 4px !important; border-bottom: 1px solid var(--border) !important; }
[data-baseweb="tab"] { background: transparent !important; border:none !important; color: var(--muted) !important; font-family:'Montserrat',sans-serif !important; font-weight:700 !important; font-size:12px !important; letter-spacing:0.08em !important; text-transform:uppercase !important; padding:10px 18px !important; border-radius:8px 8px 0 0 !important; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--gold) !important; background: rgba(233,196,0,0.08) !important; border-bottom: 2px solid var(--gold) !important; }
[data-testid="stHorizontalBlock"] > div { gap:16px !important; }

/* Sidebar labels */
.sidebar-section { font-family:'Montserrat',sans-serif; font-size:10px; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--gold); margin-bottom:4px; margin-top:16px; }

/* Third place card */
.t3-box {
  background:rgba(255,255,255,0.04); border:1px solid var(--border);
  border-radius:10px; padding:16px; text-align:center;
}
.t3-box .lbl { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; }
.t3-box .num { font-family:'Montserrat',sans-serif; font-size:32px; font-weight:900; color:var(--gold); }
.t3-box .unit { font-family:'Inter',sans-serif; font-size:11px; color:var(--muted); margin-top:2px; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(233,196,0,0.3); border-radius:10px; }

/* Dataframe dark */
[data-testid="stDataFrame"] { background:transparent !important; }
.dvn-scroller { background: var(--surface) !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── FLAGS (emoji por país) ──────────────────────────────────────────────────
FLAGS = {
    "Argentina":"🇦🇷","Spain":"🇪🇸","France":"🇫🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Brazil":"🇧🇷","Portugal":"🇵🇹","Morocco":"🇲🇦","Netherlands":"🇳🇱",
    "Germany":"🇩🇪","Croatia":"🇭🇷","Belgium":"🇧🇪","Colombia":"🇨🇴",
    "Uruguay":"🇺🇾","Mexico":"🇲🇽","USA":"🇺🇸","Switzerland":"🇨🇭",
    "Japan":"🇯🇵","Senegal":"🇸🇳","Iran":"🇮🇷","Norway":"🇳🇴",
    "South Korea":"🇰🇷","Turkey":"🇹🇷","Austria":"🇦🇹","Australia":"🇦🇺",
    "Sweden":"🇸🇪","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Algeria":"🇩🇿","Ecuador":"🇪🇨",
    "Ivory Coast":"🇨🇮","Egypt":"🇪🇬","Tunisia":"🇹🇳","Ghana":"🇬🇭",
    "Saudi Arabia":"🇸🇦","Czech Republic":"🇨🇿","South Africa":"🇿🇦",
    "Paraguay":"🇵🇾","Bosnia Herzegovina":"🇧🇦","Qatar":"🇶🇦",
    "Canada":"🇨🇦","DR Congo":"🇨🇩","Uzbekistan":"🇺🇿","Panama":"🇵🇦",
    "Jordan":"🇯🇴","Iraq":"🇮🇶","New Zealand":"🇳🇿","Haiti":"🇭🇹",
    "Curacao":"🇨🇼","Cape Verde":"🇨🇻",
}

def flag(team): return FLAGS.get(team, "🏳")

GROUP_COLORS = {
    "A":"var(--gold)","B":"var(--blue)","C":"var(--green)","D":"#ff9800",
    "E":"#e040fb","F":"var(--red)","G":"#00bcd4","H":"#8bc34a",
    "I":"#ff5722","J":"var(--gold)","K":"var(--blue)","L":"var(--green)",
}


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="label-caps" style="color:var(--gold);font-size:18px;font-weight:900;font-family:Montserrat">⚽ FIFA WC 2026</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:var(--muted);margin-top:-8px">Monte Carlo Simulator · Dixon-Coles</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="sidebar-section">Sim Control</p>', unsafe_allow_html=True)
    n_sims = st.slider("Simulaciones", 500, 10000, 2000, 500)
    seed   = st.number_input("Semilla aleatoria", 0, 9999, 42)
    st.markdown('<p class="sidebar-section">Escenario</p>', unsafe_allow_html=True)
    try:
        td_s = load_tournament_data()
        team_opts = ["(torneo completo)"] + sorted(td_s["team"].tolist())
    except Exception:
        team_opts = ["(torneo completo)"]
    exclude = st.selectbox("Simular sin equipo", team_opts, label_visibility="collapsed")
    exclude_team = None if exclude == "(torneo completo)" else exclude
    if exclude_team:
        st.markdown(f'<div style="background:rgba(255,61,0,0.12);border:1px solid rgba(255,61,0,0.3);border-radius:8px;padding:8px 12px;font-size:12px;color:var(--red)">⚡ Sin <b>{exclude_team}</b></div>', unsafe_allow_html=True)
    st.markdown("---")
    run_btn = st.button("▶ EJECUTAR SIMULACIÓN")
    st.markdown("")
    st.markdown('<p style="font-size:10px;color:var(--muted);text-align:center">Datos: FIFA Rankings jun 2026<br>Modelo: Dixon-Coles + Confederación</p>', unsafe_allow_html=True)


# ─── LOAD / RUN ──────────────────────────────────────────────────────────────
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
    with st.spinner(""):
        st.markdown('<div class="live-badge"><span class="live-dot"></span>SIMULANDO…</div>', unsafe_allow_html=True)
        res = run_monte_carlo(n_simulations=n_sims, seed=seed, verbose=False, exclude_team=exclude_team)
        generate_all_charts(res["team_probabilities"], res["group_summary"],
                            res["third_place_stats"], res["variance_table"])
    st.cache_data.clear()
    st.success(f"✅ {n_sims:,} simulaciones completadas" + (f" · sin {exclude_team}" if exclude_team else ""))
    data = {k: res[k] for k in ["team_probabilities","finals","group_summary","third_place_stats","path_to_title","variance_table"]}
else:
    data = load_saved()

if data is None:
    st.markdown('<p class="display">FIFA WORLD CUP<br><span class="gold-text">2026</span></p>', unsafe_allow_html=True)
    st.markdown('<div class="live-badge"><span class="live-dot"></span>AWAITING SIMULATION</div>', unsafe_allow_html=True)
    st.markdown('<br><p style="color:var(--muted);font-family:Inter">👈 Configura y pulsa <b style="color:var(--gold)">Ejecutar Simulación</b> para comenzar.</p>', unsafe_allow_html=True)
    try:
        td = load_tournament_data()
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="headline" style="font-size:18px">GRUPOS DEL MUNDIAL 2026</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (g, gdf) in enumerate(td.groupby("group")):
            with cols[i % 4]:
                teams_html = "".join([f'<div style="padding:4px 0;font-size:13px;color:var(--text)">{flag(t)} {t}</div>' for t in gdf["team"].tolist()])
                gc = GROUP_COLORS.get(g, "var(--gold)")
                st.markdown(f'<div class="gc" style="border-top:3px solid {gc}"><p class="label-caps" style="color:{gc}">Grupo {g}</p>{teams_html}</div>', unsafe_allow_html=True)
    except Exception: pass
    st.stop()

tp      = data["team_probabilities"]
finals  = data["finals"]
gs      = data["group_summary"]
thirds  = data["third_place_stats"]
path_df = data["path_to_title"]
var_df  = data["variance_table"]


# ─── HERO HEADER ─────────────────────────────────────────────────────────────
top  = tp.iloc[0]
fin1 = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
topf = finals.iloc[0] if not finals.empty else None

col_hero, col_badge = st.columns([3, 1])
with col_hero:
    st.markdown('<p class="display">TOURNAMENT<br><span class="gold-text">SIMULATOR</span></p>', unsafe_allow_html=True)
    st.markdown('<div class="live-badge"><span class="live-dot"></span>SIMULATION COMPLETE · DIXON-COLES MODEL</div>', unsafe_allow_html=True)
with col_badge:
    n_sim_val = 2000
    try:
        log = pd.read_csv(OUTPUTS_DIR / "simulation_log.csv")
        n_sim_val = int(log[log["metric"]=="n_simulations"]["value"].iloc[0])
    except: pass
    st.markdown(f'''<div class="metric-box" style="margin-top:12px">
        <div class="label">Simulaciones</div>
        <div class="val">{n_sim_val:,}</div>
        <div class="sub">Mundial 2026 · 48 equipos</div>
    </div>''', unsafe_allow_html=True)

st.markdown('<hr style="border-color:var(--border);margin:20px 0">', unsafe_allow_html=True)


# ─── KPI METRICS ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'''<div class="metric-box gc-gold">
    <div class="label">🏆 Gran Favorito</div>
    <div class="val">{flag(top["team"])} {top["team"]}</div>
    <div class="sub">{top["champion_pct"]:.1f}% probabilidad campeón</div>
</div>''', unsafe_allow_html=True)
c2.markdown(f'''<div class="metric-box gc-blue">
    <div class="label">🥈 Más Veces Finalista</div>
    <div class="val">{flag(fin1["team"])} {fin1["team"]}</div>
    <div class="sub">{fin1["reach_final_pct"]:.1f}% llega a la final</div>
</div>''', unsafe_allow_html=True)
c3.markdown(f'''<div class="metric-box gc-green">
    <div class="label">🎯 Final Más Probable</div>
    <div class="val" style="font-size:16px;padding-top:4px">{topf["final"] if topf is not None else "—"}</div>
    <div class="sub">{f"{topf['probability_pct']:.1f}% de simulaciones" if topf is not None else ""}</div>
</div>''', unsafe_allow_html=True)
hardg = gs.iloc[0] if not gs.empty else None
c4.markdown(f'''<div class="metric-box gc-red">
    <div class="label">💀 Grupo Más Duro</div>
    <div class="val">Grupo {hardg["group"] if hardg is not None else "—"}</div>
    <div class="sub">Rating medio {hardg["average_rating"] if hardg is not None else "—"}</div>
</div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 DASHBOARD",
    "🗂 GRUPOS",
    "🏆 BRACKET",
    "📊 FAVORITOS",
    "3️⃣ TERCEROS",
    "📉 VARIANZA",
    "⚙ ESCENARIOS",
])


# ══════════════════════════════════════════════════════
# TAB 0 — DASHBOARD: grupos + win probability (como el diseño)
# ══════════════════════════════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<p class="headline" style="font-size:20px;margin-bottom:16px">GROUP STAGE OVERVIEW</p>', unsafe_allow_html=True)
        try:
            td = load_tournament_data()
            group_list = sorted(td["group"].unique())
            # 2 grupos por fila
            for row_i in range(0, len(group_list), 2):
                gcols = st.columns(2)
                for ci, grp in enumerate(group_list[row_i:row_i+2]):
                    with gcols[ci]:
                        gc = GROUP_COLORS.get(grp, "var(--gold)")
                        gdf = td[td["group"]==grp].copy()
                        grp_tp = tp[tp["group"]==grp].set_index("team")
                        rows_html = ""
                        for rank_i, (_, row_t) in enumerate(gdf.sort_values("overall_rating",ascending=False).iterrows()):
                            t = row_t["team"]
                            pct = grp_tp.loc[t, "pass_group_stage_pct"] if t in grp_tp.index else 0
                            champ = grp_tp.loc[t, "champion_pct"] if t in grp_tp.index else 0
                            qualifier_class = "qualified-1" if rank_i==0 else ("qualified-2" if rank_i==1 else "")
                            rows_html += f'''<tr class="{qualifier_class}">
                              <td style="padding:8px 4px;display:flex;align-items:center;gap:8px">
                                <span style="font-size:16px">{flag(t)}</span>
                                <span style="font-family:Inter;font-size:13px;color:#fff">{t}</span>
                              </td>
                              <td style="text-align:center;font-size:12px;color:var(--muted)">{row_t["overall_rating"]:.0f}</td>
                              <td style="text-align:right;font-family:Montserrat;font-size:12px;font-weight:700;color:{gc}">{champ:.1f}%</td>
                            </tr>'''
                        st.markdown(f'''<div class="gc" style="border-top:3px solid {gc}">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                            <span class="label-caps" style="color:{gc};font-size:13px;font-weight:800">GROUP {grp}</span>
                            <span class="label-caps" style="font-size:9px">Rating · Campeón</span>
                          </div>
                          <table class="grp-table">
                            <thead><tr>
                              <th style="text-align:left">Equipo</th>
                              <th style="text-align:center">Rtg</th>
                              <th style="text-align:right">🏆 %</th>
                            </tr></thead>
                            <tbody>{rows_html}</tbody>
                          </table>
                        </div>''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error cargando grupos: {e}")

    with col_right:
        # Win probability panel
        st.markdown(f'''<div class="gc gc-green">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
            <span class="headline" style="font-size:16px">Win Probability</span>
            <span class="label-caps" style="background:var(--surface3);padding:3px 8px;border-radius:4px">TOP 12</span>
          </div>''', unsafe_allow_html=True)
        tp_top = tp.head(12)
        max_pct = tp_top["champion_pct"].max()
        bars_html = ""
        for _, row_p in tp_top.iterrows():
            pct = row_p["champion_pct"]
            w   = (pct / max_pct * 100) if max_pct > 0 else 0
            bars_html += f'''<div class="prob-row">
              <div style="display:flex;justify-content:space-between">
                <span class="name">{flag(row_p["team"])} {row_p["team"]}</span>
                <span class="pct">{pct:.1f}%</span>
              </div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:var(--green)"></div></div>
            </div>'''
        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

        # Reach semis
        st.markdown(f'''<div class="gc gc-gold" style="margin-top:0">
          <span class="headline" style="font-size:16px;display:block;margin-bottom:12px">Semifinal Odds</span>''', unsafe_allow_html=True)
        tp_sf = tp.nlargest(8, "reach_semis_pct")
        max_sf = tp_sf["reach_semis_pct"].max()
        sf_html = ""
        for _, row_s in tp_sf.iterrows():
            pct = row_s["reach_semis_pct"]
            w   = (pct / max_sf * 100) if max_sf > 0 else 0
            sf_html += f'''<div class="prob-row">
              <div style="display:flex;justify-content:space-between">
                <span class="name">{flag(row_s["team"])} {row_s["team"]}</span>
                <span style="font-family:Montserrat;font-size:13px;font-weight:700;color:var(--gold)">{pct:.1f}%</span>
              </div>
              <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:var(--gold)"></div></div>
            </div>'''
        st.markdown(sf_html + "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 1 — GRUPOS: desglose individual
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">FASE DE GRUPOS</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);font-size:13px;margin-bottom:20px">Probabilidades por equipo dentro de cada grupo. 🥇🥈 pasan automáticamente. El mejor 3° también clasifica.</p>', unsafe_allow_html=True)

    try:
        td = load_tournament_data()
        group_list = sorted(td["group"].unique())
        for grp in group_list:
            gc_col = GROUP_COLORS.get(grp, "var(--gold)")
            gdf = td[td["group"]==grp].copy()
            grp_tp = tp[tp["group"]==grp].set_index("team")
            rows = ""
            for rank_i, (_, row_t) in enumerate(gdf.sort_values("overall_rating", ascending=False).iterrows()):
                t = row_t["team"]
                pct   = grp_tp.loc[t,"pass_group_stage_pct"] if t in grp_tp.index else 0
                champ = grp_tp.loc[t,"champion_pct"] if t in grp_tp.index else 0
                semis = grp_tp.loc[t,"reach_semis_pct"] if t in grp_tp.index else 0
                elim  = grp_tp.loc[t,"group_exit_pct"] if t in grp_tp.index else 0
                bar_w = pct
                medal = ["🥇","🥈","🥉","4️⃣"][rank_i] if rank_i < 4 else ""
                rows += f'''<tr>
                  <td style="padding:10px 8px;display:flex;align-items:center;gap:10px">
                    <span>{medal}</span>
                    <span style="font-size:18px">{flag(t)}</span>
                    <span style="font-family:Inter;font-size:14px;color:#fff;font-weight:600">{t}</span>
                  </td>
                  <td style="text-align:center;color:var(--muted);font-size:13px">{row_t["overall_rating"]:.0f}</td>
                  <td style="text-align:center">
                    <div style="background:rgba(255,255,255,0.06);border-radius:6px;padding:3px 10px;display:inline-block">
                      <span style="font-family:Montserrat;font-weight:700;font-size:13px;color:var(--green)">{pct:.0f}%</span>
                    </div>
                  </td>
                  <td style="text-align:center;font-family:Montserrat;font-size:13px;font-weight:700;color:var(--gold)">{champ:.1f}%</td>
                  <td style="text-align:center;font-family:Montserrat;font-size:13px;color:#8bc3ff">{semis:.1f}%</td>
                  <td style="text-align:center;font-family:Montserrat;font-size:13px;color:var(--red)">{elim:.1f}%</td>
                  <td style="width:120px;padding-right:8px">
                    <div style="background:rgba(255,255,255,0.07);height:4px;border-radius:999px;overflow:hidden">
                      <div style="width:{bar_w}%;height:100%;background:{gc_col};border-radius:999px"></div>
                    </div>
                  </td>
                </tr>'''
            st.markdown(f'''<div class="gc" style="border-top:3px solid {gc_col};margin-bottom:12px">
              <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span class="headline" style="font-size:20px;color:{gc_col}">GRUPO {grp}</span>
              </div>
              <table style="width:100%;border-collapse:collapse;font-family:Inter;font-size:13px">
                <thead><tr style="border-bottom:1px solid var(--border)">
                  <th style="text-align:left;padding:0 8px 8px;color:var(--muted);font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase">Equipo</th>
                  <th style="text-align:center;padding:0 4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase">Rating</th>
                  <th style="text-align:center;padding:0 4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase">Pasa %</th>
                  <th style="text-align:center;padding:0 4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase">🏆 %</th>
                  <th style="text-align:center;padding:0 4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase">Semis %</th>
                  <th style="text-align:center;padding:0 4px 8px;color:var(--muted);font-size:10px;text-transform:uppercase">Elim %</th>
                  <th style="padding:0 8px 8px"></th>
                </tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════
# TAB 2 — BRACKET
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">KNOCKOUT BRACKET</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);font-size:13px;margin-bottom:20px">Camino más probable al título para los principales favoritos.</p>', unsafe_allow_html=True)

    if not path_df.empty:
        # Selector de equipo
        top_teams = path_df.head(12)["team"].tolist()
        sel_team  = st.selectbox("Ver bracket de", top_teams, key="bracket_sel")
        row_b = path_df[path_df["team"]==sel_team].iloc[0] if sel_team in path_df["team"].values else None

        if row_b is not None:
            round_map = {
                "Dieciseisavos_rival": ("R32","Dieciseisavos","Dieciseisavos_freq_pct"),
                "Octavos_rival":       ("R16","Octavos","Octavos_freq_pct"),
                "Cuartos_rival":       ("QF","Cuartos","Cuartos_freq_pct"),
                "Semifinal_rival":     ("SF","Semifinal","Semifinal_freq_pct"),
            }

            # Bracket visual estilo columnas
            rnd_cols = st.columns(5)
            rnd_labels = ["Dieciseisavos","Octavos","Cuartos","Semifinal","Final"]
            rnd_rivals = {}
            for col_key, (short, label, freq_col) in round_map.items():
                if col_key in row_b.index:
                    rnd_rivals[label] = (row_b[col_key], row_b.get(freq_col, 0))

            for i, (label, col) in enumerate(zip(rnd_labels, rnd_cols)):
                with col:
                    st.markdown(f'<div class="bracket-rnd-title">{label}</div>', unsafe_allow_html=True)
                    if label in rnd_rivals:
                        rival, freq = rnd_rivals[label]
                        match_html = f'''<div class="b-match">
                          <div class="team winner">{flag(sel_team)} {sel_team}</div>
                          <div style="font-size:9px;color:var(--muted);text-align:center;padding:2px 0">vs · {freq:.0f}% de las veces</div>
                          <div class="team" style="color:var(--muted)">{flag(rival)} {rival}</div>
                        </div>'''
                        st.markdown(match_html, unsafe_allow_html=True)
                    elif label == "Final":
                        fin_rival = finals.iloc[0]["final"].replace(f"{sel_team} vs ","").replace(f" vs {sel_team}","") if not finals.empty and sel_team in finals.iloc[0]["final"] else "Rival"
                        st.markdown(f'''<div class="b-match">
                          <div class="team winner">{flag(sel_team)} {sel_team}</div>
                          <div style="font-size:9px;color:var(--muted);text-align:center;padding:2px 0">vs</div>
                          <div class="team" style="color:var(--muted)">{flag(fin_rival)} {fin_rival}</div>
                        </div>''', unsafe_allow_html=True)

            # Campeón
            st.markdown(f'''<div class="b-champion" style="margin-top:16px">
              <div class="trophy">🏆</div>
              <div class="name">{flag(sel_team)} {sel_team}</div>
              <div class="sub">Campeón en el {row_b["champion_pct"]:.1f}% de simulaciones</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="headline" style="font-size:18px;margin-bottom:12px">FINALES MÁS REPETIDAS</p>', unsafe_allow_html=True)
        fin_cols = st.columns(3)
        for i, (_, row_f) in enumerate(finals.head(9).iterrows()):
            with fin_cols[i % 3]:
                teams_f = row_f["final"].split(" vs ")
                t1 = teams_f[0].strip(); t2 = teams_f[1].strip() if len(teams_f)>1 else ""
                st.markdown(f'''<div class="gc" style="text-align:center;padding:14px">
                  <div style="font-family:Montserrat;font-size:11px;font-weight:700;color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px">Final #{i+1}</div>
                  <div style="font-size:22px;margin-bottom:4px">{flag(t1)} vs {flag(t2)}</div>
                  <div style="font-family:Inter;font-size:12px;color:var(--text)">{t1} vs {t2}</div>
                  <div style="font-family:Montserrat;font-weight:700;font-size:16px;color:var(--gold);margin-top:8px">{row_f["probability_pct"]:.1f}%</div>
                </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3 — FAVORITOS tabla completa
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">PROBABILIDADES COMPLETAS</p>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([1,1])
    with col_f1:
        grp_f = st.multiselect("Filtrar por grupo", sorted(tp["group"].unique()), key="grp_fav2")
    with col_f2:
        top_n = st.slider("Top N", 10, 48, 24, key="topn2")

    filt = tp.copy()
    if grp_f: filt = filt[filt["group"].isin(grp_f)]
    filt = filt.head(top_n).reset_index(drop=True)
    filt.index += 1

    # Gráfico horizontal estilo Apex
    fig, ax = plt.subplots(figsize=(10, max(5, len(filt)*0.38)))
    fig.patch.set_facecolor("#05122d"); ax.set_facecolor("#0d1b36")
    colors = ["#e9c400" if i<1 else "#0A84FF" if i<3 else "#00E676" if i<8 else "#4d5a78" for i in range(len(filt))]
    bars = ax.barh([f"{flag(t)} {t}" for t in filt["team"]], filt["champion_pct"], color=colors[::-1], edgecolor="none", height=0.65)
    for bar, val in zip(bars, filt["champion_pct"].values[::-1]):
        if val > 0.1:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#d9e2ff")
    ax.set_xlabel("% Probabilidad de ser Campeón", color="#d0c6ab", fontsize=10)
    ax.tick_params(colors="#d9e2ff", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.1, color="#283451")
    ax.set_xlim(0, filt["champion_pct"].max()*1.3)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close(fig)

    # Tabla
    disp = filt[["team","group","overall_rating","champion_pct","reach_final_pct",
                 "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct"]].copy()
    disp.columns = ["Selección","Grupo","Rating","🏆 Campeón %","🥈 Final %","🏟 Semis %","⚔ Cuartos %","✅ Pasa grupos %","❌ Elim grupos %"]
    st.dataframe(disp.style
        .background_gradient(subset=["🏆 Campeón %"], cmap="YlOrRd")
        .background_gradient(subset=["❌ Elim grupos %"], cmap="Reds_r")
        .format({c:"{:.1f}" for c in disp.columns if "%" in c or c=="Rating"}),
        use_container_width=True, height=500)
    st.download_button("⬇ Descargar CSV", tp.to_csv(index=False).encode(), "probabilidades_wc2026.csv", "text/csv")


# ══════════════════════════════════════════════════════
# TAB 4 — MEDIA PUNTOS TERCEROS
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">MEJOR TERCER CLASIFICADO</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);font-size:13px;margin-bottom:20px">¿Con cuántos puntos te clasificas siendo tercero de grupo en el Mundial 2026?</p>', unsafe_allow_html=True)

    if thirds.empty:
        st.info("Ejecuta la simulación para ver esta información.")
    else:
        summary = thirds[thirds["categoria"].str.startswith("RESUMEN")]
        detail  = thirds[~thirds["categoria"].str.startswith("RESUMEN")]

        if not summary.empty:
            t3_cols = st.columns(len(summary))
            icons = ["📊","🔽","🔼","⚠️"]
            for i, (_, sr) in enumerate(summary.iterrows()):
                label = sr["categoria"].replace("RESUMEN - ","")
                val   = sr["puntos"]
                with t3_cols[i]:
                    ic = icons[i] if i < len(icons) else "•"
                    st.markdown(f'''<div class="t3-box">
                      <div class="lbl">{ic} {label}</div>
                      <div class="num">{val:.2f if "Clasif" in label else int(val)}</div>
                      <div class="unit">puntos</div>
                    </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabla pivot
        if not detail.empty:
            st.markdown('<p class="label-caps">Distribución de puntos · Clasificados vs Eliminados</p>', unsafe_allow_html=True)
            try:
                pivot = detail.pivot_table(index="puntos", columns="categoria", values="frecuencia_pct", aggfunc="sum").fillna(0).reset_index()
                pivot["puntos"] = pivot["puntos"].astype(int).astype(str) + " pts"
                st.dataframe(pivot, use_container_width=True)
            except Exception: pass

        st.markdown(f'''<div class="gc gc-gold" style="margin-top:16px">
          <p class="headline" style="font-size:16px;margin-bottom:8px">📌 Conclusión práctica</p>
          <p style="font-family:Inter;font-size:14px;color:var(--text);line-height:1.6">
            Con <b style="color:var(--gold)">4 o más puntos</b> la clasificación como mejor tercero es muy probable.<br>
            Con <b style="color:#8bc3ff">3 puntos</b> depende del rendimiento del resto de grupos.<br>
            Con <b style="color:var(--red)">2 puntos o menos</b> es prácticamente imposible clasificarse.
          </p>
        </div>''', unsafe_allow_html=True)

        # Gráfico si existe
        img_path = OUTPUTS_DIR / "third_place_points.png"
        if img_path.exists():
            st.image(str(img_path), use_column_width=True)


# ══════════════════════════════════════════════════════
# TAB 5 — VARIANZA
# ══════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">VARIANZA Y CONSISTENCIA</p>', unsafe_allow_html=True)

    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown('''<div class="gc gc-green">
          <p style="font-family:Montserrat;font-weight:700;font-size:13px;color:var(--green)">🟢 ALTA CONSISTENCIA</p>
          <p style="font-size:13px;color:var(--text)">Llegan lejos en casi todas las simulaciones. Pocas sorpresas.</p>
        </div>''', unsafe_allow_html=True)
    with cv2:
        st.markdown('''<div class="gc gc-red">
          <p style="font-family:Montserrat;font-weight:700;font-size:13px;color:var(--red)">🔴 ALTA VARIANZA</p>
          <p style="font-size:13px;color:var(--text)">Alto potencial de campeón pero también pueden caer en grupos.</p>
        </div>''', unsafe_allow_html=True)

    img_var = OUTPUTS_DIR / "variance_scatter.png"
    if img_var.exists():
        st.image(str(img_var), use_column_width=True)

    if not var_df.empty:
        disp_var = var_df[["team","group","overall_rating","champion_pct","reach_semis_pct","group_exit_pct","consistency_index","variance_index"]].copy()
        disp_var.columns = ["Selección","Grupo","Rating","🏆 Campeón %","Semis %","Elim grupos %","📈 Consistencia","⚡ Varianza"]
        disp_var = disp_var.reset_index(drop=True); disp_var.index += 1
        st.dataframe(disp_var.style
            .background_gradient(subset=["📈 Consistencia"], cmap="Greens")
            .background_gradient(subset=["⚡ Varianza"], cmap="Reds")
            .format({c:"{:.2f}" for c in ["📈 Consistencia","⚡ Varianza"]} | {c:"{:.1f}" for c in disp_var.columns if "%" in c or c=="Rating"}),
            use_container_width=True)


# ══════════════════════════════════════════════════════
# TAB 6 — ESCENARIOS
# ══════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="headline" style="font-size:22px;margin-bottom:4px">ANÁLISIS DE ESCENARIOS</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:var(--muted);font-size:13px;margin-bottom:20px">Usa el sidebar para excluir un equipo y ver cómo cambia el torneo.</p>', unsafe_allow_html=True)

    if exclude_team:
        st.markdown(f'''<div class="gc gc-red">
          <p class="headline" style="font-size:16px;color:var(--red)">⚡ Simulación activa: sin {flag(exclude_team)} {exclude_team}</p>
          <p style="font-size:13px;color:var(--text)">Todos los resultados de las otras pestañas reflejan este escenario.</p>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="gc">
          <p style="font-size:14px;color:var(--muted)">Selecciona un equipo en el sidebar y pulsa <b style="color:var(--gold)">Ejecutar Simulación</b>.</p>
          <p style="font-size:13px;color:var(--muted);margin-top:8px">Ejemplos de análisis posibles:</p>
          <ul style="font-size:13px;color:var(--text);margin-top:4px">
            <li>¿Cómo cambia el torneo sin Argentina?</li>
            <li>¿Quién se beneficia si Francia no juega?</li>
            <li>¿Mejoran las opciones de Brasil sin España en el cuadro?</li>
          </ul>
        </div>''', unsafe_allow_html=True)

    st.markdown('<p class="headline" style="font-size:18px;margin:16px 0 8px">TOP 10 FAVORITOS ACTUALES</p>', unsafe_allow_html=True)
    top10_rows = ""
    for i, (_, r) in enumerate(tp.head(10).iterrows()):
        bar_w = (r["champion_pct"] / tp.iloc[0]["champion_pct"] * 100) if tp.iloc[0]["champion_pct"] > 0 else 0
        medal = ["🥇","🥈","🥉"] + [f"{j}." for j in range(4,11)]
        top10_rows += f'''<tr>
          <td style="padding:10px 8px;font-size:20px">{medal[i]}</td>
          <td style="padding:10px 8px;display:flex;align-items:center;gap:10px">
            <span style="font-size:20px">{flag(r["team"])}</span>
            <span style="font-family:Inter;font-size:14px;color:#fff;font-weight:600">{r["team"]}</span>
          </td>
          <td style="text-align:center;font-size:12px;color:var(--muted)">{r["group"]}</td>
          <td style="text-align:right;font-family:Montserrat;font-weight:700;font-size:16px;color:var(--gold)">{r["champion_pct"]:.1f}%</td>
          <td style="width:150px;padding-right:16px">
            <div style="background:rgba(255,255,255,0.07);height:5px;border-radius:999px;overflow:hidden">
              <div style="width:{bar_w:.0f}%;height:100%;background:var(--gold);border-radius:999px"></div>
            </div>
          </td>
        </tr>'''
    st.markdown(f'''<div class="gc">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="border-bottom:1px solid var(--border)">
          <th style="padding:0 8px 8px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em"></th>
          <th style="padding:0 8px 8px;text-align:left;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">Equipo</th>
          <th style="padding:0 8px 8px;text-align:center;color:var(--muted);font-size:10px;text-transform:uppercase">Grupo</th>
          <th style="padding:0 8px 8px;text-align:right;color:var(--muted);font-size:10px;text-transform:uppercase">Campeón %</th>
          <th></th>
        </tr></thead>
        <tbody>{top10_rows}</tbody>
      </table>
    </div>''', unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('''<hr style="border-color:var(--border);margin:32px 0 16px">
<div style="display:flex;justify-content:space-between;align-items:center;opacity:0.5">
  <p style="font-family:Inter;font-size:11px;color:var(--muted)">© 2026 FIFA World Cup Simulation Engine · Dixon-Coles Model · Datos FIFA Rankings Jun 2026</p>
  <p style="font-family:Montserrat;font-size:11px;color:var(--gold)">⚡ All data is for experimental analysis</p>
</div>''', unsafe_allow_html=True)
