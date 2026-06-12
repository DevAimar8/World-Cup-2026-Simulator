"""
streamlit_app.py — FIFA World Cup 2026 Monte Carlo Simulator
UI: Apex Stadium · Navy + Gold + Glassmorphism
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

from src.monte_carlo import run_monte_carlo
from src.data_loader import load_tournament_data
from src.analysis import generate_group_summary
from src.visualizations import generate_all_charts
from src.config import OUTPUTS_DIR

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 | Simulator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── INJECT FONTS + CSS ──────────────────────────────────────────────────────
# Fonts first (separate call, no style tag issues)
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800;900'
    '&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>',
    unsafe_allow_html=True,
)

# CSS in one clean block
st.markdown("""<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #020f2a !important;
    color: #d9e2ff !important;
}
[data-testid="stHeader"] { background: transparent !important; display:none; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stMainBlockContainer"] { padding-top: 24px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #010d28 !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #d9e2ff !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div { background: #e9c400 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Run button ── */
.stButton > button {
    background: #e9c400 !important;
    color: #0d0a00 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
    font-size: 12px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    width: 100% !important;
    transition: box-shadow 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 22px rgba(233,196,0,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    gap: 2px !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: rgba(208,198,171,0.7) !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #e9c400 !important;
    background: rgba(233,196,0,0.07) !important;
    border-bottom: 2px solid #e9c400 !important;
}
[data-baseweb="tab-panel"] { background: transparent !important; }

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #d9e2ff !important;
    border-radius: 8px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background: transparent !important; }
.dvn-scroller { background: #0d1b36 !important; border-radius: 8px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(233,196,0,0.3); border-radius: 10px; }

/* ── Glass card ── */
.gc {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.gc-gold  { border-top: 3px solid #e9c400 !important; }
.gc-blue  { border-top: 3px solid #0A84FF !important; }
.gc-green { border-top: 3px solid #00E676 !important; }
.gc-red   { border-top: 3px solid #FF3D00 !important; }

/* ── Typography helpers ── */
.lc { font-family:'Montserrat',sans-serif; font-size:10px; font-weight:700;
      letter-spacing:0.13em; text-transform:uppercase; color:#d0c6ab; }
.hl { font-family:'Montserrat',sans-serif; font-weight:800; color:#fff; }
.dp { font-family:'Montserrat',sans-serif; font-weight:900; color:#fff;
      font-size:40px; line-height:1.1; letter-spacing:-0.02em; }

/* ── KPI box ── */
.kpi {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    height: 100%;
}
.kpi-lbl { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
           letter-spacing:0.13em; text-transform:uppercase; color:#d0c6ab; margin-bottom:8px; }
.kpi-val { font-family:'Montserrat',sans-serif; font-size:24px; font-weight:900;
           color:#fff; line-height:1.1; }
.kpi-sub { font-family:'Inter',sans-serif; font-size:11px; color:#e9c400; margin-top:6px; }

/* ── Live badge ── */
.lbadge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,61,0,0.14); color:#FF3D00;
    border:1px solid rgba(255,61,0,0.28); border-radius:9999px;
    padding:4px 12px; font-family:'Montserrat',sans-serif;
    font-size:10px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
}
.ldot { width:6px; height:6px; background:#FF3D00; border-radius:50%;
        display:inline-block; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Prob bars ── */
.prow { margin-bottom:11px; }
.prow-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.prow-name { font-family:'Inter',sans-serif; font-size:13px; font-weight:500; color:#e0e8ff; }
.prow-pct  { font-family:'Montserrat',sans-serif; font-size:13px; font-weight:700; }
.pbar-bg   { background:rgba(255,255,255,0.09); height:4px; border-radius:999px; overflow:hidden; }
.pbar-fill { height:100%; border-radius:999px; }

/* ── Group table ── */
.gt { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:13px; }
.gt th { color:#d0c6ab; font-size:9px; font-weight:700; letter-spacing:0.1em;
         text-transform:uppercase; padding:0 6px 8px; border-bottom:1px solid rgba(255,255,255,0.09); }
.gt td { padding:9px 6px; border-bottom:1px solid rgba(255,255,255,0.04); color:#d9e2ff; }
.gt tr:last-child td { border-bottom:none; }
.gt-q1 td:first-child { border-left:3px solid #e9c400; padding-left:9px; }
.gt-q2 td:first-child { border-left:3px solid #0A84FF; padding-left:9px; }

/* ── Bracket ── */
.brnd { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
        letter-spacing:0.13em; text-transform:uppercase; color:#e9c400;
        margin-bottom:10px; text-align:center; }
.bmatch {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:8px; padding:9px 12px; margin-bottom:6px;
}
.bmatch-team { font-size:12px; color:#d9e2ff; padding:3px 0;
               display:flex; justify-content:space-between; align-items:center; }
.bmatch-win  { color:#fff; font-weight:700; }
.bmatch-freq { font-size:10px; color:#8b9ab8; }
.bchamp {
    background:linear-gradient(135deg,rgba(233,196,0,0.18),rgba(233,196,0,0.04));
    border:1px solid rgba(233,196,0,0.38); border-radius:12px;
    padding:16px; text-align:center;
    box-shadow:0 0 28px rgba(233,196,0,0.18);
}
.bchamp-icon { font-size:26px; margin-bottom:4px; }
.bchamp-name { font-family:'Montserrat',sans-serif; font-size:17px;
               font-weight:900; color:#e9c400; }
.bchamp-sub  { font-size:11px; color:#8b9ab8; margin-top:3px; }

/* ── Finals card ── */
.fcard {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; padding:14px; text-align:center; margin-bottom:10px;
}
.fcard-rank { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
              letter-spacing:0.12em; text-transform:uppercase; color:#8b9ab8; margin-bottom:6px; }
.fcard-flags { font-size:22px; margin-bottom:4px; }
.fcard-names { font-family:'Inter',sans-serif; font-size:12px; color:#d9e2ff; }
.fcard-pct   { font-family:'Montserrat',sans-serif; font-weight:900;
               font-size:15px; color:#e9c400; margin-top:7px; }

/* ── T3 box ── */
.t3b {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:10px; padding:16px; text-align:center;
}
.t3b-lbl { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
           letter-spacing:0.12em; text-transform:uppercase; color:#8b9ab8; margin-bottom:4px; }
.t3b-num { font-family:'Montserrat',sans-serif; font-size:30px;
           font-weight:900; color:#e9c400; line-height:1; }
.t3b-unit { font-family:'Inter',sans-serif; font-size:11px; color:#8b9ab8; margin-top:3px; }

/* ── Ranking row ── */
.rrow {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.05);
}
.rrow:last-child { border-bottom:none; }
.rrow-left  { display:flex; align-items:center; gap:10px; }
.rrow-rank  { font-family:'Montserrat',sans-serif; font-size:11px;
              font-weight:700; color:#8b9ab8; width:20px; }
.rrow-flag  { font-size:18px; }
.rrow-name  { font-family:'Inter',sans-serif; font-size:13px;
              font-weight:500; color:#e0e8ff; }
.rrow-pct   { font-family:'Montserrat',sans-serif; font-size:14px;
              font-weight:700; }

/* ── Section header ── */
.sh {
    font-family:'Montserrat',sans-serif; font-weight:800; color:#fff;
    font-size:18px; letter-spacing:-0.01em; margin-bottom:16px;
}
.sh-sub { font-family:'Inter',sans-serif; font-size:13px;
          color:#8b9ab8; margin-top:-12px; margin-bottom:16px; }
</style>""", unsafe_allow_html=True)


# ─── FLAGS ───────────────────────────────────────────────────────────────────
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
def fl(t): return FLAGS.get(t, "🏳")

GRP_COLORS = {
    "A":"#e9c400","B":"#0A84FF","C":"#00E676","D":"#ff9800",
    "E":"#c678dd","F":"#FF3D00","G":"#00bcd4","H":"#61afef",
    "I":"#e06c75","J":"#e9c400","K":"#0A84FF","L":"#00E676",
}


# ─── LOAD / SAVE HELPERS ─────────────────────────────────────────────────────
@st.cache_data
def load_saved():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists():
        return None
    tp = pd.read_csv(p("team_probabilities.csv"))
    # v1 → v2 column rename compat
    tp = tp.rename(columns={
        "champion_probability":     "champion_pct",
        "finalist_probability":     "reach_final_pct",
        "semifinalist_probability": "reach_semis_pct",
        "quarterfinalist_prob":     "reach_quarters_pct",
        "round_of_16_probability":  "reach_round16_pct",
        "round_of_32_probability":  "reach_round32_pct",
        "group_exit_probability":   "group_exit_pct",
    })
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct",
                "reach_quarters_pct","attack_coef","defense_coef"]:
        if col not in tp.columns:
            tp[col] = (100 - tp["group_exit_pct"]).round(2) if col == "pass_group_stage_pct" else 0.0
    return {
        "team_probabilities": tp,
        "finals":        pd.read_csv(p("finals.csv")),
        "group_summary": pd.read_csv(p("group_summary.csv")),
        "third_stats":   pd.read_csv(p("third_place_stats.csv")) if p("third_place_stats.csv").exists() else pd.DataFrame(),
        "path":          pd.read_csv(p("path_to_title.csv"))     if p("path_to_title.csv").exists()     else pd.DataFrame(),
        "variance":      pd.read_csv(p("variance_table.csv"))    if p("variance_table.csv").exists()    else pd.DataFrame(),
    }


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Montserrat;font-weight:900;font-size:20px;'
        'color:#e9c400;margin:0;letter-spacing:-0.01em">⚽ FIFA WC 2026</p>'
        '<p style="font-family:Inter;font-size:11px;color:#8b9ab8;margin-top:2px">'
        'Monte Carlo Simulator · Dixon-Coles</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<p class="lc" style="color:#e9c400">Sim Control</p>', unsafe_allow_html=True)
    n_sims = st.slider("Simulaciones", 500, 10000, 2000, 500)
    seed   = st.number_input("Semilla aleatoria", 0, 9999, 42)
    st.markdown('<p class="lc" style="color:#e9c400;margin-top:12px">Escenario</p>', unsafe_allow_html=True)
    try:
        td_s      = load_tournament_data()
        team_opts = ["(torneo completo)"] + sorted(td_s["team"].tolist())
    except Exception:
        team_opts = ["(torneo completo)"]
    exclude      = st.selectbox("Simular sin equipo", team_opts, label_visibility="collapsed")
    exclude_team = None if exclude == "(torneo completo)" else exclude
    if exclude_team:
        st.markdown(
            f'<div style="background:rgba(255,61,0,0.1);border:1px solid rgba(255,61,0,0.3);'
            f'border-radius:8px;padding:8px 12px;font-size:12px;color:#FF3D00">'
            f'⚡ Sin <b>{exclude_team}</b></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    run_btn = st.button("▶ EJECUTAR SIMULACIÓN")
    st.markdown(
        '<p style="font-size:10px;color:#8b9ab8;text-align:center;margin-top:8px">'
        'Datos: FIFA Rankings jun 2026<br>Modelo: Dixon-Coles + Confederación</p>',
        unsafe_allow_html=True,
    )


# ─── RUN OR LOAD ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Ejecutando {n_sims:,} simulaciones…"):
        res = run_monte_carlo(
            n_simulations=n_sims, seed=seed,
            verbose=False, exclude_team=exclude_team,
        )
        generate_all_charts(
            res["team_probabilities"], res["group_summary"],
            res["third_place_stats"],  res["variance_table"],
        )
    st.cache_data.clear()
    lbl = f"✅ {n_sims:,} simulaciones completadas"
    if exclude_team:
        lbl += f" · sin {exclude_team}"
    st.success(lbl)
    data = {
        "team_probabilities": res["team_probabilities"],
        "finals":             res["finals"],
        "group_summary":      res["group_summary"],
        "third_stats":        res["third_place_stats"],
        "path":               res["path_to_title"],
        "variance":           res["variance_table"],
    }
else:
    data = load_saved()


# ─── NO DATA STATE ───────────────────────────────────────────────────────────
if data is None:
    st.markdown(
        '<p class="dp">FIFA WORLD CUP<br>'
        '<span style="color:#e9c400">2026</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lbadge"><span class="ldot"></span>AWAITING SIMULATION</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<br><p style="color:#8b9ab8;font-family:Inter">👈 Configura y pulsa '
        '<b style="color:#e9c400">Ejecutar Simulación</b> para comenzar.</p>',
        unsafe_allow_html=True,
    )
    try:
        td = load_tournament_data()
        st.markdown('<p class="sh" style="margin-top:24px">GRUPOS DEL MUNDIAL 2026</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (g, gdf) in enumerate(td.groupby("group")):
            with cols[i % 4]:
                gc = GRP_COLORS.get(g, "#e9c400")
                teams_html = "".join(
                    f'<div style="padding:4px 0;font-size:13px;color:#d9e2ff">{fl(t)} {t}</div>'
                    for t in gdf["team"].tolist()
                )
                st.markdown(
                    f'<div class="gc" style="border-top:3px solid {gc}">'
                    f'<p class="lc" style="color:{gc};margin-bottom:8px">Grupo {g}</p>'
                    f'{teams_html}</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass
    st.stop()


# ─── UNPACK DATA ─────────────────────────────────────────────────────────────
tp      = data["team_probabilities"]
finals  = data["finals"]
gs      = data["group_summary"]
thirds  = data["third_stats"]
path_df = data["path"]
var_df  = data["variance"]

top      = tp.iloc[0]
fin1     = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
topf     = finals.iloc[0] if not finals.empty else None
hardg    = gs.iloc[0]     if not gs.empty     else None

# Sim count from log
n_sim_logged = 2000
try:
    log = pd.read_csv(OUTPUTS_DIR / "simulation_log.csv")
    n_sim_logged = int(log[log["metric"] == "n_simulations"]["value"].iloc[0])
except Exception:
    pass


# ─── HERO ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        '<p class="dp">TOURNAMENT<br>'
        '<span style="color:#e9c400">SIMULATOR</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lbadge" style="margin-top:8px">'
        '<span class="ldot"></span>SIMULATION COMPLETE · DIXON-COLES MODEL</div>',
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f'<div class="kpi gc-gold" style="margin-top:10px">'
        f'<div class="kpi-lbl">Simulaciones</div>'
        f'<div class="kpi-val">{n_sim_logged:,}</div>'
        f'<div class="kpi-sub">Mundial 2026 · 48 equipos</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:20px 0">', unsafe_allow_html=True)


# ─── KPI ROW ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

final_label = topf["final"] if topf is not None else "—"
final_sub   = f"{topf['probability_pct']:.1f}% de simulaciones" if topf is not None else ""
hard_label  = f"Grupo {hardg['group']}" if hardg is not None else "—"
hard_sub    = f"Rating medio {hardg['average_rating']}" if hardg is not None else ""

k1.markdown(
    f'<div class="kpi gc-gold">'
    f'<div class="kpi-lbl">🏆 Gran Favorito</div>'
    f'<div class="kpi-val">{fl(top["team"])} {top["team"]}</div>'
    f'<div class="kpi-sub">{top["champion_pct"]:.1f}% probabilidad campeón</div>'
    f'</div>',
    unsafe_allow_html=True,
)
k2.markdown(
    f'<div class="kpi gc-blue">'
    f'<div class="kpi-lbl">🥈 Más Veces Finalista</div>'
    f'<div class="kpi-val">{fl(fin1["team"])} {fin1["team"]}</div>'
    f'<div class="kpi-sub">{fin1["reach_final_pct"]:.1f}% llega a la final</div>'
    f'</div>',
    unsafe_allow_html=True,
)
k3.markdown(
    f'<div class="kpi gc-green">'
    f'<div class="kpi-lbl">🎯 Final Más Probable</div>'
    f'<div class="kpi-val" style="font-size:15px;padding-top:6px">{final_label}</div>'
    f'<div class="kpi-sub">{final_sub}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
k4.markdown(
    f'<div class="kpi gc-red">'
    f'<div class="kpi-lbl">💀 Grupo Más Duro</div>'
    f'<div class="kpi-val">{hard_label}</div>'
    f'<div class="kpi-sub">{hard_sub}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠  DASHBOARD",
    "🗂  GRUPOS",
    "🏆  BRACKET",
    "📊  FAVORITOS",
    "3️⃣  TERCEROS",
    "📉  VARIANZA",
    "⚙  ESCENARIOS",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 0 · DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_grp, col_stats = st.columns([2, 1])

    # ── Groups grid ──
    with col_grp:
        st.markdown('<p class="sh">GROUP STAGE OVERVIEW</p>', unsafe_allow_html=True)
        try:
            td = load_tournament_data()
            groups = sorted(td["group"].unique())
            for row_i in range(0, len(groups), 2):
                gc1, gc2 = st.columns(2)
                for ci, grp in enumerate(groups[row_i:row_i + 2]):
                    col = gc1 if ci == 0 else gc2
                    with col:
                        color  = GRP_COLORS.get(grp, "#e9c400")
                        gdf    = td[td["group"] == grp].copy()
                        grp_tp = tp[tp["group"] == grp].set_index("team")

                        rows_html = ""
                        for rank_i, (_, rt) in enumerate(
                            gdf.sort_values("overall_rating", ascending=False).iterrows()
                        ):
                            t     = rt["team"]
                            champ = grp_tp.loc[t, "champion_pct"] if t in grp_tp.index else 0.0
                            rtg   = int(rt["overall_rating"])
                            qcls  = "gt-q1" if rank_i == 0 else ("gt-q2" if rank_i == 1 else "")
                            champ_color = "#e9c400" if champ > 5 else ("#00E676" if champ > 1 else "#8b9ab8")
                            rows_html += (
                                f'<tr class="{qcls}">'
                                f'<td style="display:flex;align-items:center;gap:8px">'
                                f'<span style="font-size:16px">{fl(t)}</span>'
                                f'<span style="font-weight:500;color:#e0e8ff">{t}</span>'
                                f'</td>'
                                f'<td style="text-align:center;color:#8b9ab8">{rtg}</td>'
                                f'<td style="text-align:right;font-family:Montserrat;font-weight:700;'
                                f'font-size:13px;color:{champ_color}">{champ:.1f}%</td>'
                                f'</tr>'
                            )
                        st.markdown(
                            f'<div class="gc" style="border-top:3px solid {color}">'
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;margin-bottom:10px">'
                            f'<span style="font-family:Montserrat;font-weight:800;'
                            f'font-size:16px;color:{color}">GROUP {grp}</span>'
                            f'<span class="lc" style="font-size:9px">Rating · Campeón %</span>'
                            f'</div>'
                            f'<table class="gt">'
                            f'<thead><tr>'
                            f'<th style="text-align:left">Equipo</th>'
                            f'<th style="text-align:center">Rtg</th>'
                            f'<th style="text-align:right">🏆 %</th>'
                            f'</tr></thead>'
                            f'<tbody>{rows_html}</tbody>'
                            f'</table></div>',
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Error cargando grupos: {e}")

    # ── Stats panel ──
    with col_stats:
        # Win probability
        st.markdown(
            '<div class="gc gc-green">'
            '<div style="display:flex;justify-content:space-between;'
            'align-items:center;margin-bottom:14px">'
            '<span style="font-family:Montserrat;font-weight:800;'
            'font-size:15px;color:#fff">Win Probability</span>'
            '<span class="lc" style="background:rgba(255,255,255,0.06);'
            'padding:3px 8px;border-radius:4px;font-size:9px">TOP 12</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        max_p = tp.head(12)["champion_pct"].max()
        bars  = ""
        for _, rp in tp.head(12).iterrows():
            pct = rp["champion_pct"]
            w   = (pct / max_p * 100) if max_p > 0 else 0
            bars += (
                f'<div class="prow">'
                f'<div class="prow-head">'
                f'<span class="prow-name">{fl(rp["team"])} {rp["team"]}</span>'
                f'<span class="prow-pct" style="color:#00E676">{pct:.1f}%</span>'
                f'</div>'
                f'<div class="pbar-bg">'
                f'<div class="pbar-fill" style="width:{w:.1f}%;background:#00E676"></div>'
                f'</div></div>'
            )
        st.markdown(bars + "</div>", unsafe_allow_html=True)

        # Semifinal odds
        tp_sf  = tp.nlargest(8, "reach_semis_pct")
        max_sf = tp_sf["reach_semis_pct"].max()
        sf_bars = ""
        for _, rs in tp_sf.iterrows():
            pct = rs["reach_semis_pct"]
            w   = (pct / max_sf * 100) if max_sf > 0 else 0
            sf_bars += (
                f'<div class="prow">'
                f'<div class="prow-head">'
                f'<span class="prow-name">{fl(rs["team"])} {rs["team"]}</span>'
                f'<span class="prow-pct" style="color:#e9c400">{pct:.1f}%</span>'
                f'</div>'
                f'<div class="pbar-bg">'
                f'<div class="pbar-fill" style="width:{w:.1f}%;background:#e9c400"></div>'
                f'</div></div>'
            )
        st.markdown(
            '<div class="gc gc-gold" style="margin-top:0">'
            '<span style="font-family:Montserrat;font-weight:800;'
            'font-size:15px;color:#fff;display:block;margin-bottom:14px">'
            'Semifinal Odds</span>'
            + sf_bars + '</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 1 · GRUPOS DESGLOSADOS
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="sh">FASE DE GRUPOS — DESGLOSE COMPLETO</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sh-sub">🥇🥈 clasifican automáticamente · '
        'Los 8 mejores 🥉 también pasan</p>',
        unsafe_allow_html=True,
    )
    try:
        td     = load_tournament_data()
        groups = sorted(td["group"].unique())
        for grp in groups:
            color  = GRP_COLORS.get(grp, "#e9c400")
            gdf    = td[td["group"] == grp].copy()
            grp_tp = tp[tp["group"] == grp].set_index("team")

            rows_html = ""
            for rank_i, (_, rt) in enumerate(
                gdf.sort_values("overall_rating", ascending=False).iterrows()
            ):
                t     = rt["team"]
                pasa  = grp_tp.loc[t, "pass_group_stage_pct"] if t in grp_tp.index else 0.0
                champ = grp_tp.loc[t, "champion_pct"]         if t in grp_tp.index else 0.0
                semis = grp_tp.loc[t, "reach_semis_pct"]      if t in grp_tp.index else 0.0
                elim  = grp_tp.loc[t, "group_exit_pct"]       if t in grp_tp.index else 0.0
                rtg   = int(rt["overall_rating"])
                medals = ["🥇","🥈","🥉","4️⃣"]
                medal  = medals[rank_i] if rank_i < 4 else ""

                pasa_color  = "#00E676" if pasa > 70 else ("#e9c400" if pasa > 40 else "#FF3D00")
                champ_color = "#e9c400" if champ > 5 else ("#8bc3ff" if champ > 1 else "#8b9ab8")
                elim_color  = "#FF3D00" if elim > 40 else ("#e9c400" if elim > 15 else "#8b9ab8")
                bar_w = pasa

                rows_html += (
                    f'<tr>'
                    f'<td style="padding:10px 8px">'
                    f'<span style="font-size:16px">{medal}</span>'
                    f'</td>'
                    f'<td style="padding:10px 6px">'
                    f'<span style="display:flex;align-items:center;gap:10px">'
                    f'<span style="font-size:17px">{fl(t)}</span>'
                    f'<span style="font-family:Inter;font-size:14px;font-weight:600;color:#e0e8ff">{t}</span>'
                    f'</span></td>'
                    f'<td style="text-align:center;color:#8b9ab8;font-size:13px">{rtg}</td>'
                    f'<td style="text-align:center">'
                    f'<span style="font-family:Montserrat;font-weight:700;font-size:13px;'
                    f'color:{pasa_color}">{pasa:.0f}%</span></td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-size:13px;'
                    f'font-weight:700;color:{champ_color}">{champ:.1f}%</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-size:13px;'
                    f'color:#8bc3ff">{semis:.1f}%</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-size:13px;'
                    f'color:{elim_color}">{elim:.1f}%</td>'
                    f'<td style="width:110px;padding-right:10px">'
                    f'<div style="background:rgba(255,255,255,0.07);height:4px;'
                    f'border-radius:999px;overflow:hidden">'
                    f'<div style="width:{bar_w:.0f}%;height:100%;background:{color};'
                    f'border-radius:999px"></div></div></td>'
                    f'</tr>'
                )

            st.markdown(
                f'<div class="gc" style="border-top:3px solid {color};margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:20px;'
                f'color:{color}">GRUPO {grp}</span>'
                f'</div>'
                f'<div style="overflow-x:auto">'
                f'<table style="width:100%;border-collapse:collapse;font-family:Inter;font-size:13px">'
                f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.09)">'
                f'<th style="padding:0 8px 8px;color:#8b9ab8;font-size:9px;'
                f'font-weight:700;text-transform:uppercase;letter-spacing:0.1em"></th>'
                f'<th style="text-align:left;padding:0 6px 8px;color:#8b9ab8;'
                f'font-size:9px;text-transform:uppercase;letter-spacing:0.1em">Equipo</th>'
                f'<th style="text-align:center;padding:0 6px 8px;color:#8b9ab8;'
                f'font-size:9px;text-transform:uppercase">Rating</th>'
                f'<th style="text-align:center;padding:0 6px 8px;color:#00E676;'
                f'font-size:9px;text-transform:uppercase">Pasa %</th>'
                f'<th style="text-align:center;padding:0 6px 8px;color:#e9c400;'
                f'font-size:9px;text-transform:uppercase">🏆 %</th>'
                f'<th style="text-align:center;padding:0 6px 8px;color:#8bc3ff;'
                f'font-size:9px;text-transform:uppercase">Semis %</th>'
                f'<th style="text-align:center;padding:0 6px 8px;color:#FF3D00;'
                f'font-size:9px;text-transform:uppercase">Elim %</th>'
                f'<th style="padding:0 10px 8px"></th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table></div></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 · BRACKET
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="sh">KNOCKOUT BRACKET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sh-sub">Camino más probable al título por equipo</p>',
        unsafe_allow_html=True,
    )

    if not path_df.empty:
        top_teams = path_df.head(16)["team"].tolist()
        sel_team  = st.selectbox("Seleccionar equipo", top_teams, key="bsel")
        row_b     = path_df[path_df["team"] == sel_team].iloc[0] if sel_team in path_df["team"].values else None

        if row_b is not None:
            ROUNDS = [
                ("Dieciseisavos", "Dieciseisavos_rival", "Dieciseisavos_freq_pct"),
                ("Octavos",       "Octavos_rival",       "Octavos_freq_pct"),
                ("Cuartos",       "Cuartos_rival",       "Cuartos_freq_pct"),
                ("Semifinal",     "Semifinal_rival",     "Semifinal_freq_pct"),
            ]
            valid_rounds = [(lbl, rc, fc) for lbl, rc, fc in ROUNDS if rc in row_b.index]

            r_cols = st.columns(len(valid_rounds) + 1)

            for i, (lbl, rc, fc) in enumerate(valid_rounds):
                rival = row_b.get(rc, "—")
                freq  = float(row_b.get(fc, 0))
                with r_cols[i]:
                    st.markdown(f'<div class="brnd">{lbl}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="bmatch">'
                        f'<div class="bmatch-team bmatch-win">'
                        f'<span>{fl(sel_team)} {sel_team}</span>'
                        f'</div>'
                        f'<div style="font-size:9px;color:#8b9ab8;text-align:center;'
                        f'padding:2px 0">vs · {freq:.0f}% de las veces</div>'
                        f'<div class="bmatch-team">'
                        f'<span style="color:#8b9ab8">{fl(rival)} {rival}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            # Champion card
            with r_cols[-1]:
                st.markdown('<div class="brnd">Final 🏆</div>', unsafe_allow_html=True)
                # Find most likely final opponent
                fin_rival = "—"
                if not finals.empty:
                    for _, fr in finals.iterrows():
                        if sel_team in fr["final"]:
                            fin_rival = fr["final"].replace(f"{sel_team} vs ", "").replace(f" vs {sel_team}", "")
                            break
                champ_pct = float(row_b["champion_pct"])
                st.markdown(
                    f'<div class="bchamp">'
                    f'<div class="bchamp-icon">🏆</div>'
                    f'<div class="bchamp-name">{fl(sel_team)} {sel_team}</div>'
                    f'<div class="bchamp-sub">vs {fl(fin_rival)} {fin_rival}</div>'
                    f'<div class="bchamp-sub" style="color:#e9c400;font-weight:700;'
                    f'font-size:14px;margin-top:6px">{champ_pct:.1f}% campeón</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Finals grid
        st.markdown('<br><p class="sh">FINALES MÁS REPETIDAS</p>', unsafe_allow_html=True)
        if not finals.empty:
            fin_cols = st.columns(3)
            for i, (_, row_f) in enumerate(finals.head(9).iterrows()):
                parts = row_f["final"].split(" vs ")
                t1 = parts[0].strip()
                t2 = parts[1].strip() if len(parts) > 1 else ""
                with fin_cols[i % 3]:
                    st.markdown(
                        f'<div class="fcard">'
                        f'<div class="fcard-rank">Final #{i+1}</div>'
                        f'<div class="fcard-flags">{fl(t1)} vs {fl(t2)}</div>'
                        f'<div class="fcard-names">{t1} vs {t2}</div>'
                        f'<div class="fcard-pct">{row_f["probability_pct"]:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Sin datos de bracket. Ejecuta la simulación.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 · FAVORITOS
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="sh">PROBABILIDADES COMPLETAS</p>', unsafe_allow_html=True)

    cf1, cf2 = st.columns([1, 1])
    with cf1:
        grp_f = st.multiselect("Filtrar grupo", sorted(tp["group"].unique()), key="gf3")
    with cf2:
        top_n = st.slider("Top N", 10, 48, 24, key="tn3")

    filt = tp.copy()
    if grp_f:
        filt = filt[filt["group"].isin(grp_f)]
    filt = filt.head(top_n).reset_index(drop=True)

    # Matplotlib chart — Apex style
    fig, ax = plt.subplots(figsize=(10, max(5, len(filt) * 0.38)))
    fig.patch.set_facecolor("#020f2a")
    ax.set_facecolor("#0d1b36")
    palette = (
        ["#e9c400"] * 1
        + ["#0A84FF"] * 2
        + ["#00E676"] * 5
        + ["#4d6080"] * max(0, len(filt) - 8)
    )
    bar_colors = palette[:len(filt)][::-1]
    labels     = [f"{fl(t)}  {t}" for t in filt["team"]][::-1]
    values     = filt["champion_pct"].values[::-1]
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="none", height=0.65)
    for bar, val in zip(bars, values):
        if val > 0.1:
            ax.text(
                bar.get_width() + 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left",
                fontsize=9, color="#d9e2ff",
                fontfamily="DejaVu Sans",
            )
    ax.set_xlabel("Probabilidad de ser Campeón (%)", color="#8b9ab8", fontsize=10)
    ax.tick_params(colors="#d9e2ff", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", alpha=0.08, color="#283451")
    ax.set_xlim(0, (filt["champion_pct"].max() or 1) * 1.3)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)

    # Table
    filt.index += 1
    disp = filt[[
        "team","group","overall_rating","champion_pct","reach_final_pct",
        "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct",
    ]].copy()
    disp.columns = [
        "Selección","Grupo","Rating",
        "🏆 Campeón %","🥈 Final %","🏟 Semis %",
        "⚔ Cuartos %","✅ Pasa grupos %","❌ Elim grupos %",
    ]
    num_cols = [c for c in disp.columns if "%" in c or c == "Rating"]
    st.dataframe(
        disp.style
            .background_gradient(subset=["🏆 Campeón %"], cmap="YlOrRd")
            .background_gradient(subset=["❌ Elim grupos %"], cmap="Reds_r")
            .format({c: "{:.1f}" for c in num_cols}),
        use_container_width=True,
        height=500,
    )
    st.download_button(
        "⬇ Descargar CSV",
        tp.to_csv(index=False).encode(),
        "probabilidades_wc2026.csv",
        "text/csv",
    )


# ══════════════════════════════════════════════════════════════════════
# TAB 4 · MEDIA PUNTOS TERCEROS
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="sh">MEJOR TERCER CLASIFICADO</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sh-sub">¿Con cuántos puntos te clasificas siendo tercero de grupo?</p>',
        unsafe_allow_html=True,
    )

    if thirds.empty:
        st.info("Ejecuta la simulación para ver esta información.")
    else:
        summary = thirds[thirds["categoria"].str.startswith("RESUMEN")].copy()
        detail  = thirds[~thirds["categoria"].str.startswith("RESUMEN")].copy()

        if not summary.empty:
            t3_cols = st.columns(len(summary))
            icons   = ["📊","🔽","🔼","⚠️"]
            for i, (_, sr) in enumerate(summary.iterrows()):
                lbl = sr["categoria"].replace("RESUMEN - ", "")
                ic  = icons[i] if i < len(icons) else "•"
                # Build value string safely — no conditional inside f-string
                raw = sr["puntos"]
                if "Clasificados" in lbl:
                    val_str = f"{float(raw):.2f}"
                else:
                    val_str = str(int(float(raw)))
                with t3_cols[i]:
                    st.markdown(
                        f'<div class="t3b">'
                        f'<div class="t3b-lbl">{ic} {lbl}</div>'
                        f'<div class="t3b-num">{val_str}</div>'
                        f'<div class="t3b-unit">puntos</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Distribution chart
        if not detail.empty:
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            fig5.patch.set_facecolor("#020f2a")
            ax5.set_facecolor("#0d1b36")
            for cat, color in [("Clasificado (top 8)", "#00E676"), ("Eliminado", "#FF3D00")]:
                sub = detail[detail["categoria"] == cat]
                if not sub.empty:
                    ax5.bar(
                        sub["puntos"].astype(str), sub["frecuencia_pct"],
                        label=cat, color=color, alpha=0.85, edgecolor="none", width=0.5,
                    )
            ax5.set_xlabel("Puntos al finalizar la fase de grupos", color="#8b9ab8", fontsize=10)
            ax5.set_ylabel("Frecuencia (%)", color="#8b9ab8", fontsize=10)
            ax5.tick_params(colors="#d9e2ff")
            ax5.spines[:].set_visible(False)
            ax5.grid(axis="y", alpha=0.1, color="#283451")
            ax5.legend(facecolor="#0d1b36", labelcolor="#d9e2ff", fontsize=10)
            ax5.set_title(
                "Distribución de puntos: clasificados vs eliminados",
                color="#d9e2ff", pad=10, fontsize=11,
            )
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)

        st.markdown(
            '<div class="gc gc-gold" style="margin-top:8px">'
            '<p class="sh" style="font-size:15px;margin-bottom:8px">📌 Conclusión práctica</p>'
            '<p style="font-family:Inter;font-size:14px;color:#d9e2ff;line-height:1.7">'
            'Con <b style="color:#e9c400">4 o más puntos</b> la clasificación como mejor '
            'tercero es muy probable.<br>'
            'Con <b style="color:#8bc3ff">3 puntos</b> depende del rendimiento del resto de grupos.<br>'
            'Con <b style="color:#FF3D00">2 puntos o menos</b> es prácticamente imposible.'
            '</p></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 5 · VARIANZA
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="sh">VARIANZA Y CONSISTENCIA</p>', unsafe_allow_html=True)

    vi1, vi2 = st.columns(2)
    vi1.markdown(
        '<div class="gc gc-green">'
        '<p style="font-family:Montserrat;font-weight:700;font-size:13px;color:#00E676">'
        '🟢 ALTA CONSISTENCIA</p>'
        '<p style="font-size:13px;color:#d9e2ff">Llegan lejos en casi todas las '
        'simulaciones. Pocas sorpresas.</p></div>',
        unsafe_allow_html=True,
    )
    vi2.markdown(
        '<div class="gc gc-red">'
        '<p style="font-family:Montserrat;font-weight:700;font-size:13px;color:#FF3D00">'
        '🔴 ALTA VARIANZA</p>'
        '<p style="font-size:13px;color:#d9e2ff">Alto potencial de campeón pero también '
        'pueden caer en grupos.</p></div>',
        unsafe_allow_html=True,
    )

    img_var = OUTPUTS_DIR / "variance_scatter.png"
    if img_var.exists():
        st.image(str(img_var), use_column_width=True)

    if not var_df.empty:
        dv = var_df[[
            "team","group","overall_rating","champion_pct",
            "reach_semis_pct","group_exit_pct","consistency_index","variance_index",
        ]].copy()
        dv.columns = [
            "Selección","Grupo","Rating","🏆 Campeón %",
            "Semis %","Elim grupos %","📈 Consistencia","⚡ Varianza",
        ]
        dv = dv.reset_index(drop=True)
        dv.index += 1
        num_dv = [c for c in dv.columns if "%" in c or c in ["Rating","📈 Consistencia","⚡ Varianza"]]
        st.dataframe(
            dv.style
                .background_gradient(subset=["📈 Consistencia"], cmap="Greens")
                .background_gradient(subset=["⚡ Varianza"],     cmap="Reds")
                .format({c: "{:.2f}" for c in ["📈 Consistencia","⚡ Varianza"]}
                        | {c: "{:.1f}" for c in dv.columns if "%" in c or c == "Rating"}),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 6 · ESCENARIOS
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="sh">ANÁLISIS DE ESCENARIOS</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sh-sub">Excluye un equipo del sidebar para ver cómo cambia el torneo</p>',
        unsafe_allow_html=True,
    )

    if exclude_team:
        st.markdown(
            f'<div class="gc gc-red">'
            f'<p style="font-family:Montserrat;font-weight:800;font-size:16px;color:#FF3D00">'
            f'⚡ Simulación activa: sin {fl(exclude_team)} {exclude_team}</p>'
            f'<p style="font-size:13px;color:#d9e2ff">Todos los resultados de las '
            f'otras pestañas reflejan este escenario.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gc">'
            '<p style="font-size:14px;color:#8b9ab8">Selecciona un equipo en el sidebar '
            'y pulsa <b style="color:#e9c400">Ejecutar Simulación</b>.</p>'
            '<p style="font-size:13px;color:#8b9ab8;margin-top:8px">Ejemplos:</p>'
            '<ul style="font-size:13px;color:#d9e2ff;margin-top:4px;line-height:2">'
            '<li>¿Cómo cambia el torneo sin Argentina?</li>'
            '<li>¿Quién se beneficia si Francia no juega?</li>'
            '<li>¿Mejoran las opciones de Brasil sin España en el cuadro?</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="sh" style="margin-top:8px">TOP 10 FAVORITOS ACTUALES</p>', unsafe_allow_html=True)
    rows_html = ""
    for i, (_, r) in enumerate(tp.head(10).iterrows()):
        bar_w = (r["champion_pct"] / tp.iloc[0]["champion_pct"] * 100) if tp.iloc[0]["champion_pct"] > 0 else 0
        medals = ["🥇","🥈","🥉"] + [f"{j}." for j in range(4,11)]
        medal  = medals[i] if i < len(medals) else f"{i+1}."
        pct_color = "#e9c400" if i == 0 else ("#0A84FF" if i < 3 else "#d9e2ff")
        rows_html += (
            f'<div class="rrow">'
            f'<div class="rrow-left">'
            f'<span class="rrow-rank">{medal}</span>'
            f'<span class="rrow-flag">{fl(r["team"])}</span>'
            f'<span class="rrow-name">{r["team"]}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<div style="width:120px;background:rgba(255,255,255,0.07);'
            f'height:4px;border-radius:999px;overflow:hidden">'
            f'<div style="width:{bar_w:.0f}%;height:100%;background:{pct_color};'
            f'border-radius:999px"></div></div>'
            f'<span class="rrow-pct" style="color:{pct_color};min-width:44px;text-align:right">'
            f'{r["champion_pct"]:.1f}%</span>'
            f'</div></div>'
        )
    st.markdown(
        f'<div class="gc" style="padding:8px 4px">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border-color:rgba(255,255,255,0.07);margin:32px 0 14px">'
    '<div style="display:flex;justify-content:space-between;align-items:center;'
    'opacity:0.45">'
    '<p style="font-family:Inter;font-size:11px;color:#8b9ab8">'
    '© 2026 FIFA World Cup Simulation Engine · Dixon-Coles Model · '
    'Datos FIFA Rankings Jun 2026</p>'
    '<p style="font-family:Montserrat;font-size:11px;color:#e9c400">'
    '⚡ For experimental analysis only</p>'
    '</div>',
    unsafe_allow_html=True,
)
