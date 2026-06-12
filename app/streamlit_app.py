"""
FIFA World Cup 2026 — Monte Carlo Simulator
UI inspirada en Apex Stadium: topbar, sidebar, grupos, bracket, teams, stats
Auto-run: 10.000 simulaciones al arrancar si no hay datos
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
from collections import defaultdict

from src.monte_carlo import run_monte_carlo
from src.data_loader import load_tournament_data
from src.visualizations import generate_all_charts
from src.config import OUTPUTS_DIR, DEFAULT_SIMULATIONS

st.set_page_config(
    page_title="FIFA WC 2026 | Simulator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# FONTS + GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;800;900'
    '&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>',
    unsafe_allow_html=True,
)

st.markdown("""<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
html, body { background:#020f2a; color:#d9e2ff; font-family:'Inter',sans-serif; overflow-x:hidden; }
[data-testid="stAppViewContainer"] { background:#020f2a !important; }
[data-testid="stHeader"]           { display:none !important; }
[data-testid="stDecoration"]       { display:none !important; }
[data-testid="stSidebarNav"]       { display:none !important; }
section[data-testid="stSidebar"]   { display:none !important; }
[data-testid="stMainBlockContainer"] { padding:0 !important; max-width:100% !important; }
[data-testid="stVerticalBlock"]    { gap:0 !important; }
.block-container { padding:0 !important; }
footer { display:none !important; }

/* ── Topbar ── */
.topbar {
    position:fixed; top:0; left:0; right:0; z-index:1000;
    height:56px;
    background:rgba(13,23,50,0.96);
    backdrop-filter:blur(16px);
    border-bottom:1px solid rgba(255,255,255,0.08);
    display:flex; align-items:center;
    padding:0 32px;
    gap:32px;
}
.topbar-brand {
    font-family:'Montserrat',sans-serif; font-weight:900; font-size:15px;
    color:#e9c400; letter-spacing:-0.01em; white-space:nowrap;
}
.topbar-nav { display:flex; gap:4px; flex:1; }
.topbar-nav a {
    font-family:'Montserrat',sans-serif; font-weight:700; font-size:12px;
    letter-spacing:0.06em; text-transform:uppercase;
    color:rgba(217,226,255,0.55);
    text-decoration:none; padding:6px 14px; border-radius:6px;
    transition:all 0.15s;
    cursor:pointer;
}
.topbar-nav a:hover { color:#fff; background:rgba(255,255,255,0.06); }
.topbar-nav a.active { color:#fff; border-bottom:2px solid #e9c400; }
.topbar-right { display:flex; align-items:center; gap:12px; margin-left:auto; }
.run-btn {
    background:#e9c400; color:#0d0a00;
    font-family:'Montserrat',sans-serif; font-weight:800; font-size:11px;
    letter-spacing:0.1em; text-transform:uppercase;
    border:none; border-radius:20px; padding:8px 18px; cursor:pointer;
    transition:box-shadow 0.2s, transform 0.1s;
    white-space:nowrap;
}
.run-btn:hover { box-shadow:0 0 20px rgba(233,196,0,0.45); transform:scale(1.03); }
.run-btn.running { background:#ff9800; animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* ── Sidebar ── */
.sidebar {
    position:fixed; left:0; top:56px; bottom:0; width:200px; z-index:900;
    background:#010d28;
    border-right:1px solid rgba(255,255,255,0.07);
    display:flex; flex-direction:column;
    padding:20px 12px;
    gap:4px;
}
.sidebar-section-title {
    font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
    letter-spacing:0.16em; text-transform:uppercase; color:#e9c400;
    padding:12px 10px 6px; opacity:0.9;
}
.sidebar-item {
    display:flex; align-items:center; gap:10px;
    padding:9px 12px; border-radius:8px; cursor:pointer;
    font-family:'Inter',sans-serif; font-size:13px; font-weight:500;
    color:rgba(217,226,255,0.55); transition:all 0.15s;
    text-decoration:none;
}
.sidebar-item:hover { color:#fff; background:rgba(255,255,255,0.06); }
.sidebar-item.active { color:#fff; background:#05e77722; }
.sidebar-item .icon { font-size:16px; width:20px; text-align:center; }
.sidebar-footer {
    margin-top:auto; padding:12px;
    font-family:'Inter',sans-serif; font-size:10px;
    color:rgba(217,226,255,0.3); line-height:1.6;
}

/* ── Main canvas ── */
.main {
    margin-left:200px;
    margin-top:56px;
    min-height:calc(100vh - 56px);
    background:#020f2a;
    padding:32px 36px 48px;
}

/* ── Section headers ── */
.section-hero { margin-bottom:28px; }
.section-hero h1 {
    font-family:'Montserrat',sans-serif; font-weight:900; font-size:36px;
    color:#fff; line-height:1.05; letter-spacing:-0.02em;
}
.section-hero h1 span { color:#e9c400; }
.section-hero p {
    font-family:'Inter',sans-serif; font-size:14px;
    color:rgba(217,226,255,0.5); margin-top:6px;
}

/* ── Glass cards ── */
.gc {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px; padding:18px 20px; margin-bottom:14px;
}
.gc-gold  { border-top:3px solid #e9c400; }
.gc-blue  { border-top:3px solid #0A84FF; }
.gc-green { border-top:3px solid #00E676; }
.gc-red   { border-top:3px solid #FF3D00; }

/* ── KPI cards ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:28px; }
.kpi-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px; padding:18px 16px;
    position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
}
.kpi-card.gold::before  { background:#e9c400; }
.kpi-card.blue::before  { background:#0A84FF; }
.kpi-card.green::before { background:#00E676; }
.kpi-card.red::before   { background:#FF3D00; }
.kpi-label { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
             letter-spacing:0.14em; text-transform:uppercase; color:rgba(208,198,171,0.7);
             margin-bottom:8px; }
.kpi-value { font-family:'Montserrat',sans-serif; font-size:22px; font-weight:900;
             color:#fff; line-height:1.1; }
.kpi-sub   { font-family:'Inter',sans-serif; font-size:12px; color:#e9c400; margin-top:5px; }

/* ── Live badge ── */
.live-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,61,0,0.14); color:#FF3D00;
    border:1px solid rgba(255,61,0,0.3); border-radius:9999px;
    padding:4px 12px;
    font-family:'Montserrat',sans-serif; font-size:9px;
    font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
}
.ldot { width:6px; height:6px; background:#FF3D00; border-radius:50%;
        display:inline-block; animation:lpulse 2s infinite; }
@keyframes lpulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Group tables ── */
.group-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
.group-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px; overflow:hidden;
}
.group-header {
    padding:12px 16px;
    display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid rgba(255,255,255,0.07);
}
.group-name {
    font-family:'Montserrat',sans-serif; font-weight:800; font-size:14px;
}
.group-table { width:100%; border-collapse:collapse; }
.group-table th {
    font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase;
    color:rgba(208,198,171,0.55); padding:8px 10px;
    border-bottom:1px solid rgba(255,255,255,0.06); text-align:center;
}
.group-table th:first-child { text-align:left; }
.group-table td { padding:9px 10px; font-size:13px; color:#d9e2ff;
                  border-bottom:1px solid rgba(255,255,255,0.03); text-align:center; }
.group-table td:first-child { text-align:left; }
.group-table tr:last-child td { border-bottom:none; }
.group-table tr.q1 td { background:rgba(233,196,0,0.04); }
.group-table tr.q1 td:first-child { border-left:3px solid #e9c400; }
.group-table tr.q2 td { background:rgba(10,132,255,0.04); }
.group-table tr.q2 td:first-child { border-left:3px solid #0A84FF; }
.team-cell { display:flex; align-items:center; gap:8px; }
.team-flag { font-size:16px; }
.team-name { font-weight:500; color:#e0e8ff; }
.pts-val { font-family:'Montserrat',sans-serif; font-weight:700; color:#e9c400; }
.champ-pct { font-family:'Montserrat',sans-serif; font-weight:700; font-size:12px; }

/* ── Probability bars ── */
.prob-section { margin-bottom:4px; }
.prob-row { margin-bottom:10px; }
.prob-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.prob-name { font-family:'Inter',sans-serif; font-size:13px; font-weight:500; color:#e0e8ff; }
.prob-pct  { font-family:'Montserrat',sans-serif; font-size:13px; font-weight:700; }
.prob-bar  { background:rgba(255,255,255,0.08); height:4px; border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; }

/* ── Bracket ── */
.bracket-hero { margin-bottom:28px; }
.bracket-hero h1 {
    font-family:'Montserrat',sans-serif; font-weight:900; font-size:42px;
    color:#fff; letter-spacing:-0.02em; text-transform:uppercase;
}
.bracket-hero h1 span { color:#e9c400; }
.bracket-hero p { font-size:14px; color:rgba(217,226,255,0.5); margin-top:6px; max-width:480px; }
.bracket-grid { display:flex; gap:8px; overflow-x:auto; padding-bottom:12px; }
.bracket-round { min-width:160px; }
.round-label {
    font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase; color:#e9c400;
    margin-bottom:10px; text-align:center;
    padding:4px 8px; background:rgba(233,196,0,0.08);
    border-radius:4px;
}
.match-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:8px; padding:10px 12px; margin-bottom:8px;
}
.match-card.highlight {
    border-color:rgba(233,196,0,0.4);
    background:rgba(233,196,0,0.05);
    box-shadow:0 0 16px rgba(233,196,0,0.12);
}
.match-team {
    display:flex; justify-content:space-between; align-items:center;
    padding:3px 0; font-size:12px;
}
.match-team.winner { color:#fff; font-weight:700; }
.match-team.loser  { color:rgba(217,226,255,0.45); }
.match-divider { height:1px; background:rgba(255,255,255,0.06); margin:3px 0; }
.match-freq { font-size:9px; color:rgba(208,198,171,0.5); text-align:center;
              margin-top:4px; font-family:'Montserrat',sans-serif; }
.champ-card {
    background:linear-gradient(135deg,rgba(233,196,0,0.18),rgba(233,196,0,0.03));
    border:1px solid rgba(233,196,0,0.4); border-radius:12px;
    padding:18px 14px; text-align:center;
    box-shadow:0 0 30px rgba(233,196,0,0.15);
}
.champ-trophy { font-size:32px; margin-bottom:6px; }
.champ-name {
    font-family:'Montserrat',sans-serif; font-weight:900; font-size:16px;
    color:#e9c400;
}
.champ-pct-big {
    font-family:'Montserrat',sans-serif; font-weight:700; font-size:13px;
    color:rgba(233,196,0,0.7); margin-top:4px;
}

/* ── Team cards ── */
.teams-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.team-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px; overflow:hidden;
    transition:border-color 0.2s, transform 0.15s;
    cursor:pointer;
}
.team-card:hover { border-color:rgba(233,196,0,0.3); transform:translateY(-2px); }
.team-card-img {
    width:100%; height:90px; object-fit:cover;
    background:linear-gradient(135deg,#0d1b36 0%,#1d2945 100%);
    position:relative; overflow:hidden;
    display:flex; align-items:center; justify-content:center;
    font-size:42px;
}
.team-card-body { padding:12px 14px 14px; }
.team-card-rank { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
                  letter-spacing:0.12em; text-transform:uppercase;
                  color:rgba(208,198,171,0.5); margin-bottom:3px; }
.team-card-name { font-family:'Montserrat',sans-serif; font-weight:800; font-size:15px;
                  color:#fff; margin-bottom:8px; }
.team-card-stats { display:flex; gap:12px; margin-bottom:10px; }
.tc-stat { text-align:center; }
.tc-stat-val { font-family:'Montserrat',sans-serif; font-weight:700; font-size:13px; color:#e9c400; }
.tc-stat-lbl { font-family:'Inter',sans-serif; font-size:9px; color:rgba(208,198,171,0.5);
               text-transform:uppercase; letter-spacing:0.08em; margin-top:1px; }
.tc-pills { display:flex; gap:6px; flex-wrap:wrap; }
.pill {
    font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
    letter-spacing:0.06em; text-transform:uppercase;
    padding:3px 8px; border-radius:20px;
}
.pill-gold   { background:rgba(233,196,0,0.15);  color:#e9c400; }
.pill-blue   { background:rgba(10,132,255,0.15); color:#0A84FF; }
.pill-green  { background:rgba(0,230,118,0.15);  color:#00E676; }
.pill-red    { background:rgba(255,61,0,0.15);   color:#FF3D00; }
.pill-gray   { background:rgba(255,255,255,0.06); color:rgba(217,226,255,0.5); }
.tc-btn {
    width:100%; background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1); border-radius:8px;
    padding:7px; font-family:'Montserrat',sans-serif; font-size:10px;
    font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
    color:rgba(217,226,255,0.6); cursor:pointer; margin-top:10px;
}
.tc-btn:hover { background:rgba(255,255,255,0.09); color:#fff; }

/* ── Stats / analytics ── */
.analytics-panel {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:14px; padding:18px 20px;
}
.ap-title { font-family:'Montserrat',sans-serif; font-weight:800; font-size:16px;
            color:#fff; margin-bottom:16px; }
.stat-row { display:flex; justify-content:space-between; align-items:center;
            padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
.stat-row:last-child { border-bottom:none; }
.stat-label { font-family:'Inter',sans-serif; font-size:13px; color:rgba(217,226,255,0.65); }
.stat-value { font-family:'Montserrat',sans-serif; font-weight:700; font-size:14px; color:#fff; }

/* ── Tabs (Streamlit override) ── */
[data-baseweb="tab-list"] {
    background:transparent !important;
    border-bottom:1px solid rgba(255,255,255,0.08) !important;
    gap:0 !important; padding:0 !important;
}
[data-baseweb="tab"] {
    background:transparent !important; border:none !important;
    color:rgba(208,198,171,0.55) !important;
    font-family:'Montserrat',sans-serif !important;
    font-weight:700 !important; font-size:11px !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    padding:12px 20px !important; border-radius:0 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color:#e9c400 !important;
    border-bottom:2px solid #e9c400 !important;
    background:rgba(233,196,0,0.05) !important;
}
[data-baseweb="tab-panel"] { padding:0 !important; background:transparent !important; }

/* ── Streamlit widget resets ── */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"]   { color:#e9c400 !important; }
.stSlider [role="slider"]  { background:#e9c400 !important; border-color:#e9c400 !important; }
.stButton > button {
    background:#e9c400 !important; color:#0d0a00 !important;
    font-family:'Montserrat',sans-serif !important; font-weight:800 !important;
    font-size:11px !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important; border:none !important;
    border-radius:20px !important; padding:10px 22px !important;
    transition:box-shadow 0.2s !important;
}
.stButton > button:hover { box-shadow:0 0 20px rgba(233,196,0,0.45) !important; }
[data-testid="stSelectbox"] div { background:rgba(255,255,255,0.05) !important;
    border-color:rgba(255,255,255,0.1) !important; color:#d9e2ff !important; }
[data-testid="stDataFrame"] { background:transparent !important; }
.dvn-scroller { background:#0d1b36 !important; border-radius:8px !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-thumb { background:rgba(233,196,0,0.25); border-radius:10px; }

/* ── Footer ── */
.app-footer {
    margin-left:200px; margin-top:0;
    border-top:1px solid rgba(255,255,255,0.06);
    padding:14px 36px;
    display:flex; justify-content:space-between; align-items:center;
    font-family:'Inter',sans-serif; font-size:11px;
    color:rgba(217,226,255,0.3);
}

/* ── Misc ── */
.divider { height:1px; background:rgba(255,255,255,0.07); margin:20px 0; }
.lc { font-family:'Montserrat',sans-serif; font-size:9px; font-weight:700;
      letter-spacing:0.14em; text-transform:uppercase; color:rgba(208,198,171,0.6); }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FLAGS
# ─────────────────────────────────────────────────────────────────────────────
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
def fl(t): return FLAGS.get(str(t), "🏳")

GC = {
    "A":"#e9c400","B":"#0A84FF","C":"#00E676","D":"#ff9800",
    "E":"#c678dd","F":"#FF3D00","G":"#00bcd4","H":"#61afef",
    "I":"#e06c75","J":"#e9c400","K":"#0A84FF","L":"#00E676",
}

# Stadium backgrounds per team (CSS gradient fallback)
TEAM_BG = {
    "Spain":"135deg,#1a0a3a,#2d1060","Argentina":"135deg,#001a3a,#003080",
    "France":"135deg,#0a1530,#1e3a6e","Brazil":"135deg,#0a2a00,#1a5500",
    "Portugal":"135deg,#2a0005,#7a0010","England":"135deg,#0a0a2a,#1a1a60",
    "Netherlands":"135deg,#2a0f00,#7a2f00","Germany":"135deg,#0a0a0a,#3a3a3a",
    "Morocco":"135deg,#2a1000,#006040","Belgium":"135deg,#2a0000,#600010",
    "Croatia":"135deg,#1a0010,#400030","Colombia":"135deg,#1a1500,#404000",
    "Uruguay":"135deg,#001025,#003060","Switzerland":"135deg,#200000,#600000",
}
def team_gradient(t):
    return TEAM_BG.get(t, "135deg,#0d1b36,#1d2945")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD / RUN
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_outputs():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists():
        return None
    tp = pd.read_csv(p("team_probabilities.csv"))
    tp = tp.rename(columns={
        "champion_probability":"champion_pct","finalist_probability":"reach_final_pct",
        "semifinalist_probability":"reach_semis_pct","quarterfinalist_prob":"reach_quarters_pct",
        "round_of_16_probability":"reach_round16_pct","round_of_32_probability":"reach_round32_pct",
        "group_exit_probability":"group_exit_pct",
    })
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct","reach_quarters_pct"]:
        if col not in tp.columns:
            tp[col] = (100 - tp["group_exit_pct"]).round(2) if col == "pass_group_stage_pct" else 0.0
    return {
        "tp":     tp,
        "finals": pd.read_csv(p("finals.csv")),
        "gs":     pd.read_csv(p("group_summary.csv")),
        "t3":     pd.read_csv(p("third_place_stats.csv")) if p("third_place_stats.csv").exists() else pd.DataFrame(),
        "path":   pd.read_csv(p("path_to_title.csv"))     if p("path_to_title.csv").exists()     else pd.DataFrame(),
        "var":    pd.read_csv(p("variance_table.csv"))    if p("variance_table.csv").exists()     else pd.DataFrame(),
    }

@st.cache_data(show_spinner=False)
def load_teams():
    return load_tournament_data()

def run_simulation_10k():
    """Ejecuta 10.000 simulaciones y guarda outputs."""
    res = run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
    generate_all_charts(
        res["team_probabilities"], res["group_summary"],
        res["third_place_stats"],  res["variance_table"],
    )
    return res

# ── Auto-run si no hay datos ──
data = load_outputs()
if data is None:
    with st.spinner("⚽ Ejecutando 10.000 simulaciones Dixon-Coles…"):
        run_simulation_10k()
    load_outputs.clear()
    load_teams.clear()
    data = load_outputs()

# State
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "running" not in st.session_state:
    st.session_state.running = False

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR (HTML — no streamlit widgets here)
# ─────────────────────────────────────────────────────────────────────────────
pages = ["dashboard","brackets","teams","stats"]
page_labels = {"dashboard":"Dashboard","brackets":"Brackets","teams":"Teams","stats":"Stats"}
cur = st.session_state.page

nav_html = "".join(
    f'<a class="{"active" if p == cur else ""}" '
    f'onclick="window.parent.document.querySelector(\'[data-testid=stApp]\').dispatchEvent('
    f'new CustomEvent(\'streamlit:setComponentValue\',{{detail:{{value:\'{p}\'}}}}));">{page_labels[p]}</a>'
    for p in pages
)

n_logged = 10000
try:
    log = pd.read_csv(OUTPUTS_DIR / "simulation_log.csv")
    n_logged = int(log[log["metric"] == "n_simulations"]["value"].iloc[0])
except Exception:
    pass

st.markdown(
    f'<div class="topbar">'
    f'<span class="topbar-brand">FIFA WORLD CUP 2026</span>'
    f'<nav class="topbar-nav">'
    f'<a class="{"active" if cur=="dashboard" else ""}">Dashboard</a>'
    f'<a class="{"active" if cur=="brackets" else ""}">Brackets</a>'
    f'<a class="{"active" if cur=="teams" else ""}">Teams</a>'
    f'<a class="{"active" if cur=="stats" else ""}">Stats</a>'
    f'</nav>'
    f'<div class="topbar-right">'
    f'<span class="lc" style="color:rgba(208,198,171,0.4)">{n_logged:,} sims</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="sidebar">'
    '<div class="sidebar-section-title">Sim Control</div>'
    '<a class="sidebar-item active"><span class="icon">🔲</span>Group Stage</a>'
    '<a class="sidebar-item"><span class="icon">🌳</span>Knockout Brackets</a>'
    '<a class="sidebar-item"><span class="icon">📊</span>Team Rankings</a>'
    '<a class="sidebar-item"><span class="icon">📋</span>Match History</a>'
    '<a class="sidebar-item"><span class="icon">⚙️</span>Sim Settings</a>'
    '<div class="sidebar-footer">'
    f'v2.0 · Dixon-Coles<br>{n_logged:,} simulations<br>FIFA Rankings Jun 2026'
    '</div></div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main">', unsafe_allow_html=True)

tp     = data["tp"]
finals = data["finals"]
gs     = data["gs"]
t3     = data["t3"]
pth    = data["path"]
var    = data["var"]

top  = tp.iloc[0]
fin1 = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
topf = finals.iloc[0] if not finals.empty else None
hg   = gs.iloc[0]     if not gs.empty     else None

# ─────────────────────────────────────────────────────────────────────────────
# TABS (Streamlit native — the only reliable way to switch views)
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_brackets, tab_teams, tab_stats = st.tabs([
    "🏠  DASHBOARD", "🌳  BRACKETS", "⭐  TEAMS", "📊  STATS"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # Hero
    st.markdown(
        '<div style="margin-bottom:20px">'
        '<h1 style="font-family:Montserrat;font-weight:900;font-size:34px;'
        'color:#fff;letter-spacing:-0.02em;margin-bottom:6px">'
        'TOURNAMENT <span style="color:#e9c400">LIVE</span></h1>'
        '<div class="live-badge"><span class="ldot"></span>'
        f'Group Stage — {n_logged:,} Simulations Complete</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    kc = st.columns(4)
    final_txt = topf["final"] if topf is not None else "—"
    final_sub = f"{topf['probability_pct']:.1f}%" if topf is not None else ""
    hg_txt    = f"Group {hg['group']}" if hg is not None else "—"
    hg_sub    = f"Rating {hg['average_rating']:.1f}" if hg is not None else ""

    for col, cls, ico, lbl, val, sub in [
        (kc[0], "gold",  "🏆", "Gran Favorito",      f"{fl(top['team'])} {top['team']}",   f"{top['champion_pct']:.1f}% probability"),
        (kc[1], "blue",  "🥈", "Más Veces Finalista", f"{fl(fin1['team'])} {fin1['team']}", f"{fin1['reach_final_pct']:.1f}% reach final"),
        (kc[2], "green", "🎯", "Final Más Probable",  final_txt,                             final_sub),
        (kc[3], "red",   "💀", "Grupo Más Duro",      hg_txt,                                hg_sub),
    ]:
        col.markdown(
            f'<div class="kpi-card {cls}">'
            f'<div class="kpi-label">{ico} {lbl}</div>'
            f'<div class="kpi-value" style="font-size:{"17px" if len(val)>14 else "22px"}">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # Two-column layout: groups + right panel
    col_groups, col_right = st.columns([7, 3], gap="large")

    with col_groups:
        st.markdown(
            '<p style="font-family:Montserrat;font-weight:800;font-size:13px;'
            'letter-spacing:0.08em;text-transform:uppercase;color:rgba(208,198,171,0.6);'
            'margin-bottom:14px">GROUP STAGE OVERVIEW</p>',
            unsafe_allow_html=True,
        )
        try:
            td = load_teams()
            groups = sorted(td["group"].unique())
            for row_i in range(0, len(groups), 2):
                gc1, gc2 = st.columns(2, gap="small")
                for ci, grp in enumerate(groups[row_i:row_i + 2]):
                    col = gc1 if ci == 0 else gc2
                    with col:
                        color  = GC.get(grp, "#e9c400")
                        gdf    = td[td["group"] == grp].copy()
                        grp_tp = tp[tp["group"] == grp].set_index("team")

                        rows = ""
                        for ri, (_, rt) in enumerate(
                            gdf.sort_values("overall_rating", ascending=False).iterrows()
                        ):
                            t     = rt["team"]
                            champ = float(grp_tp.loc[t, "champion_pct"]) if t in grp_tp.index else 0.0
                            rtg   = int(rt["overall_rating"])
                            pasa  = float(grp_tp.loc[t, "pass_group_stage_pct"]) if t in grp_tp.index else 0.0
                            qcls  = "q1" if ri == 0 else ("q2" if ri == 1 else "")
                            cc    = "#e9c400" if champ > 8 else ("#00E676" if champ > 2 else ("#8bc3ff" if champ > 0.5 else "rgba(217,226,255,0.3)"))
                            rows += (
                                f'<tr class="{qcls}">'
                                f'<td><div class="team-cell">'
                                f'<span class="team-flag">{fl(t)}</span>'
                                f'<span class="team-name">{t}</span>'
                                f'</div></td>'
                                f'<td style="color:rgba(208,198,171,0.5)">{rtg}</td>'
                                f'<td>{int(pasa)}%</td>'
                                f'<td><span style="font-family:Montserrat;font-weight:700;font-size:12px;color:{cc}">{champ:.1f}%</span></td>'
                                f'</tr>'
                            )
                        st.markdown(
                            f'<div class="group-card">'
                            f'<div class="group-header">'
                            f'<span class="group-name" style="color:{color}">GROUP {grp}</span>'
                            f'<span class="lc">Rtg · Pasa · 🏆</span>'
                            f'</div>'
                            f'<table class="group-table">'
                            f'<thead><tr>'
                            f'<th>Team</th><th>Rtg</th><th>Pass%</th><th>Win%</th>'
                            f'</tr></thead>'
                            f'<tbody>{rows}</tbody>'
                            f'</table></div>',
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Error grupos: {e}")

    with col_right:
        # Win probability
        max_p = tp.head(10)["champion_pct"].max()
        pb    = ""
        for _, rp in tp.head(10).iterrows():
            pct = float(rp["champion_pct"])
            w   = (pct / max_p * 100) if max_p > 0 else 0
            pb += (
                f'<div class="prob-row">'
                f'<div class="prob-head">'
                f'<span class="prob-name">{fl(rp["team"])} {rp["team"]}</span>'
                f'<span class="prob-pct" style="color:#00E676">{pct:.1f}%</span>'
                f'</div>'
                f'<div class="prob-bar">'
                f'<div class="prob-fill" style="width:{w:.1f}%;background:#00E676"></div>'
                f'</div></div>'
            )
        st.markdown(
            '<div class="gc gc-green">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">'
            '<span style="font-family:Montserrat;font-weight:800;font-size:15px;color:#fff">Win Probability</span>'
            '<span class="lc" style="background:rgba(255,255,255,0.06);padding:3px 8px;border-radius:4px">TOP 10</span>'
            '</div>' + pb + '</div>',
            unsafe_allow_html=True,
        )

        # Semifinal odds
        tp_sf  = tp.nlargest(8, "reach_semis_pct")
        max_sf = tp_sf["reach_semis_pct"].max()
        sb     = ""
        for _, rs in tp_sf.iterrows():
            pct = float(rs["reach_semis_pct"])
            w   = (pct / max_sf * 100) if max_sf > 0 else 0
            sb += (
                f'<div class="prob-row">'
                f'<div class="prob-head">'
                f'<span class="prob-name">{fl(rs["team"])} {rs["team"]}</span>'
                f'<span class="prob-pct" style="color:#e9c400">{pct:.1f}%</span>'
                f'</div>'
                f'<div class="prob-bar">'
                f'<div class="prob-fill" style="width:{w:.1f}%;background:#e9c400"></div>'
                f'</div></div>'
            )
        st.markdown(
            '<div class="gc gc-gold">'
            '<span style="font-family:Montserrat;font-weight:800;font-size:15px;'
            'color:#fff;display:block;margin-bottom:14px">Semifinal Odds</span>'
            + sb + '</div>',
            unsafe_allow_html=True,
        )

        # Re-run button
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("🔄 RE-RUN 10K SIMULATION"):
            with st.spinner("Ejecutando 10.000 simulaciones…"):
                run_simulation_10k()
            load_outputs.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: BRACKETS
# ══════════════════════════════════════════════════════════════════════════════
with tab_brackets:
    st.markdown(
        '<div class="bracket-hero">'
        '<h1>ROAD TO <span>GLORY</span></h1>'
        '<p>The final 32 teams battle for football immortality. '
        'Track the progression from the opening round to the MetLife Stadium Final.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_bk, col_ba = st.columns([3, 1], gap="large")

    with col_bk:
        if not pth.empty:
            top_teams = pth.head(12)["team"].tolist()
            sel = st.selectbox("Select team path", top_teams, key="bsel",
                               label_visibility="collapsed")
            row_b = pth[pth["team"] == sel].iloc[0] if sel in pth["team"].values else None

            if row_b is not None:
                ROUNDS = [
                    ("ROUND OF 32", "Dieciseisavos_rival", "Dieciseisavos_freq_pct"),
                    ("ROUND OF 16", "Octavos_rival",       "Octavos_freq_pct"),
                    ("QUARTER FINAL","Cuartos_rival",      "Cuartos_freq_pct"),
                    ("SEMI FINAL",  "Semifinal_rival",     "Semifinal_freq_pct"),
                ]
                valid = [(lbl, rc, fc) for lbl, rc, fc in ROUNDS if rc in row_b.index]
                rcols = st.columns(len(valid) + 1, gap="small")

                for i, (lbl, rc, fc) in enumerate(valid):
                    rival = str(row_b.get(rc, "TBD"))
                    freq  = float(row_b.get(fc, 0))
                    with rcols[i]:
                        st.markdown(
                            f'<div class="round-label">{lbl}</div>'
                            f'<div class="match-card highlight">'
                            f'<div class="match-team winner">'
                            f'<span>{fl(sel)} {sel}</span>'
                            f'<span style="font-family:Montserrat;font-size:14px;color:#e9c400">W</span>'
                            f'</div>'
                            f'<div class="match-divider"></div>'
                            f'<div class="match-team loser">'
                            f'<span>{fl(rival)} {rival}</span>'
                            f'</div>'
                            f'<div class="match-freq">{freq:.0f}% of wins</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Champion
                with rcols[-1]:
                    fin_rival = "TBD"
                    if not finals.empty:
                        for _, fr in finals.iterrows():
                            if sel in str(fr["final"]):
                                fin_rival = str(fr["final"]).replace(f"{sel} vs ", "").replace(f" vs {sel}", "").strip()
                                break
                    champ_pct = float(row_b["champion_pct"])
                    st.markdown(
                        f'<div class="round-label">FINAL 🏆</div>'
                        f'<div class="champ-card">'
                        f'<div class="champ-trophy">🏆</div>'
                        f'<div class="champ-name">{fl(sel)} {sel}</div>'
                        f'<div style="font-size:11px;color:rgba(217,226,255,0.5);margin-top:4px">'
                        f'vs {fl(fin_rival)} {fin_rival}</div>'
                        f'<div class="champ-pct-big">{champ_pct:.1f}% champion</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Finals grid
        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:Montserrat;font-weight:800;font-size:13px;'
            'letter-spacing:0.08em;text-transform:uppercase;color:rgba(208,198,171,0.6);'
            'margin-bottom:14px">MOST FREQUENT FINALS</p>',
            unsafe_allow_html=True,
        )
        if not finals.empty:
            fc3 = st.columns(3, gap="small")
            for i, (_, row_f) in enumerate(finals.head(9).iterrows()):
                parts  = str(row_f["final"]).split(" vs ")
                t1, t2 = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
                border = ["#e9c400","#0A84FF","#00E676"][i % 3]
                with fc3[i % 3]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.04);'
                        f'border:1px solid rgba(255,255,255,0.09);'
                        f'border-top:3px solid {border};'
                        f'border-radius:12px;padding:14px;text-align:center;margin-bottom:10px">'
                        f'<div class="lc" style="margin-bottom:6px">Final #{i+1}</div>'
                        f'<div style="font-size:24px;margin-bottom:4px">{fl(t1)} vs {fl(t2)}</div>'
                        f'<div style="font-size:12px;color:rgba(217,226,255,0.6);margin-bottom:6px">'
                        f'{t1} vs {t2}</div>'
                        f'<div style="font-family:Montserrat;font-weight:900;font-size:16px;'
                        f'color:#e9c400">{row_f["probability_pct"]:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with col_ba:
        # Analytics panel
        st.markdown(
            '<div class="analytics-panel">'
            '<div class="ap-title">TOURNAMENT ODDS</div>',
            unsafe_allow_html=True,
        )
        panel_html = ""
        for i, (_, rp) in enumerate(tp.head(10).iterrows()):
            rank_color = "#e9c400" if i == 0 else ("#0A84FF" if i < 3 else "rgba(217,226,255,0.5)")
            panel_html += (
                f'<div class="stat-row">'
                f'<span class="stat-label">'
                f'<span style="font-family:Montserrat;font-weight:700;color:{rank_color};margin-right:8px">'
                f'{i+1}.</span>'
                f'{fl(rp["team"])} {rp["team"]}</span>'
                f'<span class="stat-value" style="color:{rank_color}">'
                f'{rp["champion_pct"]:.1f}%</span>'
                f'</div>'
            )
        st.markdown(panel_html + '</div>', unsafe_allow_html=True)

        # Third place stats
        if not t3.empty:
            summary = t3[t3["categoria"].str.startswith("RESUMEN")].copy()
            if not summary.empty:
                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="analytics-panel">'
                    '<div class="ap-title">3RD PLACE QUALIFIER</div>',
                    unsafe_allow_html=True,
                )
                t3_html = ""
                for _, sr in summary.iterrows():
                    lbl = sr["categoria"].replace("RESUMEN - ", "")
                    raw = sr["puntos"]
                    val_str = f"{float(raw):.2f}" if "Clasificados" in lbl else str(int(float(raw)))
                    t3_html += (
                        f'<div class="stat-row">'
                        f'<span class="stat-label">{lbl}</span>'
                        f'<span class="stat-value" style="color:#e9c400">{val_str} pts</span>'
                        f'</div>'
                    )
                st.markdown(t3_html + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: TEAMS
# ══════════════════════════════════════════════════════════════════════════════
with tab_teams:
    st.markdown(
        '<h1 style="font-family:Montserrat;font-weight:900;font-size:34px;'
        'color:#fff;letter-spacing:-0.02em;margin-bottom:6px">'
        'PARTICIPATING TEAMS</h1>'
        '<p style="font-size:14px;color:rgba(217,226,255,0.5);margin-bottom:20px">'
        'Browse the elite 48 nations competing for glory in North America. '
        'Track rankings, squad stats, and simulation history.</p>',
        unsafe_allow_html=True,
    )

    # Filter bar
    tf1, tf2, tf3 = st.columns([2, 1, 1])
    with tf1:
        search = st.text_input("", placeholder="Find a team (e.g. Argentina, France…)",
                               label_visibility="collapsed", key="tsearch")
    with tf2:
        grp_filter = st.selectbox("Group", ["ALL"] + [f"GROUP {g}" for g in "ABCDEFGHIJKL"],
                                  label_visibility="collapsed", key="tgrp")
    with tf3:
        sort_by = st.selectbox("Sort by", ["Champion %","Rating","Semis %"],
                               label_visibility="collapsed", key="tsort")

    # Filter & sort
    show_tp = tp.copy()
    if search:
        show_tp = show_tp[show_tp["team"].str.contains(search, case=False)]
    if grp_filter != "ALL":
        grp_letter = grp_filter.replace("GROUP ", "")
        show_tp = show_tp[show_tp["group"] == grp_letter]
    sort_col = {"Champion %":"champion_pct","Rating":"overall_rating","Semis %":"reach_semis_pct"}[sort_by]
    show_tp  = show_tp.sort_values(sort_col, ascending=False)

    # Summary stats
    sc1, sc2, sc3 = st.columns(3, gap="small")
    sc1.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);'
        f'border-radius:10px;padding:12px 16px;display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:22px">🌍</span>'
        f'<div><div style="font-family:Montserrat;font-weight:700;font-size:18px;color:#fff">'
        f'{len(show_tp)} / 48</div>'
        f'<div class="lc">Teams shown</div></div></div>',
        unsafe_allow_html=True,
    )
    sc2.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);'
        f'border-radius:10px;padding:12px 16px;display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:22px">✅</span>'
        f'<div><div style="font-family:Montserrat;font-weight:700;font-size:18px;color:#00E676">'
        f'#{int(tp.iloc[0]["overall_rating"])}</div>'
        f'<div class="lc">Avg. Rating</div></div></div>',
        unsafe_allow_html=True,
    )
    sc3.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);'
        f'border-radius:10px;padding:12px 16px;display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:22px">🔵</span>'
        f'<div><div style="font-family:Montserrat;font-weight:700;font-size:18px;color:#0A84FF">'
        f'#{n_logged:,}</div>'
        f'<div class="lc">Simulations</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Team cards grid — 4 per row
    teams_list = show_tp.reset_index(drop=True)
    for row_i in range(0, min(len(teams_list), 48), 4):
        cols4 = st.columns(4, gap="small")
        for ci, (_, tr) in enumerate(teams_list.iloc[row_i:row_i + 4].iterrows()):
            t = tr["team"]
            champ  = float(tr["champion_pct"])
            semis  = float(tr["reach_semis_pct"])
            rtg    = int(tr["overall_rating"])
            pasa   = float(tr["pass_group_stage_pct"])
            grp    = tr["group"]
            bg     = team_gradient(t)
            rank_i = int(teams_list[teams_list["team"] == t].index[0])

            # Pills
            if champ > 10:   pill = '<span class="pill pill-gold">FAVORITE</span>'
            elif champ > 3:  pill = '<span class="pill pill-blue">CONTENDER</span>'
            elif pasa > 70:  pill = '<span class="pill pill-green">QUALIFIER</span>'
            else:            pill = '<span class="pill pill-gray">UNDERDOG</span>'

            if semis > 20:   pill2 = '<span class="pill pill-green">SEMI FINAL</span>'
            elif semis > 8:  pill2 = '<span class="pill pill-blue">QUARTER F.</span>'
            elif pasa > 50:  pill2 = '<span class="pill pill-gray">ROUND 16</span>'
            else:            pill2 = '<span class="pill pill-red">GROUP EXIT</span>'

            with cols4[ci]:
                st.markdown(
                    f'<div class="team-card">'
                    f'<div class="team-card-img" style="background:linear-gradient({bg})">'
                    f'<span style="font-size:46px;opacity:0.9">{fl(t)}</span>'
                    f'<div style="position:absolute;top:8px;right:10px;'
                    f'background:rgba(0,0,0,0.5);border-radius:4px;'
                    f'padding:2px 7px;font-family:Montserrat;font-weight:700;'
                    f'font-size:10px;color:rgba(208,198,171,0.8)">GRP {grp}</div>'
                    f'</div>'
                    f'<div class="team-card-body">'
                    f'<div class="team-card-rank">FIFA RANKING #{rank_i + 1} · RATING {rtg}</div>'
                    f'<div class="team-card-name">{t}</div>'
                    f'<div class="team-card-stats">'
                    f'<div class="tc-stat">'
                    f'<div class="tc-stat-val">{champ:.1f}%</div>'
                    f'<div class="tc-stat-lbl">Champion</div>'
                    f'</div>'
                    f'<div class="tc-stat">'
                    f'<div class="tc-stat-val">{semis:.1f}%</div>'
                    f'<div class="tc-stat-lbl">Semis</div>'
                    f'</div>'
                    f'<div class="tc-stat">'
                    f'<div class="tc-stat-val">{pasa:.0f}%</div>'
                    f'<div class="tc-stat-lbl">Pass grp</div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="tc-pills">{pill}{pill2}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown(
        '<h1 style="font-family:Montserrat;font-weight:900;font-size:34px;'
        'color:#fff;letter-spacing:-0.02em;margin-bottom:6px">'
        'SIMULATION <span style="color:#e9c400">ANALYTICS</span></h1>'
        '<p style="font-size:14px;color:rgba(217,226,255,0.5);margin-bottom:24px">'
        f'Based on {n_logged:,} full tournament simulations using Dixon-Coles model.</p>',
        unsafe_allow_html=True,
    )

    sa1, sa2 = st.columns([3, 2], gap="large")

    with sa1:
        # Full probability table
        st.markdown(
            '<p style="font-family:Montserrat;font-weight:800;font-size:13px;'
            'letter-spacing:0.08em;text-transform:uppercase;color:rgba(208,198,171,0.6);'
            'margin-bottom:12px">COMPLETE PROBABILITY TABLE</p>',
            unsafe_allow_html=True,
        )
        disp = tp[[
            "team","group","overall_rating","champion_pct","reach_final_pct",
            "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct",
        ]].copy()
        disp.columns = [
            "Team","Grp","Rating","🏆 Champion%","Final%",
            "Semis%","Quarters%","Pass Grp%","Elim%",
        ]
        disp = disp.reset_index(drop=True)
        disp.index += 1
        st.dataframe(
            disp.style
                .background_gradient(subset=["🏆 Champion%"], cmap="YlOrRd")
                .background_gradient(subset=["Elim%"], cmap="Reds_r")
                .format({c: "{:.1f}" for c in disp.columns if "%" in c or c == "Rating"}),
            use_container_width=True, height=500,
        )
        st.download_button(
            "⬇ Download CSV",
            tp.to_csv(index=False).encode(),
            "wc2026_probabilities.csv", "text/csv",
        )

        # Varianza
        if not var.empty:
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-family:Montserrat;font-weight:800;font-size:13px;'
                'letter-spacing:0.08em;text-transform:uppercase;color:rgba(208,198,171,0.6);'
                'margin-bottom:12px">VARIANCE & CONSISTENCY</p>',
                unsafe_allow_html=True,
            )
            img_var = OUTPUTS_DIR / "variance_scatter.png"
            if img_var.exists():
                st.image(str(img_var), use_column_width=True)

    with sa2:
        # Phase probability for a team
        st.markdown(
            '<p style="font-family:Montserrat;font-weight:800;font-size:13px;'
            'letter-spacing:0.08em;text-transform:uppercase;color:rgba(208,198,171,0.6);'
            'margin-bottom:12px">PHASE BREAKDOWN</p>',
            unsafe_allow_html=True,
        )
        sel_t = st.selectbox("Team", tp["team"].tolist(), key="statsTeam",
                             label_visibility="collapsed")
        rt = tp[tp["team"] == sel_t].iloc[0]

        phases = [
            ("Pass Groups",    float(rt["pass_group_stage_pct"]), "#00E676"),
            ("Round of 32",    float(rt.get("reach_round32_pct", 0)), "#0A84FF"),
            ("Round of 16",    float(rt.get("reach_round16_pct", 0)), "#0A84FF"),
            ("Quarters",       float(rt["reach_quarters_pct"]), "#8bc3ff"),
            ("Semis",          float(rt["reach_semis_pct"]), "#e9c400"),
            ("Final",          float(rt["reach_final_pct"]), "#ff9800"),
            ("Champion 🏆",    float(rt["champion_pct"]), "#e9c400"),
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#020f2a")
        ax.set_facecolor("#0d1b36")
        labels = [p[0] for p in phases]
        values = [p[1] for p in phases]
        colors = [p[2] for p in phases]
        bars   = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                         edgecolor="none", height=0.6)
        for bar, val in zip(bars, values[::-1]):
            if val > 0.1:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", ha="left", fontsize=9,
                        color="#d9e2ff")
        ax.set_xlim(0, max(values or [1]) * 1.35)
        ax.tick_params(colors="#d9e2ff", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.07, color="#283451")
        ax.set_title(f"{fl(sel_t)}  {sel_t}", color="#fff", pad=10,
                     fontfamily="DejaVu Sans", fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Group difficulty
        if not gs.empty:
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="lc" style="margin-bottom:10px">GROUP DIFFICULTY</p>',
                unsafe_allow_html=True,
            )
            gs_s   = gs.sort_values("average_rating", ascending=False)
            max_gr = float(gs_s["average_rating"].max())
            gd_html = ""
            for _, gr in gs_s.iterrows():
                color = GC.get(str(gr["group"]), "#e9c400")
                w     = (float(gr["average_rating"]) / max_gr * 100) if max_gr > 0 else 0
                gd_html += (
                    f'<div style="margin-bottom:9px">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                    f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;'
                    f'color:{color}">GROUP {gr["group"]}</span>'
                    f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;'
                    f'color:#d9e2ff">{gr["average_rating"]:.1f}</span>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.07);height:4px;border-radius:999px;overflow:hidden">'
                    f'<div style="width:{w:.1f}%;height:100%;background:{color};border-radius:999px"></div>'
                    f'</div></div>'
                )
            st.markdown(
                '<div class="gc">' + gd_html + '</div>',
                unsafe_allow_html=True,
            )

        # 3rd place threshold
        if not t3.empty:
            summary_t3 = t3[t3["categoria"].str.startswith("RESUMEN")].copy()
            if not summary_t3.empty:
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="lc" style="margin-bottom:10px">3RD PLACE THRESHOLD</p>',
                    unsafe_allow_html=True,
                )
                t3p_html = ""
                for _, sr in summary_t3.iterrows():
                    lbl = sr["categoria"].replace("RESUMEN - ", "")
                    raw = sr["puntos"]
                    val_str = f"{float(raw):.2f}" if "Clasificados" in lbl else str(int(float(raw)))
                    t3p_html += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;padding:8px 0;'
                        f'border-bottom:1px solid rgba(255,255,255,0.05)">'
                        f'<span style="font-size:12px;color:rgba(217,226,255,0.6)">{lbl}</span>'
                        f'<span style="font-family:Montserrat;font-weight:700;'
                        f'font-size:15px;color:#e9c400">{val_str} pts</span>'
                        f'</div>'
                    )
                st.markdown(
                    '<div class="gc">' + t3p_html + '</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Close main div + footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-footer">'
    '<span>© 2026 FIFA World Cup Simulation Engine · Dixon-Coles Model · '
    'Data: FIFA Rankings Jun 2026 · For experimental analysis only</span>'
    '<span style="color:rgba(233,196,0,0.6)">⚡ All results synced</span>'
    '</div>',
    unsafe_allow_html=True,
)
