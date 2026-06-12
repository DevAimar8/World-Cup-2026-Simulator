"""
FIFA World Cup 2026 — Monte Carlo Simulator
Navegación: st.session_state (única forma fiable en Streamlit)
UI: Apex Stadium — navy + gold
Auto-run: 10.000 simulaciones al primer arranque
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
from src.visualizations import generate_all_charts
from src.config import OUTPUTS_DIR

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 | Simulator",
    page_icon="🏆", layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Fonts ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Montserrat:wght@500;700;800;900&family=Inter:wght@400;500;600'
    '&display=swap" rel="stylesheet"/>',
    unsafe_allow_html=True,
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""<style>
/* BASE */
html,body,[data-testid="stAppViewContainer"]{background:#020f2a!important;color:#d9e2ff!important;}
[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stSidebarNav"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important;}
.block-container{padding:0!important;}
footer{display:none!important;}
[data-testid="stVerticalBlock"]{gap:0!important;}

/* TOPBAR — pure CSS, no JS needed */
.topbar{
  background:rgba(5,18,45,0.97);border-bottom:1px solid rgba(255,255,255,0.08);
  padding:0 32px;height:56px;display:flex;align-items:center;gap:28px;
  position:sticky;top:0;z-index:999;
}
.tb-brand{font-family:Montserrat,sans-serif;font-weight:900;font-size:15px;
  color:#e9c400;letter-spacing:-0.01em;white-space:nowrap;}
.tb-sims{font-family:Montserrat,sans-serif;font-size:10px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:rgba(208,198,171,0.4);
  margin-left:auto;}

/* NAV BUTTONS — styled as tabs */
.stButton>button{
  background:transparent!important;border:none!important;
  font-family:Montserrat,sans-serif!important;font-weight:700!important;
  font-size:12px!important;letter-spacing:0.08em!important;
  text-transform:uppercase!important;padding:8px 16px!important;
  color:rgba(208,198,171,0.5)!important;border-radius:0!important;
  transition:color 0.15s!important;width:auto!important;
}
.stButton>button:hover{color:#fff!important;background:rgba(255,255,255,0.05)!important;}

/* Active nav button */
.nav-active>div>button,
div[data-testid="column"] .nav-active button{
  color:#fff!important;
  border-bottom:2px solid #e9c400!important;
  background:rgba(233,196,0,0.06)!important;
}

/* RUN button override */
.run-btn>div>button{
  background:#e9c400!important;color:#0d0a00!important;
  border-radius:20px!important;padding:8px 20px!important;
  font-size:11px!important;box-shadow:none!important;
}
.run-btn>div>button:hover{box-shadow:0 0 20px rgba(233,196,0,0.4)!important;}

/* SIDEBAR */
.sidebar-box{
  background:#010d28;border-right:1px solid rgba(255,255,255,0.07);
  padding:20px 0;min-height:calc(100vh - 56px);
}
.sb-title{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.16em;text-transform:uppercase;color:#e9c400;
  padding:0 16px 8px;opacity:0.85;}
.sb-item{display:flex;align-items:center;gap:10px;padding:10px 16px;
  font-family:Inter,sans-serif;font-size:13px;font-weight:500;
  color:rgba(217,226,255,0.5);cursor:default;}
.sb-item.active{color:#fff;background:rgba(0,230,118,0.08);
  border-left:2px solid #00E676;}
.sb-item span.ic{font-size:16px;width:22px;text-align:center;}
.sb-footer{font-family:Inter,sans-serif;font-size:10px;
  color:rgba(217,226,255,0.25);padding:20px 16px;line-height:1.7;margin-top:auto;}

/* MAIN CONTENT */
.main-content{background:#020f2a;padding:28px 32px 48px;}

/* GLASS CARDS */
.gc{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;padding:18px 20px;margin-bottom:12px;}
.gc.gold{border-top:3px solid #e9c400!important;}
.gc.blue{border-top:3px solid #0A84FF!important;}
.gc.green{border-top:3px solid #00E676!important;}
.gc.red{border-top:3px solid #FF3D00!important;}

/* KPI */
.kpi{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:12px;padding:16px;position:relative;overflow:hidden;height:100%;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kpi.gold::before{background:#e9c400;}.kpi.blue::before{background:#0A84FF;}
.kpi.green::before{background:#00E676;}.kpi.red::before{background:#FF3D00;}
.kpi-l{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.13em;text-transform:uppercase;color:rgba(208,198,171,0.65);margin-bottom:7px;}
.kpi-v{font-family:Montserrat,sans-serif;font-weight:900;color:#fff;line-height:1.1;}
.kpi-s{font-family:Inter,sans-serif;font-size:11px;color:#e9c400;margin-top:5px;}

/* LIVE BADGE */
.lbadge{display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,61,0,0.13);color:#FF3D00;
  border:1px solid rgba(255,61,0,0.28);border-radius:999px;
  padding:4px 12px;font-family:Montserrat,sans-serif;font-size:9px;
  font-weight:700;letter-spacing:0.12em;text-transform:uppercase;}
.ldot{width:6px;height:6px;background:#FF3D00;border-radius:50%;
  display:inline-block;animation:lp 2s infinite;}
@keyframes lp{0%,100%{opacity:1}50%{opacity:0.2}}

/* GROUP TABLE */
.gt{width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:13px;}
.gt th{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;
  color:rgba(208,198,171,0.45);padding:0 8px 8px;
  border-bottom:1px solid rgba(255,255,255,0.07);text-align:center;}
.gt th:first-child{text-align:left;}
.gt td{padding:9px 8px;color:#d9e2ff;
  border-bottom:1px solid rgba(255,255,255,0.04);text-align:center;}
.gt td:first-child{text-align:left;}
.gt tr:last-child td{border-bottom:none;}
.gt tr.q1 td:first-child{border-left:3px solid #e9c400;padding-left:9px;}
.gt tr.q2 td:first-child{border-left:3px solid #0A84FF;padding-left:9px;}
.team-cell{display:flex;align-items:center;gap:9px;}
.tf{font-size:16px;}
.tn{font-weight:500;color:#e0e8ff;}

/* PROB BARS */
.pb{margin-bottom:11px;}
.pb-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.pb-name{font-family:Inter,sans-serif;font-size:13px;font-weight:500;color:#e0e8ff;}
.pb-pct{font-family:Montserrat,sans-serif;font-size:13px;font-weight:700;}
.pb-bg{background:rgba(255,255,255,0.08);height:4px;border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;border-radius:999px;}

/* BRACKET */
.br-lbl{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.14em;text-transform:uppercase;color:#e9c400;
  text-align:center;padding:5px 8px;
  background:rgba(233,196,0,0.08);border-radius:4px;margin-bottom:10px;}
.br-match{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:9px;padding:10px 12px;margin-bottom:7px;}
.br-match.hl{border-color:rgba(233,196,0,0.4);background:rgba(233,196,0,0.05);
  box-shadow:0 0 16px rgba(233,196,0,0.1);}
.br-t{display:flex;justify-content:space-between;align-items:center;
  padding:3px 0;font-size:12px;}
.br-t.w{color:#fff;font-weight:700;}
.br-t.l{color:rgba(217,226,255,0.4);}
.br-div{height:1px;background:rgba(255,255,255,0.06);margin:3px 0;}
.br-freq{font-size:9px;color:rgba(208,198,171,0.45);text-align:center;
  margin-top:5px;font-family:Montserrat,sans-serif;letter-spacing:0.06em;}
.br-champ{background:linear-gradient(135deg,rgba(233,196,0,0.18),rgba(233,196,0,0.03));
  border:1px solid rgba(233,196,0,0.4);border-radius:12px;
  padding:18px 14px;text-align:center;
  box-shadow:0 0 28px rgba(233,196,0,0.15);}

/* TEAM CARDS */
.tc{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;overflow:hidden;margin-bottom:14px;
  transition:border-color 0.2s,transform 0.15s;}
.tc:hover{border-color:rgba(233,196,0,0.3);transform:translateY(-2px);}
.tc-img{height:88px;display:flex;align-items:center;justify-content:center;
  font-size:44px;position:relative;overflow:hidden;}
.tc-grp{position:absolute;top:7px;right:9px;
  background:rgba(0,0,0,0.55);border-radius:4px;
  padding:2px 7px;font-family:Montserrat,sans-serif;
  font-size:9px;font-weight:700;color:rgba(208,198,171,0.75);}
.tc-body{padding:12px 14px 14px;}
.tc-rank{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;
  color:rgba(208,198,171,0.45);margin-bottom:3px;}
.tc-name{font-family:Montserrat,sans-serif;font-weight:800;font-size:15px;
  color:#fff;margin-bottom:8px;}
.tc-stats{display:flex;gap:14px;margin-bottom:9px;}
.tc-sv{font-family:Montserrat,sans-serif;font-weight:700;font-size:14px;color:#e9c400;}
.tc-sl{font-family:Inter,sans-serif;font-size:9px;color:rgba(208,198,171,0.45);
  text-transform:uppercase;letter-spacing:0.07em;margin-top:1px;}
.pill{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.06em;text-transform:uppercase;
  padding:3px 9px;border-radius:20px;display:inline-block;margin-right:4px;margin-top:2px;}
.p-gold{background:rgba(233,196,0,0.15);color:#e9c400;}
.p-blue{background:rgba(10,132,255,0.15);color:#5aafff;}
.p-green{background:rgba(0,230,118,0.15);color:#00E676;}
.p-red{background:rgba(255,61,0,0.15);color:#ff6b4a;}
.p-gray{background:rgba(255,255,255,0.07);color:rgba(217,226,255,0.45);}

/* ANALYTICS */
.ap{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;padding:18px 20px;}
.ap-t{font-family:Montserrat,sans-serif;font-weight:800;font-size:14px;
  color:#fff;margin-bottom:14px;letter-spacing:0.04em;}
.sr{display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.sr:last-child{border-bottom:none;}
.sr-l{font-family:Inter,sans-serif;font-size:13px;color:rgba(217,226,255,0.6);}
.sr-v{font-family:Montserrat,sans-serif;font-weight:700;font-size:14px;color:#fff;}

/* FINALS GRID */
.final-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:12px;padding:14px;text-align:center;margin-bottom:10px;}
.fc-rank{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.12em;text-transform:uppercase;color:rgba(208,198,171,0.4);margin-bottom:6px;}
.fc-flags{font-size:22px;margin-bottom:4px;}
.fc-names{font-family:Inter,sans-serif;font-size:12px;color:rgba(217,226,255,0.65);}
.fc-pct{font-family:Montserrat,sans-serif;font-weight:900;font-size:15px;
  color:#e9c400;margin-top:7px;}

/* SECTION TITLES */
.sh{font-family:Montserrat,sans-serif;font-weight:900;font-size:32px;
  color:#fff;letter-spacing:-0.02em;margin-bottom:6px;}
.sh span{color:#e9c400;}
.sh-sub{font-family:Inter,sans-serif;font-size:14px;
  color:rgba(217,226,255,0.45);margin-bottom:20px;}
.sec-label{font-family:Montserrat,sans-serif;font-size:10px;font-weight:700;
  letter-spacing:0.12em;text-transform:uppercase;
  color:rgba(208,198,171,0.5);margin-bottom:12px;}

/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(233,196,0,0.25);border-radius:10px;}

/* Streamlit dataframe dark */
[data-testid="stDataFrame"]{background:transparent!important;}
.dvn-scroller{background:#0d1b36!important;border-radius:8px!important;}

/* Streamlit selectbox / text_input dark */
div[data-baseweb="select"]>div{
  background:rgba(255,255,255,0.05)!important;
  border-color:rgba(255,255,255,0.1)!important;
  color:#d9e2ff!important;border-radius:8px!important;}
input[type="text"]{
  background:rgba(255,255,255,0.05)!important;
  border:1px solid rgba(255,255,255,0.1)!important;
  color:#d9e2ff!important;border-radius:8px!important;}

/* Remove streamlit default padding between columns */
[data-testid="stHorizontalBlock"]{gap:0!important;}
</style>""", unsafe_allow_html=True)

# ── FLAGS & HELPERS ───────────────────────────────────────────────────────────
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

GC = {"A":"#e9c400","B":"#0A84FF","C":"#00E676","D":"#ff9800",
      "E":"#c678dd","F":"#FF3D00","G":"#00bcd4","H":"#61afef",
      "I":"#e06c75","J":"#e9c400","K":"#0A84FF","L":"#00E676"}

TEAM_BG = {
    "Spain":"135deg,#1a0060,#2d0090","Argentina":"135deg,#001240,#003080",
    "France":"135deg,#000e28,#0a2060","Brazil":"135deg,#001800,#004000",
    "Portugal":"135deg,#280004,#6a0010","England":"135deg,#060614,#101050",
    "Netherlands":"135deg,#200800,#6a2200","Germany":"135deg,#080808,#282828",
    "Morocco":"135deg,#001810,#005030","Belgium":"135deg,#1a0000,#500008",
    "Croatia":"135deg,#100008,#380020","Colombia":"135deg,#100e00,#303000",
    "Uruguay":"135deg,#000818,#002448","Switzerland":"135deg,#180000,#500000",
}
def tg(t): return TEAM_BG.get(t, "135deg,#0a1530,#1a2a50")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_outputs():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists():
        return None
    tp = pd.read_csv(p("team_probabilities.csv")).rename(columns={
        "champion_probability":"champion_pct","finalist_probability":"reach_final_pct",
        "semifinalist_probability":"reach_semis_pct","quarterfinalist_prob":"reach_quarters_pct",
        "round_of_16_probability":"reach_round16_pct","round_of_32_probability":"reach_round32_pct",
        "group_exit_probability":"group_exit_pct",
    })
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct","reach_quarters_pct"]:
        if col not in tp.columns:
            tp[col] = (100-tp["group_exit_pct"]).round(2) if col=="pass_group_stage_pct" else 0.0
    return {
        "tp":     tp,
        "finals": pd.read_csv(p("finals.csv")),
        "gs":     pd.read_csv(p("group_summary.csv")),
        "t3":     pd.read_csv(p("third_place_stats.csv")) if p("third_place_stats.csv").exists() else pd.DataFrame(),
        "path":   pd.read_csv(p("path_to_title.csv"))     if p("path_to_title.csv").exists() else pd.DataFrame(),
        "var":    pd.read_csv(p("variance_table.csv"))    if p("variance_table.csv").exists() else pd.DataFrame(),
    }

@st.cache_data(show_spinner=False)
def get_teams(): return load_tournament_data()

# ── AUTO-RUN 10K if no data ───────────────────────────────────────────────────
if not (OUTPUTS_DIR/"team_probabilities.csv").exists():
    with st.spinner("⚽ Running 10,000 simulations (Dixon-Coles)…"):
        res = run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
        generate_all_charts(res["team_probabilities"],res["group_summary"],
                            res["third_place_stats"],res["variance_table"])
    load_outputs.clear()
    get_teams.clear()

data = load_outputs()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "dashboard"

# sim count
n_sims = 80
try:
    log = pd.read_csv(OUTPUTS_DIR/"simulation_log.csv")
    n_sims = int(log[log["metric"]=="n_simulations"]["value"].iloc[0])
except: pass

# ── TOPBAR ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="topbar">'
    f'<span class="tb-brand">FIFA WORLD CUP 2026</span>'
    f'<span class="tb-sims">{n_sims:,} SIMS</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Nav buttons in a single row under topbar
nav_cols = st.columns([2, 1, 1, 1, 1, 3])
pages = [("dashboard","Dashboard"),("brackets","Brackets"),("teams","Teams"),("stats","Stats")]
cur = st.session_state.page

# Topbar nav via styled columns
nav_bar = st.container()
with nav_bar:
    nc = st.columns([2,1,1,1,1,1,3])
    with nc[1]:
        if st.button("Dashboard", key="nav_dash"):
            st.session_state.page = "dashboard"; st.rerun()
    with nc[2]:
        if st.button("Brackets", key="nav_bk"):
            st.session_state.page = "brackets"; st.rerun()
    with nc[3]:
        if st.button("Teams", key="nav_tm"):
            st.session_state.page = "teams"; st.rerun()
    with nc[4]:
        if st.button("Stats", key="nav_st"):
            st.session_state.page = "stats"; st.rerun()
    with nc[5]:
        # Run simulation button
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_clicked = st.button("▶ Run Simulation", key="nav_run")
        st.markdown('</div>', unsafe_allow_html=True)

# Active nav highlight
active_idx = {"dashboard":1,"brackets":2,"teams":3,"stats":4}.get(cur,1)
st.markdown(
    f'<style>div[data-testid="stHorizontalBlock"] '
    f'div[data-testid="column"]:nth-child({active_idx+1}) button{{'
    f'color:#fff!important;border-bottom:2px solid #e9c400!important;'
    f'background:rgba(233,196,0,0.07)!important;}}</style>',
    unsafe_allow_html=True,
)

# Divider
st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0;">', unsafe_allow_html=True)

if run_clicked:
    with st.spinner("Running 10,000 simulations…"):
        res = run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
        generate_all_charts(res["team_probabilities"],res["group_summary"],
                            res["third_place_stats"],res["variance_table"])
    load_outputs.clear(); get_teams.clear()
    st.success("✅ 10,000 simulations complete!")
    st.rerun()

# ── LAYOUT: sidebar + main ────────────────────────────────────────────────────
if data is None:
    st.error("No data found. Click **Run Simulation** above.")
    st.stop()

tp = data["tp"]; finals = data["finals"]; gs = data["gs"]
t3 = data["t3"]; pth = data["path"]; var = data["var"]
top = tp.iloc[0]
fin1 = tp.sort_values("reach_final_pct",ascending=False).iloc[0]
topf = finals.iloc[0] if not finals.empty else None
hg   = gs.iloc[0]    if not gs.empty    else None

col_sb, col_main = st.columns([1, 6], gap="small")

with col_sb:
    page_icons = {
        "dashboard":("🔲","Group Stage"),
        "brackets": ("🌳","Knockout Brackets"),
        "teams":    ("⭐","Team Rankings"),
        "stats":    ("📊","Match History"),
    }
    sb_html = '<div class="sidebar-box"><div class="sb-title">Sim Control</div>'
    for pg, (ic, lbl) in page_icons.items():
        active_cls = " active" if cur == pg else ""
        sb_html += f'<div class="sb-item{active_cls}"><span class="ic">{ic}</span>{lbl}</div>'
    sb_html += (
        f'<div class="sb-footer">v2.0 · Dixon-Coles<br>{n_sims:,} simulations<br>'
        f'FIFA Rankings Jun 2026</div></div>'
    )
    st.markdown(sb_html, unsafe_allow_html=True)

with col_main:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    if cur == "dashboard":
        # Hero
        st.markdown(
            '<div class="sh">TOURNAMENT <span>LIVE</span></div>'
            f'<div class="lbadge"><span class="ldot"></span>'
            f'Group Stage — {n_sims:,} Simulations</div>'
            '<div style="height:20px"></div>',
            unsafe_allow_html=True,
        )

        # KPI row
        k1,k2,k3,k4 = st.columns(4, gap="small")
        flab = topf["final"] if topf is not None else "—"
        fsub = f"{topf['probability_pct']:.1f}% probability" if topf is not None else ""
        hlab = f"Group {hg['group']}" if hg is not None else "—"
        hsub = f"Rating {hg['average_rating']:.1f}" if hg is not None else ""
        for col,cls,ico,lbl,val,sub in [
            (k1,"gold","🏆","Gran Favorito",f"{fl(top['team'])} {top['team']}",f"{top['champion_pct']:.1f}% probability"),
            (k2,"blue","🥈","Más Finalista", f"{fl(fin1['team'])} {fin1['team']}",f"{fin1['reach_final_pct']:.1f}% reach final"),
            (k3,"green","🎯","Final Probable",flab,fsub),
            (k4,"red","💀","Grupo Más Duro",hlab,hsub),
        ]:
            fs = "17px" if len(str(val))>15 else "21px"
            col.markdown(
                f'<div class="kpi {cls}"><div class="kpi-l">{ico} {lbl}</div>'
                f'<div class="kpi-v" style="font-size:{fs}">{val}</div>'
                f'<div class="kpi-s">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

        # Groups + right panel
        cg, cr = st.columns([2,1], gap="medium")

        with cg:
            st.markdown('<div class="sec-label">GROUP STAGE OVERVIEW</div>', unsafe_allow_html=True)
            try:
                td = get_teams()
                for ri in range(0,12,2):
                    g1,g2 = sorted(td["group"].unique())[ri:ri+2]
                    cc1,cc2 = st.columns(2, gap="small")
                    for col_g, grp in [(cc1,g1),(cc2,g2)]:
                        color = GC.get(grp,"#e9c400")
                        gdf   = td[td["group"]==grp].copy()
                        gtp   = tp[tp["group"]==grp].set_index("team")
                        rows  = ""
                        for ri2,(_, rt) in enumerate(gdf.sort_values("overall_rating",ascending=False).iterrows()):
                            t  = rt["team"]
                            ch = float(gtp.loc[t,"champion_pct"]) if t in gtp.index else 0.0
                            ps = float(gtp.loc[t,"pass_group_stage_pct"]) if t in gtp.index else 0.0
                            rtg= int(rt["overall_rating"])
                            qc = "q1" if ri2==0 else ("q2" if ri2==1 else "")
                            cc_= "#e9c400" if ch>8 else ("#00E676" if ch>2 else ("#8bc3ff" if ch>0.4 else "rgba(217,226,255,0.3)"))
                            rows+=(
                                f'<tr class="{qc}"><td>'
                                f'<div class="team-cell"><span class="tf">{fl(t)}</span>'
                                f'<span class="tn">{t}</span></div></td>'
                                f'<td style="color:rgba(208,198,171,0.5)">{rtg}</td>'
                                f'<td style="color:#00E676">{ps:.0f}%</td>'
                                f'<td><span style="font-family:Montserrat;font-weight:700;'
                                f'font-size:12px;color:{cc_}">{ch:.1f}%</span></td>'
                                f'</tr>'
                            )
                        col_g.markdown(
                            f'<div class="gc" style="border-top:3px solid {color};padding:0">'
                            f'<div style="padding:12px 16px;display:flex;justify-content:space-between;'
                            f'align-items:center;border-bottom:1px solid rgba(255,255,255,0.07)">'
                            f'<span style="font-family:Montserrat;font-weight:800;font-size:14px;'
                            f'color:{color}">GROUP {grp}</span>'
                            f'<span class="sec-label" style="margin:0;font-size:8px">Rtg · Pass · Win%</span></div>'
                            f'<div style="padding:4px 4px 8px">'
                            f'<table class="gt"><thead><tr>'
                            f'<th style="text-align:left">Team</th><th>Rtg</th><th>Pass%</th><th>Win%</th>'
                            f'</tr></thead><tbody>{rows}</tbody></table></div></div>',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Error: {e}")

        with cr:
            # Win Probability
            max_p = tp.head(10)["champion_pct"].max()
            pb=""
            for _,rp in tp.head(10).iterrows():
                p2=float(rp["champion_pct"]); w=(p2/max_p*100) if max_p>0 else 0
                pb+=(f'<div class="pb"><div class="pb-head">'
                     f'<span class="pb-name">{fl(rp["team"])} {rp["team"]}</span>'
                     f'<span class="pb-pct" style="color:#00E676">{p2:.1f}%</span></div>'
                     f'<div class="pb-bg"><div class="pb-fill" style="width:{w:.1f}%;background:#00E676"></div>'
                     f'</div></div>')
            st.markdown(
                f'<div class="gc green"><div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:14px">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:15px;color:#fff">'
                f'Win Probability</span>'
                f'<span class="sec-label" style="margin:0;background:rgba(255,255,255,0.06);'
                f'padding:3px 8px;border-radius:4px">TOP 10</span></div>{pb}</div>',
                unsafe_allow_html=True,
            )
            # Semifinal
            tp_sf=tp.nlargest(8,"reach_semis_pct"); max_sf=tp_sf["reach_semis_pct"].max()
            sb=""
            for _,rs in tp_sf.iterrows():
                p2=float(rs["reach_semis_pct"]); w=(p2/max_sf*100) if max_sf>0 else 0
                sb+=(f'<div class="pb"><div class="pb-head">'
                     f'<span class="pb-name">{fl(rs["team"])} {rs["team"]}</span>'
                     f'<span class="pb-pct" style="color:#e9c400">{p2:.1f}%</span></div>'
                     f'<div class="pb-bg"><div class="pb-fill" style="width:{w:.1f}%;background:#e9c400">'
                     f'</div></div></div>')
            st.markdown(
                f'<div class="gc gold" style="margin-top:0">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:15px;'
                f'color:#fff;display:block;margin-bottom:14px">Semifinal Odds</span>{sb}</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════
    #  BRACKETS
    # ══════════════════════════════════════════════════════════════════
    elif cur == "brackets":
        st.markdown(
            '<div class="sh">ROAD TO <span>GLORY</span></div>'
            '<div class="sh-sub">The final 32 teams battle for football immortality. '
            'Track the progression from the opening round to the MetLife Stadium Final.</div>',
            unsafe_allow_html=True,
        )

        cbk, cba = st.columns([3,1], gap="medium")

        with cbk:
            if not pth.empty:
                sel = st.selectbox("Team path", pth.head(16)["team"].tolist(),
                                   label_visibility="visible",
                                   key="bsel")
                row_b = pth[pth["team"]==sel].iloc[0] if sel in pth["team"].values else None

                if row_b is not None:
                    ROUNDS=[
                        ("ROUND OF 32","Dieciseisavos_rival","Dieciseisavos_freq_pct"),
                        ("ROUND OF 16","Octavos_rival","Octavos_freq_pct"),
                        ("QUARTER FINAL","Cuartos_rival","Cuartos_freq_pct"),
                        ("SEMI FINAL","Semifinal_rival","Semifinal_freq_pct"),
                    ]
                    valid=[(l,rc,fc) for l,rc,fc in ROUNDS if rc in row_b.index]
                    rcols=st.columns(len(valid)+1, gap="small")

                    for i,(lbl,rc,fc) in enumerate(valid):
                        rival=str(row_b.get(rc,"TBD"))
                        freq=float(row_b.get(fc,0))
                        with rcols[i]:
                            st.markdown(
                                f'<div class="br-lbl">{lbl}</div>'
                                f'<div class="br-match hl">'
                                f'<div class="br-t w"><span>{fl(sel)} {sel}</span>'
                                f'<span style="color:#e9c400;font-size:14px">W</span></div>'
                                f'<div class="br-div"></div>'
                                f'<div class="br-t l"><span>{fl(rival)} {rival}</span></div>'
                                f'<div class="br-freq">{freq:.0f}% of wins</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    with rcols[-1]:
                        fin_rival="TBD"
                        if not finals.empty:
                            for _,fr in finals.iterrows():
                                if sel in str(fr["final"]):
                                    fin_rival=str(fr["final"]).replace(f"{sel} vs ","").replace(f" vs {sel}","").strip()
                                    break
                        champ_pct=float(row_b["champion_pct"])
                        st.markdown(
                            f'<div class="br-lbl">FINAL 🏆</div>'
                            f'<div class="br-champ">'
                            f'<div style="font-size:30px;margin-bottom:6px">🏆</div>'
                            f'<div style="font-family:Montserrat;font-weight:900;font-size:16px;color:#e9c400">'
                            f'{fl(sel)} {sel}</div>'
                            f'<div style="font-size:11px;color:rgba(217,226,255,0.5);margin-top:4px">'
                            f'vs {fl(fin_rival)} {fin_rival}</div>'
                            f'<div style="font-family:Montserrat;font-weight:700;font-size:14px;'
                            f'color:rgba(233,196,0,0.8);margin-top:6px">{champ_pct:.1f}% champion</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Finals grid
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">MOST FREQUENT FINALS</div>', unsafe_allow_html=True)
            if not finals.empty:
                fc3=st.columns(3, gap="small")
                borders=["#e9c400","#0A84FF","#00E676","#ff9800","#c678dd","#FF3D00","#00bcd4","#61afef","#e06c75"]
                for i,(_,rf) in enumerate(finals.head(9).iterrows()):
                    parts=str(rf["final"]).split(" vs ")
                    t1=parts[0].strip(); t2=(parts[1].strip() if len(parts)>1 else "")
                    with fc3[i%3]:
                        st.markdown(
                            f'<div class="final-card" style="border-top:3px solid {borders[i%len(borders)]}">'
                            f'<div class="fc-rank">Final #{i+1}</div>'
                            f'<div class="fc-flags">{fl(t1)} vs {fl(t2)}</div>'
                            f'<div class="fc-names">{t1} vs {t2}</div>'
                            f'<div class="fc-pct">{rf["probability_pct"]:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        with cba:
            # Tournament odds
            st.markdown('<div class="ap"><div class="ap-t">TOURNAMENT ODDS</div>', unsafe_allow_html=True)
            odd_html=""
            for i,(_,rp) in enumerate(tp.head(10).iterrows()):
                rc="#e9c400" if i==0 else ("#0A84FF" if i<3 else "rgba(217,226,255,0.55)")
                odd_html+=(
                    f'<div class="sr"><span class="sr-l">'
                    f'<span style="font-family:Montserrat;font-weight:700;color:{rc};margin-right:8px">{i+1}.</span>'
                    f'{fl(rp["team"])} {rp["team"]}</span>'
                    f'<span class="sr-v" style="color:{rc}">{rp["champion_pct"]:.1f}%</span></div>'
                )
            st.markdown(odd_html+"</div>", unsafe_allow_html=True)

            # 3rd place
            if not t3.empty:
                summ=t3[t3["categoria"].str.startswith("RESUMEN")].copy()
                if not summ.empty:
                    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="ap"><div class="ap-t">3RD PLACE QUALIFIER</div>', unsafe_allow_html=True)
                    t3h=""
                    for _,sr in summ.iterrows():
                        lbl=sr["categoria"].replace("RESUMEN - ","")
                        raw=sr["puntos"]
                        vs=f"{float(raw):.2f}" if "Clasificados" in lbl else str(int(float(raw)))
                        t3h+=(f'<div class="sr"><span class="sr-l">{lbl}</span>'
                              f'<span class="sr-v" style="color:#e9c400">{vs} pts</span></div>')
                    st.markdown(t3h+"</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  TEAMS
    # ══════════════════════════════════════════════════════════════════
    elif cur == "teams":
        st.markdown(
            '<div class="sh">PARTICIPATING <span>TEAMS</span></div>'
            '<div class="sh-sub">Browse the elite 48 nations competing for glory. '
            'Track rankings, squad stats, and simulation history.</div>',
            unsafe_allow_html=True,
        )

        tf1,tf2,tf3=st.columns([3,2,2], gap="small")
        with tf1:
            search=st.text_input("","",placeholder="Find a team…",
                                 label_visibility="collapsed",key="ts")
        with tf2:
            grp_f=st.selectbox("Group",["ALL"]+[f"GROUP {g}" for g in "ABCDEFGHIJKL"],
                               label_visibility="collapsed",key="tg")
        with tf3:
            sort_b=st.selectbox("Sort",["Champion %","Rating","Semis %"],
                                label_visibility="collapsed",key="tb")

        show=tp.copy()
        if search: show=show[show["team"].str.contains(search,case=False)]
        if grp_f!="ALL": show=show[show["group"]==grp_f.replace("GROUP ","")]
        sc={"Champion %":"champion_pct","Rating":"overall_rating","Semis %":"reach_semis_pct"}[sort_b]
        show=show.sort_values(sc,ascending=False).reset_index(drop=True)

        for ri in range(0,min(len(show),48),4):
            c4=st.columns(4,gap="small")
            for ci,(_,tr) in enumerate(show.iloc[ri:ri+4].iterrows()):
                t=tr["team"]; ch=float(tr["champion_pct"]); sm=float(tr["reach_semis_pct"])
                rtg=int(tr["overall_rating"]); ps=float(tr["pass_group_stage_pct"])
                grp=tr["group"]; bg=tg(t)
                rank=int(show[show["team"]==t].index[0])+1
                if ch>10:  pl1=f'<span class="pill p-gold">FAVORITE</span>'
                elif ch>3: pl1=f'<span class="pill p-blue">CONTENDER</span>'
                elif ps>70:pl1=f'<span class="pill p-green">QUALIFIER</span>'
                else:      pl1=f'<span class="pill p-gray">UNDERDOG</span>'
                if sm>20:  pl2=f'<span class="pill p-green">SEMI FINAL</span>'
                elif sm>8: pl2=f'<span class="pill p-blue">QUARTER F.</span>'
                elif ps>50:pl2=f'<span class="pill p-gray">ROUND 16</span>'
                else:      pl2=f'<span class="pill p-red">GROUP EXIT</span>'
                with c4[ci]:
                    st.markdown(
                        f'<div class="tc">'
                        f'<div class="tc-img" style="background:linear-gradient({bg})">'
                        f'<span style="font-size:46px;filter:drop-shadow(0 2px 8px rgba(0,0,0,0.5))">{fl(t)}</span>'
                        f'<span class="tc-grp">GRP {grp}</span></div>'
                        f'<div class="tc-body">'
                        f'<div class="tc-rank">FIFA RANKING #{rank} · RATING {rtg}</div>'
                        f'<div class="tc-name">{t}</div>'
                        f'<div class="tc-stats">'
                        f'<div><div class="tc-sv">{ch:.1f}%</div><div class="tc-sl">Champion</div></div>'
                        f'<div><div class="tc-sv">{sm:.1f}%</div><div class="tc-sl">Semis</div></div>'
                        f'<div><div class="tc-sv">{ps:.0f}%</div><div class="tc-sl">Pass Grp</div></div>'
                        f'</div>'
                        f'{pl1}{pl2}</div></div>',
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════════════════════════════
    #  STATS
    # ══════════════════════════════════════════════════════════════════
    elif cur == "stats":
        st.markdown(
            '<div class="sh">SIMULATION <span>ANALYTICS</span></div>'
            f'<div class="sh-sub">Based on {n_sims:,} full tournament simulations · Dixon-Coles model.</div>',
            unsafe_allow_html=True,
        )

        cs1,cs2=st.columns([3,2], gap="medium")

        with cs1:
            st.markdown('<div class="sec-label">COMPLETE PROBABILITY TABLE</div>', unsafe_allow_html=True)
            disp=tp[["team","group","overall_rating","champion_pct","reach_final_pct",
                      "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct"]].copy()
            disp.columns=["Team","Grp","Rating","🏆 Champion%","Final%",
                          "Semis%","Quarters%","Pass Grp%","Elim%"]
            disp=disp.reset_index(drop=True); disp.index+=1
            st.dataframe(
                disp.style
                    .background_gradient(subset=["🏆 Champion%"],cmap="YlOrRd")
                    .background_gradient(subset=["Elim%"],cmap="Reds_r")
                    .format({c:"{:.1f}" for c in disp.columns if "%"in c or c=="Rating"}),
                use_container_width=True, height=480,
            )
            st.download_button("⬇ Download CSV",tp.to_csv(index=False).encode(),
                               "wc2026_probs.csv","text/csv")

            # Variance chart
            img_v=OUTPUTS_DIR/"variance_scatter.png"
            if img_v.exists():
                st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-label">VARIANCE — POTENTIAL VS INCONSISTENCY</div>', unsafe_allow_html=True)
                st.image(str(img_v), use_column_width=True)

        with cs2:
            # Phase breakdown
            st.markdown('<div class="sec-label">PHASE BREAKDOWN</div>', unsafe_allow_html=True)
            sel_t=st.selectbox("",tp["team"].tolist(),label_visibility="collapsed",key="stsel")
            rt=tp[tp["team"]==sel_t].iloc[0]
            phases=[
                ("Pass Groups", float(rt["pass_group_stage_pct"]),   "#00E676"),
                ("Round of 32", float(rt.get("reach_round32_pct",0)),"#0A84FF"),
                ("Round of 16", float(rt.get("reach_round16_pct",0)),"#0A84FF"),
                ("Quarters",    float(rt["reach_quarters_pct"]),      "#8bc3ff"),
                ("Semis",       float(rt["reach_semis_pct"]),         "#e9c400"),
                ("Final",       float(rt["reach_final_pct"]),         "#ff9800"),
                ("Champion 🏆", float(rt["champion_pct"]),            "#e9c400"),
            ]
            fig,ax=plt.subplots(figsize=(7,4.5))
            fig.patch.set_facecolor("#020f2a"); ax.set_facecolor("#0d1b36")
            labs=[p[0] for p in phases]; vals=[p[1] for p in phases]; cols=[p[2] for p in phases]
            bars=ax.barh(labs[::-1],vals[::-1],color=cols[::-1],edgecolor="none",height=0.58)
            for bar,val in zip(bars,vals[::-1]):
                if val>0.1:
                    ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
                            f"{val:.1f}%",va="center",ha="left",fontsize=9,color="#d9e2ff")
            ax.set_xlim(0,max(vals or [1])*1.3)
            ax.tick_params(colors="#d9e2ff",labelsize=9)
            ax.spines[:].set_visible(False)
            ax.grid(axis="x",alpha=0.07,color="#283451")
            ax.set_title(f"{fl(sel_t)}  {sel_t}",color="#fff",pad=10,
                         fontsize=12,fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Group difficulty
            if not gs.empty:
                st.markdown('<div class="sec-label" style="margin-top:16px">GROUP DIFFICULTY</div>', unsafe_allow_html=True)
                gs_s=gs.sort_values("average_rating",ascending=False)
                max_gr=float(gs_s["average_rating"].max())
                gd=""
                for _,gr in gs_s.iterrows():
                    color=GC.get(str(gr["group"]),"#e9c400")
                    w=(float(gr["average_rating"])/max_gr*100) if max_gr>0 else 0
                    gd+=(
                        f'<div style="margin-bottom:9px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;color:{color}">'
                        f'GROUP {gr["group"]}</span>'
                        f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;color:#d9e2ff">'
                        f'{gr["average_rating"]:.1f}</span></div>'
                        f'<div style="background:rgba(255,255,255,0.07);height:4px;border-radius:999px;overflow:hidden">'
                        f'<div style="width:{w:.1f}%;height:100%;background:{color};border-radius:999px">'
                        f'</div></div></div>'
                    )
                st.markdown('<div class="gc">'+gd+'</div>', unsafe_allow_html=True)

            # 3rd place threshold
            if not t3.empty:
                summ=t3[t3["categoria"].str.startswith("RESUMEN")].copy()
                if not summ.empty:
                    st.markdown('<div class="sec-label" style="margin-top:12px">3RD PLACE THRESHOLD</div>', unsafe_allow_html=True)
                    t3h=""
                    for _,sr in summ.iterrows():
                        lbl=sr["categoria"].replace("RESUMEN - ","")
                        raw=sr["puntos"]
                        vs=f"{float(raw):.2f}" if "Clasificados" in lbl else str(int(float(raw)))
                        t3h+=(f'<div style="display:flex;justify-content:space-between;align-items:center;'
                              f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)">'
                              f'<span style="font-size:12px;color:rgba(217,226,255,0.6)">{lbl}</span>'
                              f'<span style="font-family:Montserrat;font-weight:700;font-size:15px;'
                              f'color:#e9c400">{vs} pts</span></div>')
                    st.markdown('<div class="gc">'+t3h+'</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close main-content

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="margin-left:0;border-top:1px solid rgba(255,255,255,0.06);'
    'padding:12px 32px;display:flex;justify-content:space-between;align-items:center">'
    '<span style="font-family:Inter;font-size:11px;color:rgba(217,226,255,0.25)">'
    '© 2026 FIFA World Cup Simulation Engine · Dixon-Coles · FIFA Rankings Jun 2026</span>'
    '<span style="font-family:Montserrat;font-size:10px;font-weight:700;color:rgba(233,196,0,0.5)">'
    '⚡ FOR EXPERIMENTAL ANALYSIS</span>'
    '</div>',
    unsafe_allow_html=True,
)
