"""
streamlit_app.py — FIFA World Cup 2026 Prediction Model
Dashboard completo en español · Diseño Apex Stadium
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
from src.power_score import build_ratings_df, GROUPS
from src.match_simulator import win_draw_loss_probs, match_distribution
from src.config import OUTPUTS_DIR

st.set_page_config(page_title="WC 2026 | Prediction Model", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Fuentes ───────────────────────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700;800;900&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>', unsafe_allow_html=True)

# ── CSS Global ────────────────────────────────────────────────────────────────
st.markdown("""<style>
html,body,[data-testid="stAppViewContainer"]{background:#020f2a!important;color:#d9e2ff!important;}
[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stSidebarNav"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important;}
.block-container{padding:0!important;}
footer{display:none!important;}
[data-testid="stVerticalBlock"]{gap:0!important;}

.topbar{background:rgba(5,18,45,0.97);border-bottom:1px solid rgba(255,255,255,0.08);
  padding:0 32px;height:56px;display:flex;align-items:center;gap:28px;
  position:sticky;top:0;z-index:999;}
.tb-brand{font-family:Montserrat,sans-serif;font-weight:900;font-size:15px;
  color:#e9c400;letter-spacing:-0.01em;white-space:nowrap;}
.tb-sims{font-family:Montserrat,sans-serif;font-size:10px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:rgba(208,198,171,0.4);margin-left:auto;}

.stButton>button{background:transparent!important;border:none!important;
  font-family:Montserrat,sans-serif!important;font-weight:700!important;
  font-size:12px!important;letter-spacing:0.08em!important;text-transform:uppercase!important;
  padding:8px 16px!important;color:rgba(208,198,171,0.5)!important;
  border-radius:0!important;transition:color 0.15s!important;width:auto!important;}
.stButton>button:hover{color:#fff!important;background:rgba(255,255,255,0.05)!important;}
.run-btn>div>button{background:#e9c400!important;color:#0d0a00!important;
  border-radius:20px!important;padding:8px 20px!important;font-size:11px!important;}
.run-btn>div>button:hover{box-shadow:0 0 20px rgba(233,196,0,0.4)!important;}

.sidebar-box{background:#010d28;border-right:1px solid rgba(255,255,255,0.07);
  padding:20px 0;min-height:calc(100vh - 56px);}
.sb-title{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.16em;text-transform:uppercase;color:#e9c400;padding:0 16px 8px;}
.sb-item{display:flex;align-items:center;gap:10px;padding:10px 16px;
  font-family:Inter,sans-serif;font-size:13px;font-weight:500;
  color:rgba(217,226,255,0.5);}
.sb-item.active{color:#fff;background:rgba(0,230,118,0.08);border-left:2px solid #00E676;}
.sb-footer{font-family:Inter,sans-serif;font-size:10px;color:rgba(217,226,255,0.25);
  padding:20px 16px;line-height:1.7;margin-top:auto;}

.main-c{background:#020f2a;padding:28px 32px 48px;}

.gc{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;padding:18px 20px;margin-bottom:12px;}
.gc.gold{border-top:3px solid #e9c400!important;}
.gc.blue{border-top:3px solid #0A84FF!important;}
.gc.green{border-top:3px solid #00E676!important;}
.gc.red{border-top:3px solid #FF3D00!important;}

.kpi{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:12px;padding:16px;position:relative;overflow:hidden;height:100%;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kpi.gold::before{background:#e9c400;}.kpi.blue::before{background:#0A84FF;}
.kpi.green::before{background:#00E676;}.kpi.red::before{background:#FF3D00;}
.kpi-l{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.13em;text-transform:uppercase;color:rgba(208,198,171,0.65);margin-bottom:7px;}
.kpi-v{font-family:Montserrat,sans-serif;font-weight:900;color:#fff;line-height:1.1;}
.kpi-s{font-family:Inter,sans-serif;font-size:11px;color:#e9c400;margin-top:5px;}

.lbadge{display:inline-flex;align-items:center;gap:6px;background:rgba(255,61,0,0.13);
  color:#FF3D00;border:1px solid rgba(255,61,0,0.28);border-radius:999px;
  padding:4px 12px;font-family:Montserrat,sans-serif;font-size:9px;
  font-weight:700;letter-spacing:0.12em;text-transform:uppercase;}
.ldot{width:6px;height:6px;background:#FF3D00;border-radius:50%;
  display:inline-block;animation:lp 2s infinite;}
@keyframes lp{0%,100%{opacity:1}50%{opacity:0.2}}

.gt{width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:13px;}
.gt th{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:rgba(208,198,171,0.45);
  padding:0 8px 8px;border-bottom:1px solid rgba(255,255,255,0.07);text-align:center;}
.gt th:first-child{text-align:left;}
.gt td{padding:9px 8px;color:#d9e2ff;border-bottom:1px solid rgba(255,255,255,0.04);text-align:center;}
.gt td:first-child{text-align:left;}
.gt tr:last-child td{border-bottom:none;}
.gt tr.q1 td:first-child{border-left:3px solid #e9c400;padding-left:9px;}
.gt tr.q2 td:first-child{border-left:3px solid #0A84FF;padding-left:9px;}

.pb{margin-bottom:11px;}
.pb-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.pb-name{font-family:Inter,sans-serif;font-size:13px;font-weight:500;color:#e0e8ff;}
.pb-pct{font-family:Montserrat,sans-serif;font-size:13px;font-weight:700;}
.pb-bg{background:rgba(255,255,255,0.08);height:4px;border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;border-radius:999px;}

.br-lbl{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.14em;text-transform:uppercase;color:#e9c400;
  text-align:center;padding:5px 8px;background:rgba(233,196,0,0.08);
  border-radius:4px;margin-bottom:10px;}
.br-match{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:9px;padding:10px 12px;margin-bottom:7px;}
.br-match.hl{border-color:rgba(233,196,0,0.4);background:rgba(233,196,0,0.05);}
.br-t{display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:12px;}
.br-t.w{color:#fff;font-weight:700;}
.br-t.l{color:rgba(217,226,255,0.4);}
.br-div{height:1px;background:rgba(255,255,255,0.06);margin:3px 0;}
.br-freq{font-size:9px;color:rgba(208,198,171,0.45);text-align:center;margin-top:5px;}
.br-champ{background:linear-gradient(135deg,rgba(233,196,0,0.18),rgba(233,196,0,0.03));
  border:1px solid rgba(233,196,0,0.4);border-radius:12px;padding:18px 14px;
  text-align:center;box-shadow:0 0 28px rgba(233,196,0,0.15);}

.match-box{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;padding:20px;text-align:center;}
.match-teams{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;}
.match-team{font-family:Montserrat,sans-serif;font-weight:800;font-size:16px;color:#fff;}
.match-vs{font-family:Montserrat,sans-serif;font-weight:700;font-size:14px;color:rgba(208,198,171,0.4);}
.match-prob{display:flex;justify-content:space-between;gap:8px;}
.mp-box{flex:1;background:rgba(255,255,255,0.04);border-radius:8px;padding:10px;text-align:center;}
.mp-val{font-family:Montserrat,sans-serif;font-weight:900;font-size:22px;}
.mp-lbl{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:rgba(208,198,171,0.5);margin-top:3px;}
.score-pred{font-family:Montserrat,sans-serif;font-weight:900;font-size:28px;
  color:#e9c400;margin:12px 0 4px;}
.score-sub{font-family:Inter,sans-serif;font-size:11px;color:rgba(208,198,171,0.5);}

.tc{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;overflow:hidden;margin-bottom:14px;}
.tc-img{height:85px;display:flex;align-items:center;justify-content:center;
  font-size:42px;position:relative;}
.tc-grp{position:absolute;top:7px;right:9px;background:rgba(0,0,0,0.55);
  border-radius:4px;padding:2px 7px;font-family:Montserrat,sans-serif;
  font-size:9px;font-weight:700;color:rgba(208,198,171,0.75);}
.tc-body{padding:12px 14px 14px;}
.tc-rank{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;color:rgba(208,198,171,0.45);margin-bottom:3px;}
.tc-name{font-family:Montserrat,sans-serif;font-weight:800;font-size:15px;color:#fff;margin-bottom:8px;}
.tc-stats{display:flex;gap:14px;margin-bottom:9px;}
.tc-sv{font-family:Montserrat,sans-serif;font-weight:700;font-size:14px;color:#e9c400;}
.tc-sl{font-family:Inter,sans-serif;font-size:9px;color:rgba(208,198,171,0.45);
  text-transform:uppercase;letter-spacing:0.07em;margin-top:1px;}
.pill{font-family:Montserrat,sans-serif;font-size:9px;font-weight:700;
  letter-spacing:0.06em;text-transform:uppercase;padding:3px 9px;border-radius:20px;
  display:inline-block;margin-right:4px;margin-top:2px;}
.p-gold{background:rgba(233,196,0,0.15);color:#e9c400;}
.p-blue{background:rgba(10,132,255,0.15);color:#5aafff;}
.p-green{background:rgba(0,230,118,0.15);color:#00E676;}
.p-red{background:rgba(255,61,0,0.15);color:#ff6b4a;}
.p-gray{background:rgba(255,255,255,0.07);color:rgba(217,226,255,0.45);}

.ap{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
  border-radius:14px;padding:18px 20px;}
.ap-t{font-family:Montserrat,sans-serif;font-weight:800;font-size:14px;color:#fff;margin-bottom:14px;}
.sr{display:flex;justify-content:space-between;align-items:center;
  padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.sr:last-child{border-bottom:none;}
.sr-l{font-family:Inter,sans-serif;font-size:13px;color:rgba(217,226,255,0.6);}
.sr-v{font-family:Montserrat,sans-serif;font-weight:700;font-size:14px;color:#fff;}

.sh{font-family:Montserrat,sans-serif;font-weight:900;font-size:32px;
  color:#fff;letter-spacing:-0.02em;margin-bottom:6px;}
.sh span{color:#e9c400;}
.sh-sub{font-family:Inter,sans-serif;font-size:14px;color:rgba(217,226,255,0.45);margin-bottom:20px;}
.sec-label{font-family:Montserrat,sans-serif;font-size:10px;font-weight:700;
  letter-spacing:0.12em;text-transform:uppercase;color:rgba(208,198,171,0.5);margin-bottom:12px;}

div[data-baseweb="select"]>div{background:rgba(255,255,255,0.05)!important;
  border-color:rgba(255,255,255,0.1)!important;color:#d9e2ff!important;border-radius:8px!important;}
input[type="text"]{background:rgba(255,255,255,0.05)!important;
  border:1px solid rgba(255,255,255,0.1)!important;color:#d9e2ff!important;border-radius:8px!important;}
[data-testid="stDataFrame"]{background:transparent!important;}
.dvn-scroller{background:#0d1b36!important;border-radius:8px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(233,196,0,0.25);border-radius:10px;}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

GC = {"A":"#e9c400","B":"#0A84FF","C":"#00E676","D":"#ff9800","E":"#c678dd",
      "F":"#FF3D00","G":"#00bcd4","H":"#61afef","I":"#e06c75","J":"#e9c400",
      "K":"#0A84FF","L":"#00E676"}

TEAM_BG = {
    "Spain":"135deg,#1a0060,#2d0090","Argentina":"135deg,#001240,#003080",
    "France":"135deg,#000e28,#0a2060","England":"135deg,#060614,#101050",
    "Brazil":"135deg,#001800,#004000","Portugal":"135deg,#280004,#6a0010",
    "Netherlands":"135deg,#200800,#6a2200","Germany":"135deg,#080808,#282828",
    "Morocco":"135deg,#001810,#005030","Belgium":"135deg,#1a0000,#500008",
    "Croatia":"135deg,#100008,#380020","Colombia":"135deg,#100e00,#303000",
}
def tg(t): return TEAM_BG.get(t, "135deg,#0a1530,#1a2a50")


# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_outputs():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists(): return None
    tp = pd.read_csv(p("team_probabilities.csv"))
    # compatibilidad con versiones anteriores
    tp = tp.rename(columns={
        "champion_probability":"champion_pct","finalist_probability":"reach_final_pct",
        "semifinalist_probability":"reach_semis_pct","quarterfinalist_prob":"reach_quarters_pct",
        "round_of_16_probability":"reach_round16_pct","round_of_32_probability":"reach_round32_pct",
        "group_exit_probability":"group_exit_pct",
    })
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct","reach_quarters_pct","power_score"]:
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
def get_ratings():
    p = OUTPUTS_DIR.parent / "data" / "processed" / "ratings.csv"
    if p.exists(): return pd.read_csv(p)
    return build_ratings_df()

def _needs_run():
    if not (OUTPUTS_DIR / "team_probabilities.csv").exists(): return True
    try:
        log = pd.read_csv(OUTPUTS_DIR / "simulation_log.csv")
        return int(log[log["metric"] == "n_simulations"]["value"].iloc[0]) < 10000
    except: return True

# ── Auto-run 10K ──────────────────────────────────────────────────────────────
if _needs_run():
    with st.spinner("⚽ Ejecutando 10.000 simulaciones Monte Carlo + Dixon-Coles… (~1-2 min)"):
        run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
    load_outputs.clear()

data = load_outputs()

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "dashboard"
cur = st.session_state.page

n_sims = 10000
try:
    log = pd.read_csv(OUTPUTS_DIR / "simulation_log.csv")
    n_sims = int(log[log["metric"] == "n_simulations"]["value"].iloc[0])
except: pass

# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="topbar"><span class="tb-brand">FIFA WORLD CUP 2026</span>'
    f'<span class="tb-sims">{n_sims:,} SIMULACIONES</span></div>',
    unsafe_allow_html=True,
)

# Navegación
nc = st.columns([2, 1, 1, 1, 1, 1, 1, 2])
nav_items = [
    ("dashboard",  "🏠 Dashboard", 1),
    ("grupos",     "🗂 Grupos",    2),
    ("bracket",    "🌳 Bracket",   3),
    ("simulador",  "⚽ Simulador", 4),
    ("equipos",    "⭐ Equipos",   5),
    ("stats",      "📊 Stats",     6),
]
run_clicked = False
for page_id, label, col_idx in nav_items:
    with nc[col_idx]:
        if st.button(label, key=f"nav_{page_id}"):
            st.session_state.page = page_id; st.rerun()
with nc[7]:
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_clicked = st.button("▶ Re-simular", key="nav_run")
    st.markdown('</div>', unsafe_allow_html=True)

# Resaltar pestaña activa
active_idx = {"dashboard":1,"grupos":2,"bracket":3,"simulador":4,"equipos":5,"stats":6}.get(cur,1)
st.markdown(
    f'<style>div[data-testid="stHorizontalBlock"] '
    f'div[data-testid="column"]:nth-child({active_idx+1}) button{{'
    f'color:#fff!important;border-bottom:2px solid #e9c400!important;'
    f'background:rgba(233,196,0,0.07)!important;}}</style>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0">', unsafe_allow_html=True)

if run_clicked:
    with st.spinner("Re-ejecutando 10.000 simulaciones…"):
        run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
    load_outputs.clear(); get_ratings.clear()
    st.success("✅ ¡10.000 simulaciones completadas!"); st.rerun()

if data is None:
    st.error("Sin datos. Pulsa **Re-simular** arriba."); st.stop()

tp = data["tp"]; finals = data["finals"]; gs = data["gs"]
t3 = data["t3"]; pth = data["path"]; var = data["var"]
top  = tp.iloc[0]
fin1 = tp.sort_values("reach_final_pct", ascending=False).iloc[0]
topf = finals.iloc[0] if not finals.empty else None
hg   = gs.iloc[0]    if not gs.empty    else None

# ── Layout: sidebar + main ────────────────────────────────────────────────────
col_sb, col_main = st.columns([1, 6], gap="small")

with col_sb:
    icons = {"dashboard":"🔲","grupos":"🗂","bracket":"🌳",
             "simulador":"⚽","equipos":"⭐","stats":"📊"}
    labels = {"dashboard":"Dashboard","grupos":"Grupos","bracket":"Bracket",
              "simulador":"Simulador","equipos":"Equipos","stats":"Stats"}
    sb = '<div class="sidebar-box"><div class="sb-title">Navegación</div>'
    for pid, ic in icons.items():
        cls = " active" if cur == pid else ""
        sb += f'<div class="sb-item{cls}"><span>{ic}</span>{labels[pid]}</div>'
    sb += f'<div class="sb-footer">v1.0 · Dixon-Coles + Power Score<br>{n_sims:,} simulaciones<br>API-Football · FIFA Jun 2026</div></div>'
    st.markdown(sb, unsafe_allow_html=True)

with col_main:
    st.markdown('<div class="main-c">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  DASHBOARD
    # ════════════════════════════════════════════════════════
    if cur == "dashboard":
        st.markdown(
            '<div class="sh">PREDICCIÓN <span>MUNDIAL 2026</span></div>'
            f'<div class="lbadge"><span class="ldot"></span>'
            f'{n_sims:,} simulaciones · Dixon-Coles + Power Score</div>'
            '<div style="height:20px"></div>',
            unsafe_allow_html=True,
        )
        k1,k2,k3,k4 = st.columns(4, gap="small")
        fl_txt = topf["final"] if topf else "—"
        fl_sub = f"{topf['probability_pct']:.1f}% de las simulaciones" if topf else ""
        for col,cls,ico,lbl,val,sub in [
            (k1,"gold","🏆","Gran Favorito",      f"{fl(top['team'])} {top['team']}",   f"{top['champion_pct']:.2f}% probabilidad"),
            (k2,"blue","🥈","Más veces Finalista", f"{fl(fin1['team'])} {fin1['team']}", f"{fin1['reach_final_pct']:.2f}% llega a la final"),
            (k3,"green","🎯","Final más probable",  fl_txt,                               fl_sub),
            (k4,"red","💀","Grupo más duro",       f"Grupo {hg['group']}" if hg else "—", f"Rating {hg['average_rating']}" if hg else ""),
        ]:
            fs = "17px" if len(str(val)) > 15 else "21px"
            col.markdown(
                f'<div class="kpi {cls}"><div class="kpi-l">{ico} {lbl}</div>'
                f'<div class="kpi-v" style="font-size:{fs}">{val}</div>'
                f'<div class="kpi-s">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
        cg, cr = st.columns([2, 1], gap="medium")

        with cg:
            st.markdown('<div class="sec-label">RESUMEN POR GRUPOS</div>', unsafe_allow_html=True)
            for ri in range(0, 12, 2):
                g1, g2 = sorted(GROUPS.keys())[ri:ri+2]
                cc1, cc2 = st.columns(2, gap="small")
                for col_g, grp in [(cc1,g1),(cc2,g2)]:
                    color = GC.get(grp, "#e9c400")
                    teams = GROUPS[grp]
                    gtp   = tp[tp["group"] == grp].set_index("team")
                    rows  = ""
                    for ri2, t in enumerate(sorted(teams, key=lambda x: -float(gtp.loc[x,"overall_rating"]) if x in gtp.index else 0)):
                        ch  = float(gtp.loc[t,"champion_pct"])         if t in gtp.index else 0.0
                        ps  = float(gtp.loc[t,"power_score"])          if t in gtp.index else 0.0
                        rtg = int(gtp.loc[t,"overall_rating"])         if t in gtp.index else 0
                        qc  = "q1" if ri2==0 else ("q2" if ri2==1 else "")
                        cc_ = "#e9c400" if ch>8 else ("#00E676" if ch>2 else ("#8bc3ff" if ch>0.4 else "rgba(217,226,255,0.3)"))
                        rows += (
                            f'<tr class="{qc}"><td><div style="display:flex;align-items:center;gap:8px">'
                            f'<span style="font-size:16px">{fl(t)}</span>'
                            f'<span style="font-weight:500;color:#e0e8ff">{t}</span></div></td>'
                            f'<td style="color:rgba(208,198,171,0.5)">{rtg}</td>'
                            f'<td style="color:rgba(208,198,171,0.4);font-size:11px">{ps:.0f}</td>'
                            f'<td><span style="font-family:Montserrat;font-weight:700;font-size:12px;color:{cc_}">{ch:.1f}%</span></td>'
                            f'</tr>'
                        )
                    col_g.markdown(
                        f'<div class="gc" style="border-top:3px solid {color};padding:0">'
                        f'<div style="padding:12px 16px;display:flex;justify-content:space-between;'
                        f'align-items:center;border-bottom:1px solid rgba(255,255,255,0.07)">'
                        f'<span style="font-family:Montserrat;font-weight:800;font-size:14px;color:{color}">GRUPO {grp}</span>'
                        f'<span class="sec-label" style="margin:0;font-size:8px">Rtg · PS · 🏆</span></div>'
                        f'<div style="padding:4px 4px 8px"><table class="gt"><thead><tr>'
                        f'<th style="text-align:left">Equipo</th><th>Rtg</th><th>PS</th><th>Win%</th>'
                        f'</tr></thead><tbody>{rows}</tbody></table></div></div>',
                        unsafe_allow_html=True,
                    )

        with cr:
            # Win probability
            max_p = tp.head(10)["champion_pct"].max()
            pb = ""
            for _, rp in tp.head(10).iterrows():
                p2 = float(rp["champion_pct"]); w = (p2/max_p*100) if max_p>0 else 0
                pb += (f'<div class="pb"><div class="pb-head">'
                       f'<span class="pb-name">{fl(rp["team"])} {rp["team"]}</span>'
                       f'<span class="pb-pct" style="color:#00E676">{p2:.1f}%</span></div>'
                       f'<div class="pb-bg"><div class="pb-fill" style="width:{w:.1f}%;background:#00E676"></div>'
                       f'</div></div>')
            st.markdown(
                f'<div class="gc green"><div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:14px">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:15px;color:#fff">Probabilidad campeón</span>'
                f'<span class="sec-label" style="margin:0;background:rgba(255,255,255,0.06);padding:3px 8px;border-radius:4px">TOP 10</span>'
                f'</div>{pb}</div>',
                unsafe_allow_html=True,
            )
            # Semifinal odds
            tp_sf = tp.nlargest(8, "reach_semis_pct"); max_sf = tp_sf["reach_semis_pct"].max()
            sb2 = ""
            for _, rs in tp_sf.iterrows():
                p2 = float(rs["reach_semis_pct"]); w = (p2/max_sf*100) if max_sf>0 else 0
                sb2 += (f'<div class="pb"><div class="pb-head">'
                        f'<span class="pb-name">{fl(rs["team"])} {rs["team"]}</span>'
                        f'<span class="pb-pct" style="color:#e9c400">{p2:.1f}%</span></div>'
                        f'<div class="pb-bg"><div class="pb-fill" style="width:{w:.1f}%;background:#e9c400"></div>'
                        f'</div></div>')
            st.markdown(
                f'<div class="gc gold" style="margin-top:0">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:15px;color:#fff;display:block;margin-bottom:14px">'
                f'Probabilidad semifinales</span>{sb2}</div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════
    #  GRUPOS
    # ════════════════════════════════════════════════════════
    elif cur == "grupos":
        st.markdown('<div class="sh">FASE DE <span>GRUPOS</span></div><div class="sh-sub">Probabilidades por selección · 🥇🥈 clasifican automáticamente · Los 8 mejores 🥉 también pasan</div>', unsafe_allow_html=True)
        for grp in sorted(GROUPS.keys()):
            color = GC.get(grp, "#e9c400")
            teams = GROUPS[grp]
            gtp   = tp[tp["group"] == grp].set_index("team")
            rows  = ""
            for ri2, t in enumerate(sorted(teams, key=lambda x: -float(gtp.loc[x,"overall_rating"]) if x in gtp.index else 0)):
                ch  = float(gtp.loc[t,"champion_pct"])         if t in gtp.index else 0.0
                ps  = float(gtp.loc[t,"power_score"])          if t in gtp.index else 0.0
                pas = float(gtp.loc[t,"pass_group_stage_pct"]) if t in gtp.index else 0.0
                sf  = float(gtp.loc[t,"reach_semis_pct"])      if t in gtp.index else 0.0
                eli = float(gtp.loc[t,"group_exit_pct"])       if t in gtp.index else 0.0
                rtg = int(gtp.loc[t,"overall_rating"])         if t in gtp.index else 0
                medals = ["🥇","🥈","🥉","4️⃣"]
                med    = medals[ri2] if ri2 < 4 else ""
                p_col  = "#00E676" if pas>70 else ("#e9c400" if pas>40 else "#FF3D00")
                e_col  = "#FF3D00" if eli>40 else ("#e9c400" if eli>15 else "rgba(217,226,255,0.3)")
                rows += (
                    f'<tr><td style="padding:10px 8px;font-size:16px">{med}</td>'
                    f'<td style="padding:10px 6px"><div style="display:flex;align-items:center;gap:10px">'
                    f'<span style="font-size:17px">{fl(t)}</span>'
                    f'<span style="font-family:Inter;font-size:14px;font-weight:600;color:#e0e8ff">{t}</span></div></td>'
                    f'<td style="text-align:center;color:rgba(208,198,171,0.5)">{rtg}</td>'
                    f'<td style="text-align:center;color:rgba(208,198,171,0.4)">{ps:.0f}</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-weight:700;font-size:13px;color:{p_col}">{pas:.0f}%</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-weight:700;font-size:13px;color:#e9c400">{ch:.1f}%</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-size:13px;color:#8bc3ff">{sf:.1f}%</td>'
                    f'<td style="text-align:center;font-family:Montserrat;font-size:13px;color:{e_col}">{eli:.1f}%</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div class="gc" style="border-top:3px solid {color};margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
                f'<span style="font-family:Montserrat;font-weight:800;font-size:20px;color:{color}">GRUPO {grp}</span></div>'
                f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-family:Inter;font-size:13px">'
                f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.09)">'
                f'<th style="padding:0 8px 8px;color:#8b9ab8;font-size:9px;font-weight:700;text-transform:uppercase"></th>'
                f'<th style="text-align:left;padding:0 6px 8px;color:#8b9ab8;font-size:9px;text-transform:uppercase">Equipo</th>'
                f'<th style="text-align:center;color:#8b9ab8;font-size:9px;text-transform:uppercase;padding:0 6px 8px">Rating</th>'
                f'<th style="text-align:center;color:#8b9ab8;font-size:9px;text-transform:uppercase;padding:0 6px 8px">Power Score</th>'
                f'<th style="text-align:center;color:#00E676;font-size:9px;text-transform:uppercase;padding:0 6px 8px">Pasa %</th>'
                f'<th style="text-align:center;color:#e9c400;font-size:9px;text-transform:uppercase;padding:0 6px 8px">🏆 %</th>'
                f'<th style="text-align:center;color:#8bc3ff;font-size:9px;text-transform:uppercase;padding:0 6px 8px">Semis %</th>'
                f'<th style="text-align:center;color:#FF3D00;font-size:9px;text-transform:uppercase;padding:0 6px 8px">Elim %</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div></div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════
    #  BRACKET
    # ════════════════════════════════════════════════════════
    elif cur == "bracket":
        st.markdown('<div class="sh">CAMINO AL <span>TÍTULO</span></div><div class="sh-sub">Ruta más probable de cada selección hasta la final · basado en 10.000 simulaciones</div>', unsafe_allow_html=True)
        cbk, cba = st.columns([3, 1], gap="medium")
        with cbk:
            if not pth.empty:
                sel = st.selectbox("Seleccionar equipo", pth.head(16)["team"].tolist(), label_visibility="visible", key="bsel")
                row_b = pth[pth["team"] == sel].iloc[0] if sel in pth["team"].values else None
                if row_b is not None:
                    RNDS = [("DIECISEISAVOS","Dieciseisavos_rival","Dieciseisavos_freq_pct"),
                            ("OCTAVOS",      "Octavos_rival",      "Octavos_freq_pct"),
                            ("CUARTOS",      "Cuartos_rival",      "Cuartos_freq_pct"),
                            ("SEMIFINAL",    "Semifinal_rival",    "Semifinal_freq_pct")]
                    valid = [(l,rc,fc) for l,rc,fc in RNDS if rc in row_b.index]
                    rcols = st.columns(len(valid)+1, gap="small")
                    for i,(lbl,rc,fc) in enumerate(valid):
                        rival = str(row_b.get(rc,"TBD")); freq = float(row_b.get(fc,0))
                        with rcols[i]:
                            st.markdown(
                                f'<div class="br-lbl">{lbl}</div>'
                                f'<div class="br-match hl">'
                                f'<div class="br-t w"><span>{fl(sel)} {sel}</span><span style="color:#e9c400">W</span></div>'
                                f'<div class="br-div"></div>'
                                f'<div class="br-t l"><span>{fl(rival)} {rival}</span></div>'
                                f'<div class="br-freq">{freq:.0f}% de las victorias</div></div>',
                                unsafe_allow_html=True,
                            )
                    with rcols[-1]:
                        fin_rival = "TBD"
                        if not finals.empty:
                            for _,fr in finals.iterrows():
                                if sel in str(fr["final"]):
                                    fin_rival = str(fr["final"]).replace(f"{sel} vs ","").replace(f" vs {sel}","").strip(); break
                        st.markdown(
                            f'<div class="br-lbl">FINAL 🏆</div>'
                            f'<div class="br-champ">'
                            f'<div style="font-size:30px;margin-bottom:6px">🏆</div>'
                            f'<div style="font-family:Montserrat;font-weight:900;font-size:16px;color:#e9c400">{fl(sel)} {sel}</div>'
                            f'<div style="font-size:11px;color:rgba(217,226,255,0.5);margin-top:4px">vs {fl(fin_rival)} {fin_rival}</div>'
                            f'<div style="font-family:Montserrat;font-weight:700;font-size:14px;color:rgba(233,196,0,0.8);margin-top:6px">'
                            f'{float(row_b["champion_pct"]):.1f}% campeón</div></div>',
                            unsafe_allow_html=True,
                        )
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">FINALES MÁS PROBABLES</div>', unsafe_allow_html=True)
            if not finals.empty:
                borders = ["#e9c400","#0A84FF","#00E676","#ff9800","#c678dd","#FF3D00","#00bcd4","#61afef","#e06c75"]
                fc3 = st.columns(3, gap="small")
                for i,(_,rf) in enumerate(finals.head(9).iterrows()):
                    parts = str(rf["final"]).split(" vs ")
                    t1 = parts[0].strip(); t2 = parts[1].strip() if len(parts)>1 else ""
                    with fc3[i%3]:
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);'
                            f'border-top:3px solid {borders[i%len(borders)]};border-radius:12px;padding:14px;'
                            f'text-align:center;margin-bottom:10px">'
                            f'<div class="sec-label" style="margin-bottom:6px">Final #{i+1}</div>'
                            f'<div style="font-size:22px;margin-bottom:4px">{fl(t1)} vs {fl(t2)}</div>'
                            f'<div style="font-family:Inter;font-size:12px;color:rgba(217,226,255,0.65)">{t1} vs {t2}</div>'
                            f'<div style="font-family:Montserrat;font-weight:900;font-size:15px;color:#e9c400;margin-top:7px">'
                            f'{rf["probability_pct"]:.1f}%</div></div>',
                            unsafe_allow_html=True,
                        )
        with cba:
            st.markdown('<div class="ap"><div class="ap-t">CUOTAS DEL TORNEO</div>', unsafe_allow_html=True)
            odd_h = ""
            for i,(_,rp) in enumerate(tp.head(10).iterrows()):
                rc = "#e9c400" if i==0 else ("#0A84FF" if i<3 else "rgba(217,226,255,0.55)")
                odd_h += (f'<div class="sr"><span class="sr-l">'
                          f'<span style="font-family:Montserrat;font-weight:700;color:{rc};margin-right:8px">{i+1}.</span>'
                          f'{fl(rp["team"])} {rp["team"]}</span>'
                          f'<span class="sr-v" style="color:{rc}">{rp["champion_pct"]:.1f}%</span></div>')
            st.markdown(odd_h + "</div>", unsafe_allow_html=True)
            if not t3.empty:
                summ = t3[t3["categoria"].str.startswith("RESUMEN")].copy()
                if not summ.empty:
                    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="ap"><div class="ap-t">UMBRAL MEJOR 3°</div>', unsafe_allow_html=True)
                    t3h = ""
                    for _,sr in summ.iterrows():
                        lbl = sr["categoria"].replace("RESUMEN - ","")
                        raw = sr["puntos"]
                        vs  = f"{float(raw):.2f}" if "Clasificados" in lbl else str(int(float(raw)))
                        t3h += (f'<div class="sr"><span class="sr-l">{lbl}</span>'
                                f'<span class="sr-v" style="color:#e9c400">{vs} pts</span></div>')
                    st.markdown(t3h + "</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  SIMULADOR DE PARTIDOS
    # ════════════════════════════════════════════════════════
    elif cur == "simulador":
        st.markdown('<div class="sh">SIMULADOR DE <span>PARTIDOS</span></div><div class="sh-sub">Selecciona dos selecciones y calcula la predicción en tiempo real con Dixon-Coles + Power Score</div>', unsafe_allow_html=True)
        ratings_df = get_ratings()
        all_teams  = sorted(ratings_df["team"].tolist())

        sc1, sc2, sc3 = st.columns([2, 1, 2], gap="medium")
        with sc1:
            team_a = st.selectbox("Equipo A (Local)", all_teams, index=all_teams.index("Spain") if "Spain" in all_teams else 0, label_visibility="visible", key="sim_a")
        with sc2:
            st.markdown('<div style="text-align:center;padding-top:28px;font-family:Montserrat;font-weight:800;font-size:20px;color:rgba(208,198,171,0.4)">VS</div>', unsafe_allow_html=True)
        with sc3:
            team_b = st.selectbox("Equipo B (Visitante)", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 1, label_visibility="visible", key="sim_b")

        if team_a == team_b:
            st.warning("Selecciona dos equipos diferentes.")
        else:
            row_a = ratings_df[ratings_df["team"] == team_a].iloc[0]
            row_b = ratings_df[ratings_df["team"] == team_b].iloc[0]
            pred  = win_draw_loss_probs(row_a, row_b)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="match-box">'
                f'<div class="match-teams">'
                f'<div class="match-team">{fl(team_a)} {team_a}</div>'
                f'<div class="match-vs">VS</div>'
                f'<div class="match-team">{team_b} {fl(team_b)}</div>'
                f'</div>'
                f'<div class="match-prob">'
                f'<div class="mp-box"><div class="mp-val" style="color:#00E676">{pred["p_home"]:.1f}%</div><div class="mp-lbl">Victoria {team_a.split()[0]}</div></div>'
                f'<div class="mp-box"><div class="mp-val" style="color:#e9c400">{pred["p_draw"]:.1f}%</div><div class="mp-lbl">Empate</div></div>'
                f'<div class="mp-box"><div class="mp-val" style="color:#0A84FF">{pred["p_away"]:.1f}%</div><div class="mp-lbl">Victoria {team_b.split()[0]}</div></div>'
                f'</div>'
                f'<div class="score-pred">{pred["most_likely_score"]}</div>'
                f'<div class="score-sub">Marcador más probable · {pred["most_likely_prob"]:.1f}% de probabilidad</div>'
                f'<div style="margin-top:12px;display:flex;justify-content:center;gap:24px">'
                f'<div style="text-align:center"><div style="font-family:Montserrat;font-weight:700;font-size:14px;color:#e9c400">{pred["xg_home"]:.2f}</div><div class="sec-label" style="margin:0">xG {team_a.split()[0]}</div></div>'
                f'<div style="text-align:center"><div style="font-family:Montserrat;font-weight:700;font-size:14px;color:#e9c400">{pred["xg_away"]:.2f}</div><div class="sec-label" style="margin:0">xG {team_b.split()[0]}</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Distribución de marcadores
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">TOP MARCADORES MÁS PROBABLES</div>', unsafe_allow_html=True)
            dist = match_distribution(row_a, row_b).head(10)
            fig, ax = plt.subplots(figsize=(9, 3.5))
            fig.patch.set_facecolor("#020f2a"); ax.set_facecolor("#0d1b36")
            score_labels = [f"{int(r['home_goals'])}-{int(r['away_goals'])}" for _, r in dist.iterrows()]
            score_probs  = [r["probability"] * 100 for _, r in dist.iterrows()]
            colors = ["#e9c400" if i == 0 else "#0A84FF" if i < 3 else "#283451" for i in range(len(score_labels))]
            ax.bar(score_labels, score_probs, color=colors, edgecolor="none", width=0.65)
            for i, (lbl, val) in enumerate(zip(score_labels, score_probs)):
                ax.text(i, val + 0.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#d9e2ff")
            ax.set_ylabel("%", color="#8b9ab8", fontsize=9); ax.tick_params(colors="#d9e2ff", labelsize=9)
            ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.07, color="#283451")
            ax.set_title(f"{team_a} vs {team_b} — Distribución de marcadores", color="#fff", pad=10, fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Métricas comparativas
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4, gap="small")
            for col, lbl, va, vb in [
                (m1, "Rating compuesto",   f"{row_a['overall_rating']:.1f}", f"{row_b['overall_rating']:.1f}"),
                (m2, "Power Score",        f"{row_a.get('power_score',0):.1f}", f"{row_b.get('power_score',0):.1f}"),
                (m3, "ELO Rating",         str(int(row_a.get('elo_rating',0))), str(int(row_b.get('elo_rating',0)))),
                (m4, "Coef. Ataque",       f"{row_a['attack_coef']:.3f}", f"{row_b['attack_coef']:.3f}"),
            ]:
                col.markdown(
                    f'<div class="gc" style="text-align:center;padding:12px">'
                    f'<div class="sec-label" style="margin-bottom:8px">{lbl}</div>'
                    f'<div style="display:flex;justify-content:space-around">'
                    f'<div><div style="font-family:Montserrat;font-weight:700;font-size:16px;color:#00E676">{va}</div>'
                    f'<div style="font-size:10px;color:rgba(208,198,171,0.5)">{team_a.split()[0]}</div></div>'
                    f'<div><div style="font-family:Montserrat;font-weight:700;font-size:16px;color:#0A84FF">{vb}</div>'
                    f'<div style="font-size:10px;color:rgba(208,198,171,0.5)">{team_b.split()[0]}</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════
    #  EQUIPOS
    # ════════════════════════════════════════════════════════
    elif cur == "equipos":
        st.markdown('<div class="sh">SELECCIONES <span>PARTICIPANTES</span></div><div class="sh-sub">Las 48 naciones del Mundial 2026 · Rankings, Power Score y estadísticas de simulación</div>', unsafe_allow_html=True)
        tf1, tf2, tf3 = st.columns([3,2,2], gap="small")
        with tf1: search = st.text_input("","",placeholder="Buscar selección…",label_visibility="collapsed",key="ts")
        with tf2: grp_f  = st.selectbox("Grupo",["TODOS"]+[f"GRUPO {g}" for g in "ABCDEFGHIJKL"],label_visibility="collapsed",key="tg")
        with tf3: sort_b = st.selectbox("Ordenar",["Campeón %","Rating","Power Score","Semis %"],label_visibility="collapsed",key="tb")

        show = tp.copy()
        if search: show = show[show["team"].str.contains(search, case=False)]
        if grp_f != "TODOS": show = show[show["group"] == grp_f.replace("GRUPO ","")]
        sc_map = {"Campeón %":"champion_pct","Rating":"overall_rating","Power Score":"power_score","Semis %":"reach_semis_pct"}
        show = show.sort_values(sc_map[sort_b], ascending=False).reset_index(drop=True)

        for ri in range(0, min(len(show), 48), 4):
            c4 = st.columns(4, gap="small")
            for ci, (_, tr) in enumerate(show.iloc[ri:ri+4].iterrows()):
                t=tr["team"]; ch=float(tr["champion_pct"]); sm=float(tr["reach_semis_pct"])
                ps=float(tr.get("power_score",0)); rtg=int(tr["overall_rating"])
                pas=float(tr["pass_group_stage_pct"]); grp=tr["group"]; bg=tg(t)
                rank = int(show[show["team"]==t].index[0])+1
                pl1 = f'<span class="pill p-gold">FAVORITO</span>' if ch>10 else (f'<span class="pill p-blue">CONTENDIENTE</span>' if ch>3 else (f'<span class="pill p-green">CLASIFICABLE</span>' if pas>70 else f'<span class="pill p-gray">OUTSIDER</span>'))
                pl2 = f'<span class="pill p-green">SEMIFINAL</span>' if sm>20 else (f'<span class="pill p-blue">CUARTOS</span>' if sm>8 else (f'<span class="pill p-gray">16AVOS</span>' if pas>50 else f'<span class="pill p-red">GRUPOS</span>'))
                with c4[ci]:
                    st.markdown(
                        f'<div class="tc"><div class="tc-img" style="background:linear-gradient({bg})">'
                        f'<span style="font-size:42px;filter:drop-shadow(0 2px 8px rgba(0,0,0,0.5))">{fl(t)}</span>'
                        f'<span class="tc-grp">GRP {grp}</span></div>'
                        f'<div class="tc-body">'
                        f'<div class="tc-rank">RANKING #{rank} · RATING {rtg} · PS {ps:.0f}</div>'
                        f'<div class="tc-name">{t}</div>'
                        f'<div class="tc-stats">'
                        f'<div><div class="tc-sv">{ch:.1f}%</div><div class="tc-sl">Campeón</div></div>'
                        f'<div><div class="tc-sv">{sm:.1f}%</div><div class="tc-sl">Semis</div></div>'
                        f'<div><div class="tc-sv">{pas:.0f}%</div><div class="tc-sl">Pasa grp</div></div>'
                        f'</div>{pl1}{pl2}</div></div>',
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════
    #  STATS
    # ════════════════════════════════════════════════════════
    elif cur == "stats":
        st.markdown(f'<div class="sh">ANALYTICS <span>COMPLETO</span></div><div class="sh-sub">Tabla completa · {n_sims:,} simulaciones · Dixon-Coles + Power Score</div>', unsafe_allow_html=True)
        cs1, cs2 = st.columns([3, 2], gap="medium")
        with cs1:
            st.markdown('<div class="sec-label">TABLA DE PROBABILIDADES</div>', unsafe_allow_html=True)
            disp = tp[["team","group","overall_rating","power_score","champion_pct","reach_final_pct",
                        "reach_semis_pct","reach_quarters_pct","pass_group_stage_pct","group_exit_pct"]].copy()
            disp.columns = ["Selección","Grupo","Rating","Power Score","🏆 Campeón%","Final%","Semis%","Cuartos%","Pasa Grp%","Elim%"]
            disp = disp.reset_index(drop=True); disp.index += 1
            st.dataframe(
                disp.style
                    .background_gradient(subset=["🏆 Campeón%"], cmap="YlOrRd")
                    .background_gradient(subset=["Elim%"], cmap="Reds_r")
                    .format({c:"{:.1f}" for c in disp.columns if "%" in c or c in ["Rating","Power Score"]}),
                use_container_width=True, height=480,
            )
            st.download_button("⬇ Descargar CSV", tp.to_csv(index=False).encode(), "wc2026_probabilidades.csv", "text/csv")
            img_v = OUTPUTS_DIR / "variance_scatter.png"
            if img_v.exists():
                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-label">VARIANZA — POTENCIAL VS INCONSISTENCIA</div>', unsafe_allow_html=True)
                st.image(str(img_v), use_column_width=True)

        with cs2:
            st.markdown('<div class="sec-label">DESGLOSE POR FASE</div>', unsafe_allow_html=True)
            sel_t = st.selectbox("", tp["team"].tolist(), label_visibility="collapsed", key="stsel")
            rt = tp[tp["team"] == sel_t].iloc[0]
            phases = [
                ("Pasa grupos",   float(rt["pass_group_stage_pct"]),  "#00E676"),
                ("Dieciseisavos", float(rt.get("reach_round32_pct",0)),"#0A84FF"),
                ("Octavos",       float(rt.get("reach_round16_pct",0)),"#0A84FF"),
                ("Cuartos",       float(rt["reach_quarters_pct"]),     "#8bc3ff"),
                ("Semis",         float(rt["reach_semis_pct"]),        "#e9c400"),
                ("Final",         float(rt["reach_final_pct"]),        "#ff9800"),
                ("Campeón 🏆",   float(rt["champion_pct"]),           "#e9c400"),
            ]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor("#020f2a"); ax.set_facecolor("#0d1b36")
            labs=[p[0] for p in phases]; vals=[p[1] for p in phases]; cols=[p[2] for p in phases]
            bars = ax.barh(labs[::-1], vals[::-1], color=cols[::-1], edgecolor="none", height=0.58)
            for bar, val in zip(bars, vals[::-1]):
                if val > 0.1:
                    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}%",
                            va="center", ha="left", fontsize=9, color="#d9e2ff")
            ax.set_xlim(0, max(vals or [1])*1.3); ax.tick_params(colors="#d9e2ff", labelsize=9)
            ax.spines[:].set_visible(False); ax.grid(axis="x", alpha=0.07, color="#283451")
            ax.set_title(f"{fl(sel_t)}  {sel_t}", color="#fff", pad=10, fontsize=12, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Dificultad de grupos
            if not gs.empty:
                st.markdown('<div class="sec-label" style="margin-top:16px">DIFICULTAD DE GRUPOS</div>', unsafe_allow_html=True)
                gs_s = gs.sort_values("average_rating", ascending=False)
                max_gr = float(gs_s["average_rating"].max())
                gd = ""
                for _, gr in gs_s.iterrows():
                    color = GC.get(str(gr["group"]), "#e9c400")
                    w = (float(gr["average_rating"])/max_gr*100) if max_gr>0 else 0
                    gd += (f'<div style="margin-bottom:9px">'
                           f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                           f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;color:{color}">GRUPO {gr["group"]}</span>'
                           f'<span style="font-family:Montserrat;font-weight:700;font-size:12px;color:#d9e2ff">{gr["average_rating"]:.1f}</span></div>'
                           f'<div style="background:rgba(255,255,255,0.07);height:4px;border-radius:999px;overflow:hidden">'
                           f'<div style="width:{w:.1f}%;height:100%;background:{color};border-radius:999px"></div></div></div>')
                st.markdown('<div class="gc">'+gd+'</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid rgba(255,255,255,0.06);padding:12px 32px;'
    'display:flex;justify-content:space-between;align-items:center">'
    '<span style="font-family:Inter;font-size:11px;color:rgba(217,226,255,0.25)">'
    '© 2026 FIFA World Cup Prediction Model · Dixon-Coles + Power Score · API-Football · FIFA Rankings Jun 2026</span>'
    '<span style="font-family:Montserrat;font-size:10px;font-weight:700;color:rgba(233,196,0,0.5)">'
    '⚡ SOLO PARA ANÁLISIS EXPERIMENTAL</span></div>',
    unsafe_allow_html=True,
)
