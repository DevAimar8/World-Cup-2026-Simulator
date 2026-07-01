"""
streamlit_app.py
----------------
FIFA World Cup 2026 — Live Bracket + Monte Carlo Simulator
UI: Bracket visual real + simulación por partido + análisis de equipos

Comportamiento:
  - Al entrar: bracket real con resultados ya jugados, EN VIVO y pendientes
  - Click en partido: predicción Dixon-Coles + jugadores + análisis
  - Botón "Simular Mundial": 10K Monte Carlo → rellena bracket con marcadores probables
  - Refresco automático de datos a las 10:00 UTC diariamente

Autor: Aimar Esqueta
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
from datetime import datetime, timezone

from src.config import OUTPUTS_DIR
from src.live_data import (
    fetch_live_fixtures, get_team_players, get_team_analysis,
    get_live_fixtures, should_refresh_today, STAR_PLAYERS
)
from src.power_score import build_ratings_df, GROUPS
from src.match_simulator import win_draw_loss_probs, match_distribution

st.set_page_config(
    page_title="WC 2026 | Live Bracket",
    page_icon="⚽", layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Fuentes ───────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700'
    '&family=Space+Grotesk:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>',
    unsafe_allow_html=True,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
/* ── Reset ── */
html,body,[data-testid="stAppViewContainer"]{background:#07090f!important;color:#e8eaf0!important;}
[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stSidebarNav"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important;}
.block-container{padding:0!important;}
footer{display:none!important;}

/* ── Tipografía base ── */
*{font-family:'Inter',sans-serif;}
.brand-font{font-family:'Space Grotesk',sans-serif;}

/* ── Topbar ── */
.topbar{
  background:rgba(7,9,15,0.95);
  border-bottom:1px solid rgba(255,255,255,0.06);
  padding:0 28px; height:60px;
  display:flex; align-items:center; gap:24px;
  position:sticky; top:0; z-index:1000;
}
.tb-logo{
  font-family:'Space Grotesk',sans-serif; font-weight:800;
  font-size:16px; color:#fff; letter-spacing:-0.02em;
}
.tb-logo span{color:#FFCB00;}
.tb-live{
  display:inline-flex; align-items:center; gap:5px;
  background:rgba(255,50,50,0.12); color:#ff5252;
  border:1px solid rgba(255,50,50,0.25); border-radius:20px;
  padding:3px 10px; font-size:10px; font-weight:600;
  letter-spacing:0.1em; text-transform:uppercase;
}
.tb-dot{width:5px;height:5px;background:#ff5252;border-radius:50%;
  display:inline-block;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.2}}
.tb-info{font-size:12px;color:rgba(232,234,240,0.4);margin-left:auto;}

/* ── Botones Streamlit override ── */
.stButton>button{
  background:rgba(255,255,255,0.06)!important;
  border:1px solid rgba(255,255,255,0.1)!important;
  color:rgba(232,234,240,0.7)!important;
  font-family:'Inter',sans-serif!important; font-weight:500!important;
  font-size:12px!important; border-radius:8px!important;
  padding:8px 16px!important; transition:all 0.15s!important;
}
.stButton>button:hover{
  background:rgba(255,255,255,0.1)!important;
  color:#fff!important; border-color:rgba(255,255,255,0.2)!important;
}
.btn-sim>div>button{
  background:linear-gradient(135deg,#FFCB00,#FF9F00)!important;
  color:#0a0800!important; border:none!important;
  font-weight:700!important; font-size:13px!important;
  border-radius:12px!important; padding:12px 24px!important;
  letter-spacing:0.02em!important;
}
.btn-refresh>div>button{
  background:rgba(255,203,0,0.1)!important;
  color:#FFCB00!important;
  border:1px solid rgba(255,203,0,0.25)!important;
  border-radius:8px!important;
}

/* ── Sección header ── */
.sec-h{
  font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:11px; letter-spacing:0.12em; text-transform:uppercase;
  color:rgba(232,234,240,0.35); margin-bottom:12px;
}

/* ── Tarjeta de partido del bracket ── */
.match-card{
  background:#0e1118; border:1px solid rgba(255,255,255,0.07);
  border-radius:10px; overflow:hidden; cursor:pointer;
  transition:border-color 0.15s, transform 0.1s;
  margin-bottom:6px;
}
.match-card:hover{border-color:rgba(255,203,0,0.3);transform:translateY(-1px);}
.match-card.live{border-color:rgba(255,82,82,0.4);animation:liveborder 2s infinite;}
@keyframes liveborder{0%,100%{border-color:rgba(255,82,82,0.4)}50%{border-color:rgba(255,82,82,0.8)}}
.match-card.played{border-left:3px solid rgba(0,230,100,0.5);}
.match-card.pending{border-left:3px solid rgba(255,203,0,0.2);}
.match-card.simulated{border-left:3px solid rgba(100,180,255,0.5);}
.mc-header{
  padding:6px 12px 4px;
  font-size:10px; font-weight:600; letter-spacing:0.08em;
  text-transform:uppercase; color:rgba(232,234,240,0.3);
  display:flex; justify-content:space-between; align-items:center;
}
.mc-body{padding:8px 12px 10px;}
.mc-team{
  display:flex; align-items:center; justify-content:space-between;
  padding:4px 0;
}
.mc-team-info{display:flex;align-items:center;gap:8px;}
.mc-flag{font-size:18px;line-height:1;}
.mc-name{font-size:13px;font-weight:500;color:#e8eaf0;}
.mc-name.winner{font-weight:700;color:#fff;}
.mc-score{
  font-family:'Space Grotesk',sans-serif;
  font-size:18px; font-weight:700; color:#fff; min-width:24px; text-align:right;
}
.mc-score.winner-score{color:#FFCB00;}
.mc-divider{height:1px;background:rgba(255,255,255,0.05);margin:3px 0;}
.mc-date{font-size:10px;color:rgba(232,234,240,0.35);text-align:center;padding-top:4px;}
.mc-sim-score{
  font-size:10px; color:rgba(100,180,255,0.7);
  text-align:center; padding-top:2px; font-style:italic;
}
.mc-live-badge{
  background:rgba(255,82,82,0.15); color:#ff5252;
  border-radius:4px; padding:2px 6px;
  font-size:9px; font-weight:700;
}

/* ── Ronda label ── */
.round-label{
  font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:11px; letter-spacing:0.1em; text-transform:uppercase;
  color:rgba(255,203,0,0.6); margin-bottom:10px;
  padding:6px 10px; background:rgba(255,203,0,0.06);
  border-radius:6px; text-align:center;
}

/* ── Panel de detalle de partido ── */
.detail-panel{
  background:#0b0d14; border:1px solid rgba(255,255,255,0.08);
  border-radius:16px; padding:24px;
}
.dp-header{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:20px;
}
.dp-vs{
  font-family:'Space Grotesk',sans-serif; font-weight:800;
  font-size:26px; color:#fff; text-align:center;
}
.dp-vs span{color:rgba(232,234,240,0.3); font-size:16px; font-weight:400;}

/* ── Probabilidades 1X2 ── */
.probs-row{display:flex;gap:10px;margin-bottom:20px;}
.prob-box{
  flex:1; background:#0e1118; border:1px solid rgba(255,255,255,0.08);
  border-radius:10px; padding:14px 10px; text-align:center;
}
.prob-box.main{border-color:rgba(255,203,0,0.3);background:rgba(255,203,0,0.05);}
.prob-val{
  font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:24px; color:#fff;
}
.prob-val.main{color:#FFCB00;}
.prob-lbl{font-size:10px;font-weight:600;letter-spacing:0.1em;
  text-transform:uppercase;color:rgba(232,234,240,0.35);margin-top:4px;}

/* ── xG bar ── */
.xg-row{display:flex;align-items:center;gap:12px;margin:14px 0;}
.xg-label{font-size:12px;color:rgba(232,234,240,0.5);min-width:80px;}
.xg-bar-wrap{flex:1;background:rgba(255,255,255,0.07);height:6px;border-radius:999px;overflow:hidden;}
.xg-bar{height:100%;border-radius:999px;}
.xg-val{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:13px;
  color:#fff;min-width:36px;text-align:right;}

/* ── Tarjetas de jugadores ── */
.player-card{
  background:#0e1118; border:1px solid rgba(255,255,255,0.07);
  border-radius:10px; padding:14px 16px; margin-bottom:8px;
  display:flex; align-items:center; gap:14px;
}
.player-rating{
  font-family:'Space Grotesk',sans-serif; font-weight:800;
  font-size:22px; color:#FFCB00; min-width:36px;
}
.player-name{font-size:14px;font-weight:600;color:#fff;}
.player-pos{font-size:11px;color:rgba(232,234,240,0.4);margin-top:2px;}
.player-stats{display:flex;gap:12px;margin-top:6px;}
.pstat{text-align:center;}
.pstat-v{font-size:13px;font-weight:600;color:#e8eaf0;}
.pstat-l{font-size:9px;color:rgba(232,234,240,0.35);text-transform:uppercase;letter-spacing:0.08em;}

/* ── Análisis card ── */
.analysis-card{
  background:rgba(255,203,0,0.04); border:1px solid rgba(255,203,0,0.12);
  border-radius:10px; padding:16px; margin-top:8px;
}
.analysis-title{font-size:11px;font-weight:600;letter-spacing:0.1em;
  text-transform:uppercase;color:rgba(255,203,0,0.7);margin-bottom:8px;}
.analysis-text{font-size:13px;color:rgba(232,234,240,0.7);line-height:1.65;}

/* ── Score predicho ── */
.score-pred-box{
  background:#0e1118; border:1px solid rgba(100,180,255,0.2);
  border-radius:12px; padding:16px; text-align:center; margin:14px 0;
}
.score-pred-val{
  font-family:'Space Grotesk',sans-serif; font-weight:800;
  font-size:32px; color:#fff; letter-spacing:0.05em;
}
.score-pred-sub{font-size:11px;color:rgba(232,234,240,0.35);margin-top:4px;}

/* ── Grupo table ── */
.grp-card{
  background:#0e1118; border:1px solid rgba(255,255,255,0.07);
  border-radius:12px; overflow:hidden; margin-bottom:12px;
}
.grp-header{
  padding:10px 14px;
  background:rgba(255,255,255,0.03);
  border-bottom:1px solid rgba(255,255,255,0.06);
  font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:13px; color:#fff;
}
.grp-row{
  display:flex; align-items:center; padding:9px 14px;
  border-bottom:1px solid rgba(255,255,255,0.04);
  font-size:13px;
}
.grp-row:last-child{border-bottom:none;}
.grp-row.q1{background:rgba(255,203,0,0.04);}
.grp-row.q2{background:rgba(100,180,255,0.03);}
.grp-pos{color:rgba(232,234,240,0.3);min-width:20px;font-size:11px;}
.grp-flag{font-size:16px;margin:0 8px;}
.grp-name{flex:1;font-weight:500;color:#e8eaf0;}
.grp-pts{font-family:'Space Grotesk',sans-serif;font-weight:700;
  font-size:14px;color:#fff;min-width:24px;text-align:center;}
.grp-sub{font-size:11px;color:rgba(232,234,240,0.3);min-width:28px;text-align:center;}

/* ── KPIs ── */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
.kpi-card{
  background:#0e1118; border:1px solid rgba(255,255,255,0.07);
  border-radius:12px; padding:16px;
}
.kpi-label{font-size:10px;font-weight:600;letter-spacing:0.1em;
  text-transform:uppercase;color:rgba(232,234,240,0.35);margin-bottom:8px;}
.kpi-val{font-family:'Space Grotesk',sans-serif;font-weight:700;
  font-size:20px;color:#fff;line-height:1.1;}
.kpi-sub{font-size:11px;color:#FFCB00;margin-top:4px;}

/* ── Widgets Streamlit ── */
div[data-baseweb="select"]>div{background:#0e1118!important;
  border-color:rgba(255,255,255,0.1)!important;color:#e8eaf0!important;border-radius:8px!important;}
[data-testid="stDataFrame"]{background:transparent!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(255,203,0,0.2);border-radius:10px;}
[data-testid="stHorizontalBlock"]{gap:0!important;}
</style>""", unsafe_allow_html=True)

# ── FLAGS ─────────────────────────────────────────────────────────────────────
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

GC = {"A":"#FFCB00","B":"#64b4ff","C":"#00e676","D":"#ff9800","E":"#c678dd",
      "F":"#ff5252","G":"#00bcd4","H":"#64b4ff","I":"#e06c75","J":"#FFCB00",
      "K":"#64b4ff","L":"#00e676"}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "selected_match"    not in st.session_state: st.session_state.selected_match = None
if "sim_results"       not in st.session_state: st.session_state.sim_results = None
if "page"              not in st.session_state: st.session_state.page = "bracket"
if "fixtures"          not in st.session_state: st.session_state.fixtures = None
if "last_refresh"      not in st.session_state: st.session_state.last_refresh = None

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=7200, show_spinner=False)
def load_fixtures(): return fetch_live_fixtures()

@st.cache_data(show_spinner=False)
def load_ratings(): return build_ratings_df()

@st.cache_data(show_spinner=False)
def load_sim_outputs():
    p = lambda f: OUTPUTS_DIR / f
    if not p("team_probabilities.csv").exists(): return None
    tp = pd.read_csv(p("team_probabilities.csv"))
    for col in ["pass_group_stage_pct","reach_round32_pct","reach_round16_pct",
                "reach_quarters_pct","power_score"]:
        if col not in tp.columns: tp[col] = 0.0
    return {
        "tp":     tp,
        "finals": pd.read_csv(p("finals.csv")),
        "path":   pd.read_csv(p("path_to_title.csv")) if p("path_to_title.csv").exists() else pd.DataFrame(),
        "log":    pd.read_csv(p("simulation_log.csv")) if p("simulation_log.csv").exists() else pd.DataFrame(),
    }

fixtures   = load_fixtures()
ratings_df = load_ratings()
sim_data   = load_sim_outputs()

live_matches = get_live_fixtures(fixtures)
played       = [f for f in fixtures if f.get("status") == "FT"]
pending      = [f for f in fixtures if f.get("status") == "NS"]

n_sims = 0
if sim_data and not sim_data["log"].empty:
    try: n_sims = int(sim_data["log"][sim_data["log"]["metric"]=="n_simulations"]["value"].iloc[0])
    except: pass

# ── Auto-refresco diario a las 10:00 UTC ─────────────────────────────────────
if should_refresh_today():
    load_fixtures.clear()
    fixtures = load_fixtures()

# ── TOPBAR ────────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
live_html = ""
if live_matches:
    live_html = f'<span class="tb-live"><span class="tb-dot"></span>{len(live_matches)} EN VIVO</span>'

st.markdown(
    f'<div class="topbar">'
    f'<span class="tb-logo">FIFA WC<span> 2026</span></span>'
    f'{live_html}'
    f'<span class="tb-info">'
    f'{len(played)} partidos · {len(pending)} pendientes · '
    f'{now_utc.strftime("%d %b %Y %H:%M")} UTC'
    f'</span></div>',
    unsafe_allow_html=True,
)

# ── NAVEGACIÓN ────────────────────────────────────────────────────────────────
nav = st.columns([1,1,1,1,1,3])
pages = [("bracket","⚽ Bracket"), ("grupos","🗂 Grupos"),
         ("stats","📊 Stats"),    ("simulador","🎮 Simulador")]
for i,(pid,lbl) in enumerate(pages):
    with nav[i]:
        if st.button(lbl, key=f"nav_{pid}"):
            st.session_state.page = pid
            st.session_state.selected_match = None
            st.rerun()

# highlight activo
ai = {"bracket":1,"grupos":2,"stats":3,"simulador":4}.get(st.session_state.page,1)
st.markdown(
    f'<style>div[data-testid="stHorizontalBlock"] '
    f'div[data-testid="column"]:nth-child({ai}) button{{'
    f'background:rgba(255,203,0,0.1)!important;'
    f'color:#FFCB00!important;border-color:rgba(255,203,0,0.25)!important;}}</style>',
    unsafe_allow_html=True,
)
st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0">', unsafe_allow_html=True)

# ── HELPER: render match card ─────────────────────────────────────────────────
def render_match_card(fix: dict, sim_result: dict = None, key_suffix: str = "") -> bool:
    """
    Renderiza una tarjeta de partido y devuelve True si se hizo click.
    fix: dict con home, away, home_goals, away_goals, status, date
    sim_result: dict opcional con resultado simulado
    """
    status   = fix.get("status", "NS")
    home     = fix.get("home", "?")
    away     = fix.get("away", "?")
    hg       = fix.get("home_goals")
    ag       = fix.get("away_goals")
    date_raw = fix.get("date", "")

    # Formatear fecha
    try:
        dt  = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
        date_str = dt.strftime("%d %b · %H:%M")
    except:
        date_str = date_raw[:10] if date_raw else "Por confirmar"

    is_played  = status == "FT"
    is_live    = status in ("1H","HT","2H","ET","P","LIVE")
    is_pending = status == "NS"
    is_sim     = sim_result is not None

    # Determinar ganador
    h_win = is_played and hg is not None and ag is not None and hg > ag
    a_win = is_played and hg is not None and ag is not None and ag > hg

    card_cls = "played" if is_played else ("live" if is_live else ("simulated" if is_sim else "pending"))
    live_badge = '<span class="mc-live-badge">LIVE</span>' if is_live else ""
    round_short = fix.get("round","").replace("Group Stage - ","").replace("Matchday","MD")

    html = f"""
    <div class="match-card {card_cls}">
      <div class="mc-header">
        <span>{round_short}</span>
        {live_badge}
      </div>
      <div class="mc-body">
        <div class="mc-team">
          <div class="mc-team-info">
            <span class="mc-flag">{fl(home)}</span>
            <span class="mc-name {'winner' if h_win else ''}">{home}</span>
          </div>
          <span class="mc-score {'winner-score' if h_win else ''}">{hg if hg is not None else ''}</span>
        </div>
        <div class="mc-divider"></div>
        <div class="mc-team">
          <div class="mc-team-info">
            <span class="mc-flag">{fl(away)}</span>
            <span class="mc-name {'winner' if a_win else ''}">{away}</span>
          </div>
          <span class="mc-score {'winner-score' if a_win else ''}">{ag if ag is not None else ''}</span>
        </div>
    """
    if is_pending and not is_sim:
        html += f'<div class="mc-date">{date_str}</div>'
    if is_sim and sim_result:
        sh = sim_result.get("home_score","?"); sa = sim_result.get("away_score","?")
        html += f'<div class="mc-sim-score">🔮 Predicción: {sh}-{sa}</div>'
    html += "</div></div>"

    st.markdown(html, unsafe_allow_html=True)
    return st.button(f"Ver detalle →", key=f"btn_{fix.get('fixture_id','')}{key_suffix}")

# ── HELPER: panel de detalle de partido ──────────────────────────────────────
def render_match_detail(fix: dict):
    """Panel completo de análisis de un partido."""
    home = fix.get("home","?"); away = fix.get("away","?")
    status = fix.get("status","NS")
    is_played = status == "FT"

    st.markdown(f"""
    <div style="margin-bottom:20px">
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:28px;
                  color:#fff;text-align:center;margin-bottom:6px">
        {fl(home)} {home} <span style="color:rgba(232,234,240,0.25);font-weight:400">vs</span> {fl(away)} {away}
      </div>
      <div style="text-align:center;font-size:11px;color:rgba(232,234,240,0.35);
                  letter-spacing:0.1em;text-transform:uppercase">
        {'Partido jugado · Resultado real' if is_played else 'Predicción Dixon-Coles + Power Score'}
      </div>
    </div>""", unsafe_allow_html=True)

    if is_played:
        # Mostrar resultado real
        hg = fix.get("home_goals",0); ag = fix.get("away_goals",0)
        st.markdown(f"""
        <div class="score-pred-box" style="border-color:rgba(0,230,100,0.25)">
          <div class="score-pred-val">{hg} — {ag}</div>
          <div class="score-pred-sub">Resultado final</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Calcular predicción
        r_home = ratings_df[ratings_df["team"]==home]
        r_away = ratings_df[ratings_df["team"]==away]

        if not r_home.empty and not r_away.empty:
            row_h = r_home.iloc[0]; row_a = r_away.iloc[0]
            pred  = win_draw_loss_probs(row_h, row_a)

            # Marcador más probable
            st.markdown(f"""
            <div class="score-pred-box">
              <div class="score-pred-val">{pred['most_likely_score']}</div>
              <div class="score-pred-sub">Marcador más probable · {pred['most_likely_prob']:.1f}% probabilidad</div>
            </div>""", unsafe_allow_html=True)

            # 1X2
            ph, pd2, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
            main_idx = [ph, pd2, pa].index(max(ph, pd2, pa))
            def pbox(val, lbl, is_main):
                mc = "main" if is_main else ""
                vc = "main" if is_main else ""
                return f'<div class="prob-box {mc}"><div class="prob-val {vc}">{val:.0f}%</div><div class="prob-lbl">{lbl}</div></div>'

            st.markdown(
                f'<div class="probs-row">'
                f'{pbox(ph, f"{home.split()[0]} gana", main_idx==0)}'
                f'{pbox(pd2, "Empate", main_idx==1)}'
                f'{pbox(pa, f"{away.split()[0]} gana", main_idx==2)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # xG bars
            max_xg = max(pred["xg_home"], pred["xg_away"], 0.01)
            wh = pred["xg_home"] / max_xg * 100; wa = pred["xg_away"] / max_xg * 100
            st.markdown(
                f'<div style="background:#0e1118;border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:10px;padding:14px 16px;margin-bottom:14px">'
                f'<div style="font-size:10px;font-weight:600;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:rgba(232,234,240,0.3);margin-bottom:10px">Goles esperados (xG)</div>'
                f'<div class="xg-row"><span class="xg-label">{fl(home)} {home.split()[0]}</span>'
                f'<div class="xg-bar-wrap"><div class="xg-bar" style="width:{wh:.0f}%;background:#FFCB00"></div></div>'
                f'<span class="xg-val">{pred["xg_home"]:.2f}</span></div>'
                f'<div class="xg-row"><span class="xg-label">{fl(away)} {away.split()[0]}</span>'
                f'<div class="xg-bar-wrap"><div class="xg-bar" style="width:{wa:.0f}%;background:#64b4ff"></div></div>'
                f'<span class="xg-val">{pred["xg_away"]:.2f}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Distribución de marcadores (gráfico)
            dist = match_distribution(row_h, row_a).head(8)
            fig, ax = plt.subplots(figsize=(8, 2.8))
            fig.patch.set_facecolor("#07090f"); ax.set_facecolor("#0e1118")
            labels = [f"{int(r['home_goals'])}-{int(r['away_goals'])}" for _, r in dist.iterrows()]
            vals   = [r["probability"]*100 for _, r in dist.iterrows()]
            colors = ["#FFCB00" if i==0 else ("#FF9F00" if i<3 else "#1e2535") for i in range(len(labels))]
            ax.bar(labels, vals, color=colors, edgecolor="none", width=0.65)
            for i,(lbl,val) in enumerate(zip(labels,vals)):
                ax.text(i, val+0.15, f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, color="#e8eaf0")
            ax.spines[:].set_visible(False); ax.tick_params(colors="#e8eaf0", labelsize=9)
            ax.grid(axis="y", alpha=0.05, color="#ffffff"); ax.set_ylim(0, max(vals)*1.35)
            fig.tight_layout(pad=0.5); st.pyplot(fig); plt.close(fig)

    # ── Jugadores destacados ──────────────────────────────────────────────────
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    col_h, col_a = st.columns(2, gap="medium")

    for col, team in [(col_h, home), (col_a, away)]:
        with col:
            players = get_team_players(team)
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:rgba(232,234,240,0.35);margin-bottom:10px">'
                f'{fl(team)} Jugadores destacados</div>',
                unsafe_allow_html=True,
            )
            for p in players:
                name = p.get("name","?"); pos = p.get("position","?")
                rtg  = p.get("rating","78"); pace = p.get("pace","75")
                pas  = p.get("passing","75")
                st.markdown(
                    f'<div class="player-card">'
                    f'<div class="player-rating">{rtg}</div>'
                    f'<div><div class="player-name">{name}</div>'
                    f'<div class="player-pos">{pos}</div>'
                    f'<div class="player-stats">'
                    f'<div class="pstat"><div class="pstat-v">{pace}</div><div class="pstat-l">Ritmo</div></div>'
                    f'<div class="pstat"><div class="pstat-v">{pas}</div><div class="pstat-l">Pase</div></div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

    # ── Análisis de ambos equipos ─────────────────────────────────────────────
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    col_ah, col_aa = st.columns(2, gap="medium")
    for col, team in [(col_ah, home), (col_aa, away)]:
        with col:
            analysis = get_team_analysis(team)
            row_t = ratings_df[ratings_df["team"]==team]
            rtg = f"{float(row_t['overall_rating'].iloc[0]):.0f}" if not row_t.empty else "—"
            ps  = f"{float(row_t['power_score'].iloc[0]):.0f}"   if not row_t.empty else "—"
            st.markdown(
                f'<div class="analysis-card">'
                f'<div class="analysis-title">{fl(team)} {team} · Rating {rtg} · PS {ps}</div>'
                f'<div class="analysis-text">{analysis}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Botón cerrar
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    if st.button("✕ Cerrar detalle", key="close_detail"):
        st.session_state.selected_match = None
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA: BRACKET
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "bracket":

    main_col, side_col = st.columns([3, 1], gap="medium")

    with side_col:
        # Botón simular mundial
        st.markdown('<div style="padding:16px 0 8px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="btn-sim">', unsafe_allow_html=True)
        sim_clicked = st.button("🔮 Simular el Mundial · 10K", key="btn_sim_world")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="btn-refresh">', unsafe_allow_html=True)
        refresh_clicked = st.button("↻ Actualizar datos", key="btn_refresh")
        st.markdown('</div>', unsafe_allow_html=True)

        if n_sims > 0:
            st.markdown(
                f'<div style="background:#0e1118;border:1px solid rgba(100,180,255,0.15);'
                f'border-radius:10px;padding:12px 14px;margin-top:12px">'
                f'<div style="font-size:10px;font-weight:600;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:rgba(100,180,255,0.5);margin-bottom:8px">Última simulación</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-weight:700;'
                f'font-size:20px;color:#64b4ff">{n_sims:,}</div>'
                f'<div style="font-size:11px;color:rgba(232,234,240,0.3);margin-top:2px">simulaciones Monte Carlo</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if sim_data:
            tp_side = sim_data["tp"].head(5)
            st.markdown(
                '<div style="height:14px"></div>'
                '<div class="sec-h">Top favoritos</div>',
                unsafe_allow_html=True,
            )
            for i, (_, r) in enumerate(tp_side.iterrows()):
                bar_w = r["champion_pct"] / tp_side["champion_pct"].max() * 100
                rank_c = "#FFCB00" if i==0 else ("#64b4ff" if i<3 else "rgba(232,234,240,0.4)")
                st.markdown(
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">'
                    f'<span style="font-size:13px;color:#e8eaf0">{fl(r["team"])} {r["team"]}</span>'
                    f'<span style="font-family:Space Grotesk,sans-serif;font-weight:700;'
                    f'font-size:13px;color:{rank_c}">{r["champion_pct"]:.1f}%</span></div>'
                    f'<div style="background:rgba(255,255,255,0.06);height:3px;border-radius:999px;overflow:hidden">'
                    f'<div style="width:{bar_w:.0f}%;height:100%;background:{rank_c};border-radius:999px"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Leyenda
        st.markdown(
            '<div style="height:14px"></div>'
            '<div style="background:#0e1118;border:1px solid rgba(255,255,255,0.07);'
            'border-radius:10px;padding:12px 14px">'
            '<div class="sec-h">Leyenda</div>'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:rgba(232,234,240,0.5)">'
            '<div style="width:3px;height:14px;background:rgba(0,230,100,0.5);border-radius:2px"></div>Partido jugado</div>'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:rgba(232,234,240,0.5)">'
            '<div style="width:3px;height:14px;background:rgba(255,82,82,0.5);border-radius:2px"></div>En vivo</div>'
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:rgba(232,234,240,0.5)">'
            '<div style="width:3px;height:14px;background:rgba(255,203,0,0.3);border-radius:2px"></div>Pendiente</div>'
            '<div style="display:flex;align-items:center;gap:8px;font-size:12px;color:rgba(232,234,240,0.5)">'
            '<div style="width:3px;height:14px;background:rgba(100,180,255,0.5);border-radius:2px"></div>Predicción simulada</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Simular si se pulsó el botón ─────────────────────────────────────────
    if sim_clicked:
        with st.spinner("🔮 Ejecutando 10.000 simulaciones Monte Carlo…"):
            from src.monte_carlo import run_monte_carlo
            res = run_monte_carlo(n_simulations=10000, seed=42, verbose=False)
            st.session_state.sim_results = res
        load_sim_outputs.clear()
        sim_data = load_sim_outputs()
        st.success("✅ 10.000 simulaciones completadas"); st.rerun()

    if refresh_clicked:
        load_fixtures.clear()
        fixtures = load_fixtures()
        st.success("✅ Datos actualizados"); st.rerun()

    with main_col:
        # Si hay un partido seleccionado, mostrar detalle
        if st.session_state.selected_match is not None:
            st.markdown('<div style="padding-top:16px"></div>', unsafe_allow_html=True)
            render_match_detail(st.session_state.selected_match)
        else:
            # Bracket: fase de grupos
            st.markdown('<div style="padding-top:16px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-h">⚽ Fase de grupos — Partidos</div>', unsafe_allow_html=True)

            # Agrupar partidos por grupo
            groups_order = list("ABCDEFGHIJKL")
            fix_by_group = {}
            for f in fixtures:
                g = f.get("group","?")
                fix_by_group.setdefault(g, []).append(f)

            # Simular resultados pendientes si hay sim_data
            sim_scores = {}
            if sim_data and "path" in sim_data:
                pass  # se usan en el render

            # Mostrar grupos en grid de 3 columnas
            for row_i in range(0, len(groups_order), 3):
                grp_cols = st.columns(3, gap="small")
                for ci, grp in enumerate(groups_order[row_i:row_i+3]):
                    with grp_cols[ci]:
                        color = GC.get(grp, "#FFCB00")
                        st.markdown(
                            f'<div style="font-family:Space Grotesk,sans-serif;font-weight:700;'
                            f'font-size:12px;letter-spacing:0.1em;text-transform:uppercase;'
                            f'color:{color};margin-bottom:8px;padding:6px 10px;'
                            f'background:rgba(255,255,255,0.03);border-radius:6px">Grupo {grp}</div>',
                            unsafe_allow_html=True,
                        )
                        for fix in fix_by_group.get(grp, []):
                            clicked = render_match_card(fix, key_suffix=f"_{grp}_{fix.get('fixture_id','')}")
                            if clicked:
                                st.session_state.selected_match = fix
                                st.rerun()

            # Fase eliminatoria simulada (si existe)
            if sim_data and not sim_data["path"].empty:
                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-h">🏆 Eliminatorias — Bracket simulado</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div style="background:#0e1118;border:1px solid rgba(100,180,255,0.15);'
                    'border-radius:10px;padding:12px 16px;font-size:12px;'
                    'color:rgba(100,180,255,0.7);margin-bottom:14px">'
                    '🔮 Basado en 10.000 simulaciones Monte Carlo. '
                    'Los marcadores muestran el resultado más probable según el modelo Dixon-Coles.'
                    '</div>',
                    unsafe_allow_html=True,
                )

                pth = sim_data["path"]
                tp_full = sim_data["tp"]
                champ = tp_full.iloc[0]["team"]

                # Mostrar top 8 caminos más probables en grid
                rounds_display = ["Dieciseisavos","Octavos","Cuartos","Semifinal"]
                r_cols = st.columns(len(rounds_display)+1, gap="small")

                for ci, rnd in enumerate(rounds_display):
                    with r_cols[ci]:
                        st.markdown(f'<div class="round-label">{rnd}</div>', unsafe_allow_html=True)
                        shown = set()
                        for _, pr in pth.head(8).iterrows():
                            team = pr["team"]
                            rival = str(pr.get(f"{rnd}_rival","TBD"))
                            freq  = float(pr.get(f"{rnd}_freq_pct",0))
                            if team in shown: continue
                            shown.add(team); shown.add(rival)
                            fix_sim = {
                                "fixture_id": f"sim_{rnd}_{team}",
                                "home": team, "away": rival,
                                "home_goals": None, "away_goals": None,
                                "status": "NS", "round": rnd,
                                "date": "",
                            }
                            sim_res = {"home_score":"?","away_score":"?"}
                            clicked = render_match_card(fix_sim, sim_res, key_suffix=f"_sim_{rnd}_{team}")
                            if clicked:
                                st.session_state.selected_match = fix_sim
                                st.rerun()

                with r_cols[-1]:
                    st.markdown('<div class="round-label">🏆 Final</div>', unsafe_allow_html=True)
                    fin_row = sim_data["finals"].iloc[0] if not sim_data["finals"].empty else None
                    if fin_row is not None:
                        parts = str(fin_row["final"]).split(" vs ")
                        t1 = parts[0].strip(); t2 = parts[1].strip() if len(parts)>1 else "?"
                        champ_pct = float(tp_full[tp_full["team"]==t1]["champion_pct"].values[0]) if t1 in tp_full["team"].values else 0
                        st.markdown(
                            f'<div style="background:rgba(255,203,0,0.08);border:1px solid rgba(255,203,0,0.3);'
                            f'border-radius:10px;padding:16px;text-align:center">'
                            f'<div style="font-size:24px;margin-bottom:8px">🏆</div>'
                            f'<div style="font-family:Space Grotesk,sans-serif;font-weight:700;'
                            f'font-size:15px;color:#FFCB00;margin-bottom:4px">{fl(t1)} {t1}</div>'
                            f'<div style="font-size:11px;color:rgba(232,234,240,0.4)">vs {fl(t2)} {t2}</div>'
                            f'<div style="font-family:Space Grotesk,sans-serif;font-weight:700;'
                            f'font-size:13px;color:#FFCB00;margin-top:8px">{champ_pct:.1f}% campeón</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA: GRUPOS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "grupos":
    st.markdown('<div style="padding:20px 24px 0"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Space Grotesk,sans-serif;font-weight:800;'
        'font-size:26px;color:#fff;margin-bottom:4px">Fase de Grupos</div>'
        '<div style="font-size:13px;color:rgba(232,234,240,0.4);margin-bottom:20px">'
        'Clasificación actualizada · Los 2 primeros + 8 mejores terceros pasan a eliminatorias</div>',
        unsafe_allow_html=True,
    )

    # Calcular standings básicos desde fixtures jugados
    standings = {}
    for f in fixtures:
        if f.get("status") != "FT": continue
        grp = f.get("group","?"); h = f.get("home","?"); a = f.get("away","?")
        hg = f.get("home_goals",0) or 0; ag = f.get("away_goals",0) or 0
        for team,gf,gc in [(h,hg,ag),(a,ag,hg)]:
            if team not in standings: standings[team]={"pts":0,"gd":0,"gf":0,"played":0,"group":grp}
            standings[team]["played"]+=1; standings[team]["gf"]+=gf; standings[team]["gd"]+=(gf-gc)
            if gf>gc: standings[team]["pts"]+=3
            elif gf==gc: standings[team]["pts"]+=1

    for row_i in range(0,12,3):
        g3 = st.columns(3, gap="small")
        for ci, grp in enumerate(list("ABCDEFGHIJKL")[row_i:row_i+3]):
            with g3[ci]:
                color = GC.get(grp,"#FFCB00")
                teams_in_group = GROUPS.get(grp,[])
                grp_html = (
                    f'<div class="grp-card">'
                    f'<div class="grp-header" style="border-left:3px solid {color};padding-left:12px">'
                    f'Grupo {grp}</div>'
                )
                # Ordenar equipos del grupo
                def sort_key(t):
                    s = standings.get(t,{"pts":0,"gd":0,"gf":0})
                    return (-s["pts"],-s["gd"],-s["gf"])
                sorted_teams = sorted(teams_in_group, key=sort_key)
                for ri,t in enumerate(sorted_teams):
                    s = standings.get(t,{"pts":0,"gd":0,"gf":0,"played":0})
                    qcls = "q1" if ri==0 else ("q2" if ri==1 else "")
                    sim_pct = ""
                    if sim_data:
                        r = sim_data["tp"][sim_data["tp"]["team"]==t]
                        if not r.empty:
                            sim_pct = f'<span style="font-size:10px;color:rgba(255,203,0,0.5)">{r.iloc[0]["champion_pct"]:.1f}%🏆</span>'
                    grp_html+=(
                        f'<div class="grp-row {qcls}">'
                        f'<span class="grp-pos">{ri+1}</span>'
                        f'<span class="grp-flag">{fl(t)}</span>'
                        f'<span class="grp-name">{t}</span>'
                        f'{sim_pct}'
                        f'<span class="grp-sub" style="color:rgba(232,234,240,0.4)">{s["gd"]:+d}</span>'
                        f'<span class="grp-pts">{s["pts"]}</span>'
                        f'</div>'
                    )
                grp_html+='</div>'
                st.markdown(grp_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA: STATS
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "stats":
    st.markdown('<div style="padding:20px 24px 0"></div>', unsafe_allow_html=True)
    if not sim_data:
        st.info("Pulsa 'Simular el Mundial' en el bracket para ver las estadísticas completas.")
    else:
        tp = sim_data["tp"]
        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-weight:800;'
            'font-size:26px;color:#fff;margin-bottom:20px">Análisis de Simulación</div>',
            unsafe_allow_html=True,
        )
        # KPIs
        top = tp.iloc[0]; fin1=tp.sort_values("reach_final_pct",ascending=False).iloc[0]
        kc = st.columns(4, gap="small")
        for col, lbl, val, sub in [
            (kc[0],"Gran Favorito", f"{fl(top['team'])} {top['team']}",f"{top['champion_pct']:.1f}%"),
            (kc[1],"Más Finalista",  f"{fl(fin1['team'])} {fin1['team']}",f"{fin1['reach_final_pct']:.1f}%"),
            (kc[2],"Simulaciones",   f"{n_sims:,}","Monte Carlo"),
            (kc[3],"Modelo",         "Dixon-Coles","+ Power Score"),
        ]:
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-val">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        # Tabla completa
        disp = tp[["team","group","overall_rating","power_score","champion_pct",
                   "reach_final_pct","reach_semis_pct","reach_quarters_pct",
                   "pass_group_stage_pct","group_exit_pct"]].copy()
        disp.columns=["Selección","Grupo","Rating","PS","🏆%","Final%","Semis%","Cuartos%","Pasa%","Elim%"]
        disp=disp.reset_index(drop=True); disp.index+=1
        st.dataframe(
            disp.style.background_gradient(subset=["🏆%"],cmap="YlOrRd")
                      .background_gradient(subset=["Elim%"],cmap="Reds_r")
                      .format({c:"{:.1f}" for c in disp.columns if "%" in c or c in ["Rating","PS"]}),
            use_container_width=True, height=500,
        )
        st.download_button("⬇ Descargar CSV", tp.to_csv(index=False).encode(),
                           "wc2026_stats.csv","text/csv")


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA: SIMULADOR DE PARTIDO
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "simulador":
    st.markdown('<div style="padding:20px 24px 0"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Space Grotesk,sans-serif;font-weight:800;'
        'font-size:26px;color:#fff;margin-bottom:4px">Simulador de Partido</div>'
        '<div style="font-size:13px;color:rgba(232,234,240,0.4);margin-bottom:20px">'
        'Predicción Dixon-Coles + Power Score en tiempo real</div>',
        unsafe_allow_html=True,
    )
    all_teams = sorted(ratings_df["team"].tolist())
    sc1,sc2,sc3 = st.columns([2,1,2], gap="medium")
    with sc1:
        ta = st.selectbox("Equipo A",all_teams,
                          index=all_teams.index("Spain") if "Spain" in all_teams else 0,
                          key="sim_a")
    with sc2:
        st.markdown('<div style="text-align:center;padding-top:28px;font-family:Space Grotesk,sans-serif;'
                    'font-weight:800;font-size:18px;color:rgba(232,234,240,0.25)">VS</div>',
                    unsafe_allow_html=True)
    with sc3:
        tb = st.selectbox("Equipo B",all_teams,
                          index=all_teams.index("Argentina") if "Argentina" in all_teams else 1,
                          key="sim_b")

    if ta != tb:
        fix_manual = {"fixture_id":"manual","home":ta,"away":tb,
                      "home_goals":None,"away_goals":None,"status":"NS","round":"Amistoso","date":""}
        render_match_detail(fix_manual)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid rgba(255,255,255,0.05);padding:14px 28px;'
    'display:flex;justify-content:space-between;align-items:center;margin-top:24px">'
    '<span style="font-size:11px;color:rgba(232,234,240,0.2)">'
    '© 2026 FIFA World Cup Prediction Model · Aimar Esqueta · Dixon-Coles + Power Score</span>'
    '<span style="font-size:11px;color:rgba(255,203,0,0.3)">Solo análisis experimental</span>'
    '</div>',
    unsafe_allow_html=True,
)
