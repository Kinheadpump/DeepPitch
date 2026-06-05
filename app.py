import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.api_client import LiveOracleAPI
from src.lineup_scanner import LiveLineupScanner
import difflib

def get_real_team_name(api_name, known_teams):
    """Übersetzt API-Teamnamen unscharf in die korrekten Datenbank-Schlüssel"""
    # 1. Harte manuelle Zuordnungen für bekannte API-Sonderfälle
    manual_map = {
        "USA": "United States",
        "Korea Republic": "South Korea",
        "Cote d'Ivoire": "Ivory Coast",
        "Bosnia and Herzegovina": "Bosnia",
        "IR Iran": "Iran",
        "North Macedonia": "Macedonia"
    }
    if api_name in manual_map:
        return manual_map[api_name]
        
    # 2. Unscharfe Suche (Fuzzy Matching) für leichte Schreibfehler
    matches = difflib.get_close_matches(api_name, known_teams, n=1, cutoff=0.55)
    return matches[0] if matches else api_name

# 1. PAGE CONFIGURATION (Clean & Professional)
st.set_page_config(page_title="DeepPitch Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS um Standard-Streamlit Margins leicht zu reduzieren (für einen Dashboard-Look)
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GLOBALES GLOSSAR (Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("### Methodik & Glossar")
    st.markdown("""
    **Expected Value (Edge):** Die positive mathematische Differenz zwischen der vom DeepPitch-Modell berechneten Wahrscheinlichkeit und der vom Markt (Buchmacher) implizierten Wahrscheinlichkeit.
    
    **Target Allocation:** Eine konservativ skalierte Risiko-Gewichtung (**Quarter-Kelly Criterion**) basierend auf dem detektierten Edge. Das System schützt das Kapital durch ein hartes Cap von maximal **3% der Bankroll** pro Trade.
    """)

# ==========================================
# 2. CORE ENGINE INITIALIZATION
# ==========================================
@st.cache_resource(show_spinner="Initialisiere Backend-Modelle...")
def initialize_engine():
    brain_path = "data/deeppitch_brain.pkl"
    if not os.path.exists(brain_path):
        st.error("Systemfehler: Modelldaten ('deeppitch_brain.pkl') nicht gefunden.")
        st.stop()
        
    brain = joblib.load(brain_path)
    
    try:
        my_football_key = st.secrets["FOOTBALL_DATA_KEY"]
        my_sports_key = st.secrets.get("API_SPORTS_KEY", "DEMO")
    except FileNotFoundError:
        st.warning("Systemwarnung: Secrets nicht gefunden. Fallback-Modus aktiv.")
        my_football_key = "DEMO"
        my_sports_key = "DEMO"
        
    api = LiveOracleAPI(api_key=my_football_key) 
    scanner = LiveLineupScanner(fifa_csv_path="data/fifa_players_cloud.csv", api_key=my_sports_key)
    
    return brain['backtester'], brain['model'], brain['fifa'], api, scanner

# ==========================================
# 3. FINANCIAL ADVISOR WIDGET (Minimalist)
# ==========================================
def render_kelly_advisor(bankroll, probs_ml, elo_h, elo_a, team_h, team_a, live_odds=None):
    ai_confidence = max(probs_ml['home_win'], probs_ml['away_win'])
    ai_raw_pred = 2 if probs_ml['home_win'] > probs_ml['away_win'] else 0
    team_name = team_h if ai_raw_pred == 2 else team_a
    
    is_live = False
    if live_odds and live_odds[2] > 1.0 and live_odds[0] > 1.0:
        bookie_odds = live_odds[ai_raw_pred]
        is_live = True
    else:
        elo_win_prob = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        elo_loss_prob = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
        sum_p = elo_win_prob + 0.25 + elo_loss_prob
        
        odds_dict = {
            2: 1 / ((elo_win_prob / sum_p) * 1.05),
            1: 1 / ((0.25 / sum_p) * 1.05),
            0: 1 / ((elo_loss_prob / sum_p) * 1.05)
        }
        bookie_odds = odds_dict[ai_raw_pred]

    st.markdown("##### Capital Allocation (Risk-Adjusted)")
    
    if ai_confidence >= 0.50:
        p = ai_confidence
        b = bookie_odds - 1
        q = 1 - p
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction > 0:
            # GEFIXTE MATHEMATIK: Quarter-Kelly (0.25) und 3% Max Cap
            stake_pct = min(kelly_fraction * 0.25, 0.03)
            recommended_stake = bankroll * stake_pct
            edge = (p - 1/bookie_odds) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Model Implied Prob.", f"{p*100:.1f}%")
            c2.metric("Market Odds" if is_live else "Simulated Odds", f"{bookie_odds:.2f}")
            c3.metric("Expected Value (Edge)", f"+{edge:.1f}%")
            
            st.success(f"Signal: BUY {team_name.upper()} | Target Allocation: {recommended_stake:.2f} € ({stake_pct*100:.1f}%)")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Model Implied Prob.", f"{p*100:.1f}%")
            c2.metric("Market Odds" if is_live else "Simulated Odds", f"{bookie_odds:.2f}")
            st.warning("Signal: NEUTRAL | Grund: Negativer Expected Value.")
    else:
        st.error("Signal: NEUTRAL | Grund: Konfidenz-Level unterhalb des Schwellenwerts (< 50%).")

# ==========================================
# 4. MAIN UI ROUTING
# ==========================================
st.title("DeepPitch Analytics")
st.markdown("Quantitative Match Analysis & Market Pricing Engine")

bt, model, fifa_ratings, api, scanner = initialize_engine()

tab1, tab2 = st.tabs(["Live Markets", "Quantitative Sandbox"])

# ---------------------------------------------------------
# TAB 1: LIVE-ORAKEL MODUS
# ---------------------------------------------------------
with tab1:
    col_header, col_action = st.columns([3, 1])
    with col_header:
        st.markdown("#### Upcoming Market Events")
        st.caption("Scannt API-Endpoints nach regulären Länderspielen der nächsten 10 Tage.")
    with col_action:
        bankroll_live = st.number_input("Portfolio Size (€)", min_value=10, value=1000, step=100, key="bk_live")
        fetch_button = st.button("Marktdaten abrufen", use_container_width=True)

    st.divider()

    if fetch_button:
        with st.spinner("Synchronisiere mit Provider..."):
            matches = api.get_upcoming_matches(days_ahead=10)
            
            if not matches:
                st.info("Keine handelbaren Events im definierten Zeitfenster gefunden.")
            else:
                db_teams = list(fifa_ratings.keys()) # Alle bekannten Datenbank-Teams laden
                
                for match in matches:
                    # --- FIX: Namen vor der Bewertung übersetzen! ---
                    raw_h, raw_a = match['home_team'], match['away_team']
                    team_h = get_real_team_name(raw_h, db_teams)
                    team_a = get_real_team_name(raw_a, db_teams)
                    # -----------------------------------------------
                    
                    stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
                    stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
                    elo_h = bt.elo.ratings.get(team_h, 1500.0)
                    elo_a = bt.elo.ratings.get(team_a, 1500.0)
                    elo_diff = elo_h - elo_a
                    att_diff, mid_diff, def_diff = stats_h['ATT'] - stats_a['ATT'], stats_h['MID'] - stats_a['MID'], stats_h['DEF'] - stats_a['DEF']
                    
                    probs_p = bt.poisson.predict_match_probabilities(team_h, team_a, True, elo_diff=elo_diff, att_diff=att_diff, def_diff=def_diff)
                    probs_ml = model.predict_probabilities(elo_diff, probs_p['home_win'] - probs_p['away_win'], 0.0, att_diff, mid_diff, def_diff)
                    pred_h, pred_a = bt.poisson.get_smart_score(probs_p['matrix'], probs_ml)
                    
                    with st.container(border=True):
                        c_match, c_metrics, c_kelly = st.columns([1, 1.5, 2])
                        
                        with c_match:
                            st.markdown(f"**{team_h} vs. {team_a}**")
                            st.caption(f"{match['date']} | {match['competition']}")
                            st.markdown(f"Projiziertes Modal-Ergebnis: **{pred_h} - {pred_a}**")
                        
                        with c_metrics:
                            cm1, cm2, cm3 = st.columns(3)
                            cm1.metric(team_h, f"{probs_ml['home_win']*100:.1f}%")
                            cm2.metric("Draw", f"{probs_ml['draw']*100:.1f}%")
                            cm3.metric(team_a, f"{probs_ml['away_win']*100:.1f}%")
                        
                        with c_kelly:
                            render_kelly_advisor(bankroll_live, probs_ml, elo_h, elo_a, team_h, team_a)

# ---------------------------------------------------------
# TAB 2: MANUELLE ANALYSE (LABOR)
# ---------------------------------------------------------
with tab2:
    col_input, col_output = st.columns([1.2, 1], gap="large")

    with col_input:
        st.markdown("#### Scenario Parameters")
        
        with st.container(border=True):
            st.markdown("**Entities**")
            all_teams = sorted(list(fifa_ratings.keys()))
            team_h = st.selectbox("Heim-Entity (Team A)", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 0)
            team_a = st.selectbox("Auswärts-Entity (Team B)", all_teams, index=all_teams.index("Spain") if "Spain" in all_teams else 1)
        
        with st.container(border=True):
            st.markdown("**Real-Time Data Override**")
            use_live_lineup = st.checkbox(f"Live Lineup-Sync (API) für {team_h}", help="Überschreibt Baseline-Ratings mit verifizierter Startelf (T-60 Min).")
        
        with st.container(border=True):
            st.markdown("**Market Input (Optional)**")
            bankroll_manual = st.number_input("Portfolio Size (€)", min_value=10, value=1000, step=100, key="bk_manual")
            st.caption("Quoten-Override (0.0 = Baseline-Simulation)")
            c_q1, c_qX, c_q2 = st.columns(3)
            real_odds_h = c_q1.number_input("Quote Heim", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            real_odds_d = c_qX.number_input("Quote Draw", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            real_odds_a = c_q2.number_input("Quote Auswärts", min_value=0.0, value=0.0, step=0.1, format="%.2f")

        with st.container(border=True):
            st.markdown("**Environment Constraints**")
            is_neutral = st.checkbox("Neutraler Austragungsort", value=True)
            form_diff = st.slider("Form-Momentum Delta", -1.0, 1.0, 0.0, step=0.1)

    with col_output:
        st.markdown("#### Analytical Output")
        
        if team_h != team_a:
            stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
            stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
            elo_h = bt.elo.ratings.get(team_h, 1500.0)
            elo_a = bt.elo.ratings.get(team_a, 1500.0)
            elo_diff = elo_h - elo_a
            
            # --- LIVE LINEUP OVERRIDE ---
            if use_live_lineup:
                with st.spinner("Abgleich der Entity-Datenbank (Fuzzy Matching)..."):
                    live_stats_h, match_logs = scanner.get_live_squad_rating(team_h)
                    
                    if live_stats_h:
                        st.info(f"Lineup-Sync erfolgreich. Baseline-Rating ({stats_h['ATT']:.1f}) wurde durch Live-Daten ({live_stats_h['ATT']:.1f}) ersetzt.")
                        stats_h = live_stats_h 
                        
                        with st.expander("System-Logs (Entity Resolution)"):
                            for log in match_logs:
                                st.text(log)
                    else:
                        st.error(match_logs[0])
            # ---------------------------------------
            
            att_diff, mid_diff, def_diff = stats_h['ATT'] - stats_a['ATT'], stats_h['MID'] - stats_a['MID'], stats_h['DEF'] - stats_a['DEF']
            
            probs_p = bt.poisson.predict_match_probabilities(team_h, team_a, is_neutral, elo_diff=elo_diff, att_diff=att_diff, def_diff=def_diff)
            probs_ml = model.predict_probabilities(elo_diff, probs_p['home_win'] - probs_p['away_win'], form_diff, att_diff, mid_diff, def_diff)
            pred_h, pred_a = bt.poisson.get_smart_score(probs_p['matrix'], probs_ml)
            
            with st.container(border=True):
                st.markdown("##### Prediction Vector")
                c1, c2, c3 = st.columns(3)
                c1.metric(team_h, f"{probs_ml['home_win']*100:.1f}%")
                c2.metric("Draw", f"{probs_ml['draw']*100:.1f}%")
                c3.metric(team_a, f"{probs_ml['away_win']*100:.1f}%")
                
                st.markdown(f"Projiziertes Modal-Ergebnis: **{pred_h} - {pred_a}**")
            
            with st.container(border=True):
                live_odds_dict = {2: real_odds_h, 1: real_odds_d, 0: real_odds_a}
                render_kelly_advisor(bankroll_manual, probs_ml, elo_h, elo_a, team_h, team_a, live_odds=live_odds_dict)