import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.api_client import LiveOracleAPI

st.set_page_config(page_title="DeepPitch V5 - Orakel", page_icon="🔮", layout="wide")

# ==========================================
# 1. CORE ENGINE (Gehirn laden)
# ==========================================
@st.cache_resource
def initialize_engine():
    brain_path = "data/deeppitch_v5_brain.pkl"
    if not os.path.exists(brain_path):
        st.error("🧠 **Gehirn nicht gefunden!** Bitte führe zuerst `python train.py` aus.")
        st.stop()
        
    brain = joblib.load(brain_path)
    # API-Key hier eintragen, wenn du Live-Daten für echte Spiele willst!
    api = LiveOracleAPI(api_key="DEMO") 
    
    return brain['backtester'], brain['model'], brain['fifa'], api

# ==========================================
# 2. FINANCIAL ADVISOR WIDGET
# ==========================================
def render_kelly_advisor(bankroll, probs_ml, elo_h, elo_a, team_h, team_a):
    ai_confidence = max(probs_ml['home_win'], probs_ml['away_win'])
    ai_raw_pred = 2 if probs_ml['home_win'] > probs_ml['away_win'] else 0
    team_name = team_h if ai_raw_pred == 2 else team_a
    
    # Buchmacher-Quote simulieren (Inkl. 5% Marge)
    elo_win_prob = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    elo_loss_prob = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
    sum_p = elo_win_prob + 0.25 + elo_loss_prob
    
    odds_dict = {
        2: 1 / ((elo_win_prob / sum_p) * 1.05),
        1: 1 / ((0.25 / sum_p) * 1.05),
        0: 1 / ((elo_loss_prob / sum_p) * 1.05)
    }
    bookie_odds = odds_dict[ai_raw_pred]
    
    st.markdown("### 🛡️ Money Management")
    
    if ai_confidence >= 0.50:
        p = ai_confidence
        b = bookie_odds - 1
        q = 1 - p
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction > 0:
            stake_pct = min(kelly_fraction * 0.5, 0.10) # Half-Kelly, Max 10%
            recommended_stake = bankroll * stake_pct
            
            st.success(f"🔥 **VALUE GEFUNDEN:** Edge von {(p - 1/bookie_odds)*100:.1f}%")
            
            c1, c2 = st.columns(2)
            c1.metric("KI Siegchance", f"{p*100:.1f}%", help="Die von unserem V5-Modell berechnete ECHTE Wahrscheinlichkeit.")
            c2.metric("Buchmacher Quote", f"{bookie_odds:.2f}", help="Die simulierte Standard-Quote auf dem Markt (inkl. 5% Marge des Buchmachers).")
            
            st.info(f"💰 **Kelly-Empfehlung:** Setze **{recommended_stake:.2f} €** auf **Sieg {team_name}**.")
        else:
            st.warning(f"⚠️ **Kein Value!** KI sieht zwar {team_name} vorne, aber die Buchmacher-Quote ({bookie_odds:.2f}) ist mathematisch zu schlecht. **Keine Wette.**")
    else:
        st.error("❌ **FINGER WEG:** Reiner Münzwurf. Beide Teams sind zu gleichauf.")

    # Der einklappbare Erklär-Bereich (Kein Feature-Creep, da standardmäßig versteckt!)
    with st.expander("ℹ️ Was bedeuten diese Finanz-Metriken?"):
        st.markdown("""
        * **Der Edge (Value):** Buchmacher berechnen Quoten meist stur nach historischen Daten. Unsere KI nutzt aber *aktuelle Taktik- und Kaderwerte*. Liegt die KI-Chance höher als die vom Buchmacher vermutete Chance, haben wir einen "Edge" (einen unfairen Vorteil gegenüber dem System).
        * **Buchmacher Quote:** Zeigt an, wie viel Geld du zurückbekommst. Bei einer Quote von 2.00 würdest du für 10€ Einsatz genau 20€ zurückerhalten (10€ Reingewinn).
        * **Kelly-Empfehlung:** Eine Finanz-Formel, die genau berechnet, wie viel von deinem Kontostand du riskieren darfst. Ein riesiger *Edge* bedeutet einen hohen Einsatz. Wir nutzen **"Half-Kelly"** (die Empfehlung wird halbiert), um dich vor Pechsträhnen zu schützen, während dein Konto trotzdem stetig wächst.
        """)

# ==========================================
# 3. MAIN UI ROUTING
# ==========================================
st.title("🏆 DeepPitch V5: Das Live-Orakel")
st.markdown("Willkommen im Cockpit. Nutze die KI, um Ineffizienzen in den Buchmacher-Quoten aufzuspüren.")

with st.spinner("Lade KI-Gedächtnis..."):
    bt, model, fifa_ratings, api = initialize_engine()

tab1, tab2 = st.tabs(["🔴 LIVE: Kommende Spiele", "🔬 Manuelle Analyse (Labor)"])

# ---------------------------------------------------------
# TAB 1: LIVE-ORAKEL MODUS
# ---------------------------------------------------------
with tab1:
    st.header("📡 Live-Feed: Echte Länderspiele")
    bankroll_live = st.number_input("Dein aktueller Wett-Kontostand (€)", min_value=10, value=1000, step=100, key="bk_live", help="Gib dein Startkapital ein, damit die Kelly-Formel deinen perfekten Wetteinsatz berechnen kann.")
    
    if st.button("🔄 Länderspiele der nächsten 10 Tage abrufen"):
        matches = api.get_upcoming_matches(days_ahead=10)
        
        if not matches:
            st.warning("📭 Keine internationalen Spiele in den nächsten 10 Tagen gefunden (oder API-Limit erreicht).")
        else:
            for match in matches:
                team_h, team_a = match['home_team'], match['away_team']
                
                stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
                stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
                elo_h = bt.elo.ratings.get(team_h, 1500.0)
                elo_a = bt.elo.ratings.get(team_a, 1500.0)
                elo_diff = elo_h - elo_a
                att_diff, mid_diff, def_diff = stats_h['ATT'] - stats_a['ATT'], stats_h['MID'] - stats_a['MID'], stats_h['DEF'] - stats_a['DEF']
                
                probs_p = bt.poisson.predict_match_probabilities(team_h, team_a, True, elo_diff=elo_diff, att_diff=att_diff, def_diff=def_diff)
                probs_ml = model.predict_probabilities(elo_diff, probs_p['home_win'] - probs_p['away_win'], 0.0, 0.0, att_diff, mid_diff, def_diff)
                pred_h, pred_a = bt.poisson.get_smart_score(probs_p['matrix'], probs_ml)
                
                st.markdown("---")
                col_info, col_metrics, col_kelly = st.columns([1, 1.5, 2])
                
                with col_info:
                    st.subheader(f"🏟️ {team_h} vs. {team_a}")
                    st.caption(f"📅 {match['date']} | {match['competition']}")
                    st.info(f"⚽ KI-Tipp: **{pred_h}:{pred_a}**")
                
                with col_metrics:
                    c1, c2, c3 = st.columns(3)
                    c1.metric(team_h, f"{probs_ml['home_win']*100:.1f}%")
                    c2.metric("Remis", f"{probs_ml['draw']*100:.1f}%")
                    c3.metric(team_a, f"{probs_ml['away_win']*100:.1f}%")
                    st.progress(probs_ml['home_win'])
                
                with col_kelly:
                    render_kelly_advisor(bankroll_live, probs_ml, elo_h, elo_a, team_h, team_a)

# ---------------------------------------------------------
# TAB 2: MANUELLE ANALYSE (LABOR)
# ---------------------------------------------------------
with tab2:
    st.header("🔬 Taktisches Test-Labor")
    st.markdown("Simuliere jedes beliebige Spiel. Ändere die Parameter, um zu sehen, wie die KI darauf reagiert.")
    
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### 1. Die Teams")
        all_teams = sorted(list(fifa_ratings.keys()))
        team_h = st.selectbox("Heimteam (Team A)", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 0)
        team_a = st.selectbox("Auswärtsteam (Team B)", all_teams, index=all_teams.index("Spain") if "Spain" in all_teams else 1)
        
        st.divider()
        st.markdown("#### 2. Rahmenbedingungen")
        is_neutral = st.checkbox(
            "Neutraler Austragungsort", 
            value=True, 
            help="Setze ein Häkchen, wenn das Spiel auf neutralem Boden stattfindet (z.B. bei einer WM-Endrunde in einem Gastland). Dies entfernt den statistischen 'Heimvorteil' der Fans und des Stadions."
        )
        
        wm_continent = st.selectbox(
            "Auf welchem Kontinent wird gespielt?", 
            ['Europe', 'South America', 'Asia', 'Africa', 'North America'],
            help="Teams, die in ihrer eigenen Zeitzone und auf ihrem Heimatkontinent spielen, erhalten historisch gesehen einen winzigen Leistungs-Boost."
        )
        
        form_diff = st.slider(
            "Form-Vorteil der letzten 5 Spiele", 
            -1.0, 1.0, 0.0, step=0.1,
            help="Hat ein Team gerade einen perfekten Lauf? Positiv (> 0) = Heimteam ist in Topform. Negativ (< 0) = Auswärtsteam dominiert."
        )
        
        st.divider()
        st.markdown("#### 3. Finanzen")
        bankroll_manual = st.number_input("Dein aktueller Wett-Kontostand (€)", min_value=10, value=1000, step=100, key="bk_manual", help="Dein virtuelles oder echtes Startkapital für die Kelly-Berechnung.")

    with col_output:
        st.markdown("#### 📊 KI-Analyse-Ergebnis")
        if team_h != team_a:
            stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
            stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
            elo_h = bt.elo.ratings.get(team_h, 1500.0)
            elo_a = bt.elo.ratings.get(team_a, 1500.0)
            elo_diff = elo_h - elo_a
            continent_diff = (1 if bt._get_continent(team_h) == wm_continent else 0) - (1 if bt._get_continent(team_a) == wm_continent else 0)
            
            att_diff, mid_diff, def_diff = stats_h['ATT'] - stats_a['ATT'], stats_h['MID'] - stats_a['MID'], stats_h['DEF'] - stats_a['DEF']
            
            probs_p = bt.poisson.predict_match_probabilities(team_h, team_a, is_neutral, elo_diff=elo_diff, att_diff=att_diff, def_diff=def_diff)
            probs_ml = model.predict_probabilities(elo_diff, probs_p['home_win'] - probs_p['away_win'], form_diff, continent_diff, att_diff, mid_diff, def_diff)
            pred_h, pred_a = bt.poisson.get_smart_score(probs_p['matrix'], probs_ml)
            
            # Schicke Visualisierung
            c1, c2, c3 = st.columns(3)
            c1.metric(team_h, f"{probs_ml['home_win']*100:.1f}%")
            c2.metric("Remis", f"{probs_ml['draw']*100:.1f}%")
            c3.metric(team_a, f"{probs_ml['away_win']*100:.1f}%")
            
            st.info(f"⚽ Exakter KI-Ergebnis Tipp: **{pred_h} : {pred_a}**")
            st.divider()
            
            render_kelly_advisor(bankroll_manual, probs_ml, elo_h, elo_a, team_h, team_a)