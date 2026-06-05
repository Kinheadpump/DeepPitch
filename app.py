import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.api_client import LiveOracleAPI
from src.lineup_scanner import LiveLineupScanner

st.set_page_config(page_title="DeepPitch V5 - Orakel", page_icon="🔮", layout="wide")

# ==========================================
# 1. CORE ENGINE (Gehirn laden)
# ==========================================
@st.cache_resource
def initialize_engine():
    brain_path = "data/deeppitch_brain.pkl"
    if not os.path.exists(brain_path):
        st.error("🧠 Gehirn nicht gefunden! Bitte führe zuerst `python train.py` aus.")
        st.stop()
        
    brain = joblib.load(brain_path)
    try:
        my_football_key = st.secrets["FOOTBALL_DATA_KEY"]
        my_sports_key = st.secrets.get("API_SPORTS_KEY", "DEMO")
    except FileNotFoundError:
        st.warning("⚠️ Secrets nicht gefunden! Nutze Fallback-Modus.")
        my_football_key = "DEMO"
        my_sports_key = "DEMO"
        
    api = LiveOracleAPI(api_key=my_football_key) 
    
    # Der Scanner bekommt nun die Lizenz, ins echte Internet zu gehen!
    scanner = LiveLineupScanner(fifa_csv_path="data/fifa_players_cloud.csv", api_key=my_sports_key)
    
    return brain['backtester'], brain['model'], brain['fifa'], api, scanner

# ==========================================
# 2. FINANCIAL ADVISOR WIDGET (JETZT MIT LIVE-QUOTEN)
# ==========================================
def render_kelly_advisor(bankroll, probs_ml, elo_h, elo_a, team_h, team_a, live_odds=None):
    ai_confidence = max(probs_ml['home_win'], probs_ml['away_win'])
    ai_raw_pred = 2 if probs_ml['home_win'] > probs_ml['away_win'] else 0
    team_name = team_h if ai_raw_pred == 2 else team_a
    
    # Entweder echte Quoten nutzen (falls angegeben) oder simulieren
    if live_odds and live_odds[2] > 1.0 and live_odds[0] > 1.0:
        bookie_odds = live_odds[ai_raw_pred]
        is_live = True
        st.markdown("### ⚡ Live-Market Money Management")
    else:
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
        is_live = False
        st.markdown("### 🛡️ Money Management (Simulierte Quoten)")
    
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
            c1.metric("KI Siegchance", f"{p*100:.1f}%")
            if is_live:
                c2.metric("Echte Buchmacher Quote", f"{bookie_odds:.2f}", "Live Markt-Daten")
            else:
                c2.metric("Simulierte Quote", f"{bookie_odds:.2f}")
            
            st.info(f"💰 **Kelly-Empfehlung:** Setze **{recommended_stake:.2f} €** auf **Sieg {team_name}**.")
        else:
            st.warning(f"⚠️ **Kein Value!** KI sieht zwar {team_name} vorne, aber die Quote ({bookie_odds:.2f}) ist zu schlecht. **Finger weg.**")
    else:
        st.error("❌ **FINGER WEG:** Reiner Münzwurf. Beide Teams sind zu gleichauf.")

    with st.expander("ℹ️ Was bedeuten diese Finanz-Metriken?"):
        st.markdown("""
        * **Der Edge (Value):** Liegt die KI-Chance höher als die vom Buchmacher vermutete Chance, haben wir einen Edge.
        * **Kelly-Empfehlung:** Eine Finanz-Formel, die genau berechnet, wie viel von deinem Kontostand du riskieren darfst (wir nutzen *Half-Kelly* zur Absicherung).
        * **Live vs. Simuliert:** Wenn du echte Quoten in das Labor eingibst, wird die Empfehlung millimetergenau auf den aktuellen Markt angepasst.
        """)

# ==========================================
# 3. MAIN UI ROUTING
# ==========================================
st.title("🏆 DeepPitch V5: Das Live-Orakel")
st.markdown("Willkommen im Cockpit. Nutze die KI, um Ineffizienzen in den Buchmacher-Quoten aufzuspüren.")

with st.spinner("Lade KI-Gedächtnis..."):
    bt, model, fifa_ratings, api, scanner = initialize_engine()

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
        
        # --- NEU: LIVE LINEUP SCANNER ---
        st.markdown("#### 📡 Live-Startelf Scanner")
        st.caption("Prüft die offiziellen Aufstellungen 60 Min vor Anpfiff.")
        use_live_lineup = st.checkbox(f"🚨 Ausfall-Szenario für {team_h} simulieren", help="Überschreibt die historischen FIFA-Ratings mit den Werten der 11 Spieler, die laut API WIRKLICH auf dem Platz stehen.")
        # --------------------------------
        
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
        st.markdown("#### 3. Finanzen & Markt")
        bankroll_manual = st.number_input("Dein aktueller Wett-Kontostand (€)", min_value=10, value=1000, step=100, key="bk_manual")
        
        st.caption("Optional: Echte Buchmacher-Quoten eingeben (0.0 = Simulation nutzen)")
        c_q1, c_qX, c_q2 = st.columns(3)
        real_odds_h = c_q1.number_input("Sieg Team A", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        real_odds_d = c_qX.number_input("Remis (X)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        real_odds_a = c_q2.number_input("Sieg Team B", min_value=0.0, value=0.0, step=0.1, format="%.2f")

    with col_output:
        st.markdown("#### 📊 KI-Analyse-Ergebnis")
        if team_h != team_a:
            stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
            stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
            
            # --- NEU: LIVE LINEUP ÜBERSCHREIBEN ---
            if use_live_lineup:
                with st.spinner("Fuzzy Matching der Live-Aufstellung..."):
                    live_stats_h, match_logs = scanner.get_live_squad_rating(team_h)
                    
                    if live_stats_h:
                        st.warning(f"⚠️ **ACHTUNG: {team_h} spielt mit B-Elf!**\nHistorisches Rating ({stats_h['ATT']:.1f}) wurde durch Live-Aufstellung ({live_stats_h['ATT']:.1f}) überschrieben.")
                        stats_h = live_stats_h # HIER ÜBERSCHREIBEN WIR DIE THEORIE MIT DER REALITÄT!
                        
                        with st.expander("🔍 Fuzzy Matcher Logs ansehen"):
                            for log in match_logs:
                                st.text(log)
                    else:
                        st.error("API hat keine Live-Daten für dieses Team (Demo funktioniert nur für 'Germany').")
            # ---------------------------------------
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
            
            # Live Quoten Dictionary (2=Heim, 1=Remis, 0=Auswärts)
            live_odds_dict = {2: real_odds_h, 1: real_odds_d, 0: real_odds_a}
            
            render_kelly_advisor(bankroll_manual, probs_ml, elo_h, elo_a, team_h, team_a, live_odds=live_odds_dict)