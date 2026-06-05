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
    **Model Implied Prob. (Implizite Modell-Wahrscheinlichkeit):** Die rohe, ungeschönte Wahrscheinlichkeit, die die DeepPitch-KI für den Spielausgang berechnet. Sie basiert auf dem Ensemble aus Tor-Verteilungen (Poisson), Kadertiefe (FIFA-Daten), Form-Momentum und historischer Spielstärke (Elo).

    **Simulated Odds / Market Odds (Markt-Quoten):** Der Preis, den der Buchmacher für ein Ereignis zahlt. 
    * *Live Markets:* Hier zieht die App die echten, aktuellen Quoten aus der API.
    * *Simulated:* Hier erschafft das System einen künstlichen "Sharp Bookmaker" (Pinnacle-Proxy), der extrem schlaue Quoten mit einer harten Marge von 3,5 % (Vig) simuliert, um das Modell unter härtesten Bedingungen zu testen.

    **Expected Value / Edge (Erwartungswert / Vorteil):** Der mathematische Vorteil gegenüber dem Markt. Ein Edge entsteht nur dann, wenn die *Model Implied Prob.* höher ist als die Wahrscheinlichkeit, die der Buchmacher in seiner Quote versteckt hat. (Bsp: Quote 2.0 impliziert 50 %. Sagt das Modell 55 %, hast du 5 % Edge). Negative Edges werden vom System blockiert (NEUTRAL).
    
    **Target Allocation (Risk-Adjusted):** Eine konservativ skalierte Risiko-Gewichtung (**Quarter-Kelly Criterion**) basierend auf der Größe deines Bankrolls und dem detektierten Edge. Das System schützt das Kapital durch ein hartes Limit (Cap) von maximal **3 % der Bankroll** pro Trade.
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

tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Labor & Sandbox", 
    "📡 Live-Markt Scanner", 
    "📊 KI-Architektur", 
    "🏆 WM 2026 Orakel"
])

# ---------------------------------------------------------
# TAB 1: MANUELLE ANALYSE (LABOR)
# ---------------------------------------------------------
with tab1:
    st.header("🧪 Labor & Sandbox")
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

# ---------------------------------------------------------
# TAB 2: LIVE-ORAKEL MODUS
# ---------------------------------------------------------
with tab2:
    st.header("📡 Live-Markt Scanner")
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

# ------------------------------------------
# TAB 3: FEATURE ANALYTICS (NEU)
# ------------------------------------------
with tab3:
    st.header("🧠 Feature Importance (Gehirn-Scan)")
    st.markdown("Diese Live-Analyse zeigt direkt aus dem neuronalen Netz, wie stark die KI verschiedene Datenpunkte gewichtet, um ihre Vorhersagen zu treffen.")
    
    # --- 🛠️ FIX: Den echten Random Forest aus der Hülle entpacken ---
    rf_model = model
    if hasattr(model, 'model'):
        rf_model = model.model
    elif hasattr(model, 'rf'):
        rf_model = model.rf
    elif hasattr(model, 'estimator'): # Falls es ein CalibratedClassifier ist
        rf_model = model.estimator
    # ----------------------------------------------------------------
        
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        # Die exakte Reihenfolge aus deiner Pipeline
        features = ['Elo Differenz', 'Poisson (Tore)', 'Form Momentum', 'Angriff (FIFA)', 'Mittelfeld (FIFA)', 'Abwehr (FIFA)']
        
        df_imp = pd.DataFrame({"Faktor": features, "Einfluss (%)": importances * 100})
        df_imp = df_imp.sort_values(by="Einfluss (%)", ascending=False)
        df_plot = df_imp.set_index("Faktor")
        st.bar_chart(df_plot, color="#1f77b4")   
        
        # --- NEU: Die Erklärungen im ausklappbaren Menü ---
        with st.expander("📚 Was bedeuten diese Faktoren genau?"):
            st.markdown("""
            * **Poisson (Tore):** Berechnet die stochastische Wahrscheinlichkeit für die genaue Anzahl an Toren, basierend auf der jüngsten offensiven und defensiven Stärke beider Teams.
            * **Elo Differenz:** Das Elo-Rating bewertet die generelle "Klasse" einer Nation über Jahre hinweg.
            * **Mittelfeld (FIFA):** Die Relevanz eines guten Mittelfeldes
            * **Angriff (FIFA):** Die Qualität der Stürmer. 
            * **Abwehr (FIFA):** Die individuelle Klasse von Torwart und Verteidigern, um gegnerische Angriffe im Eins-gegen-Eins zu stoppen.
            * **Form Momentum:** Der "Hot-Streak"-Faktor. Ein exponentiell gewichtetes Rating der letzten 5 Spiele.
            """)
    else:
        st.error(f"Feature Importance konnte nicht geladen werden. Aktueller Modell-Typ: {type(rf_model).__name__}")


# ------------------------------------------
# TAB 4: WM 2026 ORAKEL (NEU)
# ------------------------------------------
with tab4:
    st.header("🏆 WM 2026 Monte-Carlo Orakel")
    st.markdown("Simuliert den kompletten Turnierbaum (Gruppenphase bis Finale) basierend auf den offiziellen Auslosungen und den mathematischen Modellen der KI.")
    
    sims = st.slider("Anzahl der zu simulierenden Weltmeisterschaften:", min_value=1000, max_value=20000, value=5000, step=1000)
    
    if st.button("🔮 Starte Turnier-Simulation"):
        with st.spinner(f"Simuliere {sims} komplette Weltmeisterschaften im Hintergrund (Dauert ca. 5-10 Sekunden)..."):
            # Wir importieren die Funktion live aus deinem anderen Skript
            from simulate_wm2026 import run_wm_real_structure
            
            # Ausführung und Daten-Abfang
            sorted_champs, sorted_exits = run_wm_real_structure(num_tournaments=sims)
            
            st.success("Simulation abgeschlossen!")
            
            col1, col2 = st.columns(2)
            
            # Ausgabe 1: Wer wird Weltmeister?
            with col1:
                st.subheader("🥇 Favoriten: Weltmeister")
                df_champs = pd.DataFrame(sorted_champs, columns=["Nation", "Titel"])
                df_champs["Wahrscheinlichkeit"] = (df_champs["Titel"] / sims * 100).apply(lambda x: f"{x:.2f} %")
                # Wir blenden die rohe Anzahl der Titel aus und zeigen nur die Prozente
                st.dataframe(df_champs.head(10).drop(columns=["Titel"]), use_container_width=True)
                
            # Ausgabe 2: Wer fliegt raus?
            with col2:
                st.subheader("⚠️ Risiko: Vorrunden-Aus")
                df_exits = pd.DataFrame(sorted_exits, columns=["Nation", "Exits"])
                df_exits["Risiko"] = (df_exits["Exits"] / sims * 100).apply(lambda x: f"{x:.1f} %")
                st.dataframe(df_exits.head(10).drop(columns=["Exits"]), use_container_width=True)