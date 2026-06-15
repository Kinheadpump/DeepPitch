import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.api_client import LiveOracleAPI
from src.lineup_scanner import LiveLineupScanner
import difflib
import altair as alt

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
    matches = difflib.get_close_matches(api_name, known_teams, n=1, cutoff=0.85)
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
    **Tipp & P-Wert:** Das wahrscheinlichste Ergebnis (z.B. 1:0) innerhalb des von der KI vorhergesagten Spielausgangs. Der P-Wert (z.B. P = 13.5%) ist die absolute Wahrscheinlichkeit *dieses einen Ergebnisses* laut dem Poisson-Modell. **Wichtig:** Da Fußball ein Niedrigtor-Sport ist, liegt der P-Wert fast immer zwischen 10–18 %. Der historische Trefferquoten-Wert von ~30 % (aus dem Stresstest) ist höher, weil das System nur Spiele auswählt, bei denen die Modell-Überzeugung hoch ist — es handelt sich um eine selektive Genauigkeit auf einem gefilterten Subsatz von Spielen, nicht um eine Einzelspiel-Garantie.

    **Model Implied Prob. (Implizite Modell-Wahrscheinlichkeit):** Die Wahrscheinlichkeit, die die KI für den 1X2-Spielausgang berechnet. Sie basiert auf dem Ensemble aus Tor-Verteilungen (Poisson), Kadertiefe (FIFA-Daten), Form-Momentum und historischer Spielstärke (Elo).

    **Simulated Odds / Market Odds (Markt-Quoten):** Der Preis, den der Buchmacher für ein Ereignis zahlt.
    * *Live Markets:* Echte, aktuelle Quoten aus der API.
    * *Simulated:* Ein künstlicher "Sharp Bookmaker" (Pinnacle-Proxy) mit 3,5 % Marge, um das Modell unter realen Bedingungen zu testen.

    **Expected Value / Edge (Erwartungswert / Vorteil):** Ein Edge entsteht nur wenn die *Model Implied Prob.* die im Buchmacher-Preis versteckte Wahrscheinlichkeit übersteigt. (Bsp: Quote 2.0 impliziert 50 %. Sagt das Modell 55 %, hast du 5 % Edge). Negative Edges werden blockiert (NEUTRAL).

    **Target Allocation (Risk-Adjusted):** **Quarter-Kelly Criterion** — konservativ skaliert mit hartem Cap von maximal **3 % der Bankroll** pro Trade.
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
    except (FileNotFoundError, KeyError):
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
    # Select best outcome — now includes draw
    best_key = max(probs_ml, key=probs_ml.get)
    outcome_map = {
        'home_win': (2, team_h),
        'draw':     (1, 'Unentschieden'),
        'away_win': (0, team_a),
    }
    ai_raw_pred, team_name = outcome_map[best_key]
    ai_confidence = probs_ml[best_key]

    is_live = False
    if live_odds and len(live_odds) > 2 and live_odds[ai_raw_pred] > 1.0:
        bookie_odds = live_odds[ai_raw_pred]
        is_live = True
    else:
        elo_win_prob = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        elo_loss_prob = 1 / (1 + 10 ** ((elo_h - elo_a) / 400))
        sum_p = elo_win_prob + 0.25 + elo_loss_prob
        odds_dict = {
            2: 1 / ((elo_win_prob / sum_p) * 1.05),
            1: 1 / ((0.25 / sum_p) * 1.05),
            0: 1 / ((elo_loss_prob / sum_p) * 1.05),
        }
        bookie_odds = odds_dict[ai_raw_pred]

    st.markdown("##### Capital Allocation (Risk-Adjusted)")

    if ai_confidence >= 0.50:
        p = ai_confidence
        b = bookie_odds - 1
        q = 1 - p
        kelly_fraction = (p * b - q) / b

        if kelly_fraction > 0:
            stake_pct = min(kelly_fraction * 0.25, 0.03)
            recommended_stake = bankroll * stake_pct
            edge = (p - 1 / bookie_odds) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Model Implied Prob.", f"{p*100:.1f}%")
            c2.metric("Market Odds" if is_live else "Simulated Odds", f"{bookie_odds:.2f}")
            c3.metric("Expected Value (Edge)", f"+{edge:.1f}%")
            outcome_label = {'home_win': 'HEIMSIEG', 'draw': 'UNENTSCHIEDEN', 'away_win': 'AUSWÄRTSSIEG'}[best_key]
            st.success(f"Signal: BUY {team_name.upper()} ({outcome_label}) | Target Allocation: {recommended_stake:.2f} € ({stake_pct*100:.1f}%)")
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

with tab1:
        st.header("🧪 Labor & Sandbox")
        st.markdown("Simuliere hypothetische Matchups zwischen beliebigen Nationen, um die Mathematik des Modells zu testen.")
        
        # --- UI Eingabefelder ---
        db_teams = sorted(list(fifa_ratings.keys()))
        col_h, col_a, col_bk = st.columns([2, 2, 1])
        
        with col_h:
            # Wählt Deutschland als Standard, falls vorhanden
            default_h = db_teams.index("Germany") if "Germany" in db_teams else 0
            team_h = st.selectbox("Heimteam (Team 1)", db_teams, index=default_h)
            
        with col_a:
            # Wählt Frankreich als Standard, falls vorhanden
            default_a = db_teams.index("France") if "France" in db_teams else 1
            team_a = st.selectbox("Auswärtsteam (Team 2)", db_teams, index=default_a)
            
        with col_bk:
            bankroll_sandbox = st.number_input("Portfolio (€)", min_value=10, value=1000, step=100, key="bk_sandbox")

        col_form_h, col_form_a, col_neutral = st.columns([2, 2, 1])
        with col_form_h:
            form_h_val = st.slider("Form Team 1 (-1=schlecht, +1=gut)", -1.0, 1.0, 0.0, 0.1, key="form_h")
        with col_form_a:
            form_a_val = st.slider("Form Team 2 (-1=schlecht, +1=gut)", -1.0, 1.0, 0.0, 0.1, key="form_a")
        with col_neutral:
            is_neutral_sandbox = st.checkbox("Neutraler Spielort", value=True)

        _tournament_weight_map = {
            'FIFA World Cup': 1.0,
            'UEFA Euro / Copa América': 0.67,
            'Sonstiges Turnier': 0.33,
        }
        tournament_type_sandbox = st.selectbox(
            "Turnier-Typ",
            list(_tournament_weight_map.keys()),
            index=0,
            key="tournament_sandbox",
        )
        tournament_weight_sandbox = _tournament_weight_map[tournament_type_sandbox]

        st.divider()

        # --- Simulations-Button ---
        if st.button("🧪 Prognose erstellen", use_container_width=True):
            if team_h == team_a:
                st.error("Bitte wähle zwei unterschiedliche Teams aus.")
            else:
                with st.spinner("Berechne Quant-Metriken und Wahrscheinlichkeiten..."):
                    # 1. Daten abrufen
                    stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
                    stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
                    elo_h = bt.elo.ratings.get(team_h, 1500.0)
                    elo_a = bt.elo.ratings.get(team_a, 1500.0)
                    
                    elo_diff = elo_h - elo_a
                    elo_avg  = (elo_h + elo_a) / 2.0
                    att_diff = stats_h['ATT'] - stats_a['ATT']
                    mid_diff = stats_h['MID'] - stats_a['MID']
                    def_diff = stats_h['DEF'] - stats_a['DEF']
                    att_def_h = stats_h['ATT'] - stats_a['DEF']
                    att_def_a = stats_a['ATT'] - stats_h['DEF']

                    # 2. Poisson-Matrix berechnen (enthält lambdas und xG-Wahrscheinlichkeiten)
                    probs_p = bt.poisson.predict_match_probabilities(
                        team_h, team_a, is_neutral_sandbox, elo_diff, att_def_h, att_def_a
                    )
                    lambda_h = probs_p['lambda_h']
                    lambda_a = probs_p['lambda_a']

                    # 3. Modell-Wahrscheinlichkeiten berechnen
                    # Load real att/def form from end-of-training trackers.
                    # Slider overrides only the GD-based form_diff.
                    stored_forms = bt.get_form_diffs(team_h, team_a)
                    form_diff_sandbox = form_h_val - form_a_val  # slider override
                    probs_ml = model.predict_probabilities(
                        elo_diff, elo_avg,
                        probs_p['home_win'] - probs_p['away_win'], form_diff_sandbox,
                        stored_forms['att_form_diff'], stored_forms['def_form_diff'],
                        att_diff, mid_diff, def_diff,
                        int(is_neutral_sandbox), tournament_weight_sandbox,
                    )

                    # Two-stage score: ML picks outcome region, Poisson argmax picks score within it
                    _ml_blend = {'home_win': probs_ml['home_win'], 'draw': probs_ml['draw'], 'away_win': probs_ml['away_win']}
                    proj_goals_h, proj_goals_a = bt.poisson.get_smart_score(probs_p['matrix'], _ml_blend)
                    # Probability of this exact scoreline from the ML-adjusted matrix
                    _blended_m = bt.poisson._blend_matrix(probs_p['matrix'], _ml_blend)
                    tipp_prob = float(_blended_m[proj_goals_h, proj_goals_a]) * 100

                    # --- 🌟 WUNDERSCHÖNES UI-RENDERING ---
                    with st.container(border=True):
                        st.subheader(f"{team_h} vs. {team_a}")
                        st.caption(" Manuelle Labor-Simulation")

                        top1, top2 = st.columns(2)
                        top1.metric("Erwartete Tore (xG)", f"{lambda_h:.2f} – {lambda_a:.2f}")
                        top2.metric("Tipp", f"{proj_goals_h} : {proj_goals_a}", f"P = {tipp_prob:.1f}%")

                        # Modell-Wahrscheinlichkeiten
                        st.markdown("#####  Modell-Wahrscheinlichkeiten (1X2)")
                        cm1, cm2, cm3 = st.columns(3)
                        cm1.metric(f"Heimsieg ({team_h})", f"{probs_ml['home_win']*100:.1f}%")
                        cm2.metric("Unentschieden", f"{probs_ml['draw']*100:.1f}%")
                        cm3.metric(f"Auswärtssieg ({team_a})", f"{probs_ml['away_win']*100:.1f}%")
                        
                        st.divider()
                        
                        render_kelly_advisor(bankroll_sandbox, probs_ml, elo_h, elo_a, team_h, team_a)

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

        st.info("ℹ️ Form-Werte basieren auf dem letzten Trainingsstand. Spielort wird als neutral behandelt (API liefert keine Venue-Daten).")

        if fetch_button:
            with st.spinner("Synchronisiere mit Provider..."):
                matches = api.get_upcoming_matches(days_ahead=10)

                if matches is None:
                    st.error("⚠️ API-Verbindungsfehler. Bitte API-Key und Netzwerkverbindung prüfen.")
                elif not matches:
                    st.info("Keine handelbaren Events im definierten Zeitfenster gefunden.")
                else:
                    db_teams = list(fifa_ratings.keys())
                    
                    for match in matches:
                        raw_h, raw_a = match['home_team'], match['away_team']
                        team_h = get_real_team_name(raw_h, db_teams)
                        team_a = get_real_team_name(raw_a, db_teams)
                        
                        stats_h = fifa_ratings.get(team_h, bt.FALLBACK_RATING)
                        stats_a = fifa_ratings.get(team_a, bt.FALLBACK_RATING)
                        elo_h = bt.elo.ratings.get(team_h, 1500.0)
                        elo_a = bt.elo.ratings.get(team_a, 1500.0)
                        elo_diff = elo_h - elo_a
                        elo_avg  = (elo_h + elo_a) / 2.0
                        att_diff = stats_h['ATT'] - stats_a['ATT']
                        mid_diff = stats_h['MID'] - stats_a['MID']
                        def_diff = stats_h['DEF'] - stats_a['DEF']
                        att_def_h = stats_h['ATT'] - stats_a['DEF']
                        att_def_a = stats_a['ATT'] - stats_h['DEF']

                        # Derive tournament weight from competition name
                        _comp = match.get('competition', '')
                        _tw_live = {
                            'FIFA World Cup': 1.0,
                            'UEFA Euro': 0.67, 'Copa América': 0.67,
                            'African Cup of Nations': 0.67, 'AFC Asian Cup': 0.67,
                        }
                        _live_tw = next((v for k, v in _tw_live.items() if k.lower() in _comp.lower()), 0.5)

                        # 1. Poisson-Matrix und Wahrscheinlichkeiten berechnen
                        probs_p = bt.poisson.predict_match_probabilities(
                            team_h, team_a, True, elo_diff, att_def_h, att_def_a
                        )
                        lambda_h = probs_p['lambda_h']
                        lambda_a = probs_p['lambda_a']

                        # 2. ML-Wahrscheinlichkeiten berechnen
                        live_forms = bt.get_form_diffs(team_h, team_a)
                        probs_ml = model.predict_probabilities(
                            elo_diff, elo_avg,
                            probs_p['home_win'] - probs_p['away_win'],
                            live_forms['form_diff'],
                            live_forms['att_form_diff'], live_forms['def_form_diff'],
                            att_diff, mid_diff, def_diff,
                            1, _live_tw,
                        )

                        # Two-stage score: ML picks outcome region, Poisson argmax picks score within it
                        _ml_blend = {'home_win': probs_ml['home_win'], 'draw': probs_ml['draw'], 'away_win': probs_ml['away_win']}
                        proj_goals_h, proj_goals_a = bt.poisson.get_smart_score(probs_p['matrix'], _ml_blend)
                        _blended_m = bt.poisson._blend_matrix(probs_p['matrix'], _ml_blend)
                        tipp_prob = float(_blended_m[proj_goals_h, proj_goals_a]) * 100

                        # --- 🌟 WUNDERSCHÖNES UI-RENDERING ---
                        with st.container(border=True):
                            st.subheader(f"🏟️ {team_h} vs. {team_a}")
                            st.caption(f"📅 {match['date']} | 🏆 {match['competition']}")

                            top1, top2 = st.columns(2)
                            top1.metric("Erwartete Tore (xG)", f"{lambda_h:.2f} – {lambda_a:.2f}")
                            top2.metric("Tipp", f"{proj_goals_h} : {proj_goals_a}", f"P = {tipp_prob:.1f}%")

                            st.markdown("##### 📊 Modell-Wahrscheinlichkeiten (1X2)")
                            cm1, cm2, cm3 = st.columns(3)
                            cm1.metric(f"Heimsieg ({team_h})", f"{probs_ml['home_win']*100:.1f}%")
                            cm2.metric("Unentschieden", f"{probs_ml['draw']*100:.1f}%")
                            cm3.metric(f"Auswärtssieg ({team_a})", f"{probs_ml['away_win']*100:.1f}%")
                            
                            st.divider()
                            
                            # Capital Allocation / Kelly Check
                            render_kelly_advisor(bankroll_live, probs_ml, elo_h, elo_a, team_h, team_a)

# ------------------------------------------
# TAB 3: FEATURE ANALYTICS (NEU)
# ------------------------------------------
with tab3:
    st.header("🧠 Feature Importance (Gehirn-Scan)")
    st.markdown("Diese Live-Analyse zeigt direkt aus dem neuronalen Netz, wie stark die KI verschiedene Datenpunkte gewichtet, um ihre Vorhersagen zu treffen.")
    
    importances_dict = model.get_feature_importances()

    _feature_labels = {
        'elo_diff':          'Elo Differenz',
        'elo_avg':           'Elo Niveau (Avg)',
        'poisson_diff':      'Poisson (Tore)',
        'form_diff':         'Form Momentum',
        'att_form_diff':     'Angriffs-Form',
        'def_form_diff':     'Abwehr-Form',
        'att_diff':          'Angriff (FIFA)',
        'mid_diff':          'Mittelfeld (FIFA)',
        'def_diff':          'Abwehr (FIFA)',
        'is_neutral':        'Neutraler Spielort',
        'tournament_weight': 'Turnier-Gewicht',
    }

    if importances_dict:
        df_imp = pd.DataFrame([
            {"Faktor": _feature_labels.get(k, k), "Einfluss (%)": v * 100}
            for k, v in importances_dict.items()
        ])
        df_imp = df_imp.sort_values(by="Einfluss (%)", ascending=False)

        chart = alt.Chart(df_imp).mark_bar().encode(
            x=alt.X("Faktor", sort=list(df_imp["Faktor"])),
            y="Einfluss (%)",
            color=alt.value("#1f77b4")
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        with st.expander("📚 Was bedeuten diese Faktoren genau?"):
            st.markdown("""
            * **Elo Differenz:** Die relative Stärke zwischen den Teams — Kernindikator für den Favoriten.
            * **Elo Niveau (Avg):** Absolutes Stärkeniveau beider Teams — unterscheidet Elite- von Mittelfeld-Duellen.
            * **Poisson (Tore):** Stochastische Torerwartung basierend auf Angriff/Abwehr-Stärke beider Teams.
            * **Form Momentum:** Gewichtetes Tordifferenz-Rating der letzten 5 Spiele (4:0 zählt mehr als 1:0).
            * **Angriffs-Form:** Tore-erzielte Form der letzten 5 Spiele — unabhängig von Gegentoren.
            * **Abwehr-Form:** Tore-kassierte Form der letzten 5 Spiele — erkennt defensive Serien.
            * **Angriff / Mittelfeld / Abwehr (FIFA):** Kaderqualität in drei Mannschaftsteilen.
            * **Neutraler Spielort:** 1 = kein Heimvorteil, 0 = Heimspiel (+0.2 xG für Gastgeber).
            * **Turnier-Gewicht:** WM=1.0, Kontinentalturnier=0.67, Sonstiges=0.33 — gewichtet Spielbedeutung.
            """)
    else:
        st.error("Feature Importance konnte nicht geladen werden. Bitte das Modell neu trainieren.")


# ------------------------------------------
# TAB 4: WM 2026 ORAKEL (NEU)
# ------------------------------------------
with tab4:
    st.header("🏆 WM 2026 Monte-Carlo Orakel")
    st.markdown("Simuliert den kompletten Turnierbaum (Gruppenphase bis Finale) basierend auf den offiziellen Auslosungen und den mathematischen Modellen der KI.")
    
    sims = st.slider("Anzahl der zu simulierenden Weltmeisterschaften:", min_value=1000, max_value=20000, value=5000, step=1000)
    
    if st.button("🔮 Starte Turnier-Simulation"):
        try:
            from simulate_wm2026 import run_wm_real_structure
        except ModuleNotFoundError:
            st.error("❌ simulate_wm2026.py nicht gefunden. Bitte die Datei im Projektordner anlegen.")
            st.stop()
        with st.spinner(f"Simuliere {sims} komplette Weltmeisterschaften im Hintergrund (Dauert ca. 5-10 Sekunden)..."):
            
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