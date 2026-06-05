import joblib

def run_monte_carlo_simulation():
    print("Initialisiere DeepPitch Risk Management Environment...")
    
    try:
        # Lade das gespeicherte Gehirn (Modell, Backtester und FIFA-Ratings)
        brain = joblib.load("data/deeppitch_brain.pkl")
        bt = brain['backtester']
        model = brain['model']
        fifa_ratings = brain['fifa']
    except FileNotFoundError:
        print("Kritischer Fehler: 'data/deeppitch_brain.pkl' nicht gefunden.")
        print("Bitte führe zuerst 'python train.py' aus, um das Modell zu kompilieren.")
        return

    # 1. Rohdaten laden
    df = bt.loader.load_data(start_year=2000)
    
    # 2. Pre-Match Elo-Werte berechnen
    df = bt.elo.calculate_historical_elo(df)
    
    # 3. Den historischen Feature-Vektor (die 2.100 Turnierspiele) generieren
    df_features = bt._generate_historical_features(df, fifa_ratings)
    
    # 4. Die Monte-Carlo Risiko-Engine starten
    bt.stresstest(
        model=model, 
        df_features=df_features, 
        initial_bankroll=10000.0, 
        num_simulations=1000
    )

if __name__ == "__main__":
    run_monte_carlo_simulation()