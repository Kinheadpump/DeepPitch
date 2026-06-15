# DEPRECATED — do not import this module.
# WorldCupPredictor references PoissonBaseModel (removed) and the old
# MetaMachineLearningModel signature (4 args vs current 7). It will crash on import.
# The active pipeline lives in src/backtester.py + src/models.py.
raise ImportError(
    "src/predictor.py is a legacy stub and has been disabled. "
    "Use src/backtester.py and src/models.py instead."
)

import pandas as pd
from src.data_loader import DataLoader
from src.features import EloCalculator
from src.models import PoissonBaseModel, MetaMachineLearningModel

class WorldCupPredictor:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.elo_calculator = EloCalculator()
        self.poisson_model = PoissonBaseModel()
        self.ml_model = MetaMachineLearningModel() # Unser neuer Cheftrainer

    def initialize(self):
        """Baut die komplette KI-Pipeline auf."""
        # 1. Daten laden (Wir nehmen ab 2010, damit das ML-Modell modernen Fußball lernt)
        df = self.data_loader.load_data(start_year=2010)
        
        # 2. Elo-Ratings berechnen (Chronologisch)
        df_with_elo = self.elo_calculator.calculate_historical_elo(df)
        
        # 3. Poisson trainieren (Statistische Basis)
        self.poisson_model.train(df_with_elo)
        
        # 4. Trainingsdaten für das ML-Modell generieren
        print("[System] Generiere KI-Trainingsdaten (Feature Engineering)...")
        ml_data = []
        for index, row in df_with_elo.iterrows():
            team_h = row['home_team']
            team_a = row['away_team']
            
            # Poisson fragen
            xg_h, xg_a = self.poisson_model.get_team_stat(team_h, 'attack_strength_home') * self.poisson_model.avg_home_goals * self.poisson_model.get_team_stat(team_a, 'defense_strength_away'), \
                         self.poisson_model.get_team_stat(team_a, 'attack_strength_away') * self.poisson_model.avg_away_goals * self.poisson_model.get_team_stat(team_h, 'defense_strength_home')
            
            # Echtes Ergebnis ermitteln (2=Heim, 1=X, 0=Auswärts)
            if row['home_score'] > row['away_score']: outcome = 2
            elif row['home_score'] == row['away_score']: outcome = 1
            else: outcome = 0
                
            ml_data.append({
                'elo_home': row['elo_home'],
                'elo_away': row['elo_away'],
                'xg_home': xg_h,
                'xg_away': xg_a,
                'outcome': outcome
            })
            
        # 5. ML-Modell trainieren
        ml_df = pd.DataFrame(ml_data)
        self.ml_model.train(ml_df)
        print("[System] 🔥 Predictor ist initialisiert und feuerbereit!\n")

    def predict_match(self, team_a: str, team_b: str):
        # 1. Metriken abrufen
        elo_a = self.elo_calculator.get_rating(team_a)
        elo_b = self.elo_calculator.get_rating(team_b)
        xg_a, xg_b = self.poisson_model.predict_expected_goals(team_a, team_b)
        
        # 2. Basis-Wahrscheinlichkeiten
        poisson_probs = self.poisson_model.predict_match_probabilities(xg_a, xg_b)
        
        # 3. Finale ML-Entscheidung (Die Magie!)
        ml_probs = self.ml_model.predict_probabilities(elo_a, elo_b, xg_a, xg_b)
        
        # Output formatieren
        print("=" * 60)
        print(f"🤖 KI-SPIELVORHERSAGE: {team_a} vs. {team_b}")
        print("=" * 60)
        print(f"📊 [Rohdaten der Experten]")
        print(f"  Elo-Rating:    {team_a} ({elo_a:.0f}) | {team_b} ({elo_b:.0f})")
        print(f"  Erwartete Tore:{team_a} ({xg_a:.2f}) | {team_b} ({xg_b:.2f})")
        print(f"  Wahrscheinlichstes Ergebnis (Poisson): {poisson_probs['most_likely_score']}")
        print("-" * 60)
        print(f"🧠 [Finale Machine Learning Wahrscheinlichkeiten]")
        print(f"  Sieg {team_a}:      {ml_probs['home_win'] * 100:.1f}%")
        print(f"  Unentschieden:      {ml_probs['draw'] * 100:.1f}%")
        print(f"  Sieg {team_b}:      {ml_probs['away_win'] * 100:.1f}%")
        print("=" * 60)
        
        return ml_probs