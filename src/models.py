import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class DynamicRollingPoisson:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.team_stats = {}
        # Dixon-Coles Rho-Faktor (Industrie-Standard liegt bei ca. -0.13 bis -0.15)
        # Ein negativer Wert erhöht die Wahrscheinlichkeit für niedrige Unentschieden (0:0, 1:1)
        self.rho = -0.13 

    def update_match(self, home_team, away_team, home_goals, away_goals, elo_home, elo_away):
        for team in [home_team, away_team]:
            if team not in self.team_stats:
                self.team_stats[team] = {'goals_scored': [], 'goals_conceded': [], 'elo_opponents': []}

        self.team_stats[home_team]['goals_scored'].append(home_goals)
        self.team_stats[home_team]['goals_conceded'].append(away_goals)
        self.team_stats[home_team]['elo_opponents'].append(elo_away)

        self.team_stats[away_team]['goals_scored'].append(away_goals)
        self.team_stats[away_team]['goals_conceded'].append(home_goals)
        self.team_stats[away_team]['elo_opponents'].append(elo_home)

        for team in [home_team, away_team]:
            if len(self.team_stats[team]['goals_scored']) > self.window_size:
                self.team_stats[team]['goals_scored'].pop(0)
                self.team_stats[team]['goals_conceded'].pop(0)
                self.team_stats[team]['elo_opponents'].pop(0)

    def _calculate_xg(self, team, is_home, is_neutral):
        if team not in self.team_stats or len(self.team_stats[team]['goals_scored']) < 5:
            return 1.4 if is_home and not is_neutral else 1.1

        avg_scored = np.mean(self.team_stats[team]['goals_scored'])
        avg_conceded = np.mean(self.team_stats[team]['goals_conceded'])
        
        # Heimvorteil dynamisch berechnen
        home_adv = 1.15 if is_home and not is_neutral else 1.0
        return ((avg_scored + avg_conceded) / 2) * home_adv

    def predict_match_probabilities(self, home_team, away_team, is_neutral, elo_diff=0, att_diff=0, def_diff=0):
        xg_h = self._calculate_xg(home_team, True, is_neutral)
        xg_a = self._calculate_xg(away_team, False, is_neutral)

        xg_h += (elo_diff / 400) + (att_diff / 20)
        xg_a -= (elo_diff / 400) - (def_diff / 20)

        xg_h = max(0.05, min(5.0, xg_h))
        xg_a = max(0.05, min(5.0, xg_a))

        h_probs = [poisson.pmf(i, xg_h) for i in range(6)]
        a_probs = [poisson.pmf(i, xg_a) for i in range(6)]
        matrix = np.outer(h_probs, a_probs)

        matrix[0, 0] *= max(0, 1 - xg_h * xg_a * self.rho)
        matrix[1, 0] *= max(0, 1 + xg_a * self.rho)
        matrix[0, 1] *= max(0, 1 + xg_h * self.rho)
        matrix[1, 1] *= max(0, 1 - self.rho)
        
        matrix /= np.sum(matrix)

        # NEU: Wir berechnen keinen dummen Tor-Tipp mehr hier, sondern geben die rohe Matrix zurück!
        return {
            "home_win": np.tril(matrix, -1).sum(),
            "draw": np.trace(matrix),
            "away_win": np.triu(matrix, 1).sum(),
            "matrix": matrix # Die Matrix wird an das Smart-Orakel übergeben
        }

    def get_smart_score(self, matrix, probs_ml):
        """State-of-the-Art ML-Guided Matrix Masking für hochpräzise exakte Tore."""
        # 1. Welches Ergebnis (1, X, 2) hält die KI für am wahrscheinlichsten?
        best_outcome = max(probs_ml, key=probs_ml.get)
        
        # 2. Maske erstellen (wir nullen die Matrix aus)
        masked_matrix = np.zeros_like(matrix)
        
        # 3. Nur die Zellen behalten, die der KI recht geben!
        if best_outcome == 'home_win':
            masked_matrix = np.tril(matrix, -1)
        elif best_outcome == 'away_win':
            masked_matrix = np.triu(matrix, 1)
        else:
            masked_matrix = np.diag(np.diag(matrix))
            
            # Sondereingriff für dynamische Unentschieden (2:2, 3:3)
            # Wir berechnen die statistisch zu erwartenden Gesamttore
            expected_goals = np.sum(matrix * np.add.outer(np.arange(6), np.arange(6)))
            
            if expected_goals > 2.6:
                # Extrem offensives Spiel: 0:0 und 1:1 brutal abwerten -> erzwingt 2:2!
                masked_matrix[0, 0] *= 0.05 
                masked_matrix[1, 1] *= 0.3 
            elif expected_goals > 2.1:
                # Moderat offensives Spiel: 0:0 abwerten
                masked_matrix[0, 0] *= 0.4
                
        # 4. Das wahrscheinlichste Ergebnis aus der nun perfekt gefilterten Matrix suchen
        best_idx = np.unravel_index(np.argmax(masked_matrix, axis=None), masked_matrix.shape)
        return best_idx[0], best_idx[1]

class MetaMachineLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        # NEU: 7 Features! Der alte 'squad_diff' wurde in ATT, MID, DEF aufgespalten
        self.feature_names = ['elo_diff', 'poisson_diff', 'form_diff', 'continent_adv_diff', 'att_diff', 'mid_diff', 'def_diff']
        self.best_params = {}

    def train(self, df_features: pd.DataFrame):
        X = df_features[self.feature_names]
        y = df_features['outcome']
        X_scaled = self.scaler.fit_transform(X)

        # Non-lineares Hyperparameter-Grid für den Random Forest
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6, 8],          # Kontrolliert die Komplexität der Wenn-Dann-Regeln
            'min_samples_split': [5, 10]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_scaled, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_trained = True

    def predict_probabilities(self, elo_diff: float, poisson_diff: float, form_diff: float, continent_adv_diff: float, att_diff: float, mid_diff: float, def_diff: float) -> dict:
        if not self.is_trained:
            raise RuntimeError("Modell muss trainiert werden!")
            
        X_pred = pd.DataFrame([[elo_diff, poisson_diff, form_diff, continent_adv_diff, att_diff, mid_diff, def_diff]], columns=self.feature_names)
        X_pred_scaled = self.scaler.transform(X_pred)
        probs = self.model.predict_proba(X_pred_scaled)[0]
        
        classes = self.model.classes_
        prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
        for c, p in zip(classes, probs):
            prob_dict[c] = p
            
        return {"away_win": prob_dict[0], "draw": prob_dict[1], "home_win": prob_dict[2]}