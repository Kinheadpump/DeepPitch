# DEPRECATED — superseded by EloSystem inside src/backtester.py.
# This file is kept for reference only and is not imported by the active pipeline.

import pandas as pd

class EloRatingSystem:
    """
    Dynamisches Elo-Rating-System mit turnierbasierter Gewichtung (K-Faktor).
    """
    def __init__(self, base_k_factor=40):
        self.base_k_factor = base_k_factor
        self.ratings = {}

    def _get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Berechnet die Gewinnwahrscheinlichkeit nach der mathematischen Elo-Formel."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def calculate_historical_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geht chronologisch durch alle Spiele und berechnet die Ratings."""
        print("[Features] Berechne dynamische Elo-Ratings für alle Teams...")
        elo_home_list = []
        elo_away_list = []

        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Neue Teams starten mit einem Standard-Rating von 1500
            if home not in self.ratings: self.ratings[home] = 1500.0
            if away not in self.ratings: self.ratings[away] = 1500.0

            # Aktuelles Rating vor dem Spiel abspeichern (das sind unsere Features!)
            elo_home_list.append(self.ratings[home])
            elo_away_list.append(self.ratings[away])

            # Tatsächliches Ergebnis bestimmen (1 = Heimsieg, 0.5 = Remis, 0 = Auswärtssieg)
            if row['home_score'] > row['away_score']:
                actual_home = 1.0
            elif row['home_score'] < row['away_score']:
                actual_home = 0.0
            else:
                actual_home = 0.5

            exp_home = self._get_expected_score(self.ratings[home], self.ratings[away])
            
            # WICHTIG: Dynamischer K-Faktor! Ein WM-Spiel ändert das Rating massiv,
            # ein Freundschaftsspiel ändert es kaum.
            if 'FIFA World Cup' in row['tournament']:
                k = 60  # WM-Endrunde
            elif 'qualification' in row['tournament'].lower():
                k = 50  # Qualifikation
            else:
                k = self.base_k_factor  # Freundschaftsspiele (40)

            # Elo Ratings updaten (Gedächtnis)
            self.ratings[home] += k * (actual_home - exp_home)
            self.ratings[away] += k * ((1 - actual_home) - (1 - exp_home))

        # Die Features an den DataFrame anhängen
        df['elo_home'] = elo_home_list
        df['elo_away'] = elo_away_list
        print("[Features] Elo-Ratings erfolgreich berechnet.")
        
        return df