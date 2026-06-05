import pandas as pd
from thefuzz import process

class LiveLineupScanner:
    def __init__(self, fifa_csv_path="data/fifa_players.csv"):
        print("[Scanner] Lade 1.2 GB Spieler-Datenbank in den RAM für Fuzzy-Matching...")
        # Wir laden nur die Spalten, die wir wirklich brauchen, um RAM zu sparen!
        try:
            self.df_players = pd.read_csv(fifa_csv_path, usecols=['short_name', 'long_name', 'nationality_name', 'overall', 'shooting', 'passing', 'defending'], low_memory=False)
        except ValueError:
            # Fallback für ältere FIFA-Datensätze mit leicht anderen Spaltennamen
            self.df_players = pd.read_csv(fifa_csv_path, usecols=['short_name', 'long_name', 'nationality', 'overall', 'shooting', 'passing', 'defending'], low_memory=False)
            self.df_players.rename(columns={'nationality': 'nationality_name'}, inplace=True)

    def fetch_live_lineup_api(self, team_name):
        """
        HIER WÜRDE DER ECHTE API-SPORTS CALL STEHEN.
        Für unser Tutorial simulieren wir einen drastischen Ausfall:
        Deutschland spielt OHNE seine Top-Stars (Müller, Musiala, Rüdiger fehlen).
        """
        if team_name == "Germany":
            return ["M. Neuer", "N. Süle", "M. Ginter", "R. Gosens", "T. Kehrer", 
                    "E. Can", "L. Goretzka", "J. Brandt", 
                    "T. Werner", "K. Volland", "M. Reus"]
        else:
            return [] # Demo unterstützt aktuell nur Deutschland

    def get_live_squad_rating(self, team_name):
        lineup_names = self.fetch_live_lineup_api(team_name)
        if not lineup_names:
            return None, None
            
        print(f"\n🔍 [Fuzzy Matcher] Scanne Startelf für {team_name}...")
        df_nation = self.df_players[self.df_players['nationality_name'] == team_name]
        
        if df_nation.empty:
            return None, None

        # Liste aller Namen dieser Nation
        nation_names = df_nation['short_name'].dropna().tolist() + df_nation['long_name'].dropna().tolist()
        nation_names = list(set([str(x) for x in nation_names]))

        matched_players = []
        match_logs = []
        
        for name in lineup_names:
            # MAGIE: TheFuzz sucht den ähnlichsten Namen in der Datenbank!
            best_match, score = process.extractOne(name, nation_names)
            
            if score > 75: # 75% Übereinstimmung reicht uns
                player_data = df_nation[(df_nation['short_name'] == best_match) | (df_nation['long_name'] == best_match)].iloc[0]
                matched_players.append(player_data)
                match_logs.append(f"✔️ {name} -> {best_match} (Score: {score}%) | OVR: {player_data['overall']}")
            else:
                match_logs.append(f"❌ {name} -> Nicht gefunden (Bester: {best_match} mit {score}%)")

        if not matched_players:
            return None, match_logs
            
        df_lineup = pd.DataFrame(matched_players)
        
        # Live-Aggregation der 11 Spieler auf dem Rasen
        live_stats = {
            'ATT': df_lineup['overall'].mean() + (df_lineup['shooting'].mean() * 0.15),
            'MID': df_lineup['overall'].mean() + (df_lineup['passing'].mean() * 0.15),
            'DEF': df_lineup['overall'].mean() + (df_lineup['defending'].mean() * 0.15)
        }
        
        # NaN Werte bereinigen (falls Torhüter keine Shooting-Werte haben)
        live_stats = {k: round(v, 2) if pd.notnull(v) else 75.0 for k, v in live_stats.items()}
        
        return live_stats, match_logs