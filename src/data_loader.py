import pandas as pd
import os

class DataLoader:
    """Lädt und bereinigt die historischen Spieldaten für das Vorhersagemodell."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self, start_year: int = 2000, exclude_friendlies: bool = True) -> pd.DataFrame:
        """
        Lädt die CSV-Datei und führt eine erste Bereinigung durch.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[Fehler] Die Datei {self.file_path} wurde nicht gefunden. Hast du sie heruntergeladen?")
            
        print(f"[Data] Lade historische Daten aus {self.file_path}...")
        df = pd.read_csv(self.file_path)

        df['date'] = pd.to_datetime(df['date'])

        df = df[df['date'].dt.year >= start_year].copy()

        if exclude_friendlies:
            df = df[df['tournament'] != 'Friendly'].copy()

        required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral', 'tournament']        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"[Fehler] Die erwartete Spalte '{col}' fehlt in der CSV.")

        df = df[required_columns]

        df = df.dropna(subset=['home_score', 'away_score']).copy()

        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)

        print(f"[Data] Erfolgreich {len(df)} Spiele ab dem Jahr {start_year} geladen.")
        return df
    
    def load_fifa_ratings(self, fifa_filepath="data/fifa_players.csv", year=None) -> dict:
        """Lädt FIFA-Spielerdaten und berechnet granulare Werte (ATT, MID, DEF) der besten Spieler."""
        print(f"[Data] Lade granulare Spieler-Qualitäten (FIFA Ratings) aus {fifa_filepath}...")
        try:
            # 1. Spalten auslesen: Wir brauchen jetzt auch die Positionen!
            # Hinweis: In Kaggle-Datensätzen heißt die Spalte meist 'player_positions' oder 'club_position'
            df_header = pd.read_csv(fifa_filepath, nrows=0)
            cols = df_header.columns.tolist()
            
            nat_col = next((c for c in cols if 'nation' in c.lower() and 'id' not in c.lower()), None)
            skill_col = next((c for c in cols if 'overall' in c.lower()), None)
            pos_col = next((c for c in cols if 'position' in c.lower() and 'id' not in c.lower()), None)
            
            if not all([nat_col, skill_col, pos_col]):
                print(f"[Warnung] Spalten fehlen! Gefunden: Nation={nat_col}, Skill={skill_col}, Pos={pos_col}")
                return {}

            # 2. Nur benötigte Daten in den RAM laden
            df = pd.read_csv(fifa_filepath, usecols=[nat_col, skill_col, pos_col])
            
            # 3. Positionen mappen (Sturm, Mittelfeld, Abwehr)
            def map_position(pos_str):
                pos = str(pos_str).split(',')[0].strip().upper() # Falls jemand "ST, LW" spielt, nimm das Erste
                if pos in ['ST', 'CF', 'RW', 'LW', 'RF', 'LF']: return 'ATT'
                if pos in ['CM', 'CAM', 'CDM', 'RM', 'LM']: return 'MID'
                if pos in ['CB', 'RB', 'LB', 'RWB', 'LWB']: return 'DEF'
                return 'MID' # Torhüter und Unbekannte wandern ins defensive Mittelfeld (Fallback)
                
            df['Line'] = df[pos_col].apply(map_position)
            
            # 4. DER PROFI-TRICK: Nur die besten 5 Spieler pro Land & Mannschaftsteil (Verhindert Verwässerung!)
            # Wir sortieren alle Spieler absteigend nach Stärke...
            df_sorted = df.sort_values(by=[nat_col, 'Line', skill_col], ascending=[True, True, False])
            # ...und nehmen nur die Top 5 pro Kategorie!
            top_players = df_sorted.groupby([nat_col, 'Line']).head(5)
            
            # 5. Durchschnitt der Top 5 berechnen und als Dictionary formatieren
            squad_stats = top_players.groupby([nat_col, 'Line'])[skill_col].mean().unstack().to_dict('index')
            
            # Namens-Korrekturen
            name_mapping = {
                "United States": "USA", "Korea Republic": "South Korea", 
                "IR Iran": "Iran", "Czech Republic": "Czechia", 
                "Côte d'Ivoire": "Ivory Coast"
            }
            
            final_ratings = {}
            for nation, stats in squad_stats.items():
                real_name = name_mapping.get(nation, nation)
                # Falls eine Nation keine 5 Spieler für eine Position hat, Fallback auf 70
                final_ratings[real_name] = {
                    'ATT': stats.get('ATT', 70.0),
                    'MID': stats.get('MID', 70.0),
                    'DEF': stats.get('DEF', 70.0)
                }
                
            print(f"[Data] Erfolgreich granulare Kader-Ratings für {len(final_ratings)} Nationen berechnet.")
            return final_ratings
        except Exception as e:
            print(f"[Warnung] FIFA-Daten konnten nicht geladen werden: {e}")
            return {}