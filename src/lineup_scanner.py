import pandas as pd
import requests
import datetime
from thefuzz import process

class LiveLineupScanner:
    def __init__(self, fifa_csv_path="data/fifa_players_lite.csv", api_key=None):
        self.api_key = api_key
        print("[Scanner] Lade Cloud-optimierte Spieler-Datenbank für Fuzzy-Matching...")
        self.df_players = pd.read_csv(fifa_csv_path, low_memory=False)
        
        # Headers für API-Sports
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }

    def fetch_live_lineup_api(self, team_name):
        """Holt die echte Startelf vom API-Sports Server."""
        if not self.api_key or self.api_key == "DEMO":
            return None, "Fehler: Kein API-Sports Key hinterlegt."

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        print(f"📡 Suche heutige Spiele für {team_name} ({today})...")

        try:
            # 1. Das heutige Spiel suchen
            url_fixtures = f"https://v3.football.api-sports.io/fixtures?date={today}"
            res_fix = requests.get(url_fixtures, headers=self.headers, timeout=10).json()

            fixture_id, team_id = None, None
            
            # Fehler abfangen (z.B. API-Limit erreicht)
            if 'response' not in res_fix:
                return None, f"API Fehler: {res_fix.get('message', 'Unbekannter Fehler')}"

            for match in res_fix['response']:
                # Wir suchen nach dem Teamnamen (Heim oder Auswärts)
                if match['teams']['home']['name'] == team_name:
                    fixture_id = match['fixture']['id']
                    team_id = match['teams']['home']['id']
                    break
                elif match['teams']['away']['name'] == team_name:
                    fixture_id = match['fixture']['id']
                    team_id = match['teams']['away']['id']
                    break

            if not fixture_id:
                return None, f"Es wurde heute kein Spiel für '{team_name}' gefunden."

            # 2. Die Aufstellung für dieses Spiel abrufen
            url_lineup = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}&team={team_id}"
            res_lineup = requests.get(url_lineup, headers=self.headers, timeout=10).json()

            if not res_lineup.get('response'):
                return None, "Aufstellung noch nicht veröffentlicht (idR. 60 Min vor Anpfiff)."

            # 3. Die 11 Namen extrahieren
            starting_xi = []
            for player_entry in res_lineup['response'][0]['startXI']:
                starting_xi.append(player_entry['player']['name'])

            return starting_xi, "OK"

        except Exception as e:
            return None, f"Netzwerk/API Fehler: {str(e)}"

    def get_live_squad_rating(self, team_name):
        # API aufrufen
        lineup_names, api_status = self.fetch_live_lineup_api(team_name)
        
        if not lineup_names:
            return None, [f"❌ API-Abbruch: {api_status}"]
            
        print(f"\n🔍 [Fuzzy Matcher] Scanne {len(lineup_names)} Spieler für {team_name}...")
        df_nation = self.df_players[self.df_players['nationality_name'] == team_name]
        
        if df_nation.empty:
            return None, ["❌ Keine Spieler dieser Nation in der FIFA-Datenbank gefunden."]

        nation_names = df_nation['short_name'].dropna().tolist() + df_nation['long_name'].dropna().tolist()
        nation_names = list(set([str(x) for x in nation_names]))

        if not nation_names:
            return None, ["❌ Keine Spielernamen für diese Nation in der FIFA-Datenbank gefunden."]

        matched_players = []
        match_logs = [f"✅ API-Daten erfolgreich abgerufen: {api_status}"]

        for name in lineup_names:
            result = process.extractOne(name, nation_names)
            if result is None:
                match_logs.append(f"⚠️ {name} -> Kein Match gefunden. Wird übersprungen.")
                continue
            best_match, score = result
            if score > 75:
                player_data = df_nation[(df_nation['short_name'] == best_match) | (df_nation['long_name'] == best_match)].iloc[0]
                matched_players.append(player_data)
                match_logs.append(f"✔️ {name} -> {best_match} (Score: {score}%) | OVR: {player_data['overall']}")
            else:
                match_logs.append(f"⚠️ {name} -> Nicht sicher gefunden (Bester: {best_match} mit {score}%). Wird übersprungen.")

        if not matched_players:
            return None, match_logs
            
        df_lineup = pd.DataFrame(matched_players)
        
        live_stats = {
            'ATT': df_lineup['overall'].mean() + (df_lineup['shooting'].mean() * 0.15),
            'MID': df_lineup['overall'].mean() + (df_lineup['passing'].mean() * 0.15),
            'DEF': df_lineup['overall'].mean() + (df_lineup['defending'].mean() * 0.15)
        }
        
        live_stats = {k: round(v, 2) if pd.notnull(v) else 75.0 for k, v in live_stats.items()}
        return live_stats, match_logs