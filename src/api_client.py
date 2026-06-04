import requests
from datetime import datetime, timedelta

class LiveOracleAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.headers = {'X-Auth-Token': self.api_key} if self.api_key and self.api_key != "DEMO" else {}

    def get_upcoming_matches(self, days_ahead=10):
        """Holt echte kommende Länderspiele der nächsten X Tage."""
        if not self.headers:
            print("[API] ⚠️ Kein API-Key hinterlegt. Bitte trage deinen Key in der app.py ein.")
            return []

        try:
            # Wir definieren das Zeitfenster (Heute bis in 14 Tagen)
            date_from = datetime.now().strftime("%Y-%m-%d")
            date_to = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            # API Anfrage mit Zeitfilter
            url = f"https://api.football-data.org/v4/matches?dateFrom={date_from}&dateTo={date_to}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"[API ERROR] {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            matches = []
            
            for match in data.get('matches', []):
                comp_type = match['competition'].get('type', '')
                comp_name = match['competition'].get('name', '')
                
                # Filtern nach echten Länderspielen (WM, EM, Nations League, Copa etc.)
                if comp_type == 'CUP' or 'INTERNATIONAL' in comp_type or 'World Cup' in comp_name or 'Euro' in comp_name:
                    matches.append({
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'date': match['utcDate'][:10],
                        'competition': comp_name
                    })
            
            if not matches:
                print(f"[API INFO] Keine internationalen Spiele in den nächsten {days_ahead} Tagen gefunden.")
            else:
                print(f"[API] Erfolgreich {len(matches)} kommende Spiele gefunden!")
                
            return matches[:15] # Maximal 15 Spiele zurückgeben, um das UI nicht zu sprengen
            
        except Exception as e:
            print(f"[API ERROR] {e}")
            return []