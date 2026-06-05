import pandas as pd
import numpy as np
from scipy.stats import poisson
import os

class DataLoader:
    def load_data(self, start_year=2000, file_path="data/international_matches.csv"):
        print(f"[Data] Lade historische Spiele aus {file_path} (ab Jahr {start_year})...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year >= start_year].copy()

    def load_fifa_ratings(self, file_path="data/fifa_players.csv"):
        """Lädt und aggregiert die granularen FIFA-Ratings pro Nation"""
        # OOM SCHUTZ & DUPLIKAT-FILTER
        if "fifa_players.csv" in file_path and os.path.exists("data/fifa_players_cloud.csv"):
            print("🛡️ [Data-Shield] 1.2 GB Monster erkannt! Leite auf deduplizierte 'fifa_players_cloud.csv' um...")
            file_path = "data/fifa_players_cloud.csv"
            
        print(f"[Data] Lade granulare Spieler-Qualitäten (FIFA Ratings) aus {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # Fallback für verschiedene Spaltennamen
        if 'nationality' in df.columns and 'nationality_name' not in df.columns:
            df.rename(columns={'nationality': 'nationality_name'}, inplace=True)
            
        nation_stats = {}
        for nation, group in df.groupby('nationality_name'):
            top = group.sort_values('overall', ascending=False).head(23)
            
            att = top['overall'].mean() + (top['shooting'].mean() * 0.15)
            mid = top['overall'].mean() + (top['passing'].mean() * 0.15)
            dfn = top['overall'].mean() + (top['defending'].mean() * 0.15)
            
            nation_stats[nation] = {
                'ATT': round(att, 2) if pd.notna(att) else 75.0,
                'MID': round(mid, 2) if pd.notna(mid) else 75.0,
                'DEF': round(dfn, 2) if pd.notna(dfn) else 75.0
            }
            
        print(f"[Data] Erfolgreich granulare Kader-Ratings für {len(nation_stats)} Nationen berechnet.")
        return nation_stats

class EloSystem:
    def __init__(self, k=20):
        self.ratings = {}
        self.k = k

    def get_rating(self, team):
        return self.ratings.get(team, 1500.0)

    def update(self, team_h, team_a, goals_h, goals_a, is_neutral=True):
        r_h = self.get_rating(team_h)
        r_a = self.get_rating(team_a)
        
        r_h_adj = r_h + (0 if is_neutral else 100)
        
        e_h = 1 / (1 + 10 ** ((r_a - r_h_adj) / 400))
        e_a = 1 / (1 + 10 ** ((r_h_adj - r_a) / 400))
        
        if goals_h > goals_a: s_h, s_a = 1.0, 0.0
        elif goals_h < goals_a: s_h, s_a = 0.0, 1.0
        else: s_h, s_a = 0.5, 0.5
            
        mov = abs(goals_h - goals_a)
        g_mult = np.log(mov + 1) if mov > 0 else 1.0
        
        self.ratings[team_h] = r_h + self.k * g_mult * (s_h - e_h)
        self.ratings[team_a] = r_a + self.k * g_mult * (s_a - e_a)

    def calculate_historical_elo(self, df):
        """Berechnet den Pre-Match Elo-Wert für den wasserdichten Backtest"""
        print("[Features] Berechne dynamische Elo-Ratings für alle Teams...")
        df = df.sort_values('date').reset_index(drop=True)
        elo_h_list = []
        elo_a_list = []

        for _, row in df.iterrows():
            team_h = row['home_team']
            team_a = row['away_team']

            elo_h_list.append(self.get_rating(team_h))
            elo_a_list.append(self.get_rating(team_a))

            self.update(team_h, team_a, row['home_score'], row['away_score'], row['neutral'])

        df['elo_h'] = elo_h_list
        df['elo_a'] = elo_a_list
        print("[Features] Elo-Ratings erfolgreich berechnet.")
        return df

class PoissonEngine:
    def __init__(self):
        self.max_goals = 7

    def predict_match_probabilities(self, team_h, team_a, is_neutral, elo_diff, att_diff, def_diff):
        base_hg = 1.3
        base_ag = 1.1 if not is_neutral else 1.3
        
        lambda_h = base_hg + (elo_diff * 0.001) + (att_diff * 0.015)
        lambda_a = base_ag - (elo_diff * 0.001) - (def_diff * 0.015)
        
        lambda_h = max(0.1, lambda_h)
        lambda_a = max(0.1, lambda_a)
        
        matrix = np.zeros((self.max_goals, self.max_goals))
        for i in range(self.max_goals):
            for j in range(self.max_goals):
                matrix[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
        
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        
        total = home_win + draw + away_win
        
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total,
            'matrix': matrix / total
        }

    def get_smart_score(self, matrix, probs_ml=None):
        flat_idx = np.argmax(matrix)
        goals_h, goals_a = np.unravel_index(flat_idx, matrix.shape)
        return int(goals_h), int(goals_a)

class Backtester:
    def __init__(self):
        self.FALLBACK_RATING = {'ATT': 75.0, 'MID': 75.0, 'DEF': 75.0}
        self.loader = DataLoader()
        self.elo = EloSystem()
        self.poisson = PoissonEngine()
        self.continent_map = {}

    def _get_continent(self, team):
        return self.continent_map.get(team, "Europe")

    def _generate_historical_features(self, df, fifa_dict):
        print("[Pipeline] Mapping globaler Kontinental-Strukturen...")
        df = df.sort_values('date').reset_index(drop=True)
        
        # Pre-Mapping aller Kontinente zur Vermeidung von Cold-Starts
        for _, row in df.iterrows():
            t = str(row.get('tournament', ''))
            h, a = row['home_team'], row['away_team']
            if 'Euro' in t: self.continent_map[h], self.continent_map[a] = 'Europe', 'Europe'
            elif 'Copa' in t: self.continent_map[h], self.continent_map[a] = 'South America', 'South America'
            elif 'African' in t or 'Africa' in t: self.continent_map[h], self.continent_map[a] = 'Africa', 'Africa'
            elif 'Asian' in t or 'AFC' in t: self.continent_map[h], self.continent_map[a] = 'Asia', 'Asia'
            elif 'CONCACAF' in t or 'Gold Cup' in t: self.continent_map[h], self.continent_map[a] = 'North America', 'North America'

        print("[Pipeline] Berechne mathematische Feature-Vektoren (Poisson, Form, Kontinent-Vorteil)...")
        features = []
        team_form_tracker = {}

        for _, row in df.iterrows():
            team_h, team_a = row['home_team'], row['away_team']
            goals_h, goals_a = row['home_score'], row['away_score']
            is_neutral = int(row['neutral'])
            elo_h, elo_a = row['elo_h'], row['elo_a']
            
            stats_h = fifa_dict.get(team_h, self.FALLBACK_RATING)
            stats_a = fifa_dict.get(team_a, self.FALLBACK_RATING)
            
            elo_diff = elo_h - elo_a
            att_diff = stats_h['ATT'] - stats_a['ATT']
            mid_diff = stats_h['MID'] - stats_a['MID']
            def_diff = stats_h['DEF'] - stats_a['DEF']
            
            # 1. Feature: Projections Vector (Poisson Difference)
            probs_p = self.poisson.predict_match_probabilities(team_h, team_a, bool(is_neutral), elo_diff, att_diff, def_diff)
            poisson_diff = probs_p['home_win'] - probs_p['away_win']
            
            # 2. Feature: Rollierendes Form-Delta (Letzte 5 Spiele, normiert auf [-1, 1])
            form_h = np.mean(team_form_tracker[team_h]) if team_h in team_form_tracker and team_form_tracker[team_h] else 0.0
            form_a = np.mean(team_form_tracker[team_a]) if team_a in team_form_tracker and team_form_tracker[team_a] else 0.0
            form_diff = form_h - form_a
            
            # 3. Feature: Continent Advantage Difference
            t_name = str(row['tournament'])
            if 'Euro' in t_name: t_cont = 'Europe'
            elif 'Copa' in t_name: t_cont = 'South America'
            elif 'African' in t_name or 'Africa' in t_name: t_cont = 'Africa'
            elif 'Asian' in t_name or 'AFC' in t_name: t_cont = 'Asia'
            elif 'Gold Cup' in t_name or 'CONCACAF' in t_name: t_cont = 'North America'
            else: t_cont = self._get_continent(team_h) if not is_neutral else 'Other'
                
            adv_h = 1 if self._get_continent(team_h) == t_cont else 0
            adv_a = 1 if self._get_continent(team_a) == t_cont else 0
            continent_adv_diff = adv_h - adv_a
            
            if goals_h > goals_a: target = 2
            elif goals_h == goals_a: target = 1
            else: target = 0
                
            features.append({
                'date': row['date'],
                'tournament': row['tournament'],
                'home_team': team_h,
                'away_team': team_a,
                'is_neutral': is_neutral,
                'elo_h': elo_h,
                'elo_a': elo_a,
                'elo_diff': elo_diff,
                'att_diff': att_diff,
                'mid_diff': mid_diff,
                'def_diff': def_diff,
                'poisson_diff': poisson_diff,
                'form_diff': form_diff,
                'continent_adv_diff': continent_adv_diff,
                'outcome': target  # <--- HIER IST DER FIX!
            })
            
            # ZUKUNFTS-UPDATE
            res_h = 1.0 if goals_h > goals_a else (-1.0 if goals_h < goals_a else 0.0)
            res_a = -res_h
            if team_h not in team_form_tracker: team_form_tracker[team_h] = []
            if team_a not in team_form_tracker: team_form_tracker[team_a] = []
            team_form_tracker[team_h].append(res_h)
            team_form_tracker[team_a].append(res_a)
            if len(team_form_tracker[team_h]) > 5: team_form_tracker[team_h].pop(0)
            if len(team_form_tracker[team_a]) > 5: team_form_tracker[team_a].pop(0)
            
        df_features = pd.DataFrame(features)
        
        # Filter auf Major-Turniere zur finalen Extraktion
        major_tournaments = [
            'FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations',
            'CONCACAF Championship', 'Gold Cup', 'AFC Asian Cup'
        ]
        df_tradeable = df_features[df_features['tournament'].isin(major_tournaments)].copy()
        print(f"[Pipeline] Filter aktiv: {len(df_tradeable)} handelbare Major-Turnierspiele extrahiert.")
        
        return df_tradeable
    def stresstest(self, model, df_features, initial_bankroll=10000.0, num_simulations=1000):
        print("\n" + "="*60)
        print("🎲 INITIATING MONTE CARLO RISK SIMULATION (STRESS TEST)")
        print("="*60)
        print(f"[Risk] Analysiere Trades mit {num_simulations} permutierten Pfaden...")
        
        trades = []
        
        # 1. Signale und Edges extrahieren
        for idx, row in df_features.iterrows():
            # Wir nutzen exakt die Schnittstelle, die dein Modell versteht!
            try:
                if hasattr(model, 'predict_probabilities'):
                    probs = model.predict_probabilities(
                        row['elo_diff'], 
                        row['poisson_diff'], 
                        row['form_diff'], 
                        row['continent_adv_diff'], 
                        row['att_diff'], 
                        row['mid_diff'], 
                        row['def_diff']
                    )
                    prob_a = probs['away_win']
                    prob_d = probs['draw']
                    prob_h = probs['home_win']
                elif hasattr(model, 'predict_proba'):
                    # Fallback für nackte scikit-learn Modelle
                    cols = ['elo_diff', 'poisson_diff', 'form_diff', 'continent_adv_diff', 'att_diff', 'mid_diff', 'def_diff']
                    X = row[cols].to_frame().T
                    p = model.predict_proba(X)[0]
                    prob_a, prob_d, prob_h = p[0], p[1], p[2]
                else:
                    print("⚠️ Kritisches Problem: Modell-Schnittstelle nicht erkannt.")
                    return
            except Exception as e:
                print(f"❌ Fehler bei Prediction (Spiel {idx}): {e}")
                continue

            # Approximierte echte Buchmacher-Wahrscheinlichkeit (aus Elo)
            elo_win_prob = 1 / (1 + 10 ** ((-row['elo_diff']) / 400))
            elo_loss_prob = 1 / (1 + 10 ** ((row['elo_diff']) / 400))
            sum_p = elo_win_prob + 0.25 + elo_loss_prob
            
            # Quoten mit simulierter 5% Buchmacher-Marge (Vig)
            odds_h = 1 / ((elo_win_prob / sum_p) * 1.05)
            odds_a = 1 / ((elo_loss_prob / sum_p) * 1.05)
            
            ai_pred = np.argmax([prob_a, prob_d, prob_h])
            outcome = row['outcome']
            
            # Signal Home (Modell ist sicher UND sieht positiven Erwartungswert gegen den Markt)
            if ai_pred == 2 and prob_h > 0.50: 
                edge = prob_h - (1/odds_h)
                if edge > 0:
                    b = odds_h - 1
                    q = 1 - prob_h
                    k_fraction = ((prob_h * b - q) / b) * 0.25 # Quarter-Kelly (gedrosseltes Risiko)
                    k_fraction = min(max(k_fraction, 0), 0.03) # Hartes Cap: Max 3% Bankroll pro Trade
                    won = (outcome == 2)
                    trades.append({'stake_pct': k_fraction, 'odds': odds_h, 'won': won})
                    
            # Signal Away
            elif ai_pred == 0 and prob_a > 0.50: 
                edge = prob_a - (1/odds_a)
                if edge > 0:
                    b = odds_a - 1
                    q = 1 - prob_a
                    k_fraction = ((prob_a * b - q) / b) * 0.25 # Quarter-Kelly (gedrosseltes Risiko)
                    k_fraction = min(max(k_fraction, 0), 0.03) # Hartes Cap: Max 3% Bankroll pro Trade
                    won = (outcome == 0)
                    trades.append({'stake_pct': k_fraction, 'odds': odds_a, 'won': won})

        if not trades:
            print("[Risk] ⚠️ Keine profitablen Edges gefunden. Das Modell ist aktuell zu konservativ.")
            return
            
        print(f"[Risk] {len(trades)} profitable Handelssignale im Datensatz identifiziert.")
        print("[Risk] Starte Permutationen...")
        
        # 2. Monte Carlo Engine
        results = []
        max_drawdowns = []
        bankruptcies = 0
        
        for i in range(num_simulations):
            np.random.shuffle(trades) 
            current_br = initial_bankroll
            peak_br = initial_bankroll
            max_dd = 0.0
            
            for t in trades:
                stake = current_br * t['stake_pct']
                if t['won']:
                    current_br += stake * (t['odds'] - 1)
                else:
                    current_br -= stake
                    
                if current_br > peak_br: peak_br = current_br
                dd = (peak_br - current_br) / peak_br
                if dd > max_dd: max_dd = dd
                
                # Bankrottgrenze bei 90% Verlust des Startkapitals
                if current_br < initial_bankroll * 0.1: 
                    bankruptcies += 1
                    break
                    
            results.append(current_br)
            max_drawdowns.append(max_dd)
            
        avg_br = np.mean(results)
        avg_dd = np.mean(max_drawdowns)
        max_dd_overall = np.max(max_drawdowns)
        risk_of_ruin = (bankruptcies / num_simulations) * 100
        roi = ((avg_br - initial_bankroll) / initial_bankroll) * 100
        
        print("\n📊 MONTE CARLO RESULTATE (1.000 Pfade simuliert)")
        print(f"➜ Startkapital:        {initial_bankroll:,.2f} €")
        print(f"➜ Erwartungswert:      {avg_br:,.2f} € (ROI: {roi:+.1f}%)")
        print(f"➜ Durchschn. Drawdown: -{avg_dd*100:.1f}%")
        print(f"➜ Worst-Case Drawdown: -{max_dd_overall*100:.1f}%")
        print(f"➜ Risk of Ruin:        {risk_of_ruin:.2f}% (Gefahr des Totalverlusts)")
        print("============================================================\n")
        
        return {'roi': roi, 'max_drawdown': max_dd_overall, 'risk_of_ruin': risk_of_ruin}