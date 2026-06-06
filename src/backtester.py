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
        if "fifa_players.csv" in file_path and os.path.exists("data/fifa_players_cloud.csv"):
            file_path = "data/fifa_players_cloud.csv"
            
        print(f"[Data] Lade granulare Spieler-Qualitäten (FIFA Ratings) aus {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        
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
        df = df.sort_values('date').reset_index(drop=True)
        elo_h_list, elo_a_list = [], []

        for _, row in df.iterrows():
            team_h, team_a = row['home_team'], row['away_team']
            elo_h_list.append(self.get_rating(team_h))
            elo_a_list.append(self.get_rating(team_a))
            self.update(team_h, team_a, row['home_score'], row['away_score'], row['neutral'])

        df['elo_h'] = elo_h_list
        df['elo_a'] = elo_a_list
        return df

class PoissonEngine:
    def __init__(self):
        self.max_goals = 10  # Erhöht auf 10 für extreme Mismatches (Tail-Risk Protection)

    def predict_match_probabilities(self, team_h, team_a, is_neutral, elo_diff, att_diff, def_diff):
        base_hg = 1.3
        base_ag = 1.1 if not is_neutral else 1.3
        
        lambda_h = max(0.1, base_hg + (elo_diff * 0.001) + (att_diff * 0.015))
        lambda_a = max(0.1, base_ag - (elo_diff * 0.001) - (def_diff * 0.015))
        
        matrix = np.zeros((self.max_goals, self.max_goals))
        for i in range(self.max_goals):
            for j in range(self.max_goals):
                matrix[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
        
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total, 'draw': draw / total, 'away_win': away_win / total, 'matrix': matrix / total
        }

    def get_smart_score(self, probs_matrix, probs_ml=None):
        """
        Berechnet das projizierte Ergebnis basierend auf dem Erwartungswert (xG).
        Anstatt das höchste Einzel-Ergebnis (Modus) zu nehmen, werden die xG
        mathematisch exakt aus der Matrix extrahiert und kaufmännisch gerundet.
        """
        xg_h = 0.0
        xg_a = 0.0
        
        # Iteriere durch die Matrix (meist 6x6 Tore)
        for i in range(probs_matrix.shape[0]):
            for j in range(probs_matrix.shape[1]):
                prob = probs_matrix[i, j]
                xg_h += i * prob  # Tore Team H * Wahrscheinlichkeit
                xg_a += j * prob  # Tore Team A * Wahrscheinlichkeit
                
        # Runden auf das logische, menschenlesbare Endergebnis
        proj_goals_h = int(round(xg_h))
        proj_goals_a = int(round(xg_a))
        
        return proj_goals_h, proj_goals_a

class Backtester:
    def __init__(self):
        self.FALLBACK_RATING = {'ATT': 75.0, 'MID': 75.0, 'DEF': 75.0}
        self.loader = DataLoader()
        self.elo = EloSystem()
        self.poisson = PoissonEngine()

    def _calc_weighted_form(self, form_list):
        if not form_list:
            return 0.0
        weights = np.array([1, 2, 3, 4, 5][-len(form_list):])
        return np.sum(np.array(form_list) * weights) / np.sum(weights)

    def _generate_historical_features(self, df, fifa_dict):
        print("[Pipeline] Berechne mathematische Feature-Vektoren (mit Weighted Form)...")
        df = df.sort_values('date').reset_index(drop=True)
        features = []
        team_form_tracker = {}

        for _, row in df.iterrows():
            # --- NEUER FIX: Ignoriere Spiele ohne echtes Ergebnis (Zukunft oder Abbruch) ---
            if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                continue
                
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
            
            probs_p = self.poisson.predict_match_probabilities(team_h, team_a, bool(is_neutral), elo_diff, att_diff, def_diff)
            poisson_diff = probs_p['home_win'] - probs_p['away_win']
            
            form_h = self._calc_weighted_form(team_form_tracker.get(team_h, []))
            form_a = self._calc_weighted_form(team_form_tracker.get(team_a, []))
            form_diff = form_h - form_a
            
            if goals_h > goals_a: target = 2
            elif goals_h == goals_a: target = 1
            else: target = 0
                
            features.append({
                'date': row['date'],
                'tournament': row['tournament'],
                'home_team': team_h,
                'away_team': team_a,
                'home_score': goals_h,  # <--- Echtes Ergebnis gesichert
                'away_score': goals_a,  # <--- Echtes Ergebnis gesichert
                'is_neutral': is_neutral,
                'elo_h': elo_h,
                'elo_a': elo_a,
                'elo_diff': elo_diff,
                'att_diff': att_diff,
                'mid_diff': mid_diff,
                'def_diff': def_diff,
                'poisson_diff': poisson_diff,
                'form_diff': form_diff,
                'outcome': target
            })
            
            res_h = 1.0 if goals_h > goals_a else (-1.0 if goals_h < goals_a else 0.0)
            res_a = -res_h
            if team_h not in team_form_tracker: team_form_tracker[team_h] = []
            if team_a not in team_form_tracker: team_form_tracker[team_a] = []
            team_form_tracker[team_h].append(res_h)
            team_form_tracker[team_a].append(res_a)
            if len(team_form_tracker[team_h]) > 5: team_form_tracker[team_h].pop(0)
            if len(team_form_tracker[team_a]) > 5: team_form_tracker[team_a].pop(0)
            
        df_features = pd.DataFrame(features)
        
        major_tournaments = [
            'FIFA World Cup', 'UEFA Euro', 'Copa América', 'African Cup of Nations',
            'CONCACAF Championship', 'Gold Cup', 'AFC Asian Cup'
        ]
        df_tradeable = df_features[df_features['tournament'].isin(major_tournaments)].copy()
        print(f"[Pipeline] Filter aktiv: {len(df_tradeable)} handelbare Major-Turnierspiele extrahiert.")
        return df_tradeable

    def stresstest(self, model, df_features, initial_bankroll=10000.0, num_simulations=1000):
        print("\n" + "="*60)
        print("🎲 INITIATING MONTE CARLO RISK & ACCURACY AUDIT")
        print("="*60)
        
        trades = []
        exact_hits = 0
        near_misses = 0
        total_traded_matches = 0
        
        for idx, row in df_features.iterrows():
            try:
                if hasattr(model, 'predict_probabilities'):
                    probs = model.predict_probabilities(
                        row['elo_diff'], row['poisson_diff'], row['form_diff'], 
                        row['att_diff'], row['mid_diff'], row['def_diff']
                    )
                    prob_a, prob_d, prob_h = probs['away_win'], probs['draw'], probs['home_win']
                elif hasattr(model, 'predict_proba'):
                    cols = ['elo_diff', 'poisson_diff', 'form_diff', 'att_diff', 'mid_diff', 'def_diff']
                    X = row[cols].to_frame().T
                    p = model.predict_proba(X)[0]
                    prob_a, prob_d, prob_h = p[0], p[1], p[2]
                else:
                    return
            except Exception:
                continue

            base_h = 1 / (1 + 10 ** ((-row['elo_diff']) / 400))
            base_a = 1 / (1 + 10 ** ((row['elo_diff']) / 400))
            base_d = max(0.15, 1 - base_h - base_a)
            
            sharp_h = max(0.05, min(0.95, base_h + (row['poisson_diff'] * 0.15) + (row['form_diff'] * 0.08)))
            sharp_a = max(0.05, min(0.95, base_a - (row['poisson_diff'] * 0.15) - (row['form_diff'] * 0.08)))
            
            sum_p = sharp_h + base_d + sharp_a
            norm_h, norm_a = sharp_h / sum_p, sharp_a / sum_p
            
            pinnacle_vig = 1.035
            odds_h, odds_a = 1 / (norm_h * pinnacle_vig), 1 / (norm_a * pinnacle_vig)
            
            ai_pred = np.argmax([prob_a, prob_d, prob_h])
            outcome = row['outcome']
            
            is_trade = False
            chosen_odds = 0.0
            is_win = False
            
            if ai_pred == 2 and prob_h > 0.50 and (prob_h - (1/odds_h)) > 0:
                k_fraction = min(max(((prob_h * (odds_h - 1) - (1 - prob_h)) / (odds_h - 1)) * 0.25, 0), 0.03)
                chosen_odds = odds_h
                is_win = (outcome == 2)
                is_trade = True
            elif ai_pred == 0 and prob_a > 0.50 and (prob_a - (1/odds_a)) > 0:
                k_fraction = min(max(((prob_a * (odds_a - 1) - (1 - prob_a)) / (odds_a - 1)) * 0.25, 0), 0.03)
                chosen_odds = odds_a
                is_win = (outcome == 0)
                is_trade = True

            if is_trade:
                total_traded_matches += 1
                trades.append({'stake_pct': k_fraction, 'odds': chosen_odds, 'won': is_win})
                
                probs_p = self.poisson.predict_match_probabilities(
                    row['home_team'], row['away_team'], bool(row['is_neutral']), 
                    row['elo_diff'], row['att_diff'], row['def_diff']
                )
                pred_h, pred_a = self.poisson.get_smart_score(probs_p['matrix'], {'home_win': prob_h, 'draw': prob_d, 'away_win': prob_a})
                
                actual_h = int(row['home_score'])
                actual_a = int(row['away_score'])
                
                if pred_h == actual_h and pred_a == actual_a:
                    exact_hits += 1
                elif (abs(pred_h - actual_h) + abs(pred_a - actual_a)) == 1:
                    near_misses += 1

        if not trades:
            print("[Risk] ⚠️ Keine Edges gegen Pinnacle-Proxy gefunden.")
            return
            
        print(f"[Risk] {len(trades)} Trades analysiert.")
        
        results, max_drawdowns, bankruptcies = [], [], 0
        for i in range(num_simulations):
            np.random.shuffle(trades) 
            current_br, peak_br, max_dd = initial_bankroll, initial_bankroll, 0.0
            for t in trades:
                stake = current_br * t['stake_pct']
                if t['won']: current_br += stake * (t['odds'] - 1)
                else: current_br -= stake
                if current_br > peak_br: peak_br = current_br
                dd = (peak_br - current_br) / peak_br
                if dd > max_dd: max_dd = dd
                if current_br < initial_bankroll * 0.1: 
                    bankruptcies += 1
                    break
            results.append(current_br)
            max_drawdowns.append(max_dd)
            
        avg_br = np.mean(results)
        roi = ((avg_br - initial_bankroll) / initial_bankroll) * 100
        
        hit_rate_exact = (exact_hits / total_traded_matches) * 100 if total_traded_matches > 0 else 0
        hit_rate_near = (near_misses / total_traded_matches) * 100 if total_traded_matches > 0 else 0
        
        print(f"\n📊 SCOUTING & ACCURACY AUDIT ({total_traded_matches} gesetzte Spiele)")
        print(f"🎯 Exakte Ergebnistreffer:   {exact_hits} ({hit_rate_exact:.1f}%)")
        print(f"🥈 Nur 1 Tor daneben:        {near_misses} ({hit_rate_near:.1f}%)")
        print(f"📈 Summe (Präzisions-Fokus): {exact_hits + near_misses} ({(hit_rate_exact + hit_rate_near):.1f}%)")
        print("-" * 60)
        print(f"📊 FINANCIAL PERFORMANCE\n➜ Erwartungswert: {avg_br:,.2f} € (ROI: {roi:+.1f}%)\n➜ Worst-Case Drawdown: -{np.max(max_drawdowns)*100:.1f}%\n")
        
        return {'roi': roi, 'max_drawdown': np.max(max_drawdowns), 'risk_of_ruin': (bankruptcies / num_simulations) * 100}