import pandas as pd
import numpy as np
from scipy.stats import poisson
from collections import deque
import os

class DataLoader:
    def load_data(self, start_year=2000, file_path="data/international_matches.csv"):
        print(f"[Data] Lade historische Spiele aus {file_path} (ab Jahr {start_year})...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['home_score', 'away_score'])  # guard against NaN scores
        return df[df['date'].dt.year >= start_year].copy()

    def load_fifa_ratings(self, file_path="data/fifa_players.csv"):
        if "fifa_players.csv" in file_path and os.path.exists("data/fifa_players_cloud.csv"):
            file_path = "data/fifa_players_cloud.csv"

        print(f"[Data] Lade granulare Spieler-Qualitäten (FIFA Ratings) aus {file_path}...")
        cols = ['nationality_name', 'nationality', 'overall', 'shooting', 'passing', 'defending']
        df = pd.read_csv(file_path, usecols=lambda c: c in cols, low_memory=False)

        if 'nationality' in df.columns and 'nationality_name' not in df.columns:
            df.rename(columns={'nationality': 'nationality_name'}, inplace=True)

        # Sort once globally then use groupby — avoids per-group sort
        df = df.sort_values('overall', ascending=False)

        nation_stats = {}
        for nation, group in df.groupby('nationality_name', sort=False):
            top = group.head(23)
            overall_m = top['overall'].mean()
            att = overall_m + top['shooting'].mean() * 0.15
            mid = overall_m + top['passing'].mean() * 0.15
            dfn = overall_m + top['defending'].mean() * 0.15
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

        for row in df.itertuples(index=False):
            elo_h_list.append(self.get_rating(row.home_team))
            elo_a_list.append(self.get_rating(row.away_team))
            self.update(row.home_team, row.away_team, row.home_score, row.away_score, row.neutral)

        df['elo_h'] = elo_h_list
        df['elo_a'] = elo_a_list
        return df

class PoissonEngine:
    def __init__(self):
        self.max_goals = 10  # Erhöht auf 10 für extreme Mismatches (Tail-Risk Protection)
        # Goal-scoring parameters — starting defaults, overwritten by calibrate()
        self.base_hg         = 1.35
        self.base_ag         = 1.10   # non-neutral venue (away disadvantage)
        self.base_ag_neutral = 1.35   # neutral venue
        self.elo_coeff       = 0.0015
        self.att_def_coeff   = 0.010

    def calibrate(self, df):
        """Fit Poisson goal-scoring parameters via Maximum Likelihood Estimation.

        Uses all historical games with known scores. With ~7000+ games and only
        5 parameters this is massively overidentified — no overfitting is possible.

        Required df columns: home_score, away_score, elo_diff,
                             att_def_h, att_def_a, is_neutral
        """
        from scipy.optimize import minimize

        sub = df.dropna(subset=['home_score', 'away_score']).copy()
        sub = sub[(sub['home_score'] >= 0) & (sub['home_score'] <= 15) &
                  (sub['away_score'] >= 0) & (sub['away_score'] <= 15)]

        gh  = sub['home_score'].values.astype(int)
        ga  = sub['away_score'].values.astype(int)
        ed  = sub['elo_diff'].values.astype(float)
        adh = sub['att_def_h'].values.astype(float)
        ada = sub['att_def_a'].values.astype(float)
        neu = sub['is_neutral'].values.astype(float)

        def neg_ll(params):
            bh, ba, bn, ec, ac = params
            base_ag_v = np.where(neu, bn, ba)
            lh = np.maximum(0.1, bh + ec * ed  + ac * adh)
            la = np.maximum(0.1, base_ag_v - ec * ed + ac * ada)
            ll = (poisson.logpmf(gh, lh) + poisson.logpmf(ga, la)).sum()
            return -ll

        x0     = [self.base_hg, self.base_ag, self.base_ag_neutral,
                  self.elo_coeff, self.att_def_coeff]
        bounds = [(0.50, 3.0), (0.30, 2.5), (0.50, 2.5),
                  (0.0002, 0.006), (0.001, 0.050)]

        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-14})

        if result.fun < neg_ll(x0):   # accept only if it actually improved
            bh, ba, bn, ec, ac = result.x
            self.base_hg, self.base_ag, self.base_ag_neutral = bh, ba, bn
            self.elo_coeff, self.att_def_coeff = ec, ac
            print(f"[Poisson] MLE abgeschlossen: base_h={bh:.4f} base_a={ba:.4f} "
                  f"base_n={bn:.4f} elo={ec:.5f} att_def={ac:.4f}")
        else:
            print(f"[Poisson] MLE nicht konvergiert — behalte Start-Parameter.")

    def predict_match_probabilities(self, team_h, team_a, is_neutral, elo_diff,
                                    att_def_h, att_def_a):
        """
        att_def_h = home_ATT - away_DEF  (home scorer vs away goalkeeper/defence)
        att_def_a = away_ATT - home_DEF  (away scorer vs home goalkeeper/defence)
        Parameters are learned by MLE in calibrate(); stored as instance attributes.
        """
        base_ag  = self.base_ag_neutral if is_neutral else self.base_ag
        lambda_h = max(0.1, self.base_hg + (elo_diff * self.elo_coeff) + (att_def_h * self.att_def_coeff))
        lambda_a = max(0.1, base_ag      - (elo_diff * self.elo_coeff) + (att_def_a * self.att_def_coeff))

        goals_range = np.arange(self.max_goals)
        matrix = np.outer(poisson.pmf(goals_range, lambda_h), poisson.pmf(goals_range, lambda_a))

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        total = home_win + draw + away_win
        return {
            'home_win': home_win / total, 'draw': draw / total, 'away_win': away_win / total,
            'matrix': matrix / total, 'lambda_h': lambda_h, 'lambda_a': lambda_a
        }

    def _blend_matrix(self, probs_matrix, probs_ml):
        """Scale each Poisson outcome region by ML outcome probability, then renormalise.

        This lets the ML-predicted outcome weights shift probability mass between
        home-win / draw / away-win cells without distorting the within-region
        Poisson shape, giving a single blended distribution over all exact scores.
        """
        matrix = probs_matrix.copy()
        n = matrix.shape[0]
        rows, cols = np.indices((n, n))

        p_hw = float(np.tril(matrix, -1).sum())
        p_d  = float(np.trace(matrix))
        p_aw = float(np.triu(matrix, 1).sum())

        hw_scale = probs_ml.get('home_win', p_hw) / p_hw if p_hw > 1e-9 else 1.0
        d_scale  = probs_ml.get('draw',     p_d)  / p_d  if p_d  > 1e-9 else 1.0
        aw_scale = probs_ml.get('away_win', p_aw) / p_aw if p_aw > 1e-9 else 1.0

        scale = np.where(rows > cols, hw_scale, np.where(rows == cols, d_scale, aw_scale))
        matrix = matrix * scale
        total = matrix.sum()
        return matrix / total if total > 1e-9 else matrix

    def get_top_scores(self, probs_matrix, probs_ml, top_n=5):
        """Return the top-N most probable exact scorelines from the ML-blended matrix.

        Each entry has keys: 'Ergebnis', 'Wahrscheinlichkeit (%)', 'Faire Quote (Min.)'.
        'Faire Quote (Min.)' is the minimum bookmaker odds needed for a +EV bet on this score.
        """
        matrix = self._blend_matrix(probs_matrix, probs_ml)
        flat = matrix.flatten()
        top_indices = np.argsort(flat)[::-1][:top_n]
        results = []
        for idx in top_indices:
            h, a = np.unravel_index(idx, matrix.shape)
            prob = float(matrix[h, a])
            results.append({
                'Ergebnis':               f"{int(h)}:{int(a)}",
                'Wahrscheinlichkeit (%)': round(prob * 100, 1),
                'Faire Quote (Min.)':     round(1.0 / prob, 2) if prob > 1e-9 else 999.0,
            })
        return results

    def get_smart_score(self, probs_matrix, probs_ml=None):
        """Most probable exact scoreline within the ML-predicted outcome region.

        Two-stage: first picks the ML-predicted outcome (home_win / draw / away_win),
        then finds the highest-probability cell in that region of the Poisson matrix.
        This ensures the Tipp is always consistent with the 1X2 prediction.
        """
        if probs_ml is None:
            idx = np.unravel_index(np.argmax(probs_matrix), probs_matrix.shape)
            return int(idx[0]), int(idx[1])

        best_outcome = max(probs_ml, key=probs_ml.get)

        if best_outcome == 'home_win':
            mask = np.tril(probs_matrix, -1)
            idx = np.unravel_index(np.argmax(mask), mask.shape)
            return int(idx[0]), int(idx[1])
        elif best_outcome == 'away_win':
            mask = np.triu(probs_matrix, 1)
            idx = np.unravel_index(np.argmax(mask), mask.shape)
            return int(idx[0]), int(idx[1])
        else:
            diag = np.diag(probs_matrix)
            best = int(np.argmax(diag))
            return best, best

class Backtester:
    def __init__(self):
        self.FALLBACK_RATING = {'ATT': 75.0, 'MID': 75.0, 'DEF': 75.0}
        self.loader = DataLoader()
        self.elo = EloSystem()
        self.poisson = PoissonEngine()
        # Form states persisted after training so app.py can read real form values.
        self.team_form_state = {}   # team -> list of last ≤5 GD-based form values
        self.team_att_state  = {}   # team -> list of last ≤5 goals-scored values
        self.team_def_state  = {}   # team -> list of last ≤5 goals-conceded values

    def get_form_diffs(self, team_h, team_a):
        """Return all three form differentials using end-of-training tracker state."""
        form_h = self._calc_weighted_form(self.team_form_state.get(team_h, []))
        form_a = self._calc_weighted_form(self.team_form_state.get(team_a, []))
        att_h  = self._calc_weighted_form(self.team_att_state.get(team_h, []))
        att_a  = self._calc_weighted_form(self.team_att_state.get(team_a, []))
        def_h  = self._calc_weighted_form(self.team_def_state.get(team_h, []))
        def_a  = self._calc_weighted_form(self.team_def_state.get(team_a, []))
        return {
            'form_diff':     form_h - form_a,
            'att_form_diff': att_h  - att_a,
            'def_form_diff': def_h  - def_a,
        }

    def _calc_weighted_form(self, form_list):
        if not form_list:
            return 0.0
        weights = np.array([1, 2, 3, 4, 5][-len(form_list):])
        return np.sum(np.array(form_list) * weights) / np.sum(weights)

    def _generate_historical_features(self, df, fifa_dict):
        print("[Pipeline] Berechne mathematische Feature-Vektoren (mit Weighted Form)...")
        df = df.sort_values('date').reset_index(drop=True)
        features = []
        team_form_tracker = {}  # GD-based form [-1, 1]
        team_att_tracker  = {}  # recent goals scored (raw)
        team_def_tracker  = {}  # recent goals conceded (raw)

        # Competitive weight: WC > continental > regional
        _tournament_weights = {
            'FIFA World Cup': 1.0,
            'UEFA Euro': 0.67, 'Copa América': 0.67, 'African Cup of Nations': 0.67,
            'CONCACAF Championship': 0.33, 'Gold Cup': 0.33, 'AFC Asian Cup': 0.33,
        }

        for row in df.itertuples(index=False):
            if pd.isna(row.home_score) or pd.isna(row.away_score):
                continue

            team_h, team_a = row.home_team, row.away_team
            goals_h, goals_a = row.home_score, row.away_score
            is_neutral = int(row.neutral)
            elo_h, elo_a = row.elo_h, row.elo_a

            stats_h = fifa_dict.get(team_h, self.FALLBACK_RATING)
            stats_a = fifa_dict.get(team_a, self.FALLBACK_RATING)

            elo_diff = elo_h - elo_a
            elo_avg  = (elo_h + elo_a) / 2.0
            att_diff = stats_h['ATT'] - stats_a['ATT']
            mid_diff = stats_h['MID'] - stats_a['MID']
            def_diff = stats_h['DEF'] - stats_a['DEF']

            # Cross-paired: home scorer vs away defence; away scorer vs home defence.
            # Fixes the audit flaw where away ATT was incorrectly suppressing home goals.
            att_def_h = stats_h['ATT'] - stats_a['DEF']
            att_def_a = stats_a['ATT'] - stats_h['DEF']

            probs_p = self.poisson.predict_match_probabilities(
                team_h, team_a, bool(is_neutral), elo_diff, att_def_h, att_def_a
            )
            poisson_diff = probs_p['home_win'] - probs_p['away_win']

            for t in (team_h, team_a):
                if t not in team_form_tracker: team_form_tracker[t] = deque(maxlen=5)
                if t not in team_att_tracker:  team_att_tracker[t]  = deque(maxlen=5)
                if t not in team_def_tracker:  team_def_tracker[t]  = deque(maxlen=5)

            form_h = self._calc_weighted_form(list(team_form_tracker[team_h]))
            form_a = self._calc_weighted_form(list(team_form_tracker[team_a]))
            form_diff = form_h - form_a

            # Separate attack/defence form: scoring rate and conceding rate tracked independently.
            # A team winning 4-3 has great att_form but poor def_form — GD alone misses this.
            att_form_h = self._calc_weighted_form(list(team_att_tracker[team_h]))
            att_form_a = self._calc_weighted_form(list(team_att_tracker[team_a]))
            def_form_h = self._calc_weighted_form(list(team_def_tracker[team_h]))
            def_form_a = self._calc_weighted_form(list(team_def_tracker[team_a]))

            if goals_h > goals_a: target = 2
            elif goals_h == goals_a: target = 1
            else: target = 0

            features.append({
                'date': row.date,
                'tournament': row.tournament,
                'home_team': team_h,
                'away_team': team_a,
                'home_score': goals_h,
                'away_score': goals_a,
                'is_neutral': is_neutral,
                'elo_h': elo_h,
                'elo_a': elo_a,
                'elo_diff': elo_diff,
                'elo_avg': elo_avg,
                'att_diff': att_diff,
                'mid_diff': mid_diff,
                'def_diff': def_diff,
                'att_def_h': att_def_h,   # stored for stresstest Poisson calls
                'att_def_a': att_def_a,
                'poisson_diff': poisson_diff,
                'form_diff': form_diff,
                'att_form_diff': att_form_h - att_form_a,
                'def_form_diff': def_form_h - def_form_a,
                'tournament_weight': _tournament_weights.get(row.tournament, 0.5),
                'outcome': target
            })

            # Update all trackers AFTER recording features (no leakage)
            gd = float(goals_h - goals_a)
            res_h = float(np.clip(gd, -3, 3)) / 3.0
            res_a = float(np.clip(-gd, -3, 3)) / 3.0
            team_form_tracker[team_h].append(res_h)
            team_form_tracker[team_a].append(res_a)
            team_att_tracker[team_h].append(float(goals_h))
            team_att_tracker[team_a].append(float(goals_a))
            team_def_tracker[team_h].append(float(goals_a))   # home conceded
            team_def_tracker[team_a].append(float(goals_h))   # away conceded

        # Persist final tracker state into the Backtester so it survives pickling.
        # App.py reads these to compute real form values instead of hardcoding 0.0.
        self.team_form_state = {t: list(v) for t, v in team_form_tracker.items()}
        self.team_att_state  = {t: list(v) for t, v in team_att_tracker.items()}
        self.team_def_state  = {t: list(v) for t, v in team_def_tracker.items()}

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

        # Overall outcome accuracy tracked for ALL predictions, not just trades
        correct_outcomes = 0
        total_outcomes = 0

        for row in df_features.itertuples(index=False):
            try:
                probs = model.predict_probabilities(
                    row.elo_diff, row.elo_avg, row.poisson_diff, row.form_diff,
                    row.att_form_diff, row.def_form_diff,
                    row.att_diff, row.mid_diff, row.def_diff,
                    int(row.is_neutral), row.tournament_weight
                )
                prob_a, prob_d, prob_h = probs['away_win'], probs['draw'], probs['home_win']
            except Exception:
                continue

            ai_pred = np.argmax([prob_a, prob_d, prob_h])
            outcome = row.outcome

            total_outcomes += 1
            if ai_pred == outcome:
                correct_outcomes += 1

            base_h = 1 / (1 + 10 ** ((-row.elo_diff) / 400))
            base_a = 1 / (1 + 10 ** ((row.elo_diff) / 400))
            base_d = max(0.15, 1 - base_h - base_a)

            sharp_h = max(0.05, min(0.95, base_h + (row.poisson_diff * 0.15) + (row.form_diff * 0.08)))
            sharp_a = max(0.05, min(0.95, base_a - (row.poisson_diff * 0.15) - (row.form_diff * 0.08)))

            sum_p = sharp_h + base_d + sharp_a
            norm_h = sharp_h / sum_p
            norm_a = sharp_a / sum_p
            norm_d = base_d / sum_p

            pinnacle_vig = 1.035
            odds_h = 1 / (norm_h * pinnacle_vig)
            odds_a = 1 / (norm_a * pinnacle_vig)
            odds_d = 1 / (norm_d * pinnacle_vig)

            is_trade = False
            chosen_odds = 0.0
            is_win = False

            if ai_pred == 2 and prob_h > 0.50 and (prob_h - (1 / odds_h)) > 0:
                k_fraction = min(max(((prob_h * (odds_h - 1) - (1 - prob_h)) / (odds_h - 1)) * 0.25, 0), 0.03)
                chosen_odds = odds_h
                is_win = (outcome == 2)
                is_trade = True
            elif ai_pred == 0 and prob_a > 0.50 and (prob_a - (1 / odds_a)) > 0:
                k_fraction = min(max(((prob_a * (odds_a - 1) - (1 - prob_a)) / (odds_a - 1)) * 0.25, 0), 0.03)
                chosen_odds = odds_a
                is_win = (outcome == 0)
                is_trade = True
            elif ai_pred == 1 and prob_d > 0.40 and (prob_d - (1 / odds_d)) > 0:
                # Draw trading: higher threshold (0.40 vs 0.50) reflects greater uncertainty.
                # Validated here for the first time — closes the audit gap with render_kelly_advisor.
                k_fraction = min(max(((prob_d * (odds_d - 1) - (1 - prob_d)) / (odds_d - 1)) * 0.25, 0), 0.03)
                chosen_odds = odds_d
                is_win = (outcome == 1)
                is_trade = True

            if is_trade:
                total_traded_matches += 1
                trades.append({'stake_pct': k_fraction, 'odds': chosen_odds, 'won': is_win})

                probs_p = self.poisson.predict_match_probabilities(
                    row.home_team, row.away_team, bool(row.is_neutral),
                    row.elo_diff, row.att_def_h, row.att_def_a
                )
                pred_h, pred_a = self.poisson.get_smart_score(
                    probs_p['matrix'], {'home_win': prob_h, 'draw': prob_d, 'away_win': prob_a}
                )
                actual_h, actual_a = int(row.home_score), int(row.away_score)

                if pred_h == actual_h and pred_a == actual_a:
                    exact_hits += 1
                elif (abs(pred_h - actual_h) + abs(pred_a - actual_a)) == 1:
                    near_misses += 1

        # Always print outcome accuracy, regardless of whether trades were found
        overall_acc = (correct_outcomes / total_outcomes * 100) if total_outcomes > 0 else 0
        print(f"\n{'='*60}")
        print(f"📊 KI-TREFFERQUOTE ({total_outcomes} Spiele analysiert)")
        print(f"{'='*60}")
        print(f"🎯 Korrekte 1X2-Prognosen: {correct_outcomes}/{total_outcomes} ({overall_acc:.1f}%)")
        print(f"   ⚠️  In-Sample (Trainingsdaten) — Walk-Forward CV Out-of-Sample: ~50%")
        print(f"   (Zufalls-Baseline wäre ~33,3% bei 3 Klassen)")

        if not trades:
            print("\n[Risk] ⚠️ Keine Edges gegen Pinnacle-Proxy gefunden — Kein Financial Report.")
            return {'accuracy': overall_acc}

        print(f"\n{'='*60}")
        print(f"⚽ ERGEBNIS-PRÄZISION ({total_traded_matches} gesetzte Spiele)")
        print(f"{'='*60}")
        hit_rate_exact = (exact_hits / total_traded_matches) * 100
        hit_rate_near = (near_misses / total_traded_matches) * 100
        print(f"🎯 Exakte Ergebnistreffer:   {exact_hits} ({hit_rate_exact:.1f}%)")
        print(f"🥈 Nur 1 Tor daneben:        {near_misses} ({hit_rate_near:.1f}%)")
        print(f"📈 Summe (Präzisions-Fokus): {exact_hits + near_misses} ({hit_rate_exact + hit_rate_near:.1f}%)")

        if num_simulations <= 0:
            print("\n[Risk] num_simulations muss > 0 sein.")
            return {'accuracy': overall_acc}

        results, max_drawdowns, bankruptcies = [], [], 0
        for _ in range(num_simulations):
            np.random.shuffle(trades)
            current_br, peak_br, max_dd = initial_bankroll, initial_bankroll, 0.0
            went_bust = False
            for t in trades:
                stake = current_br * t['stake_pct']
                if t['won']: current_br += stake * (t['odds'] - 1)
                else: current_br -= stake
                if current_br > peak_br: peak_br = current_br
                if peak_br > 0:
                    dd = (peak_br - current_br) / peak_br
                    if dd > max_dd: max_dd = dd
                if current_br <= 0 or current_br < initial_bankroll * 0.1:
                    bankruptcies += 1
                    went_bust = True
                    break
            results.append(current_br if not went_bust else 0.0)
            max_drawdowns.append(max_dd)

        avg_br = np.mean(results)
        roi = ((avg_br - initial_bankroll) / initial_bankroll) * 100

        print(f"\n{'='*60}")
        print(f"📈 FINANCIAL PERFORMANCE ({num_simulations} Monte-Carlo Simulationen)")
        print(f"{'='*60}")
        print(f"➜ Erwartungswert: {avg_br:,.2f} € (ROI: {roi:+.1f}%)")
        print(f"➜ Worst-Case Drawdown: -{np.max(max_drawdowns)*100:.1f}%")
        print(f"➜ Insolvenz-Risiko: {(bankruptcies / num_simulations)*100:.1f}%")

        return {
            'accuracy': overall_acc,
            'roi': roi,
            'max_drawdown': np.max(max_drawdowns),
            'risk_of_ruin': (bankruptcies / num_simulations) * 100
        }