import pandas as pd
from sklearn.metrics import brier_score_loss
from src.data_loader import DataLoader
from src.features import EloRatingSystem
from src.models import DynamicRollingPoisson, MetaMachineLearningModel

class Backtester:
    def __init__(self):
        self.loader = DataLoader(file_path="data/international_matches.csv")
        self.elo = EloRatingSystem()
        self.poisson = DynamicRollingPoisson(window_size=15)
        self.ml_model = None
        
        self.FALLBACK_RATING = {'ATT': 70.0, 'MID': 70.0, 'DEF': 70.0}
        self.CONTINENT_MAP = {
            'Europe': ['France', 'Germany', 'Spain', 'England', 'Portugal', 'Netherlands', 'Belgium', 'Croatia', 'Denmark', 'Switzerland', 'Italy', 'Serbia', 'Poland', 'Wales', 'Sweden', 'Russia'],
            'South America': ['Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Ecuador', 'Peru', 'Chile'],
            'Asia': ['Japan', 'South Korea', 'Qatar', 'Iran', 'Saudi Arabia', 'Australia'],
            'Africa': ['Morocco', 'Senegal', 'Cameroon', 'Ghana', 'Tunisia', 'Egypt', 'Nigeria'],
            'North America': ['USA', 'Mexico', 'Canada', 'Costa Rica', 'Panama']
        }

    def _get_outcome(self, h_score, a_score):
        if h_score > a_score: return 2
        if h_score == a_score: return 1
        return 0

    def _get_continent(self, team: str) -> str:
        for continent, teams in self.CONTINENT_MAP.items():
            if team in teams: return continent
        return 'Other'

    def _calculate_weighted_form(self, recent_matches: list) -> float:
        if not recent_matches: return 0.5
        total_weight = 0.0
        weighted_score = 0.0
        for pts, weight in recent_matches:
            weighted_score += pts * weight
            total_weight += weight
        return weighted_score / total_weight if total_weight > 0 else 0.5

    def _generate_historical_features(self, df: pd.DataFrame, fifa_ratings: dict) -> pd.DataFrame:
        """
        MODUL 1: Reine chronologische Feature-Pipeline.
        HIER wird der Poisson-State mathematisch sauber Tag für Tag aufgebaut,
        OHNE Wissen aus der Zukunft!
        """
        print("\n⚡ [Pipeline] Generiere Master-Feature-Tabelle (Vasserstoff-Dicht)...")
        feature_data = []
        recent_form = {} 

        for _, row in df.iterrows():
            is_neutral = bool(row['neutral'])
            
            squad_h = fifa_ratings.get(row['home_team'], self.FALLBACK_RATING)
            squad_a = fifa_ratings.get(row['away_team'], self.FALLBACK_RATING)
            
            elo_diff = row['elo_home'] - row['elo_away']
            att_diff = squad_h['ATT'] - squad_a['ATT']
            def_diff = squad_h['DEF'] - squad_a['DEF']
            mid_diff = squad_h['MID'] - squad_a['MID']

            # SAUBERE HISTORISCHE BERECHNUNG: Genau an diesem Tag berechnet!
            probs = self.poisson.predict_match_probabilities(
                row['home_team'], row['away_team'], is_neutral,
                elo_diff=elo_diff, att_diff=att_diff, def_diff=def_diff
            )
            
            form_h = self._calculate_weighted_form(recent_form.get(row['home_team'], []))
            form_a = self._calculate_weighted_form(recent_form.get(row['away_team'], []))

            home_cont = self._get_continent(row['home_team'])
            away_cont = self._get_continent(row['away_team'])
            
            hist_continent_diff = 0
            if not is_neutral:
                cont_adv_h = 1 
                cont_adv_a = 1 if away_cont == home_cont else 0
                hist_continent_diff = cont_adv_h - cont_adv_a

            feature_data.append({
                'date': row['date'],
                'tournament': row['tournament'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'home_continent': home_cont,
                'away_continent': away_cont,
                'continent_adv_diff': hist_continent_diff,
                'elo_home': row['elo_home'],
                'elo_away': row['elo_away'],
                'elo_diff': elo_diff,
                # Wir frieren die komplette Matrix und die Differenz HIER ein!
                'poisson_diff': probs['home_win'] - probs['away_win'],
                'poisson_matrix': probs['matrix'], 
                'form_diff': form_h - form_a,
                'att_diff': att_diff, 
                'mid_diff': mid_diff, 
                'def_diff': def_diff, 
                'outcome': self._get_outcome(row['home_score'], row['away_score'])
            })
            
            # Das Gedächtnis inkrementell updaten (wächst chronologisch mit)
            self.poisson.update_match(row['home_team'], row['away_team'], row['home_score'], row['away_score'], row['elo_home'], row['elo_away'])
            
            outcome = self._get_outcome(row['home_score'], row['away_score'])
            h_pts = 1.0 if outcome == 2 else (0.5 if outcome == 1 else 0.0)
            a_pts = 1.0 if outcome == 0 else (0.5 if outcome == 1 else 0.0)
            
            weight = 3.0 if 'FIFA World Cup' in row['tournament'] else (2.0 if 'qualification' in row['tournament'].lower() else 1.0)
            
            for t, p in [(row['home_team'], h_pts), (row['away_team'], a_pts)]:
                if t not in recent_form: recent_form[t] = []
                recent_form[t].append((p, weight))
                if len(recent_form[t]) > 5: recent_form[t].pop(0)

        df_features = pd.DataFrame(feature_data)
        df_features['date'] = pd.to_datetime(df_features['date'])
        return df_features

    def _evaluate_tournament(self, df_features: pd.DataFrame, start_str: str, end_str: str, host_nation: str, host_continent: str, starting_bankroll: float) -> dict:
        t_start = pd.to_datetime(start_str)
        t_end = pd.to_datetime(end_str)
        
        train_df = df_features[df_features['date'] < t_start].copy()
        test_df = df_features[(df_features['date'] >= t_start) & 
                              (df_features['date'] <= t_end) & 
                              (df_features['tournament'] == 'FIFA World Cup')].copy()

        self.ml_model = MetaMachineLearningModel()
        self.ml_model.train(train_df)
        
        current_bankroll = starting_bankroll
        
        stats = {
            'correct_ml': 0, 'correct_base': 0, 'bets_placed': 0, 'exact_scores': 0, 
            'matches': len(test_df), 'total_stake': 0.0, 'total_return': 0.0,
            'score_tracker': {},
            'total_goal_error': 0,
            'close_misses': 0
        }
        y_true_brier, y_prob_ml = [], []
        
        for _, row in test_df.iterrows():
            actual = row['outcome']
            
            # TURNIER-VORTEIL (LIVE BERECHNET)
            cont_adv_h = 1 if row['home_continent'] == host_continent else 0
            cont_adv_a = 1 if row['away_continent'] == host_continent else 0
            
            # FIX: Wir nutzen den historisch eingefrorenen poisson_diff Wert aus row! 
            # Das verhindert jegliche Kontamination aus der Zukunft.
            historical_poisson_diff = row['poisson_diff']
            historical_matrix = row['poisson_matrix']
            
            probs_ml = self.ml_model.predict_probabilities(
                row['elo_diff'], historical_poisson_diff, row['form_diff'], 
                cont_adv_h - cont_adv_a, row['att_diff'], row['mid_diff'], row['def_diff']
            )
            
            # Buchmacher-Quoten Simulation (5% Marge)
            elo_win_prob = 1 / (1 + 10 ** ((row['elo_away'] - row['elo_home']) / 400))
            elo_loss_prob = 1 / (1 + 10 ** ((row['elo_home'] - row['elo_away']) / 400))
            sum_p = elo_win_prob + 0.25 + elo_loss_prob
            
            odds = {
                2: 1 / ((elo_win_prob / sum_p) * 1.05),
                1: 1 / ((0.25 / sum_p) * 1.05),
                0: 1 / ((elo_loss_prob / sum_p) * 1.05)
            }
            
            y_true_brier.append(1 if actual == 2 else 0)
            y_prob_ml.append(probs_ml['home_win'])
            
            pred_base = 2 if row['elo_diff'] > 0 else 0
            ai_confidence = max(probs_ml['home_win'], probs_ml['away_win'])
            ai_raw_pred = 2 if probs_ml['home_win'] > probs_ml['away_win'] else 0
            
            if ai_confidence >= 0.50:
                p = ai_confidence
                b = odds[ai_raw_pred] - 1
                q = 1 - p
                kelly_fraction = (p * b - q) / b
                
                if kelly_fraction > 0:
                    stake_pct = min(kelly_fraction * 0.5, 0.10)
                    stake = current_bankroll * stake_pct
                    
                    stats['bets_placed'] += 1
                    stats['total_stake'] += stake
                    current_bankroll -= stake
                    
                    if ai_raw_pred == actual: 
                        stats['correct_ml'] += 1
                        winnings = stake * odds[ai_raw_pred]
                        stats['total_return'] += winnings
                        current_bankroll += winnings
                        
                    if pred_base == actual: stats['correct_base'] += 1
                    
                    # FIX: Das Orakel nutzt die historisch saubere, eingefrorene Matrix!
                    pred_h, pred_a = self.poisson.get_smart_score(historical_matrix, probs_ml)
                    
                    error_h = abs(row['home_score'] - pred_h)
                    error_a = abs(row['away_score'] - pred_a)
                    total_error = error_h + error_a
                    
                    stats['total_goal_error'] += total_error
                    
                    if total_error == 0:
                        stats['exact_scores'] += 1
                        score_str = f"{int(row['home_score'])}:{int(row['away_score'])}"
                        stats['score_tracker'][score_str] = stats['score_tracker'].get(score_str, 0) + 1
                    elif total_error == 1:
                        stats['close_misses'] += 1

        stats['brier_ml'] = brier_score_loss(y_true_brier, y_prob_ml)
        stats['ending_bankroll'] = current_bankroll
        return stats

    def run_stress_test(self):
        print("="*60)
        print("🏆 STARTE LEAK-FREIEN QUANT-TEST (WATERPROOF V5) 🏆")
        print("="*60)

        df = self.loader.load_data(start_year=2000)
        df = self.elo.calculate_historical_elo(df)
        fifa_ratings = self.loader.load_fifa_ratings("data/fifa_players.csv")

        # 1. Pipeline baut ALLES chronologisch auf (Absolut sicher!)
        df_features = self._generate_historical_features(df, fifa_ratings)

        tournaments = [
            ("WM 2014 (Brasilien)", "2014-06-12", "2014-07-13", "Brazil", "South America"),
            ("WM 2018 (Russland)",  "2018-06-14", "2018-07-15", "Russia", "Europe"),
            ("WM 2022 (Katar)",     "2022-11-20", "2022-12-18", "Qatar", "Asia")
        ]

        total_bets, total_exact = 0, 0
        total_stake_all, total_return_all = 0.0, 0.0
        master_score_tracker = {}
        total_goal_error_all = 0
        total_close_misses = 0
        
        kontostand = 1000.0

        for name, start_str, end_str, host_nation, host_continent in tournaments:
            print(f"\n⏳ [{name}] Simuliere Wetten mit Kontostand: {kontostand:.2f}€...")
            stats = self._evaluate_tournament(df_features, start_str, end_str, host_nation, host_continent, kontostand)
            
            bp = stats['bets_placed']
            profit = stats['total_return'] - stats['total_stake']
            roi = (profit / stats['total_stake']) * 100 if stats['total_stake'] > 0 else 0
            
            print(f"✅ [{name}] {bp} Wetten | Umsatz: {stats['total_stake']:.2f}€ | Profit: {profit:+.2f}€ (ROI: {roi:+.2f}%)")
            
            kontostand = stats['ending_bankroll']
            
            total_bets += bp
            total_exact += stats['exact_scores']
            total_stake_all += stats['total_stake']
            total_return_all += stats['total_return']
            
            total_goal_error_all += stats['total_goal_error']
            total_close_misses += stats['close_misses']
            for score, count in stats['score_tracker'].items():
                master_score_tracker[score] = master_score_tracker.get(score, 0) + count

        final_profit = kontostand - 1000.0
        final_roi = (final_profit / total_stake_all) * 100 if total_stake_all > 0 else 0
        final_exact_rate = (total_exact / total_bets) * 100 if total_bets > 0 else 0
        
        avg_goal_error = total_goal_error_all / total_bets if total_bets > 0 else 0
        close_miss_rate = (total_close_misses / total_bets) * 100 if total_bets > 0 else 0
        
        print("\n" + "="*60)
        print("🌍 FINALES QUANT-ERGEBNIS (OHNE DATEN-LEAK) 🌍")
        print("="*60)
        print(f"Startkapital:          1000.00€")
        print(f"Gesamter Umsatz:       {total_stake_all:.2f}€")
        print(f"Reiner Netto-Profit:   {final_profit:+.2f}€")
        print(f"Finanzielle Rendite:   {final_roi:+.2f}% (ROI)")
        print(f"Endkontostand:         {kontostand:.2f}€")
        print(f"Exakte Tore getroffen: {final_exact_rate:.2f}%")
        
        print("-" * 60)
        print("📏 DISTANZ-METRIKEN (MODELL-SCHÄRFE):")
        print(f"   Ø Tor-Fehler pro Spiel: {avg_goal_error:.2f} Tore daneben")
        print(f"   Pfostenschüsse (1 Tor daneben): {total_close_misses} von {total_bets} ({close_miss_rate:.1f}%)")
        print(f"   Kombiniert (Treffer + Pfosten): {(final_exact_rate + close_miss_rate):.2f}% aller Wetten")
        
        print("-" * 60)
        print("🎯 EXAKTE ERGEBNISSE (DETAIL-ANALYSE):")
        if not master_score_tracker:
            print("   Leider wurden keine exakten Ergebnisse getroffen.")
        else:
            sorted_scores = sorted(master_score_tracker.items(), key=lambda x: x[1], reverse=True)
            for score, count in sorted_scores:
                print(f"   Ergebnis {score} -> {count} mal korrekt vorhergesagt")
        print("="*60)