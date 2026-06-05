import joblib
import numpy as np
import random
import os
import warnings
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 📋 DEIN CONTROL CENTER: DIE ECHTEN WM-GRUPPEN 2026
# =====================================================================
OFFICIAL_GROUPS = {
    "A": ["Czech Republic", "Mexico", "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar", "Switzerland"], # FIX: Offizieller FIFA Name
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["Turkey", "United States", "Paraguay", "Australia"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],             # FIX: Sonderzeichen
    "F": ["Sweden", "Netherlands", "Japan", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["Iraq", "France", "Senegal", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["DR Congo", "Portugal", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

def run_wm_real_structure(num_tournaments=1000):
    print("🌍 Lade DeepPitch Brain für OFFIZIELLES WM 2026 ORAKEL...")
    brain_path = "data/deeppitch_brain.pkl"
    if not os.path.exists(brain_path):
        print("❌ Fehler: deeppitch_brain.pkl nicht gefunden.")
        return
        
    brain = joblib.load(brain_path)
    meta_model = brain['model']
    model = meta_model.model if hasattr(meta_model, 'model') else (meta_model.rf if hasattr(meta_model, 'rf') else meta_model)
    bt = brain['backtester']
    fifa_ratings = brain['fifa']

    wm_2026_teams = []
    for group_teams in OFFICIAL_GROUPS.values():
        wm_2026_teams.extend(group_teams)

    missing_teams = [t for t in wm_2026_teams if t not in bt.elo.ratings]
    if missing_teams:
        print(f"⚠️ WARNUNG: Folgende Teams fehlen in der Elo-Datenbank: {missing_teams}")

    # -------------------------------------------------------------
    # PHASE 1: PAIRWISE PRE-CALCULATION CACHE
    # -------------------------------------------------------------
    print(f"⚡ Phase 1: Berechne KI-Wahrscheinlichkeiten für alle möglichen Paarungen vorab...")
    matchup_cache = {}
    feature_matrix = []
    
    # Bilde alle exakt 1.128 möglichen Match-Kombinationen
    pair_keys = list(combinations(wm_2026_teams, 2))

    # NEU: Ladebalken für Phase 1
    for t1, t2 in tqdm(pair_keys, desc="Generiere Paarungen", unit=" Match"):
        stats_1 = fifa_ratings.get(t1, bt.FALLBACK_RATING)
        stats_2 = fifa_ratings.get(t2, bt.FALLBACK_RATING)
        elo_1 = bt.elo.ratings.get(t1, 1500.0)
        elo_2 = bt.elo.ratings.get(t2, 1500.0)
        
        elo_diff = elo_1 - elo_2
        att_diff = stats_1['ATT'] - stats_2['ATT']
        mid_diff = stats_1['MID'] - stats_2['MID']
        def_diff = stats_1['DEF'] - stats_2['DEF']
        form_diff = 0.0
        
        probs_p = bt.poisson.predict_match_probabilities(t1, t2, True, elo_diff, att_diff, def_diff)
        poisson_diff = probs_p['home_win'] - probs_p['away_win']
        
        feature_matrix.append([elo_diff, poisson_diff, form_diff, att_diff, mid_diff, def_diff])

    print("🧠 Phase 1.5: KI-Vektorberechnung (Massive Batch-Prediction)...")
    all_probs = model.predict_proba(feature_matrix)
    
    for idx, (t1, t2) in enumerate(pair_keys):
        p = all_probs[idx]
        elo_diff = feature_matrix[idx][0]
        att_diff = feature_matrix[idx][3]
        def_diff = feature_matrix[idx][5]
        
        lambda_t1 = max(0.1, 1.3 + (elo_diff * 0.001) + (att_diff * 0.015))
        lambda_t2 = max(0.1, 1.3 - (elo_diff * 0.001) - (def_diff * 0.015))
        
        matchup_cache[(t1, t2)] = {'prob_draw': p[1], 'prob_h': p[2], 'prob_a': p[0], 'l_h': lambda_t1, 'l_a': lambda_t2}

    def simulate_match(t_h, t_a, is_knockout=False):
        if (t_h, t_a) in matchup_cache:
            c = matchup_cache[(t_h, t_a)]
            g_h = np.random.poisson(c['l_h'])
            g_a = np.random.poisson(c['l_a'])
        else:
            c = matchup_cache[(t_a, t_h)]
            g_h = np.random.poisson(c['l_a'])
            g_a = np.random.poisson(c['l_h'])
            
        if is_knockout and g_h == g_a:
            elo_h = bt.elo.ratings.get(t_h, 1500.0)
            elo_a = bt.elo.ratings.get(t_a, 1500.0)
            if np.random.rand() < (elo_h / (elo_h + elo_a)):
                g_h += 1
            else:
                g_a += 1
        return g_h, g_a

    # -------------------------------------------------------------
    # PHASE 2: MONTE CARLO TOURNAMENT TREE SIMULATION
    # -------------------------------------------------------------
    print(f"🎲 Phase 2: Simuliere {num_tournaments:,} Turniere mit OFFIZIELLER Gruppenphase...")
    
    world_champions = defaultdict(int)
    vice_champions = defaultdict(int)
    semifinalists = defaultdict(int)
    early_exits = defaultdict(int)

    # Ladebalken für Phase 2 (bereits vorhanden)
    for t_idx in tqdm(range(num_tournaments), desc="Simuliere Turniere", unit="WM"):
        qualified_to_r32 = []
        third_places = []
        
        for group_name, group_teams in OFFICIAL_GROUPS.items():
            table = {team: {'pts': 0, 'gd': 0, 'gs': 0, 'name': team} for team in group_teams}
            
            for i in range(4):
                for j in range(i+1, 4):
                    t1, t2 = group_teams[i], group_teams[j]
                    g1, g2 = simulate_match(t1, t2, is_knockout=False)
                    
                    table[t1]['gs'] += g1
                    table[t2]['gs'] += g2
                    table[t1]['gd'] += (g1 - g2)
                    table[t2]['gd'] += (g2 - g1)
                    
                    if g1 > g2: table[t1]['pts'] += 3
                    elif g2 > g1: table[t2]['pts'] += 3
                    else:
                        table[t1]['pts'] += 1
                        table[t2]['pts'] += 1
                        
            sorted_table = sorted(table.values(), key=lambda x: (x['pts'], x['gd'], x['gs']), reverse=True)
            
            qualified_to_r32.append(sorted_table[0]['name'])
            qualified_to_r32.append(sorted_table[1]['name'])
            third_places.append(sorted_table[2])
            early_exits[sorted_table[3]['name']] += 1
            
        sorted_third_places = sorted(third_places, key=lambda x: (x['pts'], x['gd'], x['gs']), reverse=True)
        for i in range(8):
            qualified_to_r32.append(sorted_third_places[i]['name'])
        for i in range(8, 12):
            early_exits[sorted_third_places[i]['name']] += 1
            
        random.shuffle(qualified_to_r32)
        
        r16_teams = []
        for i in range(0, 32, 2):
            g1, g2 = simulate_match(qualified_to_r32[i], qualified_to_r32[i+1], is_knockout=True)
            r16_teams.append(qualified_to_r32[i] if g1 > g2 else qualified_to_r32[i+1])
            
        qf_teams = []
        for i in range(0, 16, 2):
            g1, g2 = simulate_match(r16_teams[i], r16_teams[i+1], is_knockout=True)
            qf_teams.append(r16_teams[i] if g1 > g2 else r16_teams[i+1])
            
        sf_teams = []
        for i in range(0, 8, 2):
            g1, g2 = simulate_match(qf_teams[i], qf_teams[i+1], is_knockout=True)
            sf_teams.append(qf_teams[i] if g1 > g2 else qf_teams[i+1])
            
        for team in sf_teams:
            semifinalists[team] += 1
            
        g1, g2 = simulate_match(sf_teams[0], sf_teams[1], is_knockout=True)
        f1 = sf_teams[0] if g1 > g2 else sf_teams[1]
        
        g3, g4 = simulate_match(sf_teams[2], sf_teams[3], is_knockout=True)
        f2 = sf_teams[2] if g3 > g4 else sf_teams[3]
        
        gf1, gf2 = simulate_match(f1, f2, is_knockout=True)
        champion = f1 if gf1 > gf2 else f2
        runner_up = f2 if gf1 > gf2 else f1
        
        world_champions[champion] += 1
        vice_champions[runner_up] += 1

    # -------------------------------------------------------------
    # ANALYSIS OUTPUT
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("🔮 DEEPPITCH MATHEMATISCHES PROGNOSE-ORAKEL (WM 2026)")
    print("="*60)
    print(f"➜ Simulierte Weltmeisterschaften: {num_tournaments:,}")
    
    sorted_champs = sorted(world_champions.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🏆 WAHRSCHEINLICHKEIT WELTMEISTER ZU WERDEN (Top 5):")
    for i in range(min(5, len(sorted_champs))):
        team, wins = sorted_champs[i]
        prob = (wins / num_tournaments) * 100
        print(f"  {i+1}. {team:<22} ➜ {prob:.2f}% Chance")
        
    print("\n⚠️ GRUPPENPHASEN-EXIT (Höchstes Risiko für frühes Aus in Top 5):")
    sorted_exits = sorted(early_exits.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(5, len(sorted_exits))):
        team, exits = sorted_exits[i]
        prob = (exits / num_tournaments) * 100
        print(f"  {i+1}. {team:<22} ➜ Fliegt zu {prob:.1f}% in der Vorrunde raus")
    print("="*60)

    return sorted_champs, sorted_exits

if __name__ == "__main__":
    run_wm_real_structure(num_tournaments=10000)