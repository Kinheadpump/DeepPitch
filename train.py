import joblib
import os
from src.backtester import Backtester
from src.models import MetaMachineLearningModel

def train_and_save_model():
    print("="*60)
    print("🧠 STARTE DEEP LEARNING & GEDÄCHTNIS-EXTRAKTION")
    print("="*60)
    
    # 1. Die gesamte V5-Pipeline exakt 1x durchlaufen lassen
    bt = Backtester()
    df = bt.loader.load_data(start_year=2000)
    df = bt.elo.calculate_historical_elo(df)
    fifa_ratings = bt.loader.load_fifa_ratings("data/fifa_players.csv")
    
    df_features = bt._generate_historical_features(df, fifa_ratings)
    
    # 2. Den Random Forest trainieren
    print("\n🌲 Trainiere Random Forest Classifier...")
    model = MetaMachineLearningModel()
    model.train(df_features)
    
    # 3. Das Gehirn bündeln und auf die Festplatte schreiben
    print("💾 Friere das System-Gedächtnis ein (Pickling)...")
    
    # Wir speichern alles Wichtige in einem einzigen Dictionary
    brain = {
        'model': model,
        'backtester': bt, # Enthält die fertigen Elo- und Poisson-States!
        'fifa': fifa_ratings
    }
    
    # Speichern in den data/ Ordner
    save_path = "data/deeppitch_v5_brain.pkl"
    joblib.dump(brain, save_path)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ ERFOLG! Das Gehirn wurde unter '{save_path}' gespeichert ({file_size:.2f} MB).")
    print("="*60)

if __name__ == "__main__":
    train_and_save_model()