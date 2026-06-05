import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_importance():
    # 1. Gehirn laden
    brain = joblib.load("data/deeppitch_brain.pkl")
    meta_model = brain['model']
    
    # Auf das interne scikit-learn Modell zugreifen
    if hasattr(meta_model, 'model'):
        rf_model = meta_model.model
    elif hasattr(meta_model, 'rf'):
        rf_model = meta_model.rf
    else:
        rf_model = meta_model

    # 2. FIX: Features definieren (Exakt die 6 Features, die jetzt noch existieren)
    feature_names = [
        'elo_diff', 'poisson_diff', 'form_diff', 
        'att_diff', 'mid_diff', 'def_diff'
    ]
    
    # 3. Importance extrahieren
    importances = rf_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    # 4. Visualisierung (inkl. Fix für die Seaborn FutureWarning)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', palette='viridis', legend=False)
    plt.title('DeepPitch Feature Importance: Was treibt den Gewinn?')
    plt.tight_layout()
    plt.show()
    
    print(df_imp)

if __name__ == "__main__":
    analyze_model_importance()