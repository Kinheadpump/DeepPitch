import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_importance():
    # 1. Gehirn laden
    brain = joblib.load("data/deeppitch_brain.pkl")
    meta_model = brain['model']
    
    # Hier der Fix: Wir greifen auf das interne Modell zu. 
    # Wenn dein Meta-Modell das Attribut 'model' oder 'rf' besitzt, nutzen wir das:
    if hasattr(meta_model, 'model'):
        rf_model = meta_model.model
    elif hasattr(meta_model, 'rf'):
        rf_model = meta_model.rf
    else:
        # Falls es direkt das Modell ist, aber die Attribute falsch gemappt sind
        rf_model = meta_model

    # 2. Features definieren
    feature_names = [
        'elo_diff', 'poisson_diff', 'form_diff', 
        'continent_adv_diff', 'att_diff', 'mid_diff', 'def_diff'
    ]
    
    # 3. Importance extrahieren
    importances = rf_model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    # 4. Visualisierung
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title('DeepPitch Feature Importance: Was treibt den Gewinn?')
    plt.tight_layout()
    plt.show()
    
    print(df_imp)

if __name__ == "__main__":
    analyze_model_importance()