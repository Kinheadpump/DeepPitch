import pandas as pd
import os

print("⏳ Radikale Daten-Diät: Entferne historische Duplikate...")
input_file = 'data/fifa_players.csv'
output_file = 'data/fifa_players_cloud.csv' # Neuer Name für den frischen Start!

chunk_size = 50000
df_list = []

# Spalten-Erkennung
header_df = pd.read_csv(input_file, nrows=1)
if 'nationality' in header_df.columns and 'nationality_name' not in header_df.columns:
    use_cols = ['short_name', 'long_name', 'nationality', 'overall', 'shooting', 'passing', 'defending']
    rename_dict = {'nationality': 'nationality_name'}
else:
    use_cols = ['short_name', 'long_name', 'nationality_name', 'overall', 'shooting', 'passing', 'defending']
    rename_dict = None

# 1. Alle Spieler ab 65 OVR sammeln
for chunk in pd.read_csv(input_file, usecols=use_cols, chunksize=chunk_size, low_memory=False):
    if rename_dict:
        chunk.rename(columns=rename_dict, inplace=True)
    chunk_lite = chunk[chunk['overall'] >= 65]
    df_list.append(chunk_lite)
    print("✔️ Chunk gefiltert...")

print("🔄 Führe Daten zusammen und vernichte Duplikate...")
df_final = pd.concat(df_list, ignore_index=True)

# 2. Die Magie: Nach Stärke sortieren und ALLE doppelten Namen löschen!
# So behalten wir immer nur die stärkste/aktuellste Version eines Spielers.
df_final = df_final.sort_values(by='overall', ascending=False)
df_final = df_final.drop_duplicates(subset=['long_name', 'nationality_name'], keep='first')

# 3. Speichern
df_final.to_csv(output_file, index=False)
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

print(f"\n✅ ERFOLG! Datei wurde auf {len(df_final)} einzigartige Spieler geschrumpft.")
print(f"Neue Dateigröße: {file_size_mb:.2f} MB (Das passt locker auf GitHub!)")