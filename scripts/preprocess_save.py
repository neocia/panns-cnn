import os
import pandas as pd

# Configurações
METADATA_CSV = '/workspace/data/metadata_compiled.csv'  # CSV original
PROCESSED_DIR = '/workspace/data'       # Pasta com áudios processados
OUTPUT_CSV = '/workspace/data/processed_metadata.csv'   # CSV de saída

# Carregar CSV original
df_metadata = pd.read_csv(METADATA_CSV)

# Lista para armazenar os dados processados
processed_rows = []

# Percorrer as subpastas (COVID-19, healthy)
for class_name in os.listdir(PROCESSED_DIR):
    class_dir = os.path.join(PROCESSED_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    for file_name in os.listdir(class_dir):
        if not file_name.endswith('.wav'):
            continue
        
        uuid = os.path.splitext(file_name)[0]
        # Procurar a linha correspondente no CSV original
        row = df_metadata[df_metadata['uuid'] == uuid]
        if not row.empty:
            row = row.copy()
            row['processed_path'] = os.path.join(class_dir, file_name)
            processed_rows.append(row)

# Concatenar todas as linhas
if processed_rows:
    df_processed = pd.concat(processed_rows, ignore_index=True)
    df_processed.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV gerado com {len(df_processed)} linhas: {OUTPUT_CSV}")
else:
    print("Nenhum arquivo processado encontrado.")
