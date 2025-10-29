import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from sklearn.utils import resample

# Configurações
CSV_PATH = '/workspace/data/metadata_compiled.csv'
AUDIO_BASE_DIR = '/workspace/data'
OUTPUT_DIR = '/workspace/data/processed_audios'
WINDOW_DURATION = 4.0
TARGET_SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar CSV
df = pd.read_csv(CSV_PATH)

# Filtrar apenas COVID-19 e healthy
df = df[df['status'].isin(['COVID-19', 'healthy'])]

# Filtrar apenas male e female
df = df[df['gender'].isin(['male', 'female'])]

# Criar faixas etárias
bins = [0, 17, 29, 39, 49, 59, 200]
labels = ['0-17', '18-29', '30-39', '40-49', '50-59', '60+']
df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

# Criar strata = faixa etária + gênero
df['strata'] = df['age_bin'].astype(str) + "_" + df['gender'].astype(str)

# Mapear status para pastas
status_to_folder = {
    'COVID-19': 'covid',
    'healthy': 'healthy'
}

def load_random_window(filepath, duration=WINDOW_DURATION, sr=TARGET_SAMPLE_RATE):
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        required_len = int(sr * duration)
        if len(audio) < required_len:
            audio = np.pad(audio, (0, required_len - len(audio)))
            return audio
        start = random.randint(0, len(audio) - required_len)
        return audio[start:start + required_len]
    except Exception as e:
        print(f"⚠️ Erro ao processar {filepath}: {e}")
        return None

def process_class(df_class, class_name):
    out_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    count = 0
    for _, row in df_class.iterrows():
        folder_name = status_to_folder[class_name]
        class_dir = os.path.join(AUDIO_BASE_DIR, folder_name)
        filepath = os.path.join(class_dir, f"{row['uuid']}.wav")
        audio_window = load_random_window(filepath)
        if audio_window is not None:
            out_path = os.path.join(out_class_dir, f"{row['uuid']}.wav")
            sf.write(out_path, audio_window, TARGET_SAMPLE_RATE)
            count += 1
    return count


# --------------------------
# BALANCEAMENTO POR STRATA
# --------------------------
balanced_df = pd.DataFrame()

for s in df['strata'].unique():
    df_strata = df[df['strata'] == s]

    covid = df_strata[df_strata['status'] == 'COVID-19']
    healthy = df_strata[df_strata['status'] == 'healthy']

    if len(covid) == 0 or len(healthy) == 0:
        continue

    n_samples = min(len(covid), len(healthy))

    covid_bal = covid.sample(n_samples, random_state=42)
    healthy_bal = healthy.sample(n_samples, random_state=42)

    balanced_df = pd.concat([balanced_df, covid_bal, healthy_bal])

# --------------------------
# PROCESSAR ÁUDIOS BALANCEADOS
# --------------------------
processed_counts = {}

for class_name in ['COVID-19', 'healthy']:
    df_class = balanced_df[balanced_df['status'] == class_name]
    count = process_class(df_class, class_name)
    processed_counts[class_name] = count
    print(f"✅ {class_name}: {count} arquivos processados.")

# Resumo final
print("\n📊 Resumo final de arquivos processados por classe:")
for class_name, count in processed_counts.items():
    print(f"{class_name}: {count} arquivos")

print("\n🏁 Processamento finalizado! Áudios balanceados por classe estão em 'processed_audios/'.")
