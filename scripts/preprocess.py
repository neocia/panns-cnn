import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from sklearn.utils import resample

# Configurações
  #CSV_PATH = '/workspace/coswara/metadata_compiled.csv'  # CSV do CoughVid
  #AUDIO_BASE_DIR = '/workspace/coswara'                  # Base onde estão as subpastas covid/healthy
  #OUTPUT_DIR = '/workspace/coswara/processed_audios'    # Pasta para salvar áudios processados

CSV_PATH = '/workspace/coughvid/metadata_compiled.csv'  # CSV do CoughVid
AUDIO_BASE_DIR = '/workspace/coughvid'                  # Base onde estão as subpastas covid/healthy
OUTPUT_DIR = '/workspace/coughvid/processed_audios'    # Pasta para salvar áudios processados

WINDOW_DURATION = 4.0
TARGET_SAMPLE_RATE = 16000
SAMPLES_PER_CLASS = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar CSV
df = pd.read_csv(CSV_PATH)

# Mapear status para pastas
status_to_folder = {
    'COVID-19': 'covid',
    'healthy': 'healthy'
}

# Função para carregar janela aleatória com padding
def load_random_window(filepath, duration=WINDOW_DURATION, sr=TARGET_SAMPLE_RATE):
    try:
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        required_len = int(sr * duration)
        if len(audio) < required_len:
            audio = np.pad(audio, (0, required_len - len(audio)))  # padding
            return audio
        max_start = len(audio) - required_len
        start = random.randint(0, max_start)
        return audio[start:start + required_len]
    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return None

# Função para processar cada classe
def process_class(df_class, class_name):
    folder_name = status_to_folder[class_name]
    class_dir = os.path.join(AUDIO_BASE_DIR, folder_name)
    out_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(out_class_dir, exist_ok=True)

    # Garantir SAMPLES_PER_CLASS
    if len(df_class) < SAMPLES_PER_CLASS:
        df_class = resample(df_class, replace=True, n_samples=SAMPLES_PER_CLASS, random_state=42)
    else:
        df_class = df_class.sample(n=SAMPLES_PER_CLASS, random_state=42)

    count = 0
    for idx, row in df_class.iterrows():
        ##MUDAR PARA uuid(coughvid) ou id(coswara)
        original_name = f"{row['uuid']}.wav"
        filepath = os.path.join(class_dir, original_name)
        audio_window = load_random_window(filepath)
        if audio_window is not None:
            out_path = os.path.join(out_class_dir, original_name)
            sf.write(out_path, audio_window, TARGET_SAMPLE_RATE)
            count += 1
        if count >= SAMPLES_PER_CLASS:
            break
    print(f"{class_name}: {count} arquivos processados.")

# Processar cada classe
process_class(df[df['status']=='COVID-19'], 'COVID-19')
process_class(df[df['status']=='healthy'], 'healthy')

print("Processamento finalizado. Todos os áudios de 4s salvos em 'processed_audios'")
