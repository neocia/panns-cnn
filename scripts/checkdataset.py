import os
import librosa

datasets = {
    'CoughVid': '/workspace/data/coughvid_processed',
    'Coswara': '/workspace/data/coswara_processed'
}

for dataset_name, data_dir in datasets.items():
    print(f"\nVerificando dataset: {dataset_name}")
    invalid_files = []

    # Percorre todas as subpastas de labels
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue  # ignora arquivos soltos
        for fname in os.listdir(label_path):
            file_path = os.path.join(label_path, fname)
            
            # Verifica extensão
            if not fname.lower().endswith('.wav') or not os.path.isfile(file_path):
                invalid_files.append(file_path)
                continue
            
            # Tenta carregar com librosa
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True, duration=1.0)
                if y is None or len(y) == 0:
                    invalid_files.append(file_path)
            except Exception as e:
                invalid_files.append(file_path)

    if len(invalid_files) == 0:
        print("Todos os arquivos são WAV válidos e carregáveis!")
    else:
        print(f"{len(invalid_files)} arquivos inválidos encontrados:")
        for f in invalid_files:
            print(f)
