import pandas as pd

# Caminho do CSV original
csv_path = '/workspace/coswara/combined_data_mapped.csv'

# Carrega o CSV
df = pd.read_csv(csv_path)

# Remove linhas com covid_status == 'exclude'
df = df[df['covid_status'] != 'exclude']

# Substitui os valores
df['covid_status'] = df['covid_status'].replace({
    'COVID_negative': 'healthy',
    'COVID_positive': 'COVID-19'
})

# Salva o CSV atualizado
df.to_csv('/workspace/coswara/metadata_compiled.csv', index=False)

print("Valores atualizados com sucesso!")
