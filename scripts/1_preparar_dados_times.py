import pandas as pd
import numpy as np

print("Iniciando o processo de coleta e limpeza de dados dos JOGADORES...")

# 1. Carregamento dos dados
print("Carregando arquivos CSV dos jogadores...")
df_regular = pd.read_csv("dados/regular_season_totals_2010_2024.csv")
df_playoffs = pd.read_csv("dados/play_off_totals_2010_2024.csv")

print(f"Dados brutos carregados. Total de {len(df_regular) + len(df_playoffs)} registros.")

# 2. Marcar tipo de jogo
df_regular["GAME_TYPE"] = "Regular"
df_playoffs["GAME_TYPE"] = "Playoff"

# 3. Unir datasets
df_all = pd.concat([df_regular, df_playoffs], ignore_index=True)

# 4. Remover colunas completamente nulas
df_all.dropna(axis=1, how='all', inplace=True)

# 5. Converter GAME_DATE
print("Convertendo a coluna 'GAME_DATE' para formato datetime...")
df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"], errors="coerce")

# 6. Ajustar formato do ano
df_all["SEASON_YEAR"] = df_all["SEASON_YEAR"].astype(str).str[:4].astype(int)

# 7. Verificar e exibir colunas com dados faltantes
missing = df_all.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("Colunas com dados ausentes:")
    print(missing.sort_values(ascending=False))
else:
    print("Nenhum dado ausente encontrado.")

# 8. Remover duplicatas
df_all.drop_duplicates(inplace=True)

# 9. Salvar CSV limpo
df_all.to_csv("all_games_clean.csv", index=False)

# 10. Separar e salvar em pkl
df = df_all.copy()
df_regular = df[df["GAME_TYPE"] == "Regular"].copy()
df_playoffs = df[df["GAME_TYPE"] == "Playoff"].copy()

df.to_pickle("dados_completos.pkl")
df_regular.to_pickle("dados_regular.pkl")
df_playoffs.to_pickle("dados_playoffs.pkl")

print("Limpeza finalizada. Dados salvos como Pickle:")
print("  - dados_completos.pkl")
print("  - dados_regular.pkl")
print("  - dados_playoffs.pkl")
print(f"Total de registros após limpeza: {len(df)}")
