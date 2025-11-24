import pandas as pd
import os

def coletar_e_limpar_dados_jogadores():
 
    print("Iniciando o processo de coleta e limpeza de dados dos JOGADORES...")

    caminhos = {
        "reg1": "dados/regular_season_box_scores_2010_2024_part_1.csv",
        "reg2": "dados/regular_season_box_scores_2010_2024_part_2.csv",
        "reg3": "dados/regular_season_box_scores_2010_2024_part_3.csv",
        "playoffs": "dados/play_off_box_scores_2010_2024.csv"
    }
    
    arquivo_saida = 'dados_limpos.pkl'

    if os.path.exists(arquivo_saida):
        print(f"O arquivo '{arquivo_saida}' ja existe, nao e necessario fazer uma nova limpeza.")
        print("Para processar os dados novamente, apague o arquivo 'dados_limpos.pkl'.")
        return

    try:
        print("Carregando arquivos CSV dos jogadores...")
        df_reg1 = pd.read_csv(caminhos["reg1"])
        df_reg2 = pd.read_csv(caminhos["reg2"])
        df_reg3 = pd.read_csv(caminhos["reg3"])
        df_regular = pd.concat([df_reg1, df_reg2, df_reg3], ignore_index=True)
        df_regular['game_type'] = 'Regular'

        df_playoffs = pd.read_csv(caminhos["playoffs"])
        df_playoffs['game_type'] = 'Playoff'

        df_all = pd.concat([df_regular, df_playoffs], ignore_index=True)
        print(f"Dados brutos carregados. Total de {len(df_all)} registros.")

    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo nao encontrado. Verifique se os arquivos CSV estao na pasta 'dados'. Detalhe: {e}")
        return

    mapa_colunas = {
        'season_year': 'season_year', 'game_date': 'game_date', 'gameId': 'game_id',
        'teamId': 'team_id', 'teamName': 'team_name', 'personId': 'player_id',
        'personName': 'player_name', 'position': 'position', 'minutes': 'min',
        'points': 'pts', 'assists': 'ast', 'reboundsTotal': 'reb',
        'fieldGoalsPercentage': 'fg_pct', 'threePointersAttempted': 'fg3a',
        'threePointersPercentage': 'fg3_pct', 'freeThrowsPercentage': 'ft_pct',
        'turnovers': 'tov', 'plusMinusPoints': 'plus_minus', 'game_type': 'game_type'
    }

    # apenas colunas existentes sao selecionadas
    colunas_existentes = [col for col in mapa_colunas.keys() if col in df_all.columns]
    df_clean = df_all[colunas_existentes].rename(columns=mapa_colunas)


    df_clean['game_date'] = pd.to_datetime(df_clean['game_date'], errors='coerce')
    df_clean.drop_duplicates(inplace=True)

    def converter_minutos(tempo_str):
        if isinstance(tempo_str, str) and ':' in tempo_str:
            try:
                minutos, segundos = map(int, tempo_str.split(':'))
                return minutos + segundos / 60
            except (ValueError, TypeError):
                return 0
        return pd.to_numeric(tempo_str, errors='coerce')

    print("Convertendo a coluna 'minutos' para formato numerico...")
    df_clean['min'] = df_clean['min'].apply(converter_minutos)

    stats_cols = ['min', 'pts', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct', 'tov']
    df_clean[stats_cols] = df_clean[stats_cols].fillna(0)
    
    # garante que a coluna 'min' exista antes de filtrar
    if 'min' in df_clean.columns:
        df_clean = df_clean[df_clean['min'] > 0] 

    df_clean.to_pickle(arquivo_saida)
    print(f"Limpeza finalizada. Dados salvos em '{arquivo_saida}'.")
    print(f"Total de registros após limpeza: {len(df_clean)}")

if __name__ == "__main__":
    coletar_e_limpar_dados_jogadores()