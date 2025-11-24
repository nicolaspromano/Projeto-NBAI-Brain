# bibliotecas para manipulação de dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# modelos e ferramentas de ML
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# FUNCAO 1: analisar a curva de carreira
# objetivo: visualizar a trajetoria da carreira de um jogador em termo de pontos e ajustar uma curva de tendencia usando modelo de Regressao polinomial
def analisar_curva_carreira(df, nome_do_jogador):

    df_jogador = df[df['player_name'] == nome_do_jogador] # obtem apenas os dados do jogador selecionado 
    stats_por_temporada = df_jogador.groupby('season_year')['pts'].mean().reset_index() # agrupa dados por temporada e calcula media de pontos de cada uma
    stats_por_temporada = stats_por_temporada[stats_por_temporada['pts'] > 5] # remove temporadas com media de pontos menor que 5

    # garante que o jogador tenha dados o suficiente para fazer uma boa analise 
    if len(stats_por_temporada) < 5:
        return None, "Jogador não possui temporadas suficientes para análise."

    
    X = np.arange(len(stats_por_temporada)).reshape(-1, 1) # cria uma sequencia numerica para representar as temporadas.
    y = stats_por_temporada['pts'].values # média de pontos de cada temporada 

    poly_features = PolynomialFeatures(degree=3) # modelo para criar a curva - regressao polinomial de grau 3
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression() # cria modelo de regressao linear 
    model.fit(X_poly, y) # treina o modelo para encontrar a melhor curva que se ajusta aos dados
    y_pred_curva = model.predict(X_poly) # usa o modelo treinado para prever os pontos da curva de tendencia

    # visualização dos dados em forma de grafico
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(stats_por_temporada['season_year'], y, color='blue', s=100, label='Média Real da Temporada', zorder=5)
    ax.plot(stats_por_temporada['season_year'], y_pred_curva, color='red', linewidth=3, label='Curva da Carreira (Modelo ML)')
    
    ax.set_title(f'Análise da Curva da Carreira de {nome_do_jogador}', fontsize=20, fontweight='bold')
    ax.set_xlabel('Temporada', fontsize=14)
    ax.set_ylabel('Média de Pontos por Jogo', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45) 
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    return fig, None

# FUNCAO 2: detector de jogos anormais do jogador
# objetivo: usar um modelo de aprendizado nao supervisionado (Isolation Forest) para encontrar jogos com estatisticas diferentes do "comum" de um determinado jogador
def detectar_anomalias(df, nome_do_jogador):
    df_jogador = df[df['player_name'] == nome_do_jogador].copy() # obtem apenas os dados do jogador selecionado 
    features_para_analise = ['pts', 'ast', 'reb', 'fg3a', 'fg_pct', 'fg3_pct', 'tov'] # define estatisticas que serão usadas para julgar se o jogo é "normal" ou "anormal"
    df_jogador[features_para_analise] = df_jogador[features_para_analise].fillna(0) # preenche valores faltantes com 0
    X = df_jogador[features_para_analise] 

    # verifica se o jogador selecionado foi encontrado 
    if X.empty:
        return None, None, f"Jogador '{nome_do_jogador}' não encontrado."

    model = IsolationForest(contamination=0.015, random_state=42) # cria modelo IF, `contamination` diz ao modelo qual a porcentagem de dados que esperamos ser anomalias (1.5%).
    model.fit(X) # treina o modelo com os dados dos jogos do jogador

    # o modelo define 1 para jogos "normais" e -1 para jogos "anormais"
    df_jogador['anomalia'] = model.predict(X) 
    df_jogador['score_anomalia'] = model.decision_function(X) # O `decision_function` retorna um "score de anomalia", quanto menor esse score, mais anormal é o jogo 
    

    anomalias_df = df_jogador[df_jogador['anomalia'] == -1].sort_values(by='score_anomalia') # cria uma nova base de dados apenas com os jogos anormais, ordenados do mais anormal ao menos anormal
    colunas_para_exibir = ['game_date', 'pts', 'ast', 'reb', 'fg3a', 'fg_pct', 'fg3_pct', 'tov', 'score_anomalia'] # define colunas que serao exibidas na tabela de resultados

    # visualização dos dados em forma de grafico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    normais = df_jogador[df_jogador['anomalia'] == 1]
    ax.scatter(normais['pts'], normais['ast'], c='grey', alpha=0.5, label='Jogos Típicos')
    anomalias = df_jogador[df_jogador['anomalia'] == -1]
    ax.scatter(anomalias['pts'], anomalias['ast'], c='red', s=150, edgecolor='black', label='Jogos Anômalos (ML)')
    ax.set_title(f'Detecção de Anomalias na Carreira de {nome_do_jogador}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Pontos no Jogo', fontsize=12)
    ax.set_ylabel('Assistências no Jogo', fontsize=12)
    ax.legend()
    
    return anomalias_df[colunas_para_exibir], fig, None

# FUNCAO 3: previsao de media de pontos na proxima temporada
# objetivo: usar dados historicos de TODOS os jogadores para treinar o modelo e preve a média de pontos de um jogador na proxima temporada
def prever_proxima_temporada(df, nome_do_jogador):
    # agrupa os dados de TODOS os jogadores por temporada para criar uma base de dados de treino
    df_temporadas = df.groupby(['player_name', 'season_year', 'team_id']).agg(
        pts=('pts', 'mean'), min=('min', 'mean'), ast=('ast', 'mean'),
        reb=('reb', 'mean'), fg_pct=('fg_pct', 'mean'), fg3_pct=('fg3_pct', 'mean'),
        ft_pct=('ft_pct', 'mean'), tov=('tov', 'mean'), total_jogos=('game_id', 'count')
    ).reset_index()

    
    # novas colunas que vao ajudar na previsao
    df_temporadas['season_year_numeric'] = df_temporadas['season_year'].str[:4].astype(int)
    df_temporadas['time_anterior_id'] = df_temporadas.groupby('player_name')['team_id'].shift(1)
    df_temporadas['mudou_de_time'] = (df_temporadas['team_id'] != df_temporadas['time_anterior_id']).astype(int)
    df_temporadas['pts_anterior'] = df_temporadas.groupby('player_name')['pts'].shift(1) # media de pontos da temporada passada
    df_temporadas['tendencia_pts'] = df_temporadas['pts'] - df_temporadas['pts_anterior'] # media de pontos aumentou ou nao ?

    # `shift(-1)` "puxa" o dado da linha de baixo (próxima temporada) para a linha atual.
    df_temporadas['next_pts'] = df_temporadas.groupby('player_name')['pts'].shift(-1)
    df_modelo = df_temporadas.dropna(subset=['next_pts', 'time_anterior_id', 'pts_anterior']).copy()
    df_modelo['tendencia_pts'] = df_modelo['tendencia_pts'].fillna(0)

    # treinamento do modelo - define a lista de features que o modelo irá usar para fazer a previsao
    features = [
        'pts', 'min', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct', 'tov', 
        'total_jogos', 
        'season_year_numeric',
        'mudou_de_time', 
        'tendencia_pts'
    ]
    

    X = df_modelo[features] # treino
    y = df_modelo['next_pts'] # alvo do treino

    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) # cria o modelo Random Forest Regressor
    model.fit(X, y) # treina o modelo com os dados historicos de todos os jogadores 
    
    # pega os dados do jogador selecionado 
    dados_jogador = df_temporadas[df_temporadas['player_name'] == nome_do_jogador].copy()
    if dados_jogador.empty:
        return None, f"Jogador '{nome_do_jogador}' não encontrado."

    # encontra a temporada mais recente do jogador em questao
    temporada_recente = dados_jogador.sort_values(by='season_year_numeric', ascending=False).iloc[0:1]
    ano_base_num = temporada_recente['season_year_numeric'].iloc[0]
    ano_base_str = temporada_recente['season_year'].iloc[0]
    temporada_recente['tendencia_pts'] = temporada_recente['tendencia_pts'].fillna(0)
    
    # prepara os dados da ultima temporada para "alimentar" o modelo 
    dados_para_previsao = temporada_recente[features]
    previsao_pts = model.predict(dados_para_previsao)[0] # usa o modelo de treino para prever a media de pontos da proxima temporada

    # resultado da previsao 
    resultado_previsao = {
        "temporada_base": ano_base_str,
        "pts_base": temporada_recente['pts'].iloc[0],
        "temporada_previsao": f"{ano_base_num+1}-{ano_base_num+2}",
        "pts_previstos": previsao_pts
    }
    
    return resultado_previsao, None