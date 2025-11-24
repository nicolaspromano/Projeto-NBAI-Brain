import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

print("Iniciando o script de treinamento do modelo...")


# Carrega os dados que foram previamente processados.
try:
    df = pd.read_pickle('dados_completos.pkl')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE').reset_index(drop=True)
    print("Dados carregados e ordenados por data.")
except FileNotFoundError:
    print("ERRO: Arquivo 'dados_completos.pkl' não encontrado. Execute os scripts de preparação primeiro.")
    exit()



# Estatisticas principais 
stats = [
    'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV',
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS'
]

def calcular_features_avancadas(df):

    print("Calculando médias móveis e características adicionais...")
    df_features = df.copy()

    # Janela de 10 jogos
    window_size = 10 
    
    # Calculo da media movel de cada estatistica
    for stat in stats:
        df_features[f'{stat}_avg'] = df_features.groupby('TEAM_NAME')[stat].transform(
            lambda x: x.shift(1).rolling(window=window_size).mean()
        )

    # Calculo de dias de descanso
    df_features['DIAS_DESCANSO'] = df_features.groupby('TEAM_NAME')['GAME_DATE'].diff().dt.days

    # Calculo da sequencia de vitorias
    df_features['WL_numeric'] = df_features['WL'].apply(lambda x: 1 if x == 'W' else 0)
    derrota_streak_id = (df_features.groupby('TEAM_NAME')['WL_numeric'].shift(1) != df_features['WL_numeric']).cumsum()
    df_features['WINSTREAK'] = df_features.groupby(['TEAM_NAME', derrota_streak_id])['WL_numeric'].cumsum()
    df_features['WINSTREAK_anterior'] = df_features.groupby('TEAM_NAME')['WINSTREAK'].shift(1).fillna(0)

    # Remove linhas onde as medias moveis nao puderam ser calculadas.
    df_features.dropna(inplace=True)
    return df_features

df_com_features = calcular_features_avancadas(df)


print("Preparando dados para análise de jogos (casa vs. visitante)...")
# Separa os jogos em "time da casa" e "time visitante"
home = df_com_features[df_com_features['MATCHUP'].str.contains('vs')].copy()
away = df_com_features[df_com_features['MATCHUP'].str.contains('@')].copy()

# Junta os dados para criar uma linha por jogo, com as estatisticas de ambos os times.
games = pd.merge(home, away, on='GAME_ID', suffixes=('_home', '_away'))
# Define o alvo: 1 se o time da casa venceu, 0 se perdeu.
games['VENCEDOR'] = games['WL_home'].apply(lambda x: 1 if x == 'W' else 0)


# O modelo aprende melhor com a diferença entre os times.
features_finais = []
stats_avg = [f'{s}_avg' for s in stats] + ['DIAS_DESCANSO', 'WINSTREAK_anterior']

for stat in stats_avg:
    col_home = f'{stat}_home'
    col_away = f'{stat}_away'
    diff_col = f'{stat}_diff'
    games[diff_col] = games[col_home] - games[col_away]
    features_finais.append(diff_col)

# Define os dados de treino (X) e o alvo (y)
X = games[features_finais]
y = games['VENCEDOR']

# Divide os dados em conjuntos de treino e teste. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Normaliza os dados para que nenhuma feature domine as outras.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Ajustando hiperparâmetros do Random Forest...")
# Define uma grade de parametros para o GridSearchCV testar.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5]
}

# Configura o GridSearchCV para encontrar os melhores parametros usando validacao cruzada.
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3, # Validacao cruzada com 3 folds
    n_jobs=-1,
    verbose=2 # Mostra o progresso
)

# Executa a busca pelos melhores parametros.
grid_search.fit(X_train_scaled, y_train)

# Pega o melhor modelo encontrado pelo GridSearch.
best_rf_model = grid_search.best_estimator_
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

print("\nAvaliando o modelo otimizado...")
# Faz as previsões no conjunto de teste.
y_pred = best_rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f'\nAcurácia do Modelo Otimizado: {accuracy:.4f}')
print(classification_report(y_test, y_pred))


# Salva o melhor modelo encontrado para uso futuro.
joblib.dump(best_rf_model, "modelo_randomforest.pkl")
joblib.dump(scaler, "scaler.pkl") 
print("\nModelo otimizado e scaler salvos como 'modelo_randomforest.pkl' e 'scaler.pkl'")
