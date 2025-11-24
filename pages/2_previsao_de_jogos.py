import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuracao da pagina
st.set_page_config(page_title="Previsão de Jogos", page_icon="🔮", layout="wide")
st.title("🔮 Previsão de Jogos da NBA")
st.write("Escolha dois times e veja quem tem mais chances de vencer com base em um modelo de Machine Learning otimizado!")


# Carrega modelo e dados
@st.cache_data
def carregar_recursos():
    """ Carrega o modelo, o scaler e o dataset principal. """
    try:
        modelo = joblib.load("modelo_randomforest.pkl")
        scaler = joblib.load("scaler.pkl")
        df = pd.read_pickle('dados_completos.pkl')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(by='GAME_DATE').reset_index(drop=True)
        return modelo, scaler, df
    except FileNotFoundError:
        return None, None, None

modelo, scaler, df = carregar_recursos()

# Lista de estatisticas usadas no treinamento 
stats = [
    'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV',
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS'
]

# Preparacao dos dados
@st.cache_data
def preparar_dados_com_features(df_original):
    print("Executando engenharia de features (cache)...")
    df_features = df_original.copy()
    window_size = 10 
    
    for stat in stats:
        df_features[f'{stat}_avg'] = df_features.groupby('TEAM_NAME')[stat].transform(
            lambda x: x.shift(1).rolling(window=window_size).mean()
        )

    df_features['DIAS_DESCANSO'] = df_features.groupby('TEAM_NAME')['GAME_DATE'].diff().dt.days
    
    df_features['WL_numeric'] = df_features['WL'].apply(lambda x: 1 if x == 'W' else 0)
    derrota_streak_id = (df_features.groupby('TEAM_NAME')['WL_numeric'].shift(1) != df_features['WL_numeric']).cumsum()
    df_features['WINSTREAK'] = df_features.groupby(['TEAM_NAME', derrota_streak_id])['WL_numeric'].cumsum()
    df_features['WINSTREAK_anterior'] = df_features.groupby('TEAM_NAME')['WINSTREAK'].shift(1).fillna(0)

    df_features.dropna(inplace=True)
    return df_features

# Interface
if modelo is None or scaler is None or df is None:
    st.error("Erro ao carregar os recursos necessários (modelo, scaler ou dados). "
             "Por favor, execute os scripts de preparação e treinamento primeiro.")
else:
    # Prepara o DataFrame com todas as features
    df_com_features = preparar_dados_com_features(df)

    st.sidebar.header("Configure o Confronto")
    # Apenas times que tem dados suficientes para analise
    times_disponiveis = sorted(df_com_features["TEAM_NAME"].unique())

    time_a = st.sidebar.selectbox("🏀 Time da Casa (A)", times_disponiveis, index=0)
    # Time A diferente de time B
    index_b = 1 if len(times_disponiveis) > 1 else 0
    time_b = st.sidebar.selectbox("🏀 Time Visitante (B)", times_disponiveis, index=index_b)

    if st.sidebar.button("Fazer Previsão", type="primary"):
        if time_a == time_b:
            st.warning("Por favor, selecione dois times diferentes.")
        else:
            try:
                # Pega as informacoes mais recentes de cada time
                stats_a = df_com_features[df_com_features['TEAM_NAME'] == time_a].iloc[-1]
                stats_b = df_com_features[df_com_features['TEAM_NAME'] == time_b].iloc[-1]

            
                dados_para_prever_dict = {}
                stats_avg = [f'{s}_avg' for s in stats] + ['DIAS_DESCANSO', 'WINSTREAK_anterior']

                for stat in stats_avg:
                    diff_col_name = f'{stat}_diff'
                    diff_value = stats_a[stat] - stats_b[stat]
                    dados_para_prever_dict[diff_col_name] = diff_value
                
                # Cria um DataFrame de uma linha para a previsao
                dados_prontos_df = pd.DataFrame([dados_para_prever_dict])

                # Transforma os dados usando o SCALER que foi salvo
                dados_scaled = scaler.transform(dados_prontos_df)
                
                # Faz a previsao com o modelo
                predicao = modelo.predict(dados_scaled)[0]
                probabilidade = modelo.predict_proba(dados_scaled)[0]

                vencedor = time_a if predicao == 1 else time_b
                confianca = probabilidade[1] if predicao == 1 else probabilidade[0]

                # Exibe o resultado 
                st.subheader("Resultado da Previsão")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Vencedor Previsto", value=f"🏆 {vencedor}")
                with col2:
                    st.metric(label="Confiança do Modelo", value=f"{confianca:.1%}")

                with st.expander("Ver detalhes das estatísticas usadas na previsão"):
                    st.write(f"Comparando as médias dos últimos 10 jogos e a forma atual de cada time.")
                    # Cria um DataFrame para exibir os dados 
                    df_comparacao = pd.DataFrame({
                        "Estatística": [s.replace('_', ' ').title() for s in stats_avg],
                        f"{time_a}": [stats_a[s] for s in stats_avg],
                        f"{time_b}": [stats_b[s] for s in stats_avg],
                    })
                    st.dataframe(df_comparacao.set_index("Estatística").style.format("{:.2f}"))

            except IndexError:
                st.error("Não foi possível encontrar dados recentes suficientes para um ou ambos os times. "
                         "Eles podem não ter jogado o suficiente na base de dados.")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")