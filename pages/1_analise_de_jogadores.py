import streamlit as st
import pandas as pd
import time

import analises

# configuracao da pagina
st.set_page_config(
    page_title="Dashboard de Análise NBA",
    page_icon="🏀",
    layout="wide"
)


# O cache garante que os dados sejam carregados apenas uma vez, tornando o app mais rapido.
@st.cache_data
def carregar_dados(caminho):
    try:
        df = pd.read_pickle(caminho)
        return df
    except FileNotFoundError:
        return None

# carrega os dados
df_dados = carregar_dados('dados_limpos.pkl')

if df_dados is None:
    st.error("Arquivo 'dados_limpos.pkl' nao encontrado. Por favor, execute o script '0_preparar_dados_jogadores.py' primeiro.")
    st.stop() # app para caso nao tenha dados

# barra lateral
st.sidebar.title("🏀 Painel de Análise NBA")

# obter lista de jogadores com mais de 3 temporadas 
jogos_por_jogador = df_dados.groupby('player_name')['season_year'].nunique()
lista_jogadores = jogos_por_jogador[jogos_por_jogador > 3].index.sort_values().tolist()

# Selecionar jogador
jogador_selecionado = st.sidebar.selectbox(
    "Selecione um Jogador:",
    options=lista_jogadores,
    index=lista_jogadores.index("LeBron James") # Valor padrao
)

# Selecionar o tipo de analise
tipo_analise = st.sidebar.selectbox(
    "Escolha o que quer saber:",
    options=[
        "Curva da Carreira (Pontos)",
        "Desempenhos Anômalos (Jogos)",
        "Previsão para Próxima Temporada",
    ]
)

st.title(f"Análise de Desempenho: {jogador_selecionado}")
st.markdown("---")


if tipo_analise == "Curva da Carreira (Pontos)":
    st.header(f"📈 Curva da Carreira de {jogador_selecionado}")
    with st.spinner('Analisando as temporadas...'):
        fig, erro = analises.analisar_curva_carreira(df_dados, jogador_selecionado)
        if erro:
            st.warning(erro)
        else:
            st.pyplot(fig)

elif tipo_analise == "Desempenhos Anômalos (Jogos)":
    st.header(f"🚨 Jogos Anômalos de {jogador_selecionado}")
    st.markdown("Utilizando o modelo *Isolation Forest* para encontrar jogos com estatísticas fora do padrão habitual do jogador.")
    with st.spinner('Procurando por anomalias...'):
        df_anomalias, fig, erro = analises.detectar_anomalias(df_dados, jogador_selecionado)
        if erro:
            st.error(erro)
        else:
            st.subheader("Top Jogos Mais Anômalos")
            st.dataframe(df_anomalias)
            st.subheader("Dispersão: Pontos vs. Assistências")
            st.pyplot(fig)

elif tipo_analise == "Previsão para Próxima Temporada":
    st.header(f"🔮 Previsão de Pontos para {jogador_selecionado}")
    st.markdown("Usando um modelo de *Random Forest* treinado com dados de todas as temporadas para prever a média de pontos da próxima temporada.")
    with st.spinner(f'Calculando previsão para {jogador_selecionado}...'):
        resultado, erro = analises.prever_proxima_temporada(df_dados, jogador_selecionado)
        if erro:
            st.error(erro)
        else:
            col1, col2 = st.columns(2)
            col1.metric(
                label=f"Média de Pontos em {resultado['temporada_base']}",
                value=f"{resultado['pts_base']:.1f} PPG"
            )
            col2.metric(
                label=f"🔥 Previsão para {resultado['temporada_previsao']}",
                value=f"{resultado['pts_previstos']:.1f} PPG"
            )
