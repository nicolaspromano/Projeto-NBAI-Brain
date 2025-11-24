import streamlit as st

st.set_page_config(
    page_title="NBAI Brain",
    page_icon="🏀",
    layout="wide"
)

st.title("🧠🏀 Bem-vindo ao NBAI Brain")

st.markdown("---")

st.header("O que você vai encontrar aqui?")

st.markdown(
    """
    Este é um portal para análises e previsões sobre a NBA.
    
    Use o menu de navegação na barra lateral esquerda para explorar as diferentes ferramentas:
    
    ### Páginas Disponíveis:
    
    - **1️⃣ Análise de Jogadores:**
        - Visualize a curva de carreira de um jogador.
        - Veja quais foram os jogos mais anormais de um jogador.
        - Preveja a média de pontos para a próxima temporada de um jogador.
        
    - **2️⃣ Previsão de Jogos:**
        - Escolha dois times, veja qual time tem a maior probabilidade de vencer a partida.
        
    **👈 Comece selecionando uma página no menu ao lado!**
    """
)

st.sidebar.success("Selecione uma análise acima.")