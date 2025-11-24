## Autores

* Nicolas Romano
* Maria Eduarda Romana

---

# 🏀 NBAI Brain: Análise e Previsão na NBA

O **NBAI Brain** é um projeto de **Sistemas Inteligentes** desenvolvido em Python que utiliza modelos de *Machine Learning* para extrair *insights* do desempenho de jogadores e prever resultados de jogos da NBA. A aplicação é totalmente interativa e construída com **Streamlit**.

## 💡 Ferramentas de Inteligência Aplicada

O projeto é dividido em duas seções principais, cada uma utilizando uma abordagem de ML específica:

### 1. 🔍 Análise de Jogadores

Focada em *insights* de carreira por meio de:

| Análise | Modelo de ML Utilizado | Objetivo |
| :--- | :--- | :--- |
| **Curva da Carreira** | **Regressão Polinomial** | Ajusta uma curva de tendência aos pontos médios do jogador por temporada. |
| **Desempenhos Anômalos** | **Isolation Forest** (Não Supervisionado) | Identifica jogos com estatísticas (Pts, Ast, Reb, etc.) fora do padrão habitual do jogador, como *outliers*. |
| **Previsão de Pontos** | **Random Forest Regressor** | Prever a média de pontos por jogo na próxima temporada com base em métricas avançadas e tendências históricas de **todos** os jogadores. |

### 2. 🔮 Previsão de Jogos

Focada na previsão do resultado de um confronto direto:

| Análise | Modelo de ML Utilizado | Abordagem |
| :--- | :--- | :--- |
| **Previsão Vencedor** | **Random Forest Classifier** | O modelo é treinado em features baseadas na **diferença** entre as médias móveis (últimos 10 jogos) e *streaks* (sequências de vitórias/derrotas) dos times para prever o vencedor (`WIN` ou `LOSS`). |

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos

Certifique-se de ter o Python instalado (versão 3.8+ recomendada) e as bibliotecas necessárias:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn joblib tensorflow
```

### Estrutura de Dados
O projeto depende dos seguintes arquivos gerados previamente pelos scripts de preparação e treinamento:

* dados_limpos.pkl (Dados de jogadores)

* dados_completos.pkl (Dados de times)

* modelo_randomforest.pkl (Modelo de previsão de jogos)

* scaler.pkl

### Fontes de Dados

Os dados brutos utilizados para o treinamento e análise deste projeto foram coletados e compilados a partir do repositório:

* **Nome do Repositório:** NBA Data 2010-2024
* **Autor:** NocturneBear
* **Link:** [https://github.com/NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024)

### Inicialização 
Para iniciar a aplicação web interativa:
```bash
python -m streamlit run app.py
```
