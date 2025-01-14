
## 1. Introdução

Este código realiza uma análise abrangente dos dados de vendas, desde a preparação dos dados até a criação de modelos preditivos. Ele utiliza diversas bibliotecas Python para manipulação, visualização e modelagem de dados.

### 1.1. Bibliotecas Utilizadas

As seguintes bibliotecas são importadas e utilizadas neste código:

*   **`numpy`:** Para operações matemáticas e numéricas.
*   **`pandas`:** Para manipulação e análise de dados tabulares.
*   **`seaborn`:** Para visualizações estatísticas.
*   **`matplotlib`:** Para criação de gráficos e visualizações.
*   **`matplotlib.ticker`:** Para formatar os eixos dos gráficos.
*   **`IPython.display`:** Para exibir imagens e iframes.
*   **`scipy.interpolate`:** Para interpolação de dados (spline).
*   **`sklearn`:** Para modelagem e métricas de Machine Learning.
*   **`xgboost`:** Para criação de modelos XGBoost.
*   **`graphviz` e `pydotplus`:** Para visualizar árvores de decisão.
*   **`ydata_profiling`:** Para gerar relatórios de perfil dos dados.

## 2. Exploração dos Dados

### 2.1. Carregamento e Pré-processamento

*   **Carregar os dados:** A base de dados, armazenada em um arquivo CSV, é carregada em um DataFrame do pandas (`case_csv`).
*   **Visualizar colunas:** As colunas do DataFrame são exibidas para facilitar a referência (`case_csv.columns`).
*   **Entender valores únicos:** A quantidade de valores únicos em cada coluna é analisada (`case_csv.nunique()`).
*   **Visualizar os primeiros registros:** Os primeiros registros do DataFrame são exibidos (`case_csv.head()`).
*   **Converter tipos de dados:**
    *   A coluna `cod_ano` é convertida para inteiro, preenchendo valores nulos com os 4 primeiros caracteres da coluna `cod_ciclo`, e em seguida, convertida em inteiro.
    *   A coluna `vlr_desconto_real` é preenchida com zeros onde há valores nulos e, em seguida, convertida para inteiro.
*   **Transformar `cod_agrupador_sap_material`:** Os valores originais são mapeados para valores sequenciais (ex: `prod01`, `prod02`), para economizar memória no ProfileReport. Uma tabela de-para é criada para referência.
*   **Transformar `cod_ciclo`:** Os valores originais de `cod_ciclo` são mapeados para valores numéricos sequenciais (1, 2, 3...), também criando uma tabela de-para para referência.
*   **Remover linhas:** As linhas do ano de 2018 e os outliers de `vlr_preco_base` (maiores que 200) são removidas do DataFrame.
*   **Arredondar colunas:** As colunas `vlr_rbv_tabela_so_tt`, `vlr_rbv_real_so_tt`, `vlr_preco_base`, e `vlr_preco_venda` são arredondadas para duas casas decimais.
*   **Criar colunas de itens vendidos:** As colunas `itens_vendidos_base` e `itens_vendidos_real` são criadas a partir das divisões de receita pelos respectivos preços.
*   **Função `stat_coluna`:** Uma função é criada para calcular estatísticas descritivas de uma coluna específica, agrupando por outra coluna.
*   **Gerar Profile Report:** O `ProfileReport` da biblioteca `ydata_profiling` é usado para gerar um relatório de análise exploratória em HTML.

### 2.2. Visualização e Análise Exploratória

*   **Definir bins:** É definida uma quantidade máxima de ciclos (`max_bins`), que corresponde ao maior número de ciclos distintos em todos os anos.
*   **Criar coluna auxiliar `bin`:** Uma coluna `bin` é criada para ordenar os ciclos dentro de cada ano.
*   **Plotar a distribuição das colunas ao longo dos anos:** Para as colunas definidas em `Colunas_plot`, é gerado um gráfico de linha para cada ano, mostrando a média da coluna ao longo dos ciclos, agrupada por UF.
*   **Plotar a evolução do desconto:** Um gráfico de linha mostra a soma dos descontos ao longo dos ciclos.

## 3. Extração de Dados para Análise

### 3.1. Criação de Tabelas Resumo

*   **Tabela resumo geral:** Uma tabela com o número de ciclos, soma da receita real e planejada, e a soma dos itens vendidos por ano é criada.
*   **Tabela resumo por canal e UF:** Uma tabela contendo as mesmas informações da tabela resumo geral, porém agregada por ano, canal e UF é criada.
*   **Tabela top SKUs:** Uma tabela com os SKUs que mais venderam em 2023 é gerada, incluindo as informações da subcategoria, marca, e UF.
*   **Tabela para plotagem do gráfico de receita e desconto:** Uma tabela agrupada por ciclo é criada, calculando a taxa de desconto.
*   **Plotar gráfico de receita e desconto:** Gráfico de linhas e barras é plotado comparando as receitas e os descontos de 2022 e 2023.
*   **Calcular e plotar a correlação:** A correlação entre a receita real e o desconto é calculada e plotada por ano.
*    **Criação e Plotagem da representatividade por público:** O código calcula a representatividade de cada público na receita total por ano e exibe essa informação em um gráfico de áreas empilhadas.
*   **Tabela resultado por público:** Uma tabela é gerada com o número de ciclos, soma da receita real e planejada, e a soma dos itens vendidos por ano, agrupada por canal e público abordado.
*   **Tabela resultado por campanha:** Uma tabela é gerada com o número de ciclos, soma da receita real e planejada, e a soma dos itens vendidos por ano, agrupada por mecânica de consumidor e subcategoria do produto.
*   **Criação da tabela para o modelo de ML:** Uma tabela auxiliar é criada, agrupando por ano, ciclo, canal, UF, público abordado, e mecânica de consumidor, e calculando algumas métricas (preço médio e desconto) e a receita em escala logarítmica.

## 4. Criação do Modelo de Machine Learning

### 4.1. Pré-processamento dos Dados para o Modelo

*   **Label Encoding:** As colunas categóricas (canal, público, UF, mecânica) são convertidas em numéricas usando `LabelEncoder`. Tabelas de-para são criadas para referência.
*   **Criação da classe `XGBRegressorWrapper`:** Essa classe foi criada para corrigir um possível erro com a versão do xgboost e a versão do python.

### 4.2. Treinamento e Avaliação do Modelo

*   **Definição dos modelos:** Uma lista de modelos de Machine Learning para teste é criada.
*   **Definição das variáveis e do target:** As colunas que serão utilizadas como variáveis preditivas e o target (receita em escala logarítmica) são definidos.
*    **Funções para calcular as métricas:** São criadas as funções `rmse`, `mse_scorer` e `r2_scorer` para avaliar a performance dos modelos.
*   **Cross-Validation:** Os modelos são avaliados usando cross-validation com KFold. As métricas MSE, RMSE, e R² são printadas para cada modelo.
*   **Separar treino e teste:** Os dados são separados em conjuntos de treino e teste (80/20).
*   **Preparação dos dados para o XGBoost:** Os dados são convertidos em matrizes `xgb.DMatrix`.
*   **Treinamento do XGBoost:** O modelo XGBoost é treinado com parâmetros definidos.
*   **Testando o modelo:** As métricas MSE, RMSE e R² do modelo XGBoost são printadas.
*   **Calculando e plotando a correlação:** A correlação entre as variáveis é calculada e plotada usando um heatmap.
*   **Teste de otimização do parâmetro `alpha`:** Comentado no código, o teste para otimizar o parâmetro `alpha` é deixado comentado.
*   **Plotagem de dispersão e linha:** São plotados gráficos de dispersão e linha comparando os valores preditos e os valores reais.
*   **Treinamento do Random Forest:** O modelo Random Forest é treinado, as métricas são printadas e são feitos os mesmo plots do XGBoost para melhor compreensão.
*   **Visualização da árvore:** Uma das árvores do modelo Random Forest é exportada e visualizada.
*    **Plotagem da importancia das variáveis:** Um gráfico de barras horizontais mostra a importância de cada variável para o modelo XGBoost, utilizando o método "gain".

## 5. Conclusão

Este documento fornece uma visão geral do código de análise de dados de vendas. Ele descreve as etapas desde o carregamento e pré-processamento dos dados até a criação de modelos preditivos. O código utiliza diversas técnicas de análise de dados, visualização e Machine Learning para extrair insights valiosos.
