# Customer Loyalty Program for E-Commerce

<img src="https://github.com/nortonvanz/Fidelity-Program/blob/pa005_norton_vanz/images/fidelity_program.jpeg?raw=true" width=70% height=70% title="Clustering-for-Loyalty-Program" alt="project_cover_image"/>

Projeto de identificação do perfil de clientes mais valiosos, para a construção de um programa de fidelidade.

Contextualização:
Os dados do projeto foram obtidos do Kaggle, do desafio "High Value Customers Identification".

Desta forma, o contexto de negócios é fictício, porém todo o planejamento e desenvolvimento da solução é implementado seguindo todos os passos de um projeto de mercado real.

## 1. Problema de negócios
### 1.1 Problema
A Stand Out Brands é uma empresa que comercializa roupas de marca de segunda linha no modelo outlet.
Tendo atingindo a marca de 5000 clientes, o time de marketing percebeu que alguns deles compram produtos de alto ticket, com alta frequência, e contribuem de forma significativa no faturamento da empresa.

Tendo identificado esta oportunidade de aumentar o faturamento, decidiram criar um programa de fidelidade para estes clientes, chamado de "Loyals". Logo, precisam que seja criada uma estrutura que identifique o perfil dos clientes mais valiosos, bem como dos demais grupos de clientes.

Com base no programa Loyals, o time de marketing tomará ações direcionadas a este público, visando aumentar sua retenção.
Também fará ações através das redes sociais através de publicidade direcionada, visando atingir clientes com perfil similar, aumentando assim o número de clientes no programa.

### 1.2 Objetivo
Agrupar os mais de 5000 clientes em grupos por perfil de consumo, e identificar os clientes mais valiosos.

Além disso, as seguintes questões de negócio devem ser respondidas à área de marketing:

- Quem são as pessoas elegíveis para participar do programa Loyals?
- Quantos clientes farão parte do grupo?
- Quais são as principais características desses clientes?
- Qual a porcentagem de contribuição de faturamento, vinda do Loyals?
- Qual a expectativa de faturamento desse grupo para os próximos meses?
- Quais as condições para uma pessoa ser elegível ao Loyals?
- Quais as condições para uma pessoa ser removida do Loyals?
- Qual a garantia que o programa Loyals é melhor que o restante da base?
- Quais ações do time de marketing pode realizar para aumentar o faturamento?

## 2. Premissas de negócio
O time de marketing precisa visualizar os perfis de cada grupo de clientes dentro da ferramenta de visualização Metabase, já utilizada pela empresa.

## 3. Planejamento da solução
### 3.1. Produto final
O que será entregue efetivamente?
- Um dashboard dentro da ferramenta Metabase, que detalha os perfis de cada grupo de clientes.
- Respostas às questões de negócio.

### 3.2. Ferramentas
Quais ferramentas serão usadas no processo?
- Python 3.8.0;
- Jupyter Notebook;
- Git e Github;
- Coggle Mindmaps;
- Pandas Profiling, Metabase;
- Algoritmos de Clusterização;
- Técnicas de Embedding;
- Crontab e Papermill;
- Serviços AWS: S3 (armazenamento), EC2 (servidor) e RDS (banco de dados).


### 3.3 Processo
#### 3.3.1 Estratégia de solução
Com base no objetivo do projeto, trata-se de um projeto de clusterização (agrupamento).

Minha estratégia para resolver esse desafio, baseado na metodologia CRISP-DS, é detalhada pelo plano abaixo:

**Solution Planning**
- Planejamento da solução, considerando o contexto de negócio.

**Data Description:**
- Coletar dados na AWS EC2.
- Compreender o significado de cada atributo dos interessados.
- Renomear colunas, compreender dimensões e tipos dos dados.
- Identificar e tratar dados nulos.
- Padronizar tipos de dados.
- Analisar atributos através de estatística descritiva.

**Data Filtering:**
- Filtrar dados de acordo com análise da estatística descritiva.

**Feature Engineering:**
- Alterar a granularidade dos dados passando de nota fiscal emitida para cliente.
- Criar features na granularidade cliente.

**Exploratory Data Analysis I:**
- Realizar uma análise univariada com uso do Pandas Profiling, para:
  - Analisar variáveis com alta variabilidade, candidatas a insumo dos modelos.
  - Analisar outliers inconsistentes, a fim de filtrá-los nos próximos ciclos.
- Realizar uma análise bivariada, identificando visualmente variáveis sem variabilidade.

**Data Preparation:**
- Aplicar transformações nas features, facilitando o aprendizado dos modelos.

**Feature Selection:**
- Selecionar as features com maior variabilidade, visando melhorar a performance dos modelos.  
- Analisar o resultado em conjunto com a análise realizada na EDA.

**Exploratory Data Analysis II:**
- Realizar um estudo do espaço de dados, buscando um espaço mais organizado com embedding.

**Hyperparameter Fine Tuning:**
- Experimentação de modelos de clusterização com diferentes Ks (quantidade de grupos), no espaço de features e no espaço de embedding.
- Fazer um ajuste fino de hiperparâmetros em cada modelo, identificando o melhor conjunto de parâmetros para maximizar sua capacidade de aprendizagem.
- Comparação de performance dos modelos com SS Score em cada espaço.
- Escolher o melhor modelo e número de grupos, considerando performance e facilidade para tomada de decisão para o negócio.
- Realizar análise de silhueta, para verificação da qualidade da clusterização.

**Machine Learning Modeling:**
- Rodar o algoritmo escolhido com parâmetros e o número de K escolhidos, no espaço de dados escolhido.
- Confirmar performance com SS Score.

**Convert Model Performance to Business Values:**
- Criar planilha de perfil de cada grupo de clientes.
- Plotar resultados traduzindo ao time de negócios.

**Exploratory Data Analysis III:**
- Criar mindmap de hipóteses de negócio.
- Criar, priorizar e validar hipóteses.
- Responder as questões de negócio ao time de marketing.

**Deploy Modelo to Production:**
- Inserir dados no banco de dados AWS RDS.
- Planejar deploy, desenhando arquitetura da infra.
- Construir e testar a infra localmente.
- Construir e testar a infra em nuvem na AWS.
- Construir e validar o dashboard no Metabase.

**Planejamento de Infraestrutura do Projeto:**

<img src="https://github.com/nortonvanz/Fidelity-Program/blob/pa005_norton_vanz/images/architecture_planning.jpeg?raw=true" alt="architecture_planning" title="Planejamento de Infraestrutura Local e em Cloud" align="center" height="600" class="center"/>

## 4. Os 3 principais insights dos dados
Durante a análise exploratória de dados, foram gerados insights ao time de negócio, através da validação das hipóteses.

Insights são informações novas, ou que contrapõem crenças até então estabelecidas do time de negócio. São também acionáveis, possibilitando ação para direcionar resultados futuros.

**H1 - Os clientes do cluster Loyals têm uma média de produtos únicos comprados acima de 10% do total de compras.**

Verdadeiro: 50% do volume de produtos únicos veio do cluster Loyals.

* Insight de negócio: Aplicar estratégias de marketing para oferecer produtos similares para os demais clusters, tomando como base os comprados pelos Loyals, visando reduzir a distância entre ambos neste quesito.

**H2 - Os clientes do cluster Loyals apresentam um número médio de devoluções 10% abaixo da média da base total de clientes.**

Falso: o cluster Insider tem um número médio de devoluções 256% maior que a média da base total de clientes.

* Insight de negócio: Dado o número de devoluções bem acima do esperado, realizar levantamento detalhado de custos de logística reversa com área responsável, avaliando o impacto negativo frente as demais características positivas do Loyals.

**H3 - A receita mediana do cluster Loyals é 10% maior que a receita mediana de todos os clusters.**

Verdadeiro: a receita mediana do cluster Loyals é 155% (1.5x) maior do que a receita mediana de todos os clusters.

* Insight de negócio: Destinar time dedicado da área de marketing para cuidar do relacionamento com os Loyals, dada representatividade do faturamento dentro da empresa, em relação ao negócio.

# 5. Modelos de Machine Learning aplicados
Foram aplicados 4 modelos de clusterização: K-Means, GMM (Gaussian Mixture Model), HC (Hierarchical Clustering) e DBScan.

Os 4 modelos foram testados considerando 2 a 12 possíveis grupos de clientes (k). Acima de 12, dificultaria o trabalho do time de marketing.

Estes testes foram realizados tanto no espaço de features (espaço original de dados), como no espaço de embedding.

Os algoritmos utilizados para a criação dos espaços de embedding foram: PCA, UMAP, t-SNE e um embedding baseado em árvores com Random Forest.

# 6. Performance do modelo de Machine Learning
A performance dos modelos foi medida com a métrica SS (Silhouette Score), visto que ela é aplicável a todos os modelos de clusterização testados.

O melhor resultado foi obtido com o modelo HC, com k=8, no espaço de embedding gerado pelo UMAP, com SS = 0.55. Desta forma, este foi o modelo eleito para deploy em produção.

# 7. Resultados de Negócio
As questões de negócio foram respondidas dentro do Jupyter Notebook, no ciclo 8.

Referente a resultados financeiros, partimos do fato de que a receita mediana do cluster Loyals é (1.5x) maior do que a receita mediana de todos os clusters, como já exposto.

Com a premissa que o time de marketing da Stand Out Brands, através do projeto, aumentará em 10% o número de Loyals no próximo ano, teremos em 10% da base um aumento mediano de faturamento de 1.5x.

O número de clientes Loyals no último ano (373 dias) é: 1786.
O número de clientes Loyals esperado para o próximo ano é de: 1965.
Teremos portanto 179 novos clientes.

Assumindo a mesma mediana de faturamento por Loyal, a expectativa de incremento de faturamento é de $280665.

O detalhamento dos cálculos também encontra-se no Jupyter Notebook, no ciclo 8.

### Dashboard dos Grupos de Clientes no Metabase

<img src="https://github.com/nortonvanz/Fidelity-Program/blob/pa005_norton_vanz/images/loyals_dashboard.jpeg?raw=true" alt="loyals_dashboard" title="Dashboard dos Grupos de Clientes no Metabase" align="center" height="600" class="center"/>

# 8. Conclusões
Com base nos resultados de negócio, conclui-se que o objetivo do projeto foi atingido.

Com a solução de dados entregue, a Stand Out Brands possui agora um programa de fidelidade robusto e lucrativo.

Ações de marketing direcionadas para os demais grupos de clientes também poderão ser realizadas, aumentando ainda mais o alcance do trabalho desenvolvido.  

# 9. Melhorias futuras
- Criar mais features a partir das já existentes, buscando gerar mais insumos para o aprendizado dos modelos.
- Utilizar uma ferramenta para gerenciamento de ambiente virtual mais eficiente, como o Poetry.


## 10 Referências
* O Dataset foi obtido no [Kaggle](https://www.kaggle.com/vik2012kvs/high-value-customers-identification).
* A imagem utilizada é de uso livre e foi obtida no [Pexels](https://www.pexels.com/pt-br/foto/silhueta-de-pessoas-durante-o-por-do-sol-853168/).
