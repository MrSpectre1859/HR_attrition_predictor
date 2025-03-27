# Predição de Turnover (Attrition)

## 📌 Visão Geral
O **People Analytics - Predição de Saída de Funcionários** foi desenvolvido com o intuito de simular uma ferramenta interativa e inteligente de análise preditiva de turnover. Além disso, gostaria de fomentar o uso de metodologias e ferramentas tecnológicas no RH, visando uma tomada de decisão baseada em dados e menos "feeling". 

Você poderá explorar o catálogos de funcionários e cadastrar novos, realizar previsões sobre o risco de saída e interpretar os resultados. Além disso, é possível interagir com um **Chatbot** que utiliza o modelo **GPT-4-Turbo** para conversar sobre os funcionários e seus contexto.

🚀 Tecnologias Utilizadas

- **Python** 🐍
- **Streamlit** (Interface Web)
- **Pandas** (Manipulação de dados)
- **Scikit-learn** (Treinamento do modelo)
- **SHAP** (Explicabilidade do modelo)
- **OpenAI API** (Chatbot com GPT-4)
- **Plotly** & **Matplotlib** (Visualizações)
- **AWS EC2** (Hospedagem do aplicativo)

---

## 🎯 Objetivos Principais

- Desenvolver um **modelo preditivo** capaz de identificar a probabilidade de saída de umm funcionário (turnover) utilizando **Machine Learning**;
- Criar um **dashboard** com visualizações detalhadas sobre alguns indicadores utilizados comumente pelo RH e pela gestão;
- **Explicar as previsões** do modelo com o uso de **SHAP** para auxiliar a compreensão do RH,
- Implementar um **chatbot inteligente** com o modelo GPT-4, permitindo interações personalizadas;
- Fomentar o desenvolvimento de RH baseado em dados e mais tech.

---

## 🔍 Estrutura do Projeto

O projeto é composto por várias páginas:

### 1️⃣ Página Home
- Página inicial com **atalhos** para funcionalidades principais: Sobre, Cadastro de Funcionários, Catálogo, Analytics e Chat com IA.
- Interface **intuitiva** e acessível a todos os usuários.

### 2️⃣ Página Sobre
- Explicação do propósito do projeto e **desafios** do uso de **Machine Learning** em **Recursos Humanos**.
- Informações sobre o **dataset** utilizado (IBM HR Analytics Employee Attrition & Performance do Kaggle).
- Discussão sobre **questões éticas** e **viés algorítmico** no uso de IA em RH.

### 3️⃣ Página de Cadastro de Funcionários
- Adição de novos funcionários manualmente ou **geração de perfis aleatórios** para testes.
- Campos de entrada incluem: idade, cargo, departamento, salário, tempo na empresa, satisfação no trabalho, etc.
- Identificador único para cada funcionário (EmployeeNumber).

### 4️⃣ Página Catálogo de Funcionários
- Exibição de uma lista de funcionários cadastrados com **filtros interativos** por departamento, cargo e faixa salarial.
- Exibe métricas como **total de funcionários**, **média salarial**, etc.
- Opção de **remoção** de funcionários.

### 5️⃣ Página Analytics - Previsão de Turnover
- A página mais robusta do projeto, combinando **Machine Learning**, **visualizações interativas** e **explicabilidade do modelo**.

#### 🔹 Processamento de Dados
- Carregamento do **dataset** e transformação de variáveis categóricas.
- Criação da variável target binária para indicar se o funcionário saiu ou não.

#### 🔹 Modelo Preditivo
- Utilização do **Random Forest Classifier** para prever o risco de saída dos funcionários.
- O modelo é **salvo** e carregado em tempo real.

#### 🔹 Classificação de Risco
- Risco classificado em três categorias: **Baixo Risco**, **Risco Moderado** e **Alto Risco**.

#### 🔹 Data Viz
- **Histogramas** e **gráficos de pizza** para a distribuição de risco.
- **Boxplot** para a distribuição salarial por nível de risco.
- **Lista interativa** de funcionários com seus respectivos scores de risco.

#### 🔹 Explicação SHAP
- Visualizações SHAP incluem:
  - **Waterfall Plot**: Explicação individual para um funcionário.
  - **Bar Plot Global**: Importância média das variáveis.
  - **Beeswarm Plot**: Distribuição dos impactos SHAP.

---

### 6️⃣ Página Chat com IA (GPT-4)

- **Chatbot interativo** baseado no modelo **GPT-4-Turbo**, permitindo que os usuários consultem a IA para obter insights sobre os funcionários.
- O chat começa com a pergunta: "Como posso te ajudar hoje?", e o modelo responde com base no histórico e contexto dos funcionários selecionados.

#### 🔹 Funcionamento do Chatbot
- O **usuário** seleciona um funcionário e interage com a IA, que gera respostas personalizadas.
- O histórico da conversa é **armazenado** para manter a continuidade.

---

## 📌 Oportuniades de melhoria

- Otimizar a **performance** do sistema, especialmente no carregamento de dados e na explicabilidade SHAP;
- Implementar alguns dos **streamlit components** para melhorar a experiência do usuário;
- Explorar **modelos alternativos** para melhorar a precisão de Machine Learning e Deep Learning

---

## 📢 Conclusão

Este projeto simula uma solução  de **People Analytics**, combinando **inteligência artificial**, **visualização de dados** e **explicabilidade** para análise de turnover. A funcionalidade de **chatbot GPT-4-Turbo** proporciona uma interação inteligente e personalizada, oferecendo insights estratégicos baseados nos dados reais dos funcionários.

---

**Caso tenha dúvidas ou queira contribuir com o projeto**, sinta-se à vontade para abrir uma **issue**, enviar um **pull request** ou **entrar em contat via e-mail**.
