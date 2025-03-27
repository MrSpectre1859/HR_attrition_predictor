import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import shap
import matplotlib.pyplot as plt
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

# Definição de categorias reais do dataset
BUSINESS_TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
DEPARTMENTS = ["Sales", "Research & Development", "Human Resources", "Marketing"]
EDUCATION_FIELDS = ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"]
JOB_ROLES = ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
             "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]

# Inicializando os dados no session_state se não existirem
if "funcionarios" not in st.session_state:
    st.session_state.funcionarios = pd.DataFrame(columns=[
        "EmployeeNumber", "Age", "BusinessTravel", "Department", "DistanceFromHome",
        "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
        "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "Over18", "OverTime",
        "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ])

# -------------------------------------------------------------
# CONFIGURAÇÃO GERAL DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="People Analytics",
    layout="wide"
)

# -------------------------------------------------------------
# PÁGINA: HOME
# -------------------------------------------------------------
def page_home():
    st.title("Bem-vindo ao People Analytics")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1], gap="large")

    # Sobre
    with col1:
        with st.container():
            st.markdown(
                "<div style='background-color: #17428C; padding: 20px; border-radius: 10px; text-align: center;'>"
                "<h3 style='color: white;'>Sobre</h3>"
                "<p style='color: white;'>Saiba mais sobre o projeto, o dataset e os desafios de ML em RH.</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Ir para Sobre", key="btn_sobre"):
                st.session_state.page = "Sobre"

    # Cadastro de Funcionários
    with col2:
        with st.container():
            st.markdown(
                "<div style='background-color: #67B9B0; padding: 20px; border-radius: 10px; text-align: center;'>"
                "<h3 style='color: white;'>Cadastro</h3>"
                "<p style='color: white;'>Cadastre novos funcionários na base de dados</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Ir para Cadastro", key="btn_cadastro"):
                st.session_state.page = "Cadastro"
    
    with col3:
        with st.container():
            st.markdown(
                "<div style='background-color: #E5531A; padding : 20px; border-radius: 10px; text-align: center;'>"
                "<h3 style='color: white;'> Catálogo</h3>"
                "<p style='color: wite;'> Veja a lista completa de funcionários da sua empresa</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Catálogo", key="btn_catalogo"):
                st.session_state.page = 'Catálogo'

    # Analytics
    with col4:
        with st.container():
            st.markdown(
                "<div style='background-color: #CE1431; padding: 20px; border-radius: 10px; text-align: center;'>"
                "<h3 style='color: white;'>Analytics</h3>"
                "<p style='color: white;'>Visualize análises avançadas, predições e explicações do modelo</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Ir para Analytics", key="btn_analytics"):
                st.session_state.page = "Analytics"
    
    # Chat com IA
    with col5:
        with st.container():
            st.markdown(
                "<div style='background-color: #1A73E8; padding: 20px; border-radius: 10px; text-align: center;'>"
                "<h3 style='color: white;'>Chat com IA</h3>"
                "<p style='color: white;'>Converse com a IA sobre seus funcionários e previsões</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Ir para Chat", key="btn_chat"):
                st.session_state.page = "Chat"


# -------------------------------------------------------------
# PÁGINA: SOBRE
# -------------------------------------------------------------
def page_sobre():
    st.title("Sobre o Projeto")

    # Visão Geral
    st.subheader("📌 Visão Geral")
    st.write(
        "O **People Analytics - Predição de Saída de Funcionários** foi desenvolvido com o intuito de simular uma ferramenta "
        "interativa e inteligente de análise preditiva de turnover. Além disso, ela foi construída com o intuito de fomentar o uso de metodologias e "
        "ferramentas tecnológicas no RH, visando uma tomada de decisão baseada em dados e menos 'feeling'.\n\n"
        "Você poderá explorar o catálogo de funcionários e cadastrar novos, realizar previsões sobre o risco de saída e "
        "interpretar os resultados. Além disso, é possível interagir com um **Chatbot** que utiliza o modelo **GPT-4-Turbo** "
        "para conversar sobre os funcionários e seus contextos."
    )

    # Tecnologias Utilizadas
    st.subheader("🚀 Tecnologias Utilizadas")
    st.write(
        "- **Python** 🐍\n"
        "- **Streamlit** (Interface Web)\n"
        "- **Pandas** (Manipulação de dados)\n"
        "- **Scikit-learn** (Treinamento do modelo)\n"
        "- **SHAP** (Explicabilidade do modelo)\n"
        "- **OpenAI API** (Chatbot com GPT-4-Turbo)\n"
        "- **Plotly & Matplotlib** (Gráficos)\n"
        "- **AWS EC2** (Hospedagem do aplicativo)"
    )

    # Objetivos Principais
    st.subheader("🎯 Objetivos Principais")
    st.write(
        "- Desenvolver um **modelo preditivo** capaz de identificar a probabilidade de saída de um funcionário (turnover) utilizando **Machine Learning**;\n"
        "- Criar um **dashboard** com visualizações detalhadas sobre alguns indicadores utilizados comumente pelo RH e pela gestão;\n"
        "- **Explicar as previsões** do modelo com o uso de **SHAP** para auxiliar a compreensão do RH;\n"
        "- Implementar um **chatbot inteligente** com o modelo GPT-4, permitindo interações personalizadas;\n"
        "- Fomentar o desenvolvimento de RH baseado em dados e mais tech."
    )

    # Sobre o Dataset
    st.subheader("📊 Sobre o Dataset")
    st.write(
        "Os dados utilizados são do **IBM HR Analytics Employee Attrition & Performance**, disponível no Kaggle. "
        "Este dataset contém informações detalhadas sobre funcionários, incluindo idade, departamento, anos na empresa, "
        "salário, envolvimento no trabalho e se o funcionário saiu da empresa (variável target: *Attrition*)."
    )

    # Desafios do uso de Machine Learning em RH
    st.subheader("⚠️ Desafios do Uso de Machine Learning em RH")
    st.write(
        "- **Dados Sensíveis**: Privacidade e ética são fundamentais ao analisar dados de funcionários.\n"
        "- **Viés Algorítmico**: Modelos podem amplificar desigualdades existentes se não forem bem treinados.\n"
        "- **Interpretação das Predições**: Decisões não devem ser tomadas apenas com base no modelo, mas sim como um apoio à gestão."
    )

    # Sobre Mim (Foto e Contatos)
    st.subheader("👤 Sobre Mim")
    st.image("eu.jpeg", width=150, caption="Eu")

    st.write(
        "👋 Olá! Meu nome é **Alison Machado Cesário**, sou Bacharel em Psicologia pela Universidade Federal Fluminense (UFF) e pós-graduando no MBA de Data Science e Analytics pela USP/ESALQ. "
        "Sou completamente apaixonado por tecnologia, inovação, dados e estou constantemente buscando algo novo para poder aprender. Comecei na programação em 2018 estudando Python 3 através as aulas do Prof. Guanabara no Curso em Vídeo e"
        " desde então não consegui parar.\n\n"
        "Minhas competências:\n\n"
        "1. Python\n"
        "2. SQL\n"
        "3. Power BI\n"
        "4. Data Storytelling\n"
        "5. ETL\n"
        "6. Machine Learning\n"
        "7. Conexão API\n\n"
        "📍 **e-mail**: [alissonmcesario@gmail.com](mailto:alissonmcesario@gmail.com)"
        "📍 **LinkedIn**: [alissonmcesario](https://www.linkedin.com/in/alissonmcesario)\n"
        "📍 **GitHub**: [MrSpectre1859](https://github.com/MrSpectre1859)"
    )

    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=False):
        st.session_state.page = "Home"


# -------------------------------------------------------------
# PÁGINA: CADASTRO DE FUNCIONÁRIOS
# -------------------------------------------------------------
def page_cadastro():
    st.title("📋 Cadastro de Funcionários")

    # Obtendo o maior EmployeeNumber existente para garantir números únicos
    if st.session_state.funcionarios.empty:
        max_employee_number = 1
    else:
        max_employee_number = st.session_state.funcionarios["EmployeeNumber"].max() + 1

    # Seção de Cadastro de Funcionários
    st.subheader("✍️ Cadastrar Novo Funcionário")

    col1, col2, col3 = st.columns(3)

    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=70, value=30)
        distance_home = st.number_input("Distância de Casa (km)", min_value=0, max_value=100, value=10)
        education = st.slider("Educação (1-Básico a 5-Doutorado)", 1, 5, 3)
        education_field = st.selectbox("Área de Formação", EDUCATION_FIELDS)

    with col2:
        department = st.selectbox("Departamento", DEPARTMENTS)
        job_role = st.selectbox("Cargo", JOB_ROLES)
        job_level = st.slider("Nível do Cargo", 1, 5, 2)
        business_travel = st.selectbox("Frequência de Viagem", BUSINESS_TRAVEL)
        marital_status = st.selectbox("Estado Civil", MARITAL_STATUS)

    with col3:
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        over_time = st.selectbox("Faz Hora Extra?", ["Sim", "Não"])
        salary_hike = st.number_input("Aumento Salarial (%)", min_value=0, max_value=50, value=10)
        monthly_income = st.number_input("Salário Mensal", min_value=1000, max_value=50000, value=5000)
        monthly_rate = st.number_input("Taxa Mensal", min_value=1000, max_value=50000, value=10000)

    st.subheader("📊 Outros Indicadores")
    col4, col5, col6 = st.columns(3)

    with col4:
        num_companies_worked = st.number_input("Número de Empresas Trabalhadas", min_value=0, max_value=10, value=2)
        training_last_year = st.slider("Treinamentos no Último Ano", 0, 10, 2)
        stock_option_level = st.slider("Opções de Ações (0-3)", 0, 3, 1)

    with col5:
        years_at_company = st.slider("Anos na Empresa", 0, 40, 5)
        years_in_current_role = st.slider("Anos no Cargo Atual", 0, 20, 3)
        years_with_manager = st.slider("Anos com o Gestor Atual", 0, 20, 4)

    with col6:
        job_involvement = st.slider("Envolvimento no Trabalho (1-4)", 1, 4, 3)
        job_satisfaction = st.slider("Satisfação no Trabalho (1-4)", 1, 4, 3)
        environment_satisfaction = st.slider("Satisfação com o Ambiente (1-4)", 1, 4, 3)

    col_btn1, col_btn2 = st.columns([1, 1])

    # Adicionar novo funcionário
    with col_btn1:
        if st.button("➕ Adicionar Funcionário", use_container_width=True):
            novo_funcionario = pd.DataFrame([[
                max_employee_number, idade, business_travel, department, distance_home,
                education, education_field, environment_satisfaction, gender, random.randint(10, 50),
                job_involvement, job_level, job_role, job_satisfaction, marital_status,
                monthly_income, monthly_rate, num_companies_worked, over_time,
                salary_hike, 3, random.randint(1, 4), stock_option_level, years_at_company + random.randint(1, 10),
                random.randint(1, 40), training_last_year, random.randint(1, 4), years_at_company, years_in_current_role,
                years_at_company - years_in_current_role, years_with_manager
            ]], columns=st.session_state.funcionarios.columns)

            st.session_state.funcionarios = pd.concat([st.session_state.funcionarios, novo_funcionario], ignore_index=True)
            st.success(f"Funcionário adicionado com sucesso!")

    # Gerar funcionário aleatório
    with col_btn2:
        if st.button("🎲 Gerar Dados Aleatórios", use_container_width=True):
            novo_funcionario = pd.DataFrame([[  
                max_employee_number, random.randint(20, 60), random.choice(BUSINESS_TRAVEL),  
                random.choice(DEPARTMENTS), random.randint(1, 50), random.randint(1, 5), random.choice(EDUCATION_FIELDS),  
                random.randint(1, 4), random.choice(["Masculino", "Feminino"]), random.randint(10, 50), random.randint(1, 4),  
                random.randint(1, 5), random.choice(JOB_ROLES), random.randint(1, 4), random.choice(MARITAL_STATUS),  
                random.randint(3000, 20000), random.randint(5000, 30000), random.randint(0, 10),  
                random.choice(["Yes", "No"]), random.randint(0, 50), 3, random.randint(1, 4), random.randint(0, 3),  
                random.randint(1, 40),  
                random.randint(1, 40),
                random.randint(0, 10), random.randint(1, 4), random.randint(1, 40),  
                random.randint(1, 20), random.randint(0, 10), random.randint(1, 20)  
            ]], columns=st.session_state.funcionarios.columns)

            st.session_state.funcionarios = pd.concat([st.session_state.funcionarios, novo_funcionario], ignore_index=True)
            st.success("Funcionário gerado aleatoriamente!")

    st.dataframe(st.session_state.funcionarios, use_container_width=True)

    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=False):
        st.session_state.page = "Home"

# -------------------------------------------------------------
# PÁGINA: CATÁLOGO DE FUNCIONÁRIOS
# -------------------------------------------------------------

def page_catalogo():
    st.title("📋 Catálogo de Funcionários")

    # Caso ainda não existam funcionários cadastrados
    if st.session_state.funcionarios.empty:
        st.warning("⚠️ Nenhum funcionário cadastrado ainda. Adicione novos funcionários na aba Cadastro!")
        if st.button("🏠 Voltar para Home", use_container_width=True):
            st.session_state.page = "Home"
        return

    # 🔍 Filtros Interativos
    st.subheader("🔍 Filtros")
    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_departamento = st.selectbox(
            "Filtrar por Departamento",
            ["Todos"] + list(st.session_state.funcionarios["Department"].unique())
        )

    with col2:
        filtro_cargo = st.selectbox(
            "Filtrar por Cargo",
            ["Todos"] + list(st.session_state.funcionarios["JobRole"].unique())
        )

    with col3:
        faixa_salarial = st.slider(
            "Filtrar por Faixa Salarial",
            int(st.session_state.funcionarios["MonthlyIncome"].min()),
            int(st.session_state.funcionarios["MonthlyIncome"].max()),
            (int(st.session_state.funcionarios["MonthlyIncome"].min()), int(st.session_state.funcionarios["MonthlyIncome"].max()))
        )

    # Aplicando os filtros na tabela
    df_filtrado = st.session_state.funcionarios.copy()

    if filtro_departamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Department"] == filtro_departamento]

    if filtro_cargo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["JobRole"] == filtro_cargo]

    df_filtrado = df_filtrado[
        (df_filtrado["MonthlyIncome"] >= faixa_salarial[0]) & 
        (df_filtrado["MonthlyIncome"] <= faixa_salarial[1])
    ]

    # 📌 Métricas Gerais
    st.subheader("📌 Métricas Gerais")
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric("Total de Funcionários", len(df_filtrado))

    with col_m2:
        st.metric("Média Salarial", f"R$ {df_filtrado['MonthlyIncome'].mean():,.2f}")

    with col_m3:
        st.metric("Idade Média", f"{df_filtrado['Age'].mean():.1f} anos")

    # 📋 Exibição da Tabela
    st.subheader("📋 Funcionários Cadastrados")
    st.dataframe(df_filtrado, use_container_width=True)

    # 🗑️ Opção para Excluir Funcionário
    if not df_filtrado.empty:
        st.subheader("🗑️ Remover Funcionário")
        funcionario_excluir = st.selectbox("Selecione um funcionário para excluir", df_filtrado["EmployeeNumber"].tolist())

        if st.button("❌ Excluir", use_container_width=False):
            st.session_state.funcionarios = st.session_state.funcionarios[st.session_state.funcionarios["EmployeeNumber"] != funcionario_excluir]
            st.success(f"Funcionário {funcionario_excluir} removido com sucesso!")
            st.rerun(scope="app")
    
    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=False):
        st.session_state.page = "Home"

# -------------------------------------------------------------
# PÁGINA: ANALYTICS
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.pkl") 
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_data
def load_data():
    df = pd.read_csv("hr_dataset.csv")

    # ===========================================
    #           PROCESSAMENTO DE DADOS
    # ===========================================
    data_prep = df.copy()

    # Transformações manuais - Simple Encoding
    data_prep['BusinessTravel'] = data_prep['BusinessTravel'].map({'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0})
    data_prep['Gender'] = data_prep['Gender'].map({'Male': 1, 'Female': 0})
    data_prep['OverTime'] = data_prep['OverTime'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding
    columns_to_dummy = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
    data_prep = pd.get_dummies(data_prep, columns=columns_to_dummy, drop_first=True, dtype='int64')

    # Criando variável target binária: Attrition_numerical
    data_prep['Attrition_numerical'] = data_prep['Attrition'].map({'Yes': 1, 'No': 0})

    # Removendo colunas desnecessárias
    cols_to_delete = ['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount', 'Attrition']
    data_prep.drop(cols_to_delete, axis=1, inplace=True)

    return df, data_prep

model = load_model()
df_original, df_prep = load_data()

@st.cache_resource
def get_explainer():
    return shap.Explainer(model, df_prep)

@st.cache_data
def compute_shap_values():
    with st.status("⏳ Estamos preparando o sistema para você.\n\n Isso costuma levar 1 minuto!", expanded=True) as status:
        explainer = get_explainer()
        shap_values_all = explainer(df_prep)
        status.update(label="✅ Sistema carregado com sucesso. Você já pode explorar os dados!", state="complete")
    return shap_values_all[..., 1]

shap_values_class1 = compute_shap_values()

def classify_risk(probs, low_threshold=0.10, medium_threshold=0.40):
    if probs < low_threshold:
        return "Baixo"
    elif probs < medium_threshold:
        return "Moderado"
    else:
        return "Alto"

def shap_func_id(funcionario_idx):
    "Gera a explicação SHAP para um funcionário específico"
    # Obtém os valores SHAP para o conjunto de dados
    shap_values = shap_values_class1[funcionario_idx]  # Índice do funcionário
    shap_importances = np.abs(shap_values.values).flatten()
    
    # Pega os 3 principais fatores que mais influenciam a saída
    top_indices = np.argsort(shap_importances)[-3:]  # Top 3 fatores
    top_features = df_prep.columns[top_indices]

    return top_features.tolist()

def page_analytics():
    st.title("📊 Analytics - Previsão de Saída de Funcionários")
    
    # Filtros interativos
    st.subheader("🔍 Filtros")
    col1, col2 = st.columns(2)
    
    with col1:
        filtro_departamento = st.selectbox(
            "Departamento",
            ["Todos"] + list(df_original["Department"].unique())
        )
    
    with col2:
        filtro_cargo = st.selectbox(
            "Cargo",
            ["Todos"] + list(df_original["JobRole"].unique())
        )
    
    # Aplicando filtros
    df_filtrado = df_original.copy()
    
    if filtro_departamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Department"] == filtro_departamento]
    if filtro_cargo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["JobRole"] == filtro_cargo]
    
    df_filtrado_prep = df_prep.loc[df_filtrado.index].drop(columns=["Attrition_numerical"], errors="ignore")

    # Fazer previsões no conjunto filtrado
    df_filtrado["Risk_Score"] = model.predict_proba(df_filtrado_prep)[:, 1]
    df_filtrado["Risk_Level"] = df_filtrado["Risk_Score"].apply(lambda x: classify_risk(x))

    # KPIs
    st.subheader("📌 KPIs")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.metric("Baixo Risco", (df_filtrado["Risk_Level"] == "Baixo").sum())
    with col_kpi2:
        st.metric("Risco Moderado", (df_filtrado["Risk_Level"] == "Moderado").sum())
    with col_kpi3:
        st.metric("Alto Risco", (df_filtrado["Risk_Level"] == "Alto").sum())
    
    # Gráficos
    st.subheader("📊 Distribuição de Risco")
    df_filtrado["Risk_Level"] = pd.Categorical(df_filtrado["Risk_Level"], categories=["Baixo", "Moderado", "Alto"], ordered=True)
    
    fig_risco = px.histogram(df_filtrado, x="Risk_Level", color="Risk_Level", title="Distribuição de Risco")
    st.plotly_chart(fig_risco, use_container_width=True)
    
    fig_pie = px.pie(df_filtrado, names="Risk_Level", title="Proporção de Risco")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Exibir tabela de risco
    st.subheader("📋 Funcionários e Classificação de Risco")
    st.dataframe(df_filtrado[["Risk_Score", "Risk_Level"]], use_container_width=True)
    
    # Adicionando um gráfico para visualizar a distribuição salarial por nível de risco
    st.subheader("💰 Salário por Nível de Risco")
    fig_box = px.box(df_filtrado, x="Risk_Level", y="MonthlyIncome", color="Risk_Level", title="Distribuição Salarial por Nível de Risco")
    st.plotly_chart(fig_box, use_container_width=True)

    # Seleção de funcionário para análise individual
    st.subheader("🔍 Análise Individual")
    if not df_filtrado.empty:
        funcionario_id = st.selectbox("Selecione um funcionário para análise", df_filtrado.index.tolist())
        func_detalhes = df_filtrado.loc[funcionario_id]

        st.write(f"**Funcionário:** {funcionario_id}")
        st.write(f"**Score de Risco:** {func_detalhes['Risk_Score']:.2f}")
        st.write(f"**Nível de Risco:** {func_detalhes['Risk_Level']}")
    else:
        st.warning("Nenhum funcionário encontrado com os filtros selecionados.")
    
        # Explicação SHAP
    st.subheader("📌 Explicação SHAP")

    # Exibir Waterfall apenas para um funcionário selecionado
    st.subheader("📊 Waterfall SHAP para o Funcionário")
    valores_um_func = shap_values_class1[funcionario_id]
    fig1, ax1 = plt.subplots()
    shap.waterfall_plot(valores_um_func)
    st.pyplot(fig1)

    # Bar Plot Global
    st.subheader("📊 Bar Plot Global (Classe 1)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values_class1, df_prep, plot_type="bar", show=False)
    st.pyplot(fig2)

    # Beeswarm Plot Global
    st.subheader("📊 Beeswarm Plot Global (Classe 1)")
    fig3, ax3 = plt.subplots()
    shap.summary_plot(shap_values_class1, df_prep, show=False)
    st.pyplot(fig3)

    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=False):
        st.session_state.page = "Home"

# -------------------------------------------------------------
# PÁGINA : CHAT COM IA
# -------------------------------------------------------------

# Inicialize o cliente com sua chave de API
client = OpenAI(api_key="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")

# 📌 Inicializar histórico de conversa na sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "Como posso te ajudar hoje?"}]

def page_chat():
    st.title("💬 Chat com IA - People Analytics")

    # Seleção do funcionário
    funcionario_idx = st.selectbox("Selecione um funcionário", list(range(len(df_original))))

    funcionario_info = df_original.iloc[funcionario_idx].to_dict()

    # Obtém a previsão do modelo para esse funcionário
    funcionario_prep = df_prep.iloc[funcionario_idx:funcionario_idx+1].drop(columns="Attrition_numerical")
    risk_score = model.predict_proba(funcionario_prep)[:, 1][0]  # Probabilidade de saída
    risk_level = classify_risk(risk_score)  # Classificação de risco (baixo, moderado, alto)

    # Obtém os fatores SHAP mais relevantes
    fatores_importantes = shap_func_id(funcionario_idx)

    resultados_modelo = (
        f"- Probabilidade de saída: {risk_score:.2%}\n"
        f"- Classificação de risco: {risk_level}\n\n"
        f"Principais fatores que influenciam essa previsão:\n"
        f"1) {fatores_importantes[0]}\n"
        f"2) {fatores_importantes[1]}\n"
        f"3) {fatores_importantes[2]}\n"
    )

    user_message = st.text_area("Digite sua pergunta:")

    if st.button("Enviar"):
        if user_message.strip():
            # Criando o prompt inicial com dados do funcionário
            mensagem_inicial = f"Os dados do funcionário são:\n{funcionario_info}\n\nSeguem os dados da previsão do modelo e explicação SHAP:\n{resultados_modelo}\nPergunta do usuário: {user_message}"

            # Adiciona ao histórico de conversa
            st.session_state.chat_history.append({"role": "user", "content": mensagem_inicial})

            try:
                resposta = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=st.session_state.chat_history,
                    temperature=1,
                    max_tokens=2048,
                    top_p=1
                )

                resposta_texto = resposta.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": resposta_texto})

            except APIError as e:
                st.error(f"Erro da API OpenAI: {e}")
            except APIConnectionError as e:
                st.error(f"Erro de conexão com a OpenAI: {e}")
            except RateLimitError as e:
                st.error(f"Limite de requisições excedido: {e}")

    # Exibir o histórico da conversa
    for mensagem in st.session_state.chat_history:
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])

    # Botão para resetar a conversa
    if st.button("Resetar Chat"):
        st.session_state.chat_history = [{"role": "system", "content": "Como posso te ajudar hoje?"}]
        st.rerun()
    
    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=False):
        st.session_state.page = "Home"

# -------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE ROTEAMENTO
# -------------------------------------------------------------
def main():
    # Se 'page' não existe ainda, definimos como 'Home'
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Roteamento simples baseado em st.session_state.page
    if st.session_state.page == "Home":
        page_home()
    elif st.session_state.page == "Sobre":
        page_sobre()
    elif st.session_state.page == "Cadastro":
        page_cadastro()
    elif st.session_state.page == "Catálogo":
        page_catalogo()
    elif st.session_state.page == "Analytics":
        page_analytics()
    elif st.session_state.page == "Chat":
        page_chat()
    else:
        st.session_state.page = "Home"
        st.error("Página não encontrada. Voltando para a Home.")


# -------------------------------------------------------------
# PONTO DE ENTRADA
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
