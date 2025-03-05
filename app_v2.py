import streamlit as st
import pandas as pd
import random

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
    st.markdown("<p style='text-align: center;'>Selecione uma opção abaixo para navegar:</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="large")

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
                "<p style='color: white;'>Cadastre novos funcionários na base de dados.</p>"
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
                "<p style='color: wite;'> Veja a lista completa de funcionários da sua empresa.</p>"
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
                "<p style='color: white;'>Visualize análises avançadas, predições e explicações do modelo.</p>"
                "</div>",
                unsafe_allow_html=True
            )
            if st.button("Ir para Analytics", key="btn_analytics"):
                st.session_state.page = "Analytics"


# -------------------------------------------------------------
# PÁGINA: SOBRE
# -------------------------------------------------------------
def page_sobre():
    st.title("Sobre o Projeto")

    # Descrição do projeto
    st.subheader("📌 O que é este projeto?")
    st.write(
        "Este projeto foi criado para demonstrar como **People Analytics** pode ser utilizado "
        "para prever a saída de funcionários, identificar padrões de comportamento e apoiar decisões estratégicas "
        "baseadas em dados dentro do setor de Recursos Humanos."
    )

    # Explicação sobre o dataset
    st.subheader("📊 Sobre o Dataset")
    st.write(
        "Os dados utilizados são do **IBM HR Analytics Employee Attrition & Performance**, disponível no Kaggle. "
        "Este dataset contém informações detalhadas sobre funcionários, incluindo idade, departamento, anos na empresa, "
        "salário, envolvimento no trabalho e se o funcionário saiu da empresa (variável target: *Attrition*)."
    )

    # Desafios do uso de ML em RH
    st.subheader("⚠️ Desafios do Uso de Machine Learning em RH")
    st.write(
        "- **Dados Sensíveis**: Privacidade e ética são fundamentais ao analisar dados de funcionários.\n"
        "- **Viés Algorítmico**: Modelos podem amplificar desigualdades existentes se não forem bem treinados.\n"
        "- **Interpretação das Predições**: Decisões não devem ser tomadas apenas com base no modelo, mas sim como "
        "um apoio à gestão."
    )

    # Sobre Mim (Espaço para adicionar sua foto e perfil)
    st.subheader("👤 Sobre Mim")
    

    st.image("extras/eu.jpeg", width=150, caption="Eu")
    
    st.write(
            "👋 Olá! Meu nome é Alisson Machado Cesário e sou um profissional de Recursos Humanos e Ciência de Dados.\n"
            "Atuo na interseção entre tecnologia e gestão de pessoas, utilizando dados para gerar insights estratégicos.\n\n"
            "📍 **LinkedIn:** [alissonmcesario](www.linkedin.com/in/alissonmcesario)\n\n"
            "📍 **GitHub:** [MrSpectre1859](https://github.com/MrSpectre1859)"
        )

    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=True):
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
                random.randint(1, 40),  # 🔥 Adicionado: TotalWorkingYears  
                random.randint(0, 10), random.randint(1, 4), random.randint(1, 40),  
                random.randint(1, 20), random.randint(0, 10), random.randint(1, 20)  
            ]], columns=st.session_state.funcionarios.columns)

            st.session_state.funcionarios = pd.concat([st.session_state.funcionarios, novo_funcionario], ignore_index=True)
            st.success("Funcionário gerado aleatoriamente!")

    st.dataframe(st.session_state.funcionarios, use_container_width=True)

    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=True):
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

        if st.button("❌ Excluir", use_container_width=True):
            st.session_state.funcionarios = st.session_state.funcionarios[st.session_state.funcionarios["EmployeeNumber"] != funcionario_excluir]
            st.success(f"Funcionário {funcionario_excluir} removido com sucesso!")
            st.rerun(scope="app")

# -------------------------------------------------------------
# PÁGINA: ANALYTICS
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Função do score
def classify_risk(probs, low_threshold=0.10, medium_threshold=0.40):
    if probs < low_threshold:
        return "Baixo"
    elif probs < medium_threshold:
        return "Moderado"
    else:
        return "Alto"

def page_analytics():
    st.title("📊 Analytics - Previsão de Saída de Funcionários")

    # Verificar se há funcionários cadastrados
    if st.session_state.funcionarios.empty:
        st.warning("⚠️ Nenhum funcionário cadastrado ainda. Adicione novos funcionários na aba Cadastro!")
        if st.button("🏠 Voltar para Home", use_container_width=True):
            st.session_state.page = "Home"
        return
    
    # Filtros interativos
    st.subheader("🔍 Filtros")
    col1, col2 = st.columns(2)
    
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
    
    df_filtrado = st.session_state.funcionarios.copy()
    
    if filtro_departamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Department"] == filtro_departamento]
    if filtro_cargo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["JobRole"] == filtro_cargo]
    
    # Previsão de risco
    X = df_filtrado.drop(columns=["EmployeeNumber", "Over18", "JobRole", "Department", "EducationField", "MaritalStatus", "BusinessTravel", "Gender", "OverTime"], errors='ignore')
    X = X.fillna(0)  # Lidando com valores nulos
    
    df_filtrado["Risk_Score"] = model.predict_proba(X)[:, 1]
    df_filtrado["Risk_Level"] = df_filtrado["Risk_Score"].apply(lambda x: classify_risk(x))
    
    # KPIs
    st.subheader("📌 KPI Cards")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.metric("Funcionários com Baixo Risco", (df_filtrado["Risk_Level"] == "Baixo").sum())
    with col_kpi2:
        st.metric("Funcionários com Risco Moderado", (df_filtrado["Risk_Level"] == "Moderado").sum())
    with col_kpi3:
        st.metric("Funcionários com Alto Risco", (df_filtrado["Risk_Level"] == "Alto").sum())
    
    # Gráficos
    st.subheader("📊 Distribuição de Risco")
    fig_risco = px.histogram(df_filtrado, x="Risk_Level", color="Risk_Level", title="Distribuição de Risco")
    st.plotly_chart(fig_risco, use_container_width=True)
    
    # Exibir tabela de risco
    st.subheader("📋 Funcionários e Classificação de Risco")
    st.dataframe(df_filtrado[["EmployeeNumber", "Risk_Score", "Risk_Level"]], use_container_width=True)
    
    # Seleção de funcionário para análise individual
    st.subheader("🔍 Análise Individual")
    funcionario_id = st.selectbox("Selecione um funcionário para análise", df_filtrado["EmployeeNumber"].tolist())
    func_detalhes = df_filtrado[df_filtrado["EmployeeNumber"] == funcionario_id].iloc[0]
    
    st.write(f"**Funcionário:** {funcionario_id}")
    st.write(f"**Score de Risco:** {func_detalhes['Risk_Score']:.2f}")
    st.write(f"**Nível de Risco:** {func_detalhes['Risk_Level']}")
    
    # Botão para voltar à Home
    st.divider()
    if st.button("🏠 Voltar para Home", use_container_width=True):
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
    else:
        st.session_state.page = "Home"
        st.error("Página não encontrada. Voltando para a Home.")


# -------------------------------------------------------------
# PONTO DE ENTRADA
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
