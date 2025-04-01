import streamlit as st
import pandas as pd
import random
from utils import load_data

BUSINESS_TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
DEPARTMENTS = ["Sales", "Research & Development", "Human Resources", "Marketing"]
EDUCATION_FIELDS = ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"]
JOB_ROLES = ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
             "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]

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

def page_cadastro():
    df_original, _ = load_data()

    if st.session_state.funcionarios.empty:
        max_employee_number = df_original["EmployeeNumber"].max() + 1
    else:
        max_existing_id = st.session_state.funcionarios["EmployeeNumber"].max()
        max_employee_number = max(df_original["EmployeeNumber"].max(), max_existing_id) + 1

    st.subheader("✍️ Cadastrar Novo Funcionário")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Idade", min_value=18, max_value=70, value=30)
        distance_home = st.number_input("Distância de Casa (km)", min_value=0, max_value=100, value=10)
        education = st.selectbox("Educação", [1, 2, 3, 4, 5], index=2)
        education_field = st.selectbox("Área de Formação", EDUCATION_FIELDS)
        marital_status = st.selectbox("Estado Civil", MARITAL_STATUS)

    with col2:
        department = st.selectbox("Departamento", DEPARTMENTS)
        job_role = st.selectbox("Cargo", JOB_ROLES)
        job_level = st.selectbox("Nível do Cargo", [1, 2, 3, 4, 5], index=1)
        business_travel = st.selectbox("Frequência de Viagem", BUSINESS_TRAVEL)
        monthly_rate = st.number_input("Taxa Mensal", min_value=1000, max_value=50000, value=10000)

    with col3:
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        over_time = st.selectbox("Faz Hora Extra?", ["Sim", "Não"])
        salary_hike = st.number_input("Aumento Salarial (%)", min_value=0, max_value=50, value=10)
        monthly_income = st.number_input("Salário Mensal", min_value=1000, max_value=50000, value=5000)
        num_companies_worked = st.number_input("Número de Empresas Trabalhadas", min_value=0, max_value=10, value=2)

    col4, col5, col6 = st.columns(3)

    with col4:
        training_last_year = st.slider("Treinamentos no Último Ano", 0, 10, 2)
        stock_option_level = st.slider("Opções de Ações (0-3)", 0, 3, 1)
        years_with_manager = st.slider("Anos com o Gestor Atual", 0, 20, 4)


    with col5:
        years_at_company = st.slider("Anos na Empresa", 0, 40, 5)
        years_in_current_role = st.slider("Anos no Cargo Atual", 0, 20, 3)
        environment_satisfaction = st.slider("Satisfação com o Ambiente (1-4)", 1, 4, 3)


    with col6:
        job_involvement = st.slider("Envolvimento no Trabalho (1-4)", 1, 4, 3)
        job_satisfaction = st.slider("Satisfação no Trabalho (1-4)", 1, 4, 3)

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("➕ Adicionar Funcionário", use_container_width=True):
            novo_funcionario = pd.DataFrame([[
                max_employee_number, age, business_travel, department, distance_home,
                education, education_field, environment_satisfaction, gender, random.randint(10, 50),
                job_involvement, job_level, job_role, job_satisfaction, marital_status,
                monthly_income, monthly_rate, num_companies_worked, "Y", over_time,
                salary_hike, 3, random.randint(1, 4), stock_option_level, years_at_company + random.randint(1, 10),
                training_last_year, random.randint(1, 4), years_at_company, years_in_current_role,
                years_at_company - years_in_current_role, years_with_manager
            ]], columns=st.session_state.funcionarios.columns)

            st.session_state.funcionarios = pd.concat([st.session_state.funcionarios, novo_funcionario], ignore_index=True)
            st.success("Funcionário adicionado com sucesso!")

    with col_btn2:
        if st.button("🎲 Gerar Dados Aleatórios", use_container_width=True):
            novo_funcionario = pd.DataFrame([[  
                max_employee_number, random.randint(20, 60), random.choice(BUSINESS_TRAVEL),  
                random.choice(DEPARTMENTS), random.randint(1, 50), random.randint(1, 5), random.choice(EDUCATION_FIELDS),  
                random.randint(1, 4), random.choice(["Masculino", "Feminino"]), random.randint(10, 50), random.randint(1, 4),  
                random.randint(1, 5), random.choice(JOB_ROLES), random.randint(1, 4), random.choice(MARITAL_STATUS),  
                random.randint(3000, 20000), random.randint(5000, 30000), random.randint(0, 10),  
                "Y", random.choice(["Yes", "No"]), random.randint(0, 50), 3, random.randint(1, 4), random.randint(0, 3),  
                random.randint(1, 40), random.randint(0, 10), random.randint(1, 4), random.randint(1, 40),  
                random.randint(1, 20), random.randint(0, 10), random.randint(1, 20)  
            ]], columns=st.session_state.funcionarios.columns)

            st.session_state.funcionarios = pd.concat([st.session_state.funcionarios, novo_funcionario], ignore_index=True)
            st.success("Funcionário gerado aleatoriamente!")

    st.divider()


    st.dataframe(st.session_state.funcionarios, use_container_width=True)

page_cadastro()
