#########################
# app.py
#########################

# ----------------------------
# IMPORTAÇÕES
# ----------------------------
import streamlit as st
import joblib
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
import shap

from data_processing import preprocess_data  # deve conter a função preprocess_data

#########################
# 1) CACHING & CARREGAMENTO
#########################

@st.cache_data
def load_dataset():
    """Carrega o dataset bruto de HR (cacheado para não recarregar)."""
    df = pd.read_csv("../02. Datasets/hr_employee_attrition.csv")
    return df

@st.cache_resource
def load_model():
    """Carrega o modelo RandomForest já treinado (cacheado como 'resource')."""
    rf_model = joblib.load("../04. ML Model/random_forest_model.pkl")
    return rf_model

@st.cache_data
def load_feature_order():
    """Carrega a ordem das colunas."""
    with open("../03. Outputs/features_order.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    return feature_names

@st.cache_data
def get_preprocessed_data(df: pd.DataFrame):
    """
    Aplica preprocess_data em todo o dataframe.
    Em tese, se df não muda, isso fica cacheado.
    """
    return preprocess_data(df)

@st.cache_resource
def get_shap_explainer_and_values(_model, X_full):
    """
    Cria o TreeExplainer (cacheado como recurso) e calcula shap_values.
    X_full deve ser o dataset preprocessado.
    """
    shap_explainer = shap.TreeExplainer(_model)
    shap_values = shap_explainer.shap_values(X_full)  # para binário => list of 2 arrays
    return shap_explainer, shap_values


#########################
# 2) PÁGINAS
#########################

# ---------------------------------------------------------------
# PÁGINA 1: DISCLAIMER
# ---------------------------------------------------------------
def page_disclaimer():
    st.header("SOBRE O APP")
    st.markdown("""
    **Olá, seja bem-vindo(a) ao meu WebApp de People Analytics**, criado para demonstrar:
    
    1. O uso de um dataset (IBM HR) que contém informações de funcionários, incluindo quem saiu e quem permaneceu;
    2. O processo de análise e modelagem para prever a *rotatividade* (Attrition) dos colaboradores;
    3. Uma forma de **explicar** as predições que o modelo faz, usando SHAP;
    4. Um protótipo de **cadastro de novos funcionários**.
    
    ---
    **Dataset**: O arquivo `hr_employee_attrition.csv` contém informações sobre idade, gênero, departamento, tempo de empresa, etc.

    **Objetivos**:
    - Análise exploratória de dados (EDA)
    - Treinamento de um modelo de Machine Learning para prever quem possui alta probabilidade de deixar a empresa
    - Explicação das predições (SHAP)
    - Construção de um WebApp interativo em **Streamlit**.

    ---
    **Quem sou eu**: 
    - (Coloque aqui uma breve descrição sua e seu LinkedIn/GitHub.)

    Sinta-se livre para navegar pelo menu lateral e descobrir cada parte do projeto!
    """)

# ---------------------------------------------------------------
# PÁGINA 2: EDA & GRÁFICOS
# ---------------------------------------------------------------
def page_eda(df: pd.DataFrame):
    st.header("ANÁLISE EXPLORATÓRIA DOS DADOS")

    if st.checkbox("Mostrar a base de dados"):
        st.dataframe(df)

    # Gráfico | Relação de Attrition
    st.subheader("Distribuição de Attrition (Yes/No)")
    attrition_count = df["Attrition"].value_counts().reset_index()
    attrition_count.columns = ["Attrition", "Count"]
    fig_bar = px.bar(attrition_count, x="Attrition", y="Count",
                     color="Attrition", title="Contagem de Attrition (Yes/No)")
    st.plotly_chart(fig_bar)

    # Gráfico | Idade
    st.subheader("Idade (Age)")
    age_count = df["Age"].value_counts().reset_index()
    age_count.columns = ["Age", "Count"]
    fig_age = px.bar(age_count, x="Age", y="Count",
                      color="Age", title="Age")
    st.plotly_chart(fig_age)

    # Gráfico | Anos de casa
    st.subheader("Anos de casa (YearsAtCompany)")
    years_company_count = df["YearsAtCompany"].value_counts().reset_index()
    years_company_count.columns = ["YearsAtCompany", "Count"]
    fig_years_company = px.bar(years_company_count, x="YearsAtCompany", y="Count",
                      color="YearsAtCompany", title="YearsAtCompany")
    st.plotly_chart(fig_years_company)

    # Gráfico | Departamento
    st.subheader("Departamento (Departament)")
    dept_count = df["Department"].value_counts().reset_index()
    dept_count.columns = ["Department", "Count"]
    fig_dept = px.bar(dept_count, x="Department", y="Count",
                      color="Department", title="Contagem por Departamento")
    st.plotly_chart(fig_dept)

    # Gráfico | Hora extra
    st.subheader("Hora Extra (OverTime)")
    over_time_count = df["OverTime"].value_counts().reset_index()
    over_time_count.columns = ["OverTime", "Count"]
    fig_over_time = px.bar(over_time_count, x="OverTime", y="Count",
                      color="OverTime", title="OverTime", labels=["Overtime"])
    st.plotly_chart(fig_over_time)

# ---------------------------------------------------------------
# FUNÇÃO DE CLASSIFICAÇÃO DE RISCO
# ---------------------------------------------------------------
def classify_risk(prob: float, low=0.10, med=0.40):
    """Classifica risco (Baixo, Moderado, Alto) dada a probabilidade de saída."""
    if prob < low:
        return "Baixo"
    elif prob < med:
        return "Moderado"
    else:
        return "Alto"


# ---------------------------------------------------------------
# PÁGINA 3: PREDIÇÃO & EXPLICAÇÕES SHAP
# ---------------------------------------------------------------
def page_prediction_shap(df, model, feature_names):
    st.header("Predições & Explicações (Random Forest + SHAP)")

    # Preprocessar todo df (background SHAP)
    X_full = get_preprocessed_data(df)

    # Carregar/gerar Explainer e shap_values (cacheado)
    shap_explainer, shap_values = get_shap_explainer_and_values(model, X_full)

    # Gera predições no dataset completo (para subgrupos)
    y_pred_full = model.predict(X_full)
    y_proba_full = model.predict_proba(X_full)

    # Uusário | Selecionar colaborador
    employee_list = df["EmployeeNumber"].tolist()
    selected_employee = st.selectbox("Selecione o ID do Funcionário", employee_list)
    employee_data = df[df["EmployeeNumber"] == selected_employee].copy()

    # Previsão para este colaborador
    X_employee = preprocess_data(employee_data)
    prob_employee = model.predict_proba(X_employee)[0][1]
    pred_employee = model.predict(X_employee)[0]
    risk = classify_risk(prob_employee)

    st.write(f"**Probabilidade de sair**: {prob_employee:.2%}")
    st.write(f"**Classe predita**: {'Sai (1)' if pred_employee == 1 else 'Não sai (0)'}")
    st.write(f"**Nível de risco**: {risk}")

    # Explicação local (Waterfall)
    st.subheader("Explicação Local (SHAP)")
    st.markdown("Podemos visualizar quais variáveis pesaram mais para este colaborador.")
    # Precisamos do índice no df original
    idx_employee = df[df["EmployeeNumber"] == selected_employee].index[0]

    if st.checkbox("Mostrar Waterfall Plot"):
        shap.plots.waterfall(shap_values[idx_employee])
        st.pyplot(plt.gcf())

    # Explicação Global por Subgrupos
    st.subheader("Explicação Global por Subgrupos (FICA vs SAI)")
    if st.checkbox("Mostrar Explicações de Subgrupos (Bar Plot)"):
        import numpy as np
        idx_stay = np.where(y_pred_full == 0)[0]
        idx_leave = np.where(y_pred_full == 1)[0]

        st.write("#### Subgrupo: FICA (Pred=0)")
        fig_stay, ax = plt.subplots()
        shap.summary_plot(shap_values[1][idx_stay],
                          X_full.iloc[idx_stay],
                          plot_type="bar", show=False)
        st.pyplot(fig_stay)

        st.write("#### Subgrupo: SAI (Pred=1)")
        fig_leave, ax = plt.subplots()
        shap.summary_plot(shap_values[1][idx_leave],
                          X_full.iloc[idx_leave],
                          plot_type="bar", show=False)
        st.pyplot(fig_leave)

    # Gráfico | Feature Importance
    if st.checkbox("Mostrar Importância das Features"):
        st.subheader("Importância das Features")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig_imp, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Importância")
        ax.set_title("Importância das Features (Random Forest)")
        plt.tight_layout()
        st.pyplot(fig_imp)


# ---------------------------------------------------------------
# PÁGINA 4: CADASTRO DE NOVOS FUNCIONÁRIOS
# ---------------------------------------------------------------
def page_forms():
    st.header("CADASTRO DE NOVO FUNCIONÁRIO")
    st.write("Preencha os campos abaixo para cadastrar um novo funcionário")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    years_at_company = st.slider("YearsAtCompany", 0, 40, 5)
    over_time = st.selectbox("OverTime", ["Yes", "No"])
    job_level = st.number_input("JobLevel", min_value=1, max_value=10, value=1)

    if st.button("Salvar"):
        st.success("Funcionário cadastrado com sucesso!")
        new_employee_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Department": [department],
            "YearsAtCompany": [years_at_company],
            "OverTime": [over_time],
            "JobLevel": [job_level]
        })
        st.write("Novo Funcionário (Exemplo):")
        st.dataframe(new_employee_df)


# ---------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ---------------------------------------------------------------
def main():
    st.title("PEOPLE ANALYTICS | ATTRITION")
    st.sidebar.title("Páginas")

    # Carrega apenas UMA vez (cacheado)
    df = load_dataset()
    model = load_model()
    feature_names = load_feature_order()

    menu_options = ['HOME', 'ANÁLISE EXPLORATÓRIA', 'PREDIÇÕES/SHAP', 'CADASTRO']
    menu_choice = st.sidebar.radio("Ir para: ", menu_options)

    if menu_choice == 'HOME':
        page_disclaimer()
    elif menu_choice == "ANÁLISE EXPLORATÓRIA":
        page_eda(df)
    elif menu_choice == "PREDIÇÕES/SHAP":
        page_prediction_shap(df, model, feature_names)
    else:
        page_forms()


if __name__ == "__main__":
    main()
