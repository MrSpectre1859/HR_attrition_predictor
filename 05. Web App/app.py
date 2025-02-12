# ----------------------------
# IMPORTAÇÕES
# ----------------------------
import streamlit as st
import joblib
import pandas as pd
from data_processing import preprocess_data  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# FUNÇÕES DE CARREGAMENTO
# ----------------------------
def load_dataset():
    df = pd.read_csv("../02. Datasets/hr_employee_attrition.csv")
    return df

def load_model():
    rf_model = joblib.load("../04. ML Model/random_forest_model.pkl")
    return rf_model


# ----------------------------
# PÁGINA 1: COMENTÁRIOS
# ---------------------------

def page_disclaimer():
    st.header("SOBRE O APP")
    st.markdown("""
                **Olá, seja bem-vindo(a) ao meu WebApp de People Analytics, criado para demonstrar:

                1. O uso de um dataset (IBM HR) que contém informações de funcionários, incluindo quem saiu e quem permaneceu;
                2. O processo de análise e modelagem para prever a *rotatividade* (Attrition) dos colaboradores;
                3. Uma forma de **explicar** as predições que o modelo faz, usando SHAP;
                4. Um protótipo de **cadastro de novos funcionários**

                ---
                **Dataset**: O arquivo `hr_employee_attrition.csv` contém informações sobre idade, gênero, departamento, tempo de empresa, etc. Simulando muito bem\n
                uma base de dados realista que normalmente utilizamos

                **Objetivos**:
                - Análise exploratória de dados (EDA)
                - Treinamento de um modelo de Machine Learning para prever quem possui alta probabilidade de deixar a empresa
                - Explicação das predições (SHAP).
                - Construção de um WebApp interativo em **Streamlit**.

                ---
                **Quem sou eu**: 
                - (Coloque aqui uma breve descrição sua e seu LinkedIn/GitHub.)

                Sinta-se livre para navegar pelo menu lateral e descobrir cada parte do projeto!
                """)

# ----------------------------
# PÁGINA 2: EDA E GRÁFICOS
# ----------------------------

def page_eda(df):
    st.header("ANÁLISE EXPLORATÓRIA DOS DADOS")

    if st.checkbox("Mostrar a base de dados:"):
        st.dataframe(df)

    # Gráfico | Contagem de Attrition
    st.subheader("Distribuição de Attrition (Yes/No)")
    attrition_count = df['Attrition'].value_counts()
    st.bar_chart(attrition_count)

# ----------------------------
# PÁGINA 3: PREDIÇÃO E SHAP
# ----------------------------

def page_prediction_shap(df, model):
    st.header("Random Forest + SHAP")

    # Usuário | Seleção do colaborador
    employee_list = df["EmployeeNumber"].tolist()
    selected_employee = st.selectbox("ID Funcionário", employee_list)
    employee_data = df[df["EmployeeNumber"] == selected_employee].copy()

	# Modelo | Predição do colaborador
    _employee = preprocess_data(employee_data)
    employee_prob = model.predict_proba(_employee)[0][1]
    employee_pred = model.predict(_employee)[0]

    st.write(f"Probabilidade de sair: {employee_prob:.2%}")
    st.write(f"Classe_Risco: {risk_level}") #>>> Precisa resolver essa pendência
    st.write("Histórico: ", "Saiu (1)" if employee_pred == 1 else "Não saiu (0)")


def load_feature_order():
    with open("../03. Outputs/features_order.txt", "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    return feature_names

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Importância")
    ax.set_title("Importância das Features (Random Forest)")
    
    plt.tight_layout()
    st.pyplot(fig)

'''def main():
    st.title("Attrition Predictor | People Analytics")
    st.write("WebApp Streamlit :)")

    df = load_dataset()
    model = load_model()
    feature_names = load_feature_order()

    st.markdown("## Escolha um colaborador para ver a probabilidade de sair:")
    if st.checkbox("Mostrar DataFrame completo"):
        st.dataframe(df)

	# Usuário | Seleção do colaborador
    employee_list = df["EmployeeNumber"].tolist()
    selected_employee = st.selectbox("ID Funcionário", employee_list)
    employee_data = df[df["EmployeeNumber"] == selected_employee].copy()

	# Modelo | Predição do colaborador
    _employee = preprocess_data(employee_data)
    employee_prob = model.predict_proba(_employee)[0][1]
    employee_pred = model.predict(_employee)[0]

    st.write(f"Probabilidade de sair: {employee_prob:.2%}")
    st.write("Classe: ", "Sai (1)" if employee_pred == 1 else "Não sai (0)")

    # Gráfico | Feature Importance 
    if st.checkbox("Mostrar Importância das Features do Modelo"):
        plot_feature_importance(model, feature_names)
'''

# ----------------------------
# PÁGINA 4: CADASTRO
# ----------------------------

def page_forms():
    st.header("CADASTRO DE NOVO FUNCIONÁRIO")

    st.write("Preencha os campos abaixo para cadastrar um novo funcionário")

    # Usuário | Formulário
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    years_at_company = st.slider("YearsAtCompany", 0, 40, 5)
    over_time = st.selectbox("OverTime", ["Yes", "No"])
    job_level = st.number_input("JobLevel", min_value=1, max_value=10, value=1)

    # Usuário | Salvar informações
    if st.button("Salvar"):
        st.success("Funcionário cadastrado sucesso!")
        
# ----------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------

def main():
    st.title("WEBAPP DE PEOPLE ANALYTICS")
    st.sidebar.title("Navegação")

    df = load_dataset()
    model = load_model()
    feature_names = load_feature_order()

    menu_options = ['DISCLAIMER', 'ANÁLISE EXPLORATÓRIA', 'PREDIÇÕES/MODELO', 'CADASTRO']
    menu_choice = st.sidebar.radio("Ir para: ", menu_options)

    if menu_choice == 'DISCLAIMER':
        page_disclaimer()
    elif menu_choice == "ANÁLISE EXPLORATÓRIA":
        page_eda(df)
    elif menu_choice == "PREDIÇÕES/MODELO":
        page_prediction_shap(df, model)
    else:
        page_forms


if __name__ == "__main__":
    main()
