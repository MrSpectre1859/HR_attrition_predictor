import streamlit as st
import joblib
import pandas as pd
from data_processing import preprocess_data  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    df = pd.read_csv("../02. Datasets/hr_employee_attrition.csv")
    return df

def load_model():
    rf_model = joblib.load("../04. ML Model/random_forest_model.pkl")
    return rf_model

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

def main():
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

if __name__ == "__main__":
    main()
