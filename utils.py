import pandas as pd
import joblib
import shap
import streamlit as st

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
    data_prep = df.copy()

    data_prep['BusinessTravel'] = data_prep['BusinessTravel'].map({'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0})
    data_prep['Gender'] = data_prep['Gender'].map({'Male': 1, 'Female': 0})
    data_prep['OverTime'] = data_prep['OverTime'].map({'Yes': 1, 'No': 0})

    columns_to_dummy = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
    data_prep = pd.get_dummies(data_prep, columns=columns_to_dummy, drop_first=True, dtype='int64')

    data_prep['Attrition_numerical'] = data_prep['Attrition'].map({'Yes': 1, 'No': 0})

    cols_to_delete = ['Over18', 'StandardHours', 'EmployeeNumber', 'EmployeeCount', 'Attrition']
    data_prep.drop(cols_to_delete, axis=1, inplace=True)

    return df, data_prep

@st.cache_resource
def get_explainer(_model, df_prep):
    return shap.Explainer(_model, df_prep)

@st.cache_data
def compute_shap_values(_explainer, df_prep):
    return _explainer(df_prep)[..., 1]

def classify_risk(probs, low_threshold=0.10, medium_threshold=0.40):
    if probs < low_threshold:
        return "Baixo"
    elif probs < medium_threshold:
        return "Moderado"
    else:
        return "Alto"

def shap_func_id(shap_values_class1, df_prep, idx):
    shap_values = shap_values_class1[idx]
    shap_importances = abs(shap_values.values).flatten()
    top_indices = shap_importances.argsort()[-3:]
    return df_prep.columns[top_indices].tolist()