import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from utils import *

model = load_model()
df_original, df_prep = load_data()
explainer = get_explainer(model, df_prep)
shap_values_class1 = compute_shap_values(explainer, df_prep)

def page_analytics():
    st.subheader("🧹 Filtros")
    col1, col2 = st.columns(2)

    with col1:
        filtro_departamento = st.selectbox("Departamento", ["Todos"] + list(df_original["Department"].unique()))
    with col2:
        filtro_cargo = st.selectbox("Cargo", ["Todos"] + list(df_original["JobRole"].unique()))

    df_filtrado = df_original.copy()
    if filtro_departamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Department"] == filtro_departamento]
    if filtro_cargo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["JobRole"] == filtro_cargo]

    df_filtrado_prep = df_prep.loc[df_filtrado.index].drop(columns=["Attrition_numerical"], errors="ignore")

    df_filtrado["Risk_Score"] = model.predict_proba(df_filtrado_prep)[:, 1]
    df_filtrado["Risk_Level"] = df_filtrado["Risk_Score"].apply(lambda x: classify_risk(x))

    st.subheader("🚩 KPIs")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3, vertical_alignment="center")
    with col_kpi1:
        st.metric("Alto Risco", (df_filtrado["Risk_Level"] == "Alto").sum())
    with col_kpi2:
        st.metric("Risco Moderado", (df_filtrado["Risk_Level"] == "Moderado").sum())
    with col_kpi3:
        st.metric("Baixo Risco", (df_filtrado["Risk_Level"] == "Baixo").sum())

    # 📊 Distribuição de Risco
    st.subheader("🧮 Distribuição por Risco")
    df_filtrado["Risk_Level"] = pd.Categorical(df_filtrado["Risk_Level"], categories=["Baixo", "Moderado", "Alto"], ordered=True)

    color_map = {
    "Baixo": "#2ECC71",
    "Moderado": "#F1C40F",
    "Alto": "#E74C3C"
                }
    
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_risco = px.histogram(
            df_filtrado,
            x="Risk_Level",
            color="Risk_Level",
            title="Quantitativo (n)",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_risco, use_container_width=True)

    with col_g2:
        fig_pie = px.pie(
            df_filtrado,
            names="Risk_Level",
            title="Percentual (%)",
            color="Risk_Level",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 💰 Análise Salarial")
    fig_box = px.box(df_filtrado, x="Risk_Level", y="MonthlyIncome", color="Risk_Level", color_discrete_map=color_map)
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.subheader("Análise Individual por funcionário ")
    st.markdown('<i style="font-size: 16px;">Você pode utilizar essa seção para analisar a \'Saída\' das pessoas individualmente.\n'
    'Basta selecionar a pessoa no campo abaixo.</i>', unsafe_allow_html=True)
    
    col_btn_func_idx, col_res_func_idx = st.columns([0.3, 0.5], gap="large")

    with col_btn_func_idx:
        funcionario_id = st.selectbox("Selecione o funcionário", df_filtrado.index.tolist())
        
    with col_res_func_idx:
        func_detalhes = df_filtrado.loc[funcionario_id]
        st.write(f"**Funcionário:** {funcionario_id}")
        st.write(f"**Score de Risco:** {func_detalhes['Risk_Score']:.2f}")
        st.write(f"**Nível de Risco:** {func_detalhes['Risk_Level']}")

    st.subheader("📊 Waterfall SHAP",
                help="""
    Esse gráfico mostra os principais fatores que influenciaram a previsão 
    de risco de saída de um funcionário específico.\n
        > Cada barra representa uma variável;
        > Vermelho indica que aumentou e Azul que reduziu o risco de saída;
        > A combinação de fatores resulta no score final de risco."""
                )
    fig1, ax1 = plt.subplots()
    shap.waterfall_plot(shap_values_class1[funcionario_id])
    st.pyplot(fig1)

    st.header("👷 Explicaçao do modelo Random Forest")
    st.markdown('<i style="font-size: 16px;">Aqui você irá verificar alguns dados do modelo que está fazendo a previsão.\n'
    'Clique no símbolo de ajuda "?" para entender do que se trata cada gráfico.</i>', unsafe_allow_html=True)
    st.subheader("📊 Bar Plot Global (Classe 1)",
                help="""
    Esse gráfico mostra, de maneira geral, quais variáveis têm mais impacto
    na decisão do modelo ao prever alto risco de saída.

        > Barras maiores indicam maior influência no resultado;
        > Útil para entender quais fatores o modelo considera mais importantes."""
                )
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values_class1, df_prep, plot_type="bar", show=False)
    st.pyplot(fig2)

    st.subheader("📊 Beeswarm Plot Global (Classe 1)",
                help="""
    Esse gráfico mostra a distribuição do impacto de cada variável nas previsões do modelo.

        > Cada ponto representa um funcionário;
        > A cor indica o valor da variável (ex: alto ou baixo salário);
        > Mostra como diferentes valores de cada variável influenciam para mais ou para menos no risco de saída."""
                )
    fig3, ax3 = plt.subplots()
    shap.summary_plot(shap_values_class1, df_prep, show=False)
    st.pyplot(fig3)

page_analytics()