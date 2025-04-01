import streamlit as st
import pandas as pd
from utils import load_data

def page_catalogo():
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

    df_original, _ = load_data()
    df_completo = pd.concat([df_original, st.session_state.funcionarios], ignore_index=True)

    st.subheader("🧹 Filtros")
    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_departamento = st.selectbox(
            "Filtrar por Departamento",
            ["Todos"] + sorted(df_completo["Department"].dropna().unique())
        )

    with col2:
        filtro_cargo = st.selectbox(
            "Filtrar por Cargo",
            ["Todos"] + sorted(df_completo["JobRole"].dropna().unique())
        )

    with col3:
        faixa_salarial = st.slider(
            "Filtrar por Faixa Salarial",
            int(df_completo["MonthlyIncome"].min()),
            int(df_completo["MonthlyIncome"].max()),
            (
                int(df_completo["MonthlyIncome"].min()),
                int(df_completo["MonthlyIncome"].max())
            )
        )

    df_filtrado = df_completo.copy()

    if filtro_departamento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Department"] == filtro_departamento]

    if filtro_cargo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["JobRole"] == filtro_cargo]

    df_filtrado = df_filtrado[
        (df_filtrado["MonthlyIncome"] >= faixa_salarial[0]) &
        (df_filtrado["MonthlyIncome"] <= faixa_salarial[1])
    ]

    st.subheader("📊 Indicadores Estratégicos")
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric("Total de Funcionários", len(df_filtrado))

    with col_m2:
        st.metric("Média Salarial", f"R$ {df_filtrado['MonthlyIncome'].mean():,.2f}")

    with col_m3:
        st.metric("Média Idade", f"{df_filtrado['Age'].mean():.1f} anos")
    
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        # Distribuição por Gênero (ajustado para português)
        genero_counts = df_filtrado["Gender"].value_counts(normalize=True) * 100
        mapa_genero = {"Male": "Homem", "Female": "Mulher"}

        for genero, perc in genero_counts.items():
            label = mapa_genero.get(genero, genero)
            st.metric(f"{label}", f"{perc:.1f}%")

    with col_t2:
        # Tempo médio com gestor atual
        media_anos_gestor = df_filtrado["YearsWithCurrManager"].mean()
        st.metric("Tempo Médio com Gestor", f"{media_anos_gestor:.1f} anos")

        # Satisfação geral no trabalho
        satisfacao_media = df_filtrado["JobSatisfaction"].mean()
        st.metric("Satisfação Geral", f"{satisfacao_media:.1f} / 4")

    with col_t3:
        # Treinamentos no último ano
        media_treinamentos = df_filtrado["TrainingTimesLastYear"].mean()
        st.metric("Média de Treinamentos (últ. ano)", f"{media_treinamentos:.1f}")

    # Distribuição por tempo de empresa
    st.markdown("##### 🕓 Distribuição por Tempo de Casa")
    col_t4, col_t5, col_t6, col_t7 = st.columns(4)

    com_anos = df_filtrado["YearsAtCompany"]

    with col_t4:
        st.metric("Menos de 2 anos", f"{(com_anos < 2).sum()}")

    with col_t5:
        st.metric("2 a 5 anos", f"{((com_anos >= 2) & (com_anos < 5)).sum()}")

    with col_t6:
        st.metric("5 a 10 anos", f"{((com_anos >= 5) & (com_anos < 10)).sum()}")

    with col_t7:
        st.metric("Acima de 10 anos", f"{(com_anos >= 10).sum()}")

    st.divider()
    
    st.subheader("📋 Tabela de Funcionários")
    st.dataframe(df_filtrado, use_container_width=True)

    excluiveis = df_filtrado[df_filtrado["EmployeeNumber"].isin(st.session_state.funcionarios["EmployeeNumber"])]

    if not excluiveis.empty:
        st.subheader("Remover Funcionário (cadastrados manualmente)")
        funcionario_excluir = st.selectbox("Selecione um funcionário para excluir", excluiveis["EmployeeNumber"].tolist())

        if st.button("❌ Excluir", use_container_width=False):
            st.session_state.funcionarios = st.session_state.funcionarios[
                st.session_state.funcionarios["EmployeeNumber"] != funcionario_excluir
            ]
            st.success(f"Funcionário {funcionario_excluir} removido com sucesso!")
            st.rerun()

page_catalogo()
