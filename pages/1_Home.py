import streamlit as st


def page_home():
    st.subheader("⚙️ O sistema")
    st.write(
    """
    Este projeto simula uma ferramenta interativa de **People Analytics preditivo**, com foco na análise de turnover. A ideia aqui é simples:
    usar dados para **antecipar riscos de saída** de colaboradores, sem saber de antemão quem vai sair.

    A variável que tentamos prever se chama "Attrition" — que pode ser traduzida como "Desligamento".

    O modelo de machine learning foi treinado para **analisar padrões** nos dados de RH e **calcular a probabilidade de permanência ou saída** de cada funcionário, focando na tomada de decisões mais estratégica e menos baseadas em “achismo”.

    Você poderá:

    * Navegar pelo banco de funcionários ou cadastrar novos

    * Obter previsões de risco de saída

    * Interpretar os resultados de forma intuitiva

    * Conversar com um Chatbot que usa a IA do ChatGPT (GPT-4o-mini) para explorar cenários e tirar dúvidas

    Espero que você goste e que eu consiga contribuir para a inovação na sua empresa!! 😄
    """
            )

    st.subheader("🚀 Tecnologias Utilizadas")
    st.write(
        "- **Python** 🐍\n"
        "- **Streamlit** (Interface Web)\n"
        "- **Pandas** (Manipulação de dados)\n"
        "- **Scikit-learn** (Treinamento do modelo)\n"
        "- **SHAP** (Explicabilidade do modelo)\n"
        "- **OpenAI API** (Chatbot com GPT-4o-mini)\n"
        "- **Plotly & Matplotlib** (Gráficos)\n"
        "- **AWS EC2** (Hospedagem do aplicativo)"
    )

    st.subheader("🌟 Objetivos Principais")
    st.write(
        "- Desenvolver um **modelo preditivo** capaz de identificar a probabilidade de saída de um funcionário (turnover) utilizando **Machine Learning**;\n"
        "- Criar um **dashboard interativo** com visualizações detalhadas sobre alguns indicadores utilizados comumente pelo RH e pela gestão;\n"
        "- **Explicar as previsões** do modelo com o uso de SHAP para auxiliar a compreensão dos clientes que não são da área;\n"
        "- Implementar um **chatbot inteligente** com um modelo da OpenAI, permitindo interações personalizadas;\n"
        "- Fomentar o desenvolvimento de **RH baseado em dados e mais tech**."
    )

    st.subheader("📊 Sobre o Dataset")
    st.write(
        "Os dados utilizados neste projeto são do **IBM HR Analytics Employee Attrition & Performance**, "
        "o qual pode ser encontrado Kaggle [clicando aqui](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)."
        "Você irá notar que o dataset contém informações muito habituais a qualquer funcionário de RH, reforçando a ideia de que estamos coletando dados porém não estamos usando eles como deveríamos...ou pelo menos poderíamos.\n\n"
        " Alguns exemplos são: _idade, departamento, salário, quantidade de treinamentos, anos na empresa, anos na função e resultados da pesquisa de satisfação_.\n\n"
            )
    
    st.subheader("👤 Sobre Mim")
    
    with st.container():
    
        col1, col2 = st.columns([1, 2], vertical_alignment="center", gap="large")

        with col1:
            st.image("Alisson.jpeg", caption="Alisson Machado Cesário", use_container_width=True)

        with col2:
            st.write(
                "👋 Olá! Meu nome é **Alison Machado Cesário**, sou Bacharel em Psicologia pela Universidade Federal Fluminense (UFF) e pós-graduando no MBA de Data Science e Analytics pela USP/ESALQ. "
                "Sou completamente apaixonado por tecnologia, inovação, dados e estou constantemente buscando algo novo para poder aprender.\n\n"
                "Meu foco está em **transformar decisões baseadas em feeling em escolhas orientadas por dados** — principalmente em áreas que trabalham com dados do comportamento humano. "
                "Para isso, estou habituado a usar:\n\n"
                "1. Python para análise de dados\n"
                "2. SQL em Bancos de Dados relacionais\n"
                "3. Power BI para dashboard e KPI's\n"
                "4. Data Storytelling para entregar valor ao cliente\n"
                "5. Muito mais...\n\n"
                "📨 **e-mail**: [alissonmcesario@gmail.com](mailto:alissonmcesario@gmail.com)\n\n"
                "ℹ️ **LinkedIn**: [alissonmcesario](https://www.linkedin.com/in/alissonmcesario)\n\n"
                "👨🏼‍💻 **GitHub**: [MrSpectre1859](https://github.com/MrSpectre1859)",
            )

page_home()