import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from utils import *
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = load_model()
df_original, df_prep = load_data()
explainer = get_explainer(model, df_prep)

def page_chat(df_original, df_prep, model):
    if "shap_values_class1" not in st.session_state:
        with st.status("⏳ Carregando SHAP...\n\nIsso costuma levar no máximo 1 minuto.", expanded=True) as status:
            shap_values_class1 = compute_shap_values(explainer, df_prep)
            st.session_state.shap_values_class1 = shap_values_class1
            status.update(label="✅ SHAP carregado com sucesso.", state="complete")
    else:
        shap_values_class1 = st.session_state.shap_values_class1

    st.title("💬 Converse com a IA do RH")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", 
        "content": ("Você é um profissional especialista em People Analytics e Recursos Humanos, com mais de 30 anos de experiência "
    "em empresas de grande porte nacionais e multinacionais. Seu papel é analisar dados de funcionários de forma ética, "
    "estratégica e baseada em evidências, trazendo recomendações e reflexões úteis para gestores de RH e lideranças. \n\n"
    "Responda sempre de maneira empática, clara e profissional, utilizando termos de RH com explicações acessíveis quando necessário. "
    "Evite suposições sem base nos dados fornecidos. Caso precise, peça mais informações ao usuário. "
    "Você deve ajudar a interpretar métricas de turnover, engajamento, satisfação, tempo de empresa, entre outros. "
    "Sua comunicação deve transmitir confiança, autoridade e colaboração, como um verdadeiro parceiro estratégico do negócio.")}]

    funcionario_idx = st.selectbox("Selecione um funcionário", list(range(len(df_original))))
    funcionario_info = df_original.iloc[funcionario_idx].to_dict()

    funcionario_prep = df_prep.iloc[funcionario_idx:funcionario_idx+1].drop(columns="Attrition_numerical")
    risk_score = model.predict_proba(funcionario_prep)[:, 1][0]
    risk_level = classify_risk(risk_score)
    fatores_importantes = shap_func_id(shap_values_class1, df_prep, funcionario_idx)

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
            # Adiciona ao histórico apenas a pergunta do usuário para exibição
            st.session_state.chat_history.append({"role": "user", "content": user_message})

            # Constrói o contexto com os dados + pergunta, mas sem mostrar isso pro usuário
            mensagem_contextualizada = f"Os dados do funcionário são:\n{funcionario_info}\n\n" \
                                    f"Seguem os dados da previsão do modelo e explicação SHAP:\n{resultados_modelo}\n" \
                                    f"Pergunta do usuário: {user_message}"

            # Envia ao modelo com o histórico anterior e a mensagem contextualizada
            mensagens_envio = st.session_state.chat_history[:-1] + [{"role": "user", "content": mensagem_contextualizada}]

            try:
                resposta = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=mensagens_envio,
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

    for mensagem in st.session_state.chat_history:
        if mensagem["role"]  == "system":
            continue
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])

    if st.button("Resetar Chat"):
        st.session_state.chat_history = [{"role": "system", "content": "Como posso te ajudar hoje?"}]
        st.rerun()

page_chat(df_original, df_prep, model)
