import streamlit as st
import rag_agent

st.set_page_config(page_title="Agente RAG", layout="wide")
st.title("Agente RAG")

st.markdown("Faça uma pergunta sobre os PDFs carregados.")
user_query = st.text_input("Pergunta:", "O que você deseja saber?")

if st.button("Buscar resposta"):
    with st.spinner("Consultando o agente RAG e gerando resposta..."):
        resposta, retrieved_chunks = rag_agent.answer_question(user_query, k=4)
    st.subheader("Resposta do agente:")
    st.markdown(resposta)
