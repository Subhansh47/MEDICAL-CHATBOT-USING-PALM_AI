import streamlit as st
from medchatbot import get_qa_chain

st.title("Welcome to MedBot powered by PALM-AI")




question = st.text_input("Question: ")




if st.button('Search', type="primary"):
    
    chain = get_qa_chain()
    response=chain(question)

    st.header("Answer")
    st.write(response["result"])
    
    
    st.header("Source of the answer")
    st.write(response["source_documents"])





