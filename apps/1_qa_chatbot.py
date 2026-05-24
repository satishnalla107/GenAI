from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
import streamlit as st

st.title("🤖 Askbuddy - AI QnA Bot")
st.markdown("Welcome to my QnA bot with Langchain and google gemini !")

query = st.chat_input("Ask - Anything")

if query:
    st.chat_message("user").markdown(query)
    res= llm.invoke(query)
    st.chat_message("ai").markdown(res.content)