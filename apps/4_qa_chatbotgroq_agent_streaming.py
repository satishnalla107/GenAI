from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import warnings
warnings.filterwarnings("ignore")




llm = ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search = GoogleSerperAPIWrapper()
tools=[search.run]


st.subheader("QuickAnswers - Answers at the speed of thought")

if "memory" not in st.session_state:
    st.session_state.memory=MemorySaver()
    st.session_state.history = []


for message in st.session_state.history:
    role = message.get("role")  
    content = message.get("content") 
    st.chat_message(role).markdown(content)


agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=st.session_state.memory,
    system_prompt="you are amazing ai agent and you can search any question on google.",
)

####Building Web-Interface


query= st.chat_input("Ask Anything ?")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user","content": query})
    res = agent.stream(
        { "messages": [  {
            "role": "user",
            "content": query
           
        }]},
        {"configurable": {"thread_id": "1"}},
        stream_mode='messages'
        ) 
    
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message=""

    for chunk in res:
        message=message+chunk[0].content

        space.write(message)    
        st.session_state.history.append({"role": "ai","content": message})



   
    
 
