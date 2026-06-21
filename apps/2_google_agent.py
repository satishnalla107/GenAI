from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
llm = ChatGroq(
    model="openai/gpt-oss-20b",streaming=True
)
memory = MemorySaver()

search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="you are a agent and you can search any question on google.",
    checkpointer=memory,
)

while True:
    query = input("User ")
    if query.lower() in ["bye","exit","quit"]:
        print("Good bye !")
        break

    res = agent.invoke(
        { "messages": [  {
            "role": "user",
            "content": query
        }]},
        {"configurable": {"thread_id": "1"}}
        )   

    print("Ai : ",res["messages"][-1].content)
    




                    