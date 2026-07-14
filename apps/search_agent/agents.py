from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os
llm = ChatGroq(
    model="openai/gpt-oss-20b",streaming=True
)


search = GoogleSerperAPIWrapper()

@tool
def google_search(query:str)->str:
    """
    Search Google up to date information on the web

    Args: query: What to search on google

    Return : Top search results as plain text
    """

    search = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))

    return search.run(query)


llm = ChatGroq(model="openai/gpt-oss-20b",api_key=os.getenv("GROQ_API_KEY"))
SYSTEM_PROMPT="""you are amazing ai agent and you can search any question on google."""

agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt=SYSTEM_PROMPT
)

#res = agent.invoke(
#         { "messages": [  {
#             "role": "user",
#             "content": "address of vn cars ,yanam"
#         }]},
#         {"configurable": {"thread_id": "1"}}
#         ) 
# print(res["messages"][-1].content)




                    