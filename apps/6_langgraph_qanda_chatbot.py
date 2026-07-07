from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict



class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state: ChatState):
    res = llm.invoke(state["messages"])
    return {"messages": [res]}

memory = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chatBot", chatBotNode)

graph.add_edge(START, "chatBot")
graph.add_edge("chatBot", END)

graph = graph.compile(checkpointer=memory)


config = {"configurable":{"thread_id":"my-bot-1"}}

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Exiting the chatbot. Goodbye!")
        break
    res = graph.invoke({"messages":[{"role":"user", "content":query}]},
    config )

    answer=res["messages"][-1].content
    print("Ai : ", answer)