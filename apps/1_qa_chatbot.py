from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

while True :
    query =  (input("user:"))
    if query.lower() == "exit":
        break
    res = llm.invoke(query)
    print("ai:"+res.content)