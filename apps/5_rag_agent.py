from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

##Load the PDF document
loader = PyPDFLoader("../data/medical_report.pdf")
docs = loader.load()

##split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = splitter.split_documents(docs)

##Create embeddings for the document chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

##Create a vector store to store the document chunks and their embeddings

vector_db = InMemoryVectorStore.from_documents(
    documents=documents, 
    embedding=embeddings)

###create agent - tool,llm, system prompt
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_tokens=1000)



@tool
def retriever_context(query: str) :
    """Retrieve documents relevant to the query from knowledge base."""

    docs = vector_db.similarity_search(query=query, k=3)
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"
    return context



system_prompt = """
You are an AI assistant.

You have access to a retrieval tool called `retrieve_context`.

For any question related to the uploaded PDF,
ALWAYS call the tool first.

Answer ONLY from the retrieved context.

If the answer is not present in the retrieved context,
say "I couldn't find that information in the uploaded document."

Do not make up information.
"""



memory = InMemorySaver()

## create agent
agent = create_agent(
     model=llm,
     tools=[retriever_context], 
     system_prompt=system_prompt,
     checkpointer=memory )


while True:
    query = input("User: ")
    if query.lower() == 'quit'or query.lower() == 'exit' or query.lower() == 'bye':
        break

    response = agent.invoke({"messages": [{"role": "user", "content": query}]},
                            {"configurable":{"thread_id": 1}})
    result = response["messages"][-1].content
    print("AI:", result)






