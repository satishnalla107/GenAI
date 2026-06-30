from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import streamlit as st

### data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path): 

    ##Load the PDF document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ##split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.split_documents(docs)

    ##Create embeddings for the document chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ##Create a vector store to store the document chunks and their embeddings

    vector_db = Chroma.from_documents(
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
    st.session_state.agent = agent
    st.session_state.document_uploaded = True
     


### upload ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Upload a PDF document", type="pdf", accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing the document..."):
            path="./doc_files/"
            for file in uploaded:
                with open(path+file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()        




###chat ui

if st.session_state.document_uploaded and st.session_state.agent: 

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything related to the uploaded document....")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
         
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke({"messages": [{"role": "user", "content": query}]},{"configurable":{"thread_id":"1"}})                                          
        result = response["messages"][-1].content
        st.chat_message("assistant").markdown(result)
        
        st.session_state.messages.append({"role": "assistant", "content": result})




