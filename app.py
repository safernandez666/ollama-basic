from colorama import Fore, Style
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import GPT4AllEmbeddings
import os

model_local = ChatOllama(model="llama2")
DB_PATH = "vectorstores/db/"

def load_chroma_db(db_path):
    if os.path.exists(db_path):
        print("\n" + Fore.YELLOW + "Cargando ChromaDB" + Style.RESET_ALL + "\n")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings())
        retriever = vectorstore.as_retriever()
    else:
        print(Fore.YELLOW + "ChromaDB no se encuentra. Salteando." + Style.RESET_ALL)
        retriever = None
    return retriever

# Cargar ChromaDB
retriever = load_chroma_db(DB_PATH)

# Antes del RAG
print(Fore.YELLOW + "\n" + "Antes de crear el RAG" + "\n" + Style.RESET_ALL)
before_rag_template = "Who is {topic}? "
#print(Fore.YELLOW + "\n" + "Pregunta: " + Style.RESET_ALL + before_rag_template)
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Stephen Maturin"}))

# Despues del RAG
if retriever:
    print("\n" + Fore.YELLOW + "Despues de crear el RAG con ChromaDB" + "\n" + Style.RESET_ALL)
    after_rag_template = """Answer based on the following context:
    {context}
    Pregunta: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    print(after_rag_chain.invoke("Whi is Stephen Maturin? Can you comment some personality indicators?"))
else:
    print("\n" + Fore.YELLOW + "Despues de crear el RAG" + "\n" + Style.RESET_ALL)
    print("Salteo el Proceso de RAG no hay ChromaDB.")
