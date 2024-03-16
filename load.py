from colorama import Fore, Style
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.runnables import RunnablePassthrough

DATA_PATH = "data/"
DB_PATH = "vectorstores/db/"

print(Fore.RED + "\n" + "Loading PDF's" + "\n" + Style.RESET_ALL)
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"Processed {len(documents)} pdf files")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)
vectorstore.persist()

retriever = vectorstore.as_retriever()