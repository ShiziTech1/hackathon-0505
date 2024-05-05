import os

from fastapi import FastAPI
from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# load the source file from the app in
source_code_loader = DirectoryLoader('./backend/example', glob="**/*.py", loader_cls=PythonLoader, use_multithreading=True)
source_code_docs = source_code_loader.load()
print(len(source_code_docs))

log_file_loader = DirectoryLoader('./backend/example', '**/*.txt', use_multithreading=True)
log_file_docs = log_file_loader.load()
print(len(log_file_docs))

# chunking
chunk_size = 1024
chunk_overlap = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
source_code_splits = text_splitter.split_documents(source_code_docs)

# load the log file from the app in
log_file_splits = text_splitter.split_documents(log_file_docs)

# insert the documents into milvus
all_splits = source_code_splits + log_file_splits

llm = OctoAIEndpoint(
        model="codellama-34b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
    )
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

vector_store = Milvus.from_documents(
    all_splits,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": 19530},
    collection_name="logging"
)
retriever = vector_store.as_retriever()
# template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:"""
template="""
Here is the source code and the log. Please provide: 
1.) The line where the error occurs,
2.) The reason for the error, and
3.) How to fix it.
"""
prompt = ChatPromptTemplate.from_template(template)
# perform FM

# then query OctoAI for the output answer
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/ask")
async def ask(question: str):
    result = chain.invoke(question)
    return {"question": question}
