import os
from typing import List

from fastapi import FastAPI
from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# load the source file from the app in
source_code_loader = DirectoryLoader('./backend/example', glob="**/*.py", loader_cls=PythonLoader,
                                     use_multithreading=True)
source_code_docs = source_code_loader.load()
print(len(source_code_docs))

log_file_loader = DirectoryLoader('./backend/example', '**/*.txt', use_multithreading=True)
log_file_docs = log_file_loader.load()
print(len(log_file_docs))
print(log_file_docs)

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
# vector_store = Milvus.from_documents(
#     all_splits,
#     embedding=embeddings,
#     connection_args={"host": "localhost", "port": 19530},
#     collection_name="try2"
# )
# embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

all_docs = log_file_docs + source_code_docs

def format_docs(docs: List[Document]) -> str:
    return "\n \n".join([doc.page_content for doc in all_docs])


template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = ChatPromptTemplate.from_template(template)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                 """
# Here is the source code and the log. Please provide:
# 1.) The line where the error occurs,
# 2.) The reason for the error, and
# 3.) How to fix it.
#                 """
#             )
#         ),
#         HumanMessagePromptTemplate.from_template(template),
#     ]
# )


# then query OctoAI for the output answer
chain = (
        {"context": format_docs, "question": RunnablePassthrough()}
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
    test_result = """
    The error occurs on line 23, where the logger.error method is called with the exc_info parameter set to True. This parameter is used to log the exception information, but it is not set to a valid exception object.

    The reason for the error is that the exc_info parameter is not set to a valid exception object. In this case, it is set to True, which is not a valid exception object.
    
    To fix the error, you should set the exc_info parameter to a valid exception object, such as Exception or a custom exception class. For example:
    logger.error('Had an issue', exc_info=Exception, extra=extra_logging)
    
    Alternatively, you can remove the exc_info parameter altogether, if you don't need to log the exception information.
    """
    print(result)
    return {"result": result}
