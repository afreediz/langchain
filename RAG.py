import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = ChatOpenAI()
embedding = OpenAIEmbeddings()
parser = StrOutputParser()
template = """
    answer the following questoins based on the given context. if you dont know then just say i dont know
    context:{context},
    
    question:{input}
"""
prompt = ChatPromptTemplate.from_template(template=template)
contents = TextLoader("test.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(contents)
vector_store = FAISS.from_documents(docs, embedding=embedding)
retriever = vector_store.as_retriever()

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input":"who is afreedi"})

print(response)