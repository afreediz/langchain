from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.title("Research tool")
st.sidebar.title("Provided Articles URL")

file_path = 'vectordb.pkl'
vector_db = None

main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0, max_tokens=600)

template = """
answer the question based on following context :
{context}

question : {input}
"""
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)
embeddings = OpenAIEmbeddings()

def process_url():
    if(len(url) == 0):
        st.sidebar.write("url cannot be empty")
        return
    main_placeholder.text("Data loading...")
    data = UnstructuredURLLoader(urls=[url]).load()
    textsplitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.','.'],
        chunk_size = 4000
    )
    docs = textsplitter.split_documents(data)
    return docs

def create_database(docs):
    return FAISS.from_documents(docs, embeddings)

url = st.sidebar.text_input("enter the url")
process_url_button = st.sidebar.button("create knowledge base")

if process_url_button:
    docs = process_url()
    vector_db = create_database(docs)

question = st.text_input("Enter your question")
answer = st.button("get answer")

if answer:
    docs = process_url()
    vector_db = create_database(docs)
    if len(question) == 0:
        st.write("question cannot be empty")
    retriever = vector_db.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    st.write(retriever_chain.invoke({"input":question})["answer"])
    res = retriever_chain.invoke({"input":"birthdat of stephen hawking"})
    print(res)

