from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
import time
import pickle
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

urls = []

url = st.sidebar.text_input("Enter url")
urls.append(url)
print(urls)

process_url_clicked = st.sidebar.button("Process datas")
file_path = 'vectordb.pkl'

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

if process_url_clicked and len(urls[0]) != 0:
    print('clicked')
    print(urls)
    main_placeholder.text("data loading started")
    data = UnstructuredURLLoader(urls=urls).load()
    textsplitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.','.'],
        chunk_size = 4000
    )
    main_placeholder.text("datas splitting and storing, be Patient")
    docs = textsplitter.split_documents(data)
    vector_store = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    vector_store.save_local(file_path)


question = st.text_input("Enter your question")
answer = st.button("get answer")

if answer and question:
    vector_store = FAISS.load_local(file_path, embeddings,{"allow_dangerous_deserialization":True})
    retriever = vector_store.as_retriever()
    input = "{input}"
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    st.write(retriever_chain.invoke(question,allow_dangerous_deserialization=True))
    # st.title(retriever_chain.invoke({"input":"birthdat of stephen hawking"})["answer"])
    # res = retriever_chain.invoke({"input":"birthdat of stephen hawking"})
    # print(res)

    # with open(file_path, 'wb') as f:
    #     pickle.dump(vectore_store, f)
# question = 

# if question:
#     if os.path.exists(file_path):
#         with(file_path, 'rb') as f:
#             vectore_store = pickle.load(f)
#             retriever = vectore_store.as_retriever()
#             input = "birthday of einstein"
#             document_chain = create_stuff_documents_chain(llm, input)
#             retriever_chain = create_retrieval_chain(retriever, document_chain)

#             print(retriever_chain.invoke())

