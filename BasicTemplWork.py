from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

sys_template = "you have {system_job} job and your name is {system_name}"
user_template = "answer me the following question. question : {user_question}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",sys_template),
    ("user",user_template)
])

llm = ChatOpenAI()
output_parser = StrOutputParser()

# chat_prompt.invoke({"system_job":"engineer","system_name":"tuttu","user_question":"what is your name and job"})

chain = chat_prompt | llm

poem = chain.invoke({"system_job":"engineer","system_name":"tuttu","user_question":"what is your name and job"})
print(poem)