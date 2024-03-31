from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.title("Lanchain")
input_text = st.text_input("seach topic")

from langchain_openai import OpenAI

llm = OpenAI()

if input_text:
    st.write(llm(input_text))