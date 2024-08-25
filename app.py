import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# load_dotenv()

# Initialize Ollama LLM
# llm = HuggingFaceEndpoint(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.7, "max_length": 512})
parser = StrOutputParser()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5
    )


system_template = """You are an AI assistant specialized in writing personalized, professional emails. 
    Your task is to generate an email based on the provided information. 
    The email should be engaging, concise, and highlight the key benefits of the project. Use bullet pointers. Maximum words allowed are 250. Always start with Subject line."""

user_template = """Write a personalized email to {name} about the {project} project. 
    Highlight the following key benefits:
    {key_benefits}
    
    The email should be professional, engaging, and no longer than 3 paragraphs."""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)

# Create an LLMChain
email_chain = prompt_template|llm
# |parser

# Streamlit UI
st.title("Personalized Email Generator")

name = st.text_input("Recipient's Name")
project = st.text_input("Project Name")
key_benefits = st.text_area("Key Benefits (one per line)")

if st.button("Generate Email"):
    if name and project and key_benefits:
        benefits_list = key_benefits.split('\n')
        benefits_str = ", ".join(benefits_list)
        
        email = email_chain.invoke({"name": name, "project": project, "key_benefits": key_benefits})
        st.subheader("Generated Email:")
        st.text_area("", email, height=300)
    else:
        st.error("Please fill in all fields.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Enter the recipient's name.\n"
    "2. Specify the project name.\n"
    "3. List key benefits, one per line.\n"
    "4. Click 'Generate Email' to create a personalized email."
)

# About
st.sidebar.header("About")
st.sidebar.info(
    "This app uses Langchain with Huggingface to generate personalized emails "
    "based on the provided information. It demonstrates how Large Language Models "
    "can be used for dynamic content creation."
)


