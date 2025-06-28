import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

# --- Sidebar ---
st.set_page_config(page_title="Chat with MySQL DB", layout="wide")
st.sidebar.title("🔌 Database Connection")

host = st.sidebar.text_input("Host", value="localhost")
port = st.sidebar.text_input("Port", value="3306")
user = st.sidebar.text_input("Username", value="root")
password = st.sidebar.text_input("Password", type="password")
database = st.sidebar.text_input("Database", value="chinook")
connect_btn = st.sidebar.button("Connect")

# Global session state
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Gemini or OpenRouter Mistral API setup ---
os.environ["GEMINI_API"] = "YOUR_GEMINI_API_KEY"

llm = ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct",
    openai_api_key="",
    openai_api_base=""
)

# --- Prompt for generating SQL ---
sql_prompt = ChatPromptTemplate.from_template("""
You are an expert SQL assistant.
Return **only** a valid SQL query that answers the question. No explanation.

Schema:
{schema}

Question: {question}

SQL:
""".strip())

# --- Prompt for natural language response ---
nl_prompt = ChatPromptTemplate.from_template("""
Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
""")

# --- On Connect ---
if connect_btn:
    try:
        uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        db = SQLDatabase.from_uri(uri)
        db.run("SELECT 1;")
        st.session_state.db = db
        st.success("✅ Connected to database!")
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")

# --- Chat UI ---
st.title("💬 Chat with a MySQL Database")
st.write("Ask me anything about your database!")

if st.session_state.db:
    user_question = st.chat_input("Enter your question")
    if user_question:
        db = st.session_state.db

        def get_schema(_): return db.get_table_info()
        def run_query(query): return db.run(query)

        # Chain 1: Convert Question → SQL Query
        sql_chain = (
            RunnablePassthrough.assign(schema=get_schema)
            | sql_prompt
            | llm.bind(stop=["```", "SQLResult:", "\n```"])
            | StrOutputParser()
        )

        # Chain 2: Convert SQL Result → Natural Language
        full_chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=get_schema,
                response=lambda vars: run_query(vars["query"]),
            )
            | nl_prompt
            | llm
            | StrOutputParser()
        )

        # Execute
        with st.spinner("Thinking..."):
            try:
                response = full_chain.invoke({"question": user_question})
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("bot", response))
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# --- Chat Display ---
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
