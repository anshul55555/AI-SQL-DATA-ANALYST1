import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import requests

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# -----------------------------
# CONFIG
# -----------------------------
TABLE_NAME = "data"
GROQ_API_KEY = "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"  # 🔁 replace this

st.set_page_config(page_title="AI SQL Data Analyst", layout="wide")
st.title("📊 AI SQL Data Analyst Agent")

# -----------------------------
# FETCH VALID MODEL
# -----------------------------
def get_working_model(api_key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception("❌ Failed to fetch models")

    models = response.json().get("data", [])

    allowed = ["llama", "mixtral", "gemma"]
    blocked = ["embed", "moderation", "whisper", "tts", "vision", "class"]

    for m in models:
        mid = m["id"].lower()
        if any(a in mid for a in allowed) and not any(b in mid for b in blocked):
            return m["id"]

    raise Exception("❌ No valid chat model found")

# -----------------------------
# LOAD LLM
# -----------------------------
def get_llm():
    model_name = get_working_model(GROQ_API_KEY)

    st.info(f"Using model: {model_name}")

    return ChatGroq(
        temperature=0,
        model_name=model_name,
        groq_api_key=GROQ_API_KEY,
        streaming=False  # 🔥 prevents streaming errors
    )

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    conn = sqlite3.connect("temp.db")
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    st.success("✅ Data loaded into SQLite")

    db = SQLDatabase.from_uri("sqlite:///temp.db")

    llm = get_llm()

    # -----------------------------
    # STRICT PROMPT
    # -----------------------------
    prefix = """
You are an expert SQL data analyst.

STRICT FORMAT (follow exactly):

Thought:
Action:
Action Input:
Observation:
Final Answer:

Rules:
- Use SQLite syntax
- Use only given table
- No DROP, DELETE, UPDATE, INSERT
- Always use LIMIT 50
- Use AVG() for averages
"""

    # -----------------------------
    # AGENT (FIXED)
    # -----------------------------
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        prefix=prefix,
        verbose=True,
        handle_parsing_errors=True  # 🔥 critical fix
    )

    # -----------------------------
    # QUERY
    # -----------------------------
    query = st.text_input("Ask a question about your data")

    if query:

        if any(w in query.upper() for w in ["DROP", "DELETE", "UPDATE", "INSERT"]):
            st.error("❌ Unsafe query")
            st.stop()

        with st.spinner("🤖 Thinking..."):
            try:
                # -----------------------------
                # TRY 1
                # -----------------------------
                try:
                    response = agent.invoke({"input": query})
                    answer = response.get("output", "No response")

                except Exception:
                    # -----------------------------
                    # RETRY (FIX)
                    # -----------------------------
                    response = agent.invoke({"input": query})
                    answer = response.get("output", "Recovered response")

                st.subheader("📌 Answer")
                st.write(answer)

                # -----------------------------
                # VISUALIZATION
                # -----------------------------
                result_df = pd.read_sql(
                    f"SELECT * FROM {TABLE_NAME} LIMIT 100", conn
                )

                st.subheader("📊 Visualization")

                if len(result_df.columns) >= 2:
                    col1 = result_df.columns[0]
                    col2 = result_df.columns[1]

                    if pd.api.types.is_numeric_dtype(result_df[col2]):
                        fig = px.bar(result_df, x=col1, y=col2)
                    else:
                        fig = px.pie(result_df, names=col1)

                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("📄 Sample Data")
                st.dataframe(result_df)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.warning("⚠️ Showing fallback")

                fallback_df = pd.read_sql(
                    f"SELECT * FROM {TABLE_NAME} LIMIT 10", conn
                )
                st.dataframe(fallback_df)