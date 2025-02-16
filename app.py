# 📊 SALES PLAYBOOK CHAT ASSISTANT (WITH FIXED LOGGING)

from pinecone import Pinecone
import os
import streamlit as st
import uuid
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import time

# ✅ Load Environment Variables
load_dotenv()

# ✅ PAGE CONFIGURATION (FIRST COMMAND)
st.set_page_config(page_title="📊 Sales Playbook Chat Assistant", layout="wide")

# ✅ Sentence Transformer Embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ Database Connection (No caching to avoid connection issues)
def get_db_connection():
    conn = sqlite3.connect('query_logs.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row  # ✅ Enable column name access
    return conn

conn = get_db_connection()
cursor = conn.cursor()

# ✅ Create Logs Table (if not exists)
cursor.execute('''
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    comment TEXT,
    llm_model TEXT,
    response_time FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# ✅ OpenAI Client Initialization
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# ✅ Pinecone Client Initialization
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "document-index"
index = pc.Index(index_name)

# ✅ FUNCTION: Generate Embeddings
def get_embedding(text):
    return embedder.encode(text).tolist()

# ✅ FUNCTION: Search Pinecone
def search_pinecone(query, top_k=3):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"source": {"$in": ["section_text", "image_summary"]}}
    )
    retrieved_chunks = []
    for match in results["matches"]:
        metadata = match["metadata"]
        retrieved_chunks.append({
            "source": metadata.get("source", "unknown"),
            "title": metadata.get("title", ""),
            "page_number": metadata.get("page_number", ""),
            "score": match["score"],
            "content": metadata.get("content", "⚠️ No content available")
        })
    return retrieved_chunks

# ✅ FUNCTION: Build Prompt for OpenAI
def build_prompt(query, chunks):
    context = "\n\n".join(
        [f"Source: {chunk['source']}\nTitle: {chunk['title']}\nPage: {chunk['page_number']}\nContent:\n{chunk['content']}" 
         for chunk in chunks]
    )
    prompt = f"""
Use the following document excerpts to answer the user's query:

{context}

---

User Query: {query}

Answer the query based only on the provided content. If the answer cannot be determined from the content, state 'Not available from the provided excerpts'.
"""
    return prompt

# ✅ FUNCTION: Query OpenAI LLM
def query_openai(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided document chunks to answer the user's question accurately."},
                {"role": "user", "content": prompt}
            ],
            timeout=15  # Added timeout for reliability
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
        return "OpenAI API error. Please try again."

# ✅ FUNCTION: Save Query to Logs
def save_query_to_db(query, response, comment, model, response_time):
    try:
        cursor.execute('''
            INSERT INTO query_logs (query, response, comment, llm_model, response_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, response, comment, model, response_time))
        conn.commit()
        st.success("✅ Query saved to logs!")
    except Exception as e:
        st.error(f"❌ Database Insert Error: {e}")

# ✅ FUNCTION: View Logs from DB
def get_all_logs():
    try:
        cursor.execute('SELECT * FROM query_logs ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        st.error(f"❌ Database Retrieval Error: {e}")
        return []

# ✅ APP TITLE AND SIDEBAR
st.title("📊 Sales Playbook Chat Assistant")
st.caption("🔍 Chat with your Sales Playbook using Pinecone and OpenAI.")

model_option = st.sidebar.selectbox(
    "Select OpenAI Model",
    options=["gpt-4o", "gpt-4o-mini"],
    index=0
)

# ✅ Chat Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ DISPLAY MESSAGE HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ VIEW QUERY LOGS (ALWAYS VISIBLE)
with st.expander("📜 View Query Logs (Live)"):
    logs = get_all_logs()
    if not logs:
        st.info("No query logs available.")
    else:
        for log in logs:
            st.write(f"🕒 Timestamp: {log['timestamp']}")
            st.write(f"🗨️ Query: {log['query']}")
            st.write(f"🤖 Response: {log['response']}")
            st.write(f"💬 Comment: {log['comment']}")
            st.write(f"🧠 Model: {log['llm_model']} | ⏱️ Time Taken: {log['response_time']:.2f} sec")
            st.markdown("---")

# ✅ USER INPUT AND RESPONSE
if user_input := st.chat_input("Type your query..."):
    start_time = datetime.now()

    # 🟡 Show User Query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🟡 Retrieve Chunks
    with st.spinner("🔍 Searching document database..."):
        retrieved_chunks = search_pinecone(user_input, top_k=3)

    # 🟡 Build Prompt
    prompt = build_prompt(user_input, retrieved_chunks)

    # 🟡 Query LLM
    with st.spinner("🤖 Generating answer..."):
        assistant_response = query_openai(model_option, prompt)

    response_time = (datetime.now() - start_time).total_seconds()

    # 🟡 Show LLM Response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # 🟡 Show Retrieved Chunks (Optional)
    with st.expander("🔍 Retrieved Context (from Pinecone)"):
        for idx, chunk in enumerate(retrieved_chunks):
            st.write(f"**Chunk {idx+1}: {chunk['title']} (Page {chunk['page_number']})**")
            st.text_area("Content", chunk["content"], height=150)

    # ✅ Store the latest query/response details in session state for feedback
    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = model_option
    st.session_state.latest_response_time = response_time

# ✅ FEEDBACK FORM (Displayed Independently)
if "latest_query" in st.session_state and "latest_response" in st.session_state:
    with st.expander("💬 Provide Feedback on the Latest Response"):
        with st.form("feedback_form"):
            feedback = st.text_area("Your comment or feedback on this response:")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                save_query_to_db(
                    st.session_state.latest_query,
                    st.session_state.latest_response,
                    feedback,
                    st.session_state.latest_model,
                    st.session_state.latest_response_time
                )
                st.success("✅ Feedback saved! Thank you for helping us improve.")
