# sales playbook chat assistant with fixed logging

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

# load environment variables from file
load_dotenv()

# set page configuration for the streamlit app
st.set_page_config(page_title="sales playbook chat assistant", layout="wide")

# initialize sentence transformer model for text embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# function to get a database connection
def get_db_connection():
    # get the database path from an environment variable or default to query_logs.db
    db_path = os.getenv("DB_PATH", "query_logs.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # enable access to columns by name
    return conn

conn = get_db_connection()
cursor = conn.cursor()

# create the logs table if it does not exist
cursor.execute('''
create table if not exists query_logs (
    id integer primary key autoincrement,
    query text,
    response text,
    comment text,
    llm_model text,
    response_time float,
    timestamp datetime default current_timestamp
)
''')
conn.commit()

# initialize the openai client with the api key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# initialize the pinecone client with the api key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "document-index"
index = pc.Index(index_name)

# function to generate embeddings using the sentence transformer
def get_embedding(text):
    return embedder.encode(text).tolist()

# function to search the pinecone index and return retrieved chunks along with timing details
def search_pinecone_with_timing(query, top_k=3):
    start_embedding = time.perf_counter()
    query_embedding = get_embedding(query)
    embedding_time = time.perf_counter() - start_embedding

    start_pinecone = time.perf_counter()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"source": {"$in": ["section_text", "image_summary"]}}
    )
    pinecone_time = time.perf_counter() - start_pinecone

    retrieved_chunks = []
    for match in results["matches"]:
        metadata = match["metadata"]
        retrieved_chunks.append({
            "source": metadata.get("source", "unknown"),
            "title": metadata.get("title", ""),
            "page_number": metadata.get("page_number", ""),
            "score": match["score"],
            "content": metadata.get("content", "no content available")
        })
    return retrieved_chunks, embedding_time, pinecone_time

# function to build the prompt for the openai model using the query and retrieved chunks
def build_prompt(query, chunks):
    context = "\n\n".join(
        [f"source {chunk['source']} title {chunk['title']} page {chunk['page_number']} content \n{chunk['content']}" 
         for chunk in chunks]
    )
    prompt = f"""
use the following document excerpts to answer the users query

{context}

user query {query}

answer the query based only on the provided content if the answer cannot be determined from the content state not available from the provided excerpts
"""
    return prompt

# function to query the openai model with the prompt
def query_openai(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "you are a helpful assistant use the provided document chunks to answer the users question accurately"},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"openai api error {e}")
        return "openai api error please try again"

# function to save the query and its response along with feedback to the database
def save_query_to_db(query, response, comment, model, response_time):
    try:
        cursor.execute('''
            insert into query_logs (query, response, comment, llm_model, response_time)
            values (?, ?, ?, ?, ?)
        ''', (query, response, comment, model, response_time))
        conn.commit()
        st.success("query saved to logs")
    except Exception as e:
        st.error(f"database insert error {e}")

# function to retrieve all query logs from the database
def get_all_logs():
    try:
        cursor.execute('select * from query_logs order by timestamp desc')
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        st.error(f"database retrieval error {e}")
        return []

# set the title and caption for the app
st.title("sales playbook chat assistant")
st.caption("chat with your sales playbook using pinecone and openai")

# sidebar option to select the openai model
model_option = st.sidebar.selectbox(
    "select openai model",
    options=["gpt-4o", "gpt-4o-mini"],
    index=0
)

# initialize chat message history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat message history from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# section to view query logs from the database with a refresh button
with st.expander("view query logs live"):
    if st.button("refresh logs"):
        pass  # pressing the button causes a rerun of the script automatically
    logs = get_all_logs()
    if not logs:
        st.info("no query logs available")
    else:
        for log in logs:
            st.write(f"timestamp {log['timestamp']}")
            st.write(f"query {log['query']}")
            st.write(f"response {log['response']}")
            st.write(f"comment {log['comment']}")
            st.write(f"model {log['llm_model']} time taken {log['response_time']:.2f} sec")
            st.markdown("---")

# get the user input from the chat input box
if user_input := st.chat_input("type your query"):
    overall_start = time.perf_counter()

    # append the user query to the session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # retrieve relevant chunks from the pinecone index along with timing details
    with st.spinner("searching document database"):
        retrieved_chunks, embedding_time, pinecone_time = search_pinecone_with_timing(user_input, top_k=3)

    # measure prompt building time
    start_prompt = time.perf_counter()
    prompt = build_prompt(user_input, retrieved_chunks)
    prompt_time = time.perf_counter() - start_prompt

    # query the openai model with the built prompt and measure model query time
    start_model = time.perf_counter()
    assistant_response = query_openai(model_option, prompt)
    model_time = time.perf_counter() - start_model

    overall_time = time.perf_counter() - overall_start

    # append the assistant response to the session state and display it
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # optionally display the retrieved chunks from the pinecone index
    with st.expander("retrieved context from pinecone"):
        for idx, chunk in enumerate(retrieved_chunks):
            st.write(f"chunk {idx+1} {chunk['title']} page {chunk['page_number']}")
            st.text_area("content", chunk["content"], height=150)

    # display timing details for each step
    with st.expander("timing details"):
        st.write(f"embedding time {embedding_time:.2f} sec")
        st.write(f"pinecone search time {pinecone_time:.2f} sec")
        st.write(f"prompt building time {prompt_time:.2f} sec")
        st.write(f"model query time {model_time:.2f} sec")
        st.write(f"overall time {overall_time:.2f} sec")

    # save the latest query and response details in session state for feedback
    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = model_option
    st.session_state.latest_response_time = overall_time

# section for the feedback form for the latest response
if "latest_query" in st.session_state and "latest_response" in st.session_state:
    with st.expander("provide feedback on the latest response"):
        with st.form("feedback_form"):
            feedback = st.text_area("your comment or feedback on this response")
            submitted = st.form_submit_button("submit feedback")
            if submitted:
                save_query_to_db(
                    st.session_state.latest_query,
                    st.session_state.latest_response,
                    feedback,
                    st.session_state.latest_model,
                    st.session_state.latest_response_time
                )
                st.success("feedback saved thank you for helping us improve")
