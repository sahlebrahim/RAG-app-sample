#########################
# sales playbook chat assistant with fixed logging (postgres + cohere rerank)
#########################

import os
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# new: psycopg2 for postgres
import psycopg2
import psycopg2.extras

# new: cohere for reranking
import cohere

# pinecone
from pinecone import Pinecone

# openai
from openai import OpenAI

# sentence transformers
from sentence_transformers import SentenceTransformer

#########################
# 1) load environment variables
#########################
load_dotenv()

#########################
# 2) set up streamlit page
#########################
st.set_page_config(page_title="sales playbook chat assistant", layout="wide")

#########################
# 3) cohere client (for reranking)
#########################
cohere_api_key = os.getenv("COHERE_API_KEY")
if cohere_api_key is None:
    st.warning("cohere api key not set (cohere_api_key). reranking will not work.")
    co = None
else:
    # use cohere.ClientV2 if that is your version, or cohere.Client otherwise
    co = cohere.ClientV2(api_key=cohere_api_key)

#########################
# 4) openai client
#########################
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

#########################
# 5) pinecone client
#########################
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "document-index"
index = pc.Index(index_name)

#########################
# 6) sentence transformer model
#########################
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#########################
# 7) connect to heroku postgres
#########################
def get_db_connection():
    """
    connects to postgres using the heroku database_url env var
    e.g. heroku config:set DATABASE_URL=...
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL not set. please configure heroku postgres.")
    # sslmode=require is typical on heroku
    conn = psycopg2.connect(db_url, sslmode="require")
    return conn

# create the logs table if it does not exist
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        create table if not exists query_logs (
            id serial primary key,
            query text,
            response text,
            comment text,
            llm_model text,
            response_time float,
            timestamp timestamptz default now()
        )
        ''')
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"database init error: {e}")

init_db()

#########################
# 8) helper functions for database
#########################
def save_query_to_db(query, response, comment, model, response_time):
    """
    inserts a new record into the query_logs table
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            insert into query_logs (query, response, comment, llm_model, response_time)
            values (%s, %s, %s, %s, %s)
        ''', (query, response, comment, model, response_time))
        conn.commit()
        cur.close()
        conn.close()
        st.success("query saved to logs")
    except Exception as e:
        st.error(f"database insert error: {e}")

def get_all_logs():
    """
    retrieves all logs from query_logs
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('select * from query_logs order by timestamp desc')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"database retrieval error: {e}")
        return []

#########################
# 9) pinecone + cohere rerank
#########################

def get_embedding(text):
    """
    uses sentence transformer to generate embeddings
    """
    return embedder.encode(text).tolist()

def cohere_rerank(query, chunks, top_n=3):
    """
    takes a query and a list of chunk dicts, calls cohere rerank,
    and returns the top n chunks in sorted order
    each chunk is a dict with "content" plus any other metadata
    """
    if not co:
        # if cohere is not configured, just return original chunks
        return chunks[:top_n]

    # cohere rerank expects a list of strings or structured docs
    # for simplicity, we'll just pass chunk["content"] as the doc
    docs = [c["content"] for c in chunks]

    try:
        # model = "rerank-v3.5" or any other cohere model
        results = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs,
            top_n=len(chunks),  # rerank all
        )
        # results.documents is a list of reranked documents with .index, .relevance_score
        # we can reorder chunks by these indexes
        # create a map: old_index -> chunk
        indexed_chunks = {i: chunk for i, chunk in enumerate(chunks)}
        # build a new list in the order cohere provides
        reranked_chunks = []
        for doc in results.documents:
            # doc.index is the original index
            # doc.relevance_score is the new score
            chunk = indexed_chunks[doc.index]
            chunk["rerank_score"] = doc.relevance_score
            reranked_chunks.append(indexed_chunks[doc.index])

        # now we have a reranked list from most relevant to least
        return reranked_chunks[:top_n]

    except Exception as e:
        st.warning(f"cohere rerank error: {e}")
        # fallback to original top_n
        return chunks[:top_n]

def search_pinecone_with_timing(query, top_k=3):
    """
    1) do embedding
    2) query pinecone for top 15
    3) cohere rerank them
    4) slice final top_k
    returns final chunks plus timing details
    """
    start_embedding = time.perf_counter()
    query_embedding = get_embedding(query)
    embedding_time = time.perf_counter() - start_embedding

    # let's fetch 15 from pinecone, then rerank down to top_k
    pinecone_fetch = 15

    start_pinecone = time.perf_counter()
    results = index.query(
        vector=query_embedding,
        top_k=pinecone_fetch,
        include_metadata=True,
        filter={"source": {"$in": ["section_text", "image_summary"]}}
    )
    pinecone_time = time.perf_counter() - start_pinecone

    all_chunks = []
    if "matches" in results:
        for match in results["matches"]:
            metadata = match["metadata"]
            all_chunks.append({
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", ""),
                "page_number": metadata.get("page_number", ""),
                "score": match["score"],
                "content": metadata.get("content", "no content available")
            })

    # now cohere rerank
    # cohere_rerank will return the top_k final
    final_chunks = cohere_rerank(query, all_chunks, top_n=top_k)

    return final_chunks, embedding_time, pinecone_time

#########################
# 10) build prompt + query openai
#########################

def build_prompt(query, chunks):
    context = "\n\n".join(
        [f"source {c['source']} title {c['title']} page {c['page_number']} content \n{c['content']}"
         for c in chunks]
    )
    prompt = f"""
use the following document excerpts to answer the users query

{context}

user query {query}

answer the query based only on the provided content if the answer cannot be determined from the content state not available from the provided excerpts
"""
    return prompt

def query_openai(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "you are a helpful assistant use the provided document chunks "
                        "to answer the users question accurately"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=15,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"openai api error {e}")
        return "openai api error please try again"

#########################
# 11) streamlit ui
#########################

st.title("sales playbook chat assistant")
st.caption("chat with your sales playbook using pinecone, cohere (for reranking), and openai")

# model selection
model_option = st.sidebar.selectbox(
    "select openai model",
    options=["gpt-4o", "gpt-4o-mini"],
    index=0
)

# initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# show logs
with st.expander("view query logs live"):
    if st.button("refresh logs"):
        pass  # triggers rerun
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

# user query
if user_input := st.chat_input("type your query"):
    overall_start = time.perf_counter()

    # show user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # retrieve chunks (top 3 after cohere rerank)
    with st.spinner("searching document database..."):
        retrieved_chunks, embedding_time, pinecone_time = search_pinecone_with_timing(user_input, top_k=3)

    # build prompt
    start_prompt = time.perf_counter()
    prompt = build_prompt(user_input, retrieved_chunks)
    prompt_time = time.perf_counter() - start_prompt

    # query openai
    start_model = time.perf_counter()
    assistant_response = query_openai(model_option, prompt)
    model_time = time.perf_counter() - start_model

    overall_time = time.perf_counter() - overall_start

    # show assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # optionally show retrieved chunks
    with st.expander("retrieved context from pinecone"):
        for idx, chunk in enumerate(retrieved_chunks):
            score = chunk.get("rerank_score", "n/a")
            st.write(f"chunk {idx+1} score {score} title {chunk['title']} page {chunk['page_number']}")
            st.text_area("content", chunk["content"], height=150)

    # show timing details
    with st.expander("timing details"):
        st.write(f"embedding time {embedding_time:.2f} sec")
        st.write(f"pinecone search time {pinecone_time:.2f} sec")
        st.write(f"prompt building time {prompt_time:.2f} sec")
        st.write(f"model query time {model_time:.2f} sec")
        st.write(f"overall time {overall_time:.2f} sec")

    # store latest query in session for feedback
    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = model_option
    st.session_state.latest_response_time = overall_time

# feedback form
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
                st.session_state["feedback_input"] = ""  # reset the text area
                st.experimental_rerun()
