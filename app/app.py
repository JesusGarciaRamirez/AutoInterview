import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatReflective


st.set_page_config(page_title="AutoInterview")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    local_llm = "mistral:instruct"
    candiate_name = "Jesus"
    profiles_dir = "Profiles"

    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatReflective(candidate=candiate_name, llm_model=local_llm, profiles_dir=profiles_dir)
        st.session_state["index"] = False

    st.header("Welcome to AutoInterview!")

    # Create Index (Candidate´s profile)
    st.session_state["ingestion_spinner"] = st.empty()
    # Only create index the first time the app is run
    if not(st.session_state["index"]):
        with st.session_state["ingestion_spinner"], st.spinner(f"Creating index for {candiate_name}"):
            st.session_state["assistant"].create_index()
            st.session_state["index"] = True
    # Chat
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
