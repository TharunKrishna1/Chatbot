
import streamlit as st
from uuid import uuid4
from langchain_core.messages import HumanMessage

from backend import chatbot

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph Chatbot")
st.caption("Powered by LangGraph + Groq")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Chat Settings")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Current Thread ID")
    st.code(st.session_state.thread_id)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Backend configuration
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config
            )

            answer = response["messages"][-1].content

            st.markdown(answer)

    # Store AI response in frontend history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
