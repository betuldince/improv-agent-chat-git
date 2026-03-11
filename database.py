from supabase import create_client
import streamlit as st
import datetime

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)

def save_message(session_id, premise, role, message, turn_index, condition):

    data = {
        "session_id": session_id,
        "premise": premise,
        "role": role,
        "message": message,
        "turn_index": turn_index,
        "condition": condition
    }

    supabase.table("messages_improv").insert(data).execute()


def save_conversation(session_id, premise, chat):

    data = {
        "session_id": session_id,
        "premise": premise,
        "conversation": chat
    }

    supabase.table("improv_chat_1").insert(data).execute()