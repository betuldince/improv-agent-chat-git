from supabase import create_client
import streamlit as st


def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def save_round_transcripts(participant_id: str, study_condition: str, round_logs: list) -> None:
    supabase = get_supabase()

    rows = []
    for log in round_logs:
        rows.append(
            {
                "participant_id": participant_id,
                "study_condition": study_condition,
                "round_number": log["round_number"],
                "transcript": log["messages"],
            }
        )

    if rows:
        supabase.table("improv_round_transcripts").insert(rows).execute()