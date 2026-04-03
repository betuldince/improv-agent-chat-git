import streamlit as st
import time
import secrets
import string

from agent import (
    create_client,
    director_step,
    actor_reply,
    baseline_actor_reply,
    get_condition_scenario,
    get_condition_agent_type,
)
from database import save_round_transcripts


# =========================================================
# CONFIG
# =========================================================

ROUND_TIME_LIMIT = 7 * 60  # 7 minutes in seconds


# =========================================================
# HELPERS
# =========================================================

def generate_participant_id() -> str:
    letters = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(3))
    numbers = ''.join(secrets.choice(string.digits) for _ in range(4))
    return f"{letters}{numbers}"


# =========================================================
# STATE
# =========================================================

def init_state() -> None:
    defaults = {
        "study_started": False,
        "study_finished": False,
        "messages": [],
        "last_director_output": None,
        "actor_current_tactic": "",
        "user_current_tactic": "",
        "user_inferred_tactic": "",
        "round_logs": [],
        "opening_done": False,
        "processing_user_turn": False,
        "last_error": "",
        "recent_user_suggested_tactics": [],
        "recent_actor_tactics": [],
        "last_audio_id": None,
        "audio_input_version": 0,
        "round_start_time": None,
        "study_condition": None,
        "pending_user_text": None,
        "awaiting_actor_response": False,
        "generated_id": None,
        "results_saved": False,
        "selected_scenario": None,
        "agent_type": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def init_condition_from_query_params() -> None:
    if st.session_state.study_condition is not None:
        return

    params = st.query_params
    condition = params.get("condition", None)

    if isinstance(condition, list):
        condition = condition[0] if condition else None

    if condition is None:
        st.session_state.study_condition = None
        return

    condition = str(condition).strip().upper()

    if condition in ["A", "B", "C", "D"]:
        st.session_state.study_condition = condition
    else:
        st.session_state.study_condition = None


def get_current_scenario():
    return st.session_state.selected_scenario


def get_display_scenario():
    scenario = get_current_scenario().copy()
    scenario["show_to_user"] = scenario["show_to_user"].copy()

    # Dummy conditions: hide guidance
    if st.session_state.study_condition in ["A", "C"]:
        scenario["show_to_user"]["user_impelling_action"] = False
        scenario["show_to_user"]["show_tactic"] = False

    # Intelligent parent/teenager condition:  r
    if st.session_state.study_condition == "B":
        scenario["show_to_user"]["user_impelling_action"] = True
        scenario["show_to_user"]["show_tactic"] = True

    # Intelligent boss/worker condition: 
    if st.session_state.study_condition == "D":
        scenario["show_to_user"]["user_impelling_action"] = True
        scenario["show_to_user"]["show_tactic"] = True

    return scenario


def reset_round_state() -> None:
    st.session_state.messages = []
    st.session_state.last_director_output = None
    st.session_state.actor_current_tactic = ""
    st.session_state.user_current_tactic = ""
    st.session_state.user_inferred_tactic = ""
    st.session_state.opening_done = False
    st.session_state.processing_user_turn = False
    st.session_state.last_error = ""
    st.session_state.recent_user_suggested_tactics = []
    st.session_state.recent_actor_tactics = []
    st.session_state.last_audio_id = None
    st.session_state.audio_input_version = 0
    st.session_state.round_start_time = None
    st.session_state.pending_user_text = None
    st.session_state.awaiting_actor_response = False


def start_study() -> None:
    st.session_state.study_started = True
    st.session_state.study_finished = False
    st.session_state.round_logs = []
    st.session_state.generated_id = None
    st.session_state.results_saved = False

    scenario = get_condition_scenario(st.session_state.study_condition)
    agent_type = get_condition_agent_type(st.session_state.study_condition)

    st.session_state.selected_scenario = scenario
    st.session_state.agent_type = agent_type

    reset_round_state()


def archive_current_round(reason: str) -> None:
    scenario = get_current_scenario()
    st.session_state.round_logs = [
        {
            "round_number": scenario["round_number"],
            "messages": st.session_state.messages.copy(),
            "end_reason": reason,
        }
    ]


# =========================================================
# TIMER
# =========================================================

def check_round_timeout(client) -> None:
    if (
        not st.session_state.study_started
        or st.session_state.study_finished
        or st.session_state.round_start_time is None
    ):
        return

    elapsed = time.time() - st.session_state.round_start_time
    remaining = int(ROUND_TIME_LIMIT - elapsed)

    if remaining <= 0:
        archive_current_round(reason="timer_expired")
        st.session_state.study_finished = True
        st.session_state.study_started = False

        if st.session_state.generated_id is None:
            st.session_state.generated_id = generate_participant_id()

        st.rerun()


# =========================================================
# AUDIO / TRANSCRIPTION
# =========================================================

def transcribe_audio(client, audio_file) -> str:
    audio_bytes = audio_file.getvalue()
    audio_name = audio_file.name or "recording.wav"
    audio_type = getattr(audio_file, "type", None) or "audio/wav"

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(audio_name, audio_bytes, audio_type),
    )

    text = getattr(transcript, "text", "") or ""
    return text.strip()


# =========================================================
# AGENT ORCHESTRATION
# =========================================================

def open_scene_with_actor(client, scenario) -> None:
    if st.session_state.agent_type == "dummy":
        first_line = baseline_actor_reply(
            client=client,
            scenario=scenario,
            messages=[],
            opening_line=True,
        )

        st.session_state.last_director_output = None
        st.session_state.actor_current_tactic = ""
        st.session_state.user_inferred_tactic = ""
        st.session_state.user_current_tactic = ""
        st.session_state.messages.append({"role": "assistant", "content": first_line})
        st.session_state.opening_done = True
        st.session_state.round_start_time = time.time()
        return

    director_out = director_step(
        client=client,
        scenario=scenario,
        messages=[],
        recent_user_suggested_tactics=st.session_state.recent_user_suggested_tactics,
        recent_actor_tactics=st.session_state.recent_actor_tactics,
    )

    st.session_state.last_director_output = director_out
    st.session_state.actor_current_tactic = director_out["actor_tactic"]
    st.session_state.user_inferred_tactic = director_out["user_tactic_inferred"]

    if scenario["show_to_user"].get("show_tactic", True):
        st.session_state.user_current_tactic = director_out["user_tactic_suggested"]
    else:
        st.session_state.user_current_tactic = ""

    st.session_state.recent_user_suggested_tactics.append(
        director_out["user_tactic_suggested"]
    )
    st.session_state.recent_actor_tactics.append(
        director_out["actor_tactic"]
    )

    first_line = actor_reply(
        client=client,
        scenario=scenario,
        messages=[],
        actor_tactic=director_out["actor_tactic"],
        opening_line=True,
    )

    st.session_state.messages.append({"role": "assistant", "content": first_line})
    st.session_state.opening_done = True
    st.session_state.round_start_time = time.time()


def generate_pending_actor_response(client, scenario) -> None:
    if not st.session_state.awaiting_actor_response:
        return

    if st.session_state.agent_type == "dummy":
        actor_text = baseline_actor_reply(
            client=client,
            scenario=scenario,
            messages=st.session_state.messages,
            opening_line=False,
        )

        st.session_state.last_director_output = None
        st.session_state.actor_current_tactic = ""
        st.session_state.user_inferred_tactic = ""
        st.session_state.user_current_tactic = ""
        st.session_state.messages.append({"role": "assistant", "content": actor_text})
        st.session_state.awaiting_actor_response = False
        st.session_state.pending_user_text = None
        return

    director_out = director_step(
        client=client,
        scenario=scenario,
        messages=st.session_state.messages,
        recent_user_suggested_tactics=st.session_state.recent_user_suggested_tactics,
        recent_actor_tactics=st.session_state.recent_actor_tactics,
    )

    st.session_state.last_director_output = director_out
    st.session_state.actor_current_tactic = director_out["actor_tactic"]
    st.session_state.user_inferred_tactic = director_out["user_tactic_inferred"]

    if scenario["show_to_user"].get("show_tactic", True):
        st.session_state.user_current_tactic = director_out["user_tactic_suggested"]
    else:
        st.session_state.user_current_tactic = ""

    st.session_state.recent_user_suggested_tactics.append(
        director_out["user_tactic_suggested"]
    )
    st.session_state.recent_actor_tactics.append(
        director_out["actor_tactic"]
    )

    actor_text = actor_reply(
        client=client,
        scenario=scenario,
        messages=st.session_state.messages,
        actor_tactic=director_out["actor_tactic"],
        opening_line=False,
    )

    st.session_state.messages.append({"role": "assistant", "content": actor_text})
    st.session_state.awaiting_actor_response = False
    st.session_state.pending_user_text = None


# =========================================================
# UI
# =========================================================

def render_conversation(scenario) -> None:
    for msg in st.session_state.messages:
        speaker = scenario["actor_role"] if msg["role"] == "assistant" else "You"
        with st.chat_message(msg["role"]):
            st.write(f"{speaker}: {msg['content']}")


def render_sidebar(scenario) -> None:
    with st.sidebar:
        st.subheader("Assigned scenario")
        st.write(f"Your role: {scenario['user_role']}")
        st.write(f"Improv Agent role: {scenario['actor_role']}")

        st.markdown("---")
        st.write("Scenario")
        st.write(scenario["prompt"])

        if scenario["show_to_user"].get("user_impelling_action", False):
            st.markdown("---")
            st.write("Your impelling action")
            st.info(scenario["user_impelling_action"])

        if scenario["show_to_user"].get("show_tactic", True):
            st.markdown("---")
            st.write("Suggested tactic for your next move")
            st.info(st.session_state.user_current_tactic or "No suggestion yet.")


# =========================================================
# APP
# =========================================================

st.set_page_config(
    page_title="Improv Agent Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 22px;
    }

    section[data-testid="stSidebar"] * {
        font-size: 20px !important;
    }

    [data-testid="stChatMessage"] {
        font-size: 22px !important;
    }

    .stButton > button {
        font-size: 20px !important;
    }

    label, .stMarkdown, .stText, p {
        font-size: 22px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()
init_condition_from_query_params()
client = create_client(st.secrets["OPENAI_API_KEY"])

st.title("Improv Agent Study")

if not st.session_state.study_started and not st.session_state.study_finished:
    st.write("You will complete one assigned interaction scenario.")

    if st.session_state.study_condition == "A":
        st.info(
            "You have been assigned the Boss/Worker scenario. "
            "To respond, use the microphone bar under 'Your response':\n"
            "1. Click the mic icon to start recording.\n"
            "2. Speak your response out loud.\n"
            "3. Click the mic icon again to stop recording.\n"
            "4. Wait while your speech is transcribed and sent to the AI agent.\n\n"
            "At the end, you will receive a generated ID to enter in Qualtrics."
        )
    elif st.session_state.study_condition == "B":
        st.info(
            "You have been assigned the Parent/Teenager scenario. "
            "To respond, use the microphone bar under 'Your response':\n"
            "1. Click the mic icon to start recording.\n"
            "2. Speak your response out loud.\n"
            "3. Click the mic icon again to stop recording.\n"
            "4. Wait while your speech is transcribed and sent to the AI agent.\n\n"
            "At the end, you will receive a generated ID to enter in Qualtrics."
        )
    elif st.session_state.study_condition == "C":
        st.info(
            "You have been assigned the Parent/Teenager scenario. "
            "To respond, use the microphone bar under 'Your response':\n"
            "1. Click the mic icon to start recording.\n"
            "2. Speak your response out loud.\n"
            "3. Click the mic icon again to stop recording.\n"
            "4. Wait while your speech is transcribed and sent to the AI agent.\n\n"
            "At the end, you will receive a generated ID to enter in Qualtrics."
        )
    elif st.session_state.study_condition == "D":
        st.info(
            "You have been assigned the Boss/Worker scenario. "
            "To respond, use the microphone bar under 'Your response':\n"
            "1. Click the mic icon to start recording.\n"
            "2. Speak your response out loud.\n"
            "3. Click the mic icon again to stop recording.\n"
            "4. Wait while your speech is transcribed and sent to the AI agent.\n\n"
            "At the end, you will receive a generated ID to enter in Qualtrics."
        )
    else:
        st.warning("Condition not recognized. Use ?condition=A, ?condition=B, ?condition=C, or ?condition=D in the URL.")

    if st.button("Start study"):
        try:
            start_study()

            if st.session_state.selected_scenario is None or st.session_state.agent_type is None:
                st.error("Invalid condition mapping. Check your agent.py condition helpers.")
                st.stop()

            scenario = get_current_scenario()
            with st.spinner("Starting the scene..."):
                open_scene_with_actor(client, scenario)
            st.rerun()
        except Exception as e:
            st.session_state.last_error = f"Start error: {str(e)}"
            st.error(st.session_state.last_error)
    st.stop()

if st.session_state.study_finished:
    if (
        not st.session_state.results_saved
        and st.session_state.study_condition in ["A", "B", "C", "D"]
    ):
        try:
            save_round_transcripts(
                participant_id=st.session_state.generated_id,
                study_condition=st.session_state.study_condition,
                round_logs=st.session_state.round_logs,
            )
            st.session_state.results_saved = True
        except Exception as e:
            st.error(f"Could not save study results: {str(e)}")

    st.success("The study is finished.")

    st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 34px;
            font-weight: bold;
            padding: 20px 0;
        ">
            Your generated ID: {st.session_state.generated_id}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Please return to the Qualtrics tab and enter this generated ID.")
    st.stop()

scenario = get_display_scenario()

if st.session_state.awaiting_actor_response:
    render_sidebar(scenario)
    st.divider()
    render_conversation(scenario)

    with st.spinner("Generating response..."):
        generate_pending_actor_response(client, get_current_scenario())

    st.rerun()

check_round_timeout(client)

render_sidebar(scenario)

st.divider()
render_conversation(scenario)

st.subheader("Your response")
st.caption("Click the mic icon to start, click it again to stop, then wait for transcription.")

audio_value = st.audio_input(
    f"Record your reply as the {scenario['user_role']}",
    key=f"audio_input_condition_{st.session_state.study_condition}_{st.session_state.audio_input_version}",
)

if audio_value is not None:
    current_audio_id = f"{audio_value.name}-{audio_value.size}"

    if (
        current_audio_id != st.session_state.last_audio_id
        and not st.session_state.processing_user_turn
    ):
        st.session_state.processing_user_turn = True
        try:
            with st.spinner("Transcribing your recording..."):
                transcript_text = transcribe_audio(client, audio_value)

            if not transcript_text:
                st.session_state.last_error = "Transcription came back empty."
            else:
                st.session_state.last_audio_id = current_audio_id

                st.session_state.messages.append({"role": "user", "content": transcript_text})
                st.session_state.pending_user_text = transcript_text
                st.session_state.awaiting_actor_response = True

                check_round_timeout(client)

                st.session_state.audio_input_version += 1
                st.session_state.last_audio_id = None

                st.rerun()

        except Exception as e:
            st.session_state.last_error = f"Audio/transcription error: {str(e)}"
            st.error(st.session_state.last_error)
        finally:
            st.session_state.processing_user_turn = False