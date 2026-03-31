import streamlit as st
import time

from agent_app_audio import (
    SCENARIOS,
    create_client,
    director_step,
    actor_reply,
)


# =========================================================
# CONFIG
# =========================================================

ROUND_TIME_LIMIT = 100  # 7 minutes in seconds


# =========================================================
# STATE
# =========================================================

def init_state() -> None:
    defaults = {
        "study_started": False,
        "study_finished": False,
        "current_round_index": 0,
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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_scenario():
    return SCENARIOS[st.session_state.current_round_index]


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


def start_study() -> None:
    st.session_state.study_started = True
    st.session_state.study_finished = False
    st.session_state.current_round_index = 0
    st.session_state.round_logs = []
    start_round(0)


def start_round(round_index: int) -> None:
    st.session_state.current_round_index = round_index
    reset_round_state()


def archive_current_round(reason: str) -> None:
    scenario = get_current_scenario()
    st.session_state.round_logs.append(
        {
            "round_number": scenario["round_number"],
            "user_role": scenario["user_role"],
            "actor_role": scenario["actor_role"],
            "prompt": scenario["prompt"],
            "end_reason": reason,
            "messages": st.session_state.messages.copy(),
        }
    )


def move_to_next_round(reason: str = "manual_next_round") -> None:
    archive_current_round(reason=reason)

    next_index = st.session_state.current_round_index + 1
    if next_index >= len(SCENARIOS):
        st.session_state.study_finished = True
        st.session_state.study_started = False
        return

    start_round(next_index)


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
        move_to_next_round(reason="timer_expired")

        if st.session_state.study_finished:
            st.rerun()

        next_scenario = get_current_scenario()
        with st.spinner("Director is starting the next scene..."):
            open_scene_with_actor(client, next_scenario)
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
        director_note_for_actor=director_out["director_note_for_actor"],
        opening_line=True,
    )

    st.session_state.messages.append({"role": "assistant", "content": first_line})
    st.session_state.opening_done = True
    st.session_state.round_start_time = time.time()


def process_user_turn(client, scenario, user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})

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
        director_note_for_actor=director_out["director_note_for_actor"],
        opening_line=False,
    )

    st.session_state.messages.append({"role": "assistant", "content": actor_text})


# =========================================================
# UI
# =========================================================

def render_conversation(scenario) -> None:
    for msg in st.session_state.messages:
        speaker = scenario["actor_role"] if msg["role"] == "assistant" else "You"
        with st.chat_message(msg["role"]):
            st.write(f"{speaker}: {msg['content']}")


def render_sidebar(scenario, client) -> None:
    with st.sidebar:
        st.subheader(f"Round {scenario['round_number']} of {len(SCENARIOS)}")
        st.write(f"User role: {scenario['user_role']}")
        st.write(f"Actor role: {scenario['actor_role']}")

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

        st.markdown("---")
        if st.button("Next round"):
            try:
                move_to_next_round()

                if st.session_state.study_finished:
                    st.rerun()

                next_scenario = get_current_scenario()
                with st.spinner("Director is starting the next scene..."):
                    open_scene_with_actor(client, next_scenario)
                st.rerun()

            except Exception as e:
                st.session_state.last_error = f"Next round error: {str(e)}"
                st.error(st.session_state.last_error)


# =========================================================
# APP
# =========================================================

st.set_page_config(
    page_title="Improv Agent Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
client = create_client(st.secrets["OPENAI_API_KEY"])

st.title("Improv Agent Study")

if not st.session_state.study_started and not st.session_state.study_finished:
    st.write("This study has 3 rounds.")
    if st.button("Start study"):
        try:
            start_study()
            scenario = get_current_scenario()
            with st.spinner("Director is starting the scene..."):
                open_scene_with_actor(client, scenario)
            st.rerun()
        except Exception as e:
            st.session_state.last_error = f"Start error: {str(e)}"
            st.error(st.session_state.last_error)
    st.stop()

if st.session_state.study_finished:
    st.success("The study is finished.")
    st.subheader("Round summaries")

    for log in st.session_state.round_logs:
        st.write(
            f"Round {log['round_number']} | "
            f"{log['user_role']} vs {log['actor_role']} | "
            f"Ended because: {log['end_reason']}"
        )

        with st.expander(f"Transcript for round {log['round_number']}"):
            for msg in log["messages"]:
                speaker = log["actor_role"] if msg["role"] == "assistant" else "You"
                st.write(f"{speaker}: {msg['content']}")

    if st.button("Restart full study"):
        st.session_state.clear()
        st.rerun()

    st.stop()

scenario = get_current_scenario()

check_round_timeout(client)

render_sidebar(scenario, client)

st.divider()
render_conversation(scenario)

st.subheader("Your response")
audio_value = st.audio_input(
    f"Record your reply as the {scenario['user_role']}",
    key=f"audio_input_round_{scenario['round_number']}_{st.session_state.audio_input_version}",
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
                process_user_turn(client, scenario, transcript_text)

                # After each rerun-producing interaction, check whether round time is over
                check_round_timeout(client)

                # Reset the audio recorder widget after a successful turn
                st.session_state.audio_input_version += 1
                st.session_state.last_audio_id = None

                st.rerun()

        except Exception as e:
            st.session_state.last_error = f"Audio/transcription error: {str(e)}"
            st.error(st.session_state.last_error)
        finally:
            st.session_state.processing_user_turn = False