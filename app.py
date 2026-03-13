import streamlit as st
import uuid
import time

from streamlit_autorefresh import st_autorefresh

from improv_agent import (
    Orchestrator,
    ActorLLM,
    DirectorLLM,
    generate_first_line
)

from improv_agent_baseline import (
    BaselineOrchestrator,
    BaselineActorLLM
)

from database import save_message, save_conversation


MODEL = "gpt-4o-mini"

TIME_LIMIT = 100  # 10 minutes

QUALTRICS_LINK = "https://neu.co1.qualtrics.com/jfe/form/SV_3TYtAhRcMTCSe2i"


st.title("Improv AI Partner")


# -------------------------
# Condition from URL
# -------------------------

params = st.query_params

if "condition" not in params:
    st.error("Study condition not specified in the link.")
    st.stop()

condition = params["condition"]

valid_conditions = ["A", "B", "testingA", "testingB"]

if condition not in valid_conditions:
    st.error("Invalid condition.")
    st.stop()

st.session_state.condition = condition


# -------------------------
# Session State
# -------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "premise" not in st.session_state:
    st.session_state.premise = None

if "persona" not in st.session_state:
    st.session_state.persona = None

if "actor" not in st.session_state:
    st.session_state.actor = None

if "director" not in st.session_state:
    st.session_state.director = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "scenario_ready" not in st.session_state:
    st.session_state.scenario_ready = False

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "experiment_over" not in st.session_state:
    st.session_state.experiment_over = False


# -------------------------
# Orchestrator selection
# -------------------------

if st.session_state.condition in ["A", "testingA"]:
    orchestrator = BaselineOrchestrator(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
else:
    orchestrator = Orchestrator(
        api_key=st.secrets["OPENAI_API_KEY"]
    )


# -------------------------
# Generate Scenario
# -------------------------

if st.button("Generate Scenario"):

    premise = orchestrator.generate_prompt()

    st.session_state.premise = premise
    st.session_state.scenario_ready = True
    st.session_state.actor = None
    st.session_state.chat = []


# -------------------------
# Show Scenario
# -------------------------

if st.session_state.premise:

    st.subheader("Scenario")
    st.info(st.session_state.premise)


# -------------------------
# Accept / Regenerate
# -------------------------

if st.session_state.scenario_ready and st.session_state.actor is None:

    col1, col2 = st.columns(2)

    if col1.button("Accept Scenario"):

        if st.session_state.condition in ["A", "testingA"]:

            actor = BaselineActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=st.session_state.premise
            )

            opening = actor.opening_line()

        else:

            persona = orchestrator.generate_persona(
                st.session_state.premise
            )

            director = DirectorLLM(
                api_key=st.secrets["OPENAI_API_KEY"]
            )

            actor = ActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=st.session_state.premise,
                persona=persona
            )

            opening = generate_first_line(
                client=orchestrator.client,
                director=director,
                premise=st.session_state.premise,
                persona=persona
            )

            actor.history.append({
                "role": "assistant",
                "content": opening
            })

            actor.actor_turn_count = 1

            st.session_state.persona = persona
            st.session_state.director = director


        st.session_state.actor = actor

        st.session_state.chat = [
            {"role": "assistant", "content": opening}
        ]

        save_message(
            st.session_state.session_id,
            st.session_state.premise,
            "assistant",
            opening,
            1,
            st.session_state.condition
        )

        st.session_state.start_time = time.time()

        st.rerun()


    if col2.button("Regenerate Scenario"):

        premise = orchestrator.generate_prompt()
        st.session_state.premise = premise
        st.rerun()


# -------------------------
# Timer
# -------------------------

if st.session_state.start_time is not None:

    st_autorefresh(interval=1000, key="timer")

    elapsed = time.time() - st.session_state.start_time
    remaining = int(TIME_LIMIT - elapsed)

    if remaining <= 0:
        st.session_state.experiment_over = True

    else:

        minutes = remaining // 60
        seconds = remaining % 60

        st.markdown(
            f"## ⏳ Time remaining: {minutes:02d}:{seconds:02d}"
        )


# -------------------------
# End experiment
# -------------------------

if st.session_state.experiment_over:

    st.success("The improvisation session has ended.")

    st.markdown(
        """
### The experiment is now complete.

Please continue by answering the questions in **Qualtrics**.

Thank you for participating.
"""
    )

    st.link_button(
        "Continue to Qualtrics Survey",
        QUALTRICS_LINK
    )

    st.stop()


# -------------------------
# Chat Interface
# -------------------------

if st.session_state.actor:

    st.subheader("Scene")

    for msg in st.session_state.chat:

        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])


# -------------------------
# Chat input
# -------------------------

user_input = None

if not st.session_state.experiment_over:
    user_input = st.chat_input("Your response")


if user_input:

    st.session_state.chat.append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").write(user_input)

    turn_index = len(st.session_state.chat)

    save_message(
        st.session_state.session_id,
        st.session_state.premise,
        "user",
        user_input,
        turn_index,
        st.session_state.condition
    )


    with st.spinner("Actor is thinking..."):

        if st.session_state.condition in ["A", "testingA"]:

            reply = st.session_state.actor.respond(user_input)

        else:

            reply = st.session_state.actor.respond(
                user_input,
                director=st.session_state.director
            )


    st.session_state.chat.append({
        "role": "assistant",
        "content": reply
    })

    st.chat_message("assistant").write(reply)

    turn_index = len(st.session_state.chat)

    save_message(
        st.session_state.session_id,
        st.session_state.premise,
        "assistant",
        reply,
        turn_index,
        st.session_state.condition
    )


    save_conversation(
        st.session_state.session_id,
        st.session_state.premise,
        st.session_state.chat,
        st.session_state.condition
    )