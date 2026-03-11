import streamlit as st
from improv_agent import Orchestrator, ActorLLM
from improv_agent_baseline import BaselineOrchestrator,  BaselineActorLLM
import uuid

from database import save_message
from database import save_conversation
MODEL = "gpt-4o-mini"

st.title("Improv AI Partner")

# -------------------------
# Read condition from URL
# -------------------------

params = st.query_params

if "condition" not in params:
    st.error("Study condition not specified in the link.")
    st.stop()

condition = params["condition"]

if condition not in ["A", "B"]:
    st.error("Invalid condition.")
    st.stop()

st.session_state.condition = condition


# -------------------------
# Session state
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "premise" not in st.session_state:
    st.session_state.premise = None

if "persona" not in st.session_state:
    st.session_state.persona = None

if "actor" not in st.session_state:
    st.session_state.actor = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "scenario_ready" not in st.session_state:
    st.session_state.scenario_ready = False



if st.session_state.condition == "A":
    orchestrator = BaselineOrchestrator(api_key=st.secrets["OPENAI_API_KEY"])
else:
    orchestrator = Orchestrator(api_key=st.secrets["OPENAI_API_KEY"])


# -------------------------
# Generate scenario
# -------------------------

if st.button("Generate Scenario"):

    premise = orchestrator.generate_prompt()

    st.session_state.premise = premise
    st.session_state.scenario_ready = True
    st.session_state.actor = None
    st.session_state.chat = []


# -------------------------
# Always show scenario
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

        persona = orchestrator.generate_persona(st.session_state.premise)

        st.session_state.persona = persona

        if st.session_state.condition == "baseline":

            actor = BaselineActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=st.session_state.premise
            )

        else:

            persona = orchestrator.generate_persona(st.session_state.premise)
            st.session_state.persona = persona

            actor = ActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=st.session_state.premise,
                persona=persona
            )

        st.session_state.actor = actor

        opening = actor.opening_line()

        st.session_state.chat = [
            {"role": "assistant", "content": opening}
        ]
        # save opening line as first turn
        save_message(
            st.session_state.session_id,
            st.session_state.premise,
            "assistant",
            opening,
            1,
            st.session_state.condition
        )
        st.rerun()

    if col2.button("Regenerate Scenario"):

        premise = orchestrator.generate_prompt()
        st.session_state.premise = premise
        st.rerun()


# -------------------------
# Chat Interface
# -------------------------

if st.session_state.actor:

    st.subheader("Scene")

    # display existing chat
    for msg in st.session_state.chat:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])


user_input = st.chat_input("Your response")

if user_input:

    # show user message
    st.session_state.chat.append({
        "role": "user",
        "content": user_input
    })

    st.chat_message("user").write(user_input)

    turn_index = len(st.session_state.chat)

    # save user message
    save_message(
        st.session_state.session_id,
        st.session_state.premise,
        "user",
        user_input,
        turn_index,
        st.session_state.condition
    )

    with st.spinner("Actor is thinking..."):
        reply = st.session_state.actor.respond(user_input)

    # show AI message
    st.session_state.chat.append({
        "role": "assistant",
        "content": reply
    })

    st.chat_message("assistant").write(reply)

    turn_index = len(st.session_state.chat)
    # SAVE AI MESSAGE
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