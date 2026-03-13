from openai import OpenAI
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import streamlit as st

MODEL = "gpt-4o-mini"

# =========================================================
# Types and constants
# =========================================================

MoveType = Literal["hold", "explore_previous", "counteraction", "new_information"]
PhaseType = Literal["relationship_grounding", "buffer", "premise_pivot", "premise_exploration"]

TACTICS_BY_MOVE = {
    "hold": [
        "to observe",
        "to answer plainly",
        "to stay guarded"
    ],
    "explore_previous": [
        "to explain",
        "to clarify",
        "to justify",
        "to elaborate",
        "to reinterpret",
        "to confess",
        "to minimize"
    ],
    "counteraction": [
        "to blame",
        "to pressure",
        "to threaten",
        "to challenge",
        "to corner",
        "to accuse",
        "to deflect"
    ],
    "new_information": [
        "to reveal",
        "to admit",
        "to disclose",
        "to confess"
    ]
}


# =========================================================
# Structured scene state
# =========================================================

@dataclass
class InfoThread:
    id: int
    content: str
    source: str                      # "generated" | "user" | "premise"
    related_to: List[int] = field(default_factory=list)
    turns_explored: int = 0
    max_explore_turns: int = 5
    is_active: bool = True
    fully_explored: bool = False


@dataclass
class DirectorState:
    initial_premise: str
    core_objective: str
    impelling_action: str
    premise_facts: List[str]
    latent_information_pool: List[str]

    phase: PhaseType = "relationship_grounding"
    actor_turn_count: int = 0

    info_threads: List[InfoThread] = field(default_factory=list)
    active_thread_id: Optional[int] = None

    last_move: MoveType = "hold"
    last_tactic: str = ""

    min_explore_before_new_info: int = 4
    max_explore_before_forced_new_info: int = 6


# =========================================================
# Orchestrator
# =========================================================

class Orchestrator:

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def build_user_message(self):

        return """
Good examples of improv prompts:

Your improv partner is playing someone who is breaking up with you.
Your improv partner is playing your college roommate meeting you for the first time.
Your improv partner is playing your parent giving you disappointing news.
Your improv partner is playing your doctor during a routine checkup at a doctor’s office.
Your improv partner is playing someone who is making small talk with you and secretly has a crush on you.
Your improv partner is playing someone who tries to return a shirt without a receipt.
Your improv partner is playing a suspect being interrogated in a serial killer case.
Your improv partner is playing an executive negotiating merging his company with yours
Your improv partner is playing the president asking you for a favor.
Your improv partner is playing a member of royalty who is trying to impress you.
Your improv partner is playing a judge questioning you while you testify in a high-profile court case.
Your improv partner is playing your friend who is confessing that they have been conning you.
Your improv partner is playing a reporter interviewing you after you lost a professional sports championship.
Your improv partner is playing your archenemy explaining you their evil plan.
Your improv partner is playing the ghost of a loved one speaking to you.
Your improv partner is playing a scientist telling you, the president, that they have discovered the secret to eternal life.
Your improv partner is playing God, who has come to Earth to give you an interview.
Your improv partner is playing another medieval knight planning with you how to slay a dragon.
Your improv partner is playing an exorcist trying to rid you of the demon that has possessed you.
Your improv partner is playing a jeweler trying to sell you a magical but dangerous item.
Your improv partner is playing the person whose body you have mysteriously swapped with.
Your improv partner is playing an alien who wants to visit Earth for tourism.

Generate ONE new improv prompt similar to these.
"""

    def generate_prompt(self):
        SYSTEM_PROMPT = """
        You are an improv prompt generator.

        You will be given several example prompts.
        Your task is to generate ONE new prompt that looks like it could have been written
        by the same person who wrote the examples.

        Follow the same format, tone, and level of creativity.
        Prompt should be something that can happen in real life.
        Output only the prompt text and nothing else.
        """
        response = self.client.responses.create(
            model=MODEL,
            temperature=1.5,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_user_message()},
            ],
        )

        return response.output_text.strip()
    

    def generate_persona(self, premise: str):
        system_prompt = """
You generate personas for improv roleplay characters.

Output STRICT JSON ONLY with exactly these keys:
{
  "age": <integer 18-70>,
  "race": <one string: "Asian"|"Black"|"White"|"Latina"|"Middle Eastern"|"Native American"|"Mixed"|"Other">,
  "sex": <"Female"|"Male"|"Non-binary">,
  "traits": [<3 to 6 adjectives, lower-case>],
  "internal_motivation": <string>
}
"""

        user_prompt = f"""
Create a persona for the improv partner in this scenario:

SCENARIO:
{premise}

Prioritize negative or difficult personality traits when appropriate.

Constraints:
- Traits should influence the internal motivation
- internal_motivation should be 2-4 sentences
- make it specific and scene-driving
- directly relevant to the premise
- JSON only
- no markdown
- no extra keys
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.1,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        persona = json.loads(response.output_text.strip())

        with open("persona.json", "w", encoding="utf-8") as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)

        return persona

    def generate_scene_package(self, premise: str, persona: dict):
        """
        Generate structured scene control variables ONCE at the beginning.
        This is the only part where the LLM helps the director structurally.
        """
        system_prompt = """
You are generating structured scene-control data for an improvisation system.

Return STRICT JSON ONLY with exactly these keys:
{
  "core_objective": "...",
  "impelling_action": "...",
  "premise_facts": ["...", "...", "..."],
  "latent_information_pool": ["...", "...", "...", "..."]
}

Requirements:
- core_objective = what the improv partner in the premsie wants from the user in the scene
- impelling_action = what pushes the improv partner in the premsie to act now, usually starting with "to ..."
- premise_facts = 2-5 concise facts already inherent in the premise
- latent_information_pool = 4-8 concise facts/reveals/pressures that could emerge later
- latent information should support long conversation
- do not resolve the scene too quickly
- each latent item should be short and concrete
- each latent item should connect to the original premise
"""

        user_prompt = f"""
SCENE PREMISE:
{premise}

PERSONA:
{json.dumps(persona, indent=2)}

Generate structured scene package.
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=0.8,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return json.loads(response.output_text.strip())


# =========================================================
# Director controller (mostly deterministic)
# =========================================================

class DirectorController:
    def __init__(self):
        pass

    def update_phase(self, state: DirectorState):
        t = state.actor_turn_count

        if t == 0:
            state.phase = "relationship_grounding"
        elif t in [1, 2]:
            state.phase = "buffer"
        elif t == 3:
            state.phase = "premise_pivot"
        else:
            state.phase = "premise_exploration"

    def get_active_thread(self, state: DirectorState) -> Optional[InfoThread]:
        if state.active_thread_id is None:
            return None
        for thread in state.info_threads:
            if thread.id == state.active_thread_id:
                return thread
        return None

    def choose_move(self, state: DirectorState) -> MoveType:
        if state.phase == "relationship_grounding":
            return "hold"

        if state.phase == "buffer":
            return "hold"

        if state.phase == "premise_pivot":
            if state.active_thread_id is None:
                return "new_information"
            return "explore_previous"

        # premise_exploration
        active = self.get_active_thread(state)

        if active is None:
            return "new_information"

        if active.turns_explored < state.min_explore_before_new_info:
            if active.turns_explored in [1, 3]:
                return "counteraction"
            return "explore_previous"

        if active.turns_explored >= state.max_explore_before_forced_new_info:
            return "new_information"

        return random.choice(["explore_previous", "counteraction"])

    def choose_tactic(self, move: MoveType, last_tactic: str = "") -> str:
        tactics = TACTICS_BY_MOVE[move][:]
        if last_tactic in tactics and len(tactics) > 1:
            tactics.remove(last_tactic)
        return random.choice(tactics)

    def add_new_information_thread(self, state: DirectorState) -> str:
        if state.latent_information_pool:
            info = state.latent_information_pool.pop(0)
        else:
            info = "There is another consequence here the actor has not admitted yet."

        related = [state.active_thread_id] if state.active_thread_id is not None else []

        thread = InfoThread(
            id=len(state.info_threads),
            content=info,
            source="generated",
            related_to=related,
            max_explore_turns=5
        )

        state.info_threads.append(thread)
        state.active_thread_id = thread.id
        return info

    def add_user_information_thread(self, state: DirectorState, info: str):
        related = [state.active_thread_id] if state.active_thread_id is not None else []

        thread = InfoThread(
            id=len(state.info_threads),
            content=info,
            source="user",
            related_to=related,
            max_explore_turns=5
        )

        state.info_threads.append(thread)
        state.active_thread_id = thread.id

    def mark_progress(self, state: DirectorState, move: MoveType):
        active = self.get_active_thread(state)
        if active is None:
            return

        if move in ["hold", "explore_previous", "counteraction"]:
            active.turns_explored += 1
            if active.turns_explored >= active.max_explore_turns:
                active.fully_explored = True

    def render_instruction(self, phase, move, tactic, focus_thread, objective):
        if phase == "relationship_grounding":
            return "Establish ordinary interaction, relationship, and setting. Do not raise the main issue yet."

        if phase == "buffer":
            return "Continue naturally. Hint that something matters, but do not fully introduce the real issue yet."

        if phase == "premise_pivot":
            return "Turn clearly toward the real issue. Stop circling around it."

        if move == "new_information":
            return f"Reveal one new concrete fact using the tactic {tactic}. It should escalate the scene naturally."

        if move == "explore_previous":
            return f"Stay on the current thread and deepen it using the tactic {tactic}."

        if move == "counteraction":
            return f"Use the current thread to challenge, pressure, or complicate the user's position using the tactic {tactic}."

        return f"Pursue the scene objective: {objective}"

    def build_guidance(self, state: DirectorState) -> dict:
        self.update_phase(state)

        move = self.choose_move(state)
        tactic = self.choose_tactic(move, state.last_tactic)

        new_information = ""
        focus_thread = ""
        thread_source = ""
        pressure_note = ""

        active = self.get_active_thread(state)

        if move == "new_information":
            new_information = self.add_new_information_thread(state)
            focus_thread = new_information
            thread_source = "generated"
            pressure_note = f"Introduce this information naturally: {new_information}"

        elif active:
            focus_thread = active.content
            thread_source = active.source

            if move == "explore_previous":
                pressure_note = f"Deepen this thread: {focus_thread}"

            elif move == "counteraction":
                pressure_note = f"Use this thread to challenge the user: {focus_thread}"

        instruction = self.render_instruction(
            phase=state.phase,
            move=move,
            tactic=tactic,
            focus_thread=focus_thread,
            objective=state.core_objective
        )

        state.last_move = move
        state.last_tactic = tactic

        return {
            "phase": state.phase,
            "move_type": move,
            "tactic": tactic,
            "focus_thread": focus_thread,
            "thread_source": thread_source,
            "new_information": new_information,
            "pressure_note": pressure_note,
            "objective_reminder": state.core_objective,
            "impelling_action": state.impelling_action,
            "instruction": instruction,
            "must_do": self.build_must_do(move, focus_thread, new_information),
            "must_not_do": self.build_must_not_do(state.phase, move)
        }

    def build_must_do(self, move: MoveType, focus_thread: str, new_information: str) -> str:
        if move == "new_information":
            return f"Introduce this new fact: {new_information}"
        if move == "explore_previous":
            return f"Keep the scene on this thread: {focus_thread}"
        if move == "counteraction":
            return f"Pressure the user around this thread: {focus_thread}"
        return "Keep the interaction grounded and role-consistent."

    def build_must_not_do(self, phase: PhaseType, move: MoveType) -> str:
        if phase in ["relationship_grounding", "buffer"]:
            return "Do not fully resolve or fully reveal the core issue."
        if move == "new_information":
            return "Do not introduce more than one new fact."
        return "Do not switch to a different issue or resolve the scene too quickly."


# =========================================================
# Optional helper: detect if user introduced a new thread
# =========================================================

class UserInfoDetector:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def detect(self, premise: str, history: list, latest_user_message: str) -> dict:
        system_prompt = """
You detect whether the user's latest message introduced a new important piece of information
that the scene should now explore.

Return STRICT JSON ONLY:
{
  "contains_new_relevant_information": true,
  "user_information": "..."
}

Rules:
- Return true only if the user added a meaningful fact, reason, confession, stake, explanation, or secret
- user_information should be short and specific
- If nothing important was added, return:
{
  "contains_new_relevant_information": false,
  "user_information": ""
}
"""

        user_prompt = f"""
SCENE PREMISE:
{premise}

RECENT HISTORY:
{json.dumps(history[-8:], indent=2)}

LATEST USER MESSAGE:
{latest_user_message}
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=0.2,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        text = response.output_text.strip()
        try:
            return json.loads(text)
        except Exception:
            return {
                "contains_new_relevant_information": False,
                "user_information": ""
            }


# =========================================================
# Actor
# =========================================================

class ActorLLM:
    def __init__(
        self,
        api_key: str,
        premise: str,
        persona: dict,
        director_state: DirectorState,
        director_controller: DirectorController,
        use_user_info_detector: bool = True
    ):
        self.client = OpenAI(api_key=api_key)
        self.premise = premise
        self.persona = persona
        self.history = []
        self.actor_turn_count = 0

        self.director_state = director_state
        self.director_controller = director_controller
        self.user_info_detector = UserInfoDetector(api_key) if use_user_info_detector else None

    def build_system_prompt(self, guidance: dict):
        return f"""
You are an improv actor.

Stay fully in character.

SCENE PREMISE
{self.premise}

CURRENT PERSONA
{json.dumps(self.persona, indent=2)}

DIRECTOR CONTROL
{json.dumps(guidance, indent=2)}

RULES
- Respond as the character.
- Keep the interaction natural and grounded.
- This is strictly a 2-person scene.
- Do not introduce new third people into the live conversation unless already implied by the scene.
- Keep responses short (1-3 sentences).
- Follow the director control closely.
- Do not mention the director or the guidance.
- Do not narrate your actions.
- Let your internal motivation shape your behavior.
- If move_type is "new_information", organically introduce the exact new_information.
- If move_type is "explore_previous", deepen the focus_thread.
- If move_type is "counteraction", actively challenge, resist, pressure, or complicate around the focus_thread.
- Use the tactic clearly in how you respond.
- Do not resolve the whole scene too quickly.
"""

    def maybe_add_user_thread(self, user_message: str):
        if self.user_info_detector is None:
            return

        # Only check during later phases
        if self.director_state.phase not in ["premise_pivot", "premise_exploration"]:
            return

        result = self.user_info_detector.detect(
            premise=self.premise,
            history=self.history,
            latest_user_message=user_message
        )

        if result.get("contains_new_relevant_information"):
            new_info = result.get("user_information", "").strip()
            if new_info:
                self.director_controller.add_user_information_thread(
                    self.director_state,
                    new_info
                )

    def respond(self, user_message: str):
        self.history.append({
            "role": "user",
            "content": user_message
        })

        self.director_state.actor_turn_count = self.actor_turn_count
        self.director_controller.update_phase(self.director_state)

        # If user introduced an important new fact, capture it as a new thread
        self.maybe_add_user_thread(user_message)

        guidance = self.director_controller.build_guidance(self.director_state)

        print(f"\n--- Director Guidance ({self.director_state.phase}) ---\n")
        print(json.dumps(guidance, indent=2, ensure_ascii=False))

        print("\n--- Director State ---\n")
        print(json.dumps(self.serialize_director_state(), indent=2, ensure_ascii=False))
        print("\n---------------------------------------\n")

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.0,
            input=[
                {"role": "system", "content": self.build_system_prompt(guidance)},
                *self.history
            ],
        )

        reply = response.output_text.strip()

        self.history.append({
            "role": "assistant",
            "content": reply
        })

        self.director_controller.mark_progress(
            self.director_state,
            guidance["move_type"]
        )

        self.actor_turn_count += 1
        return reply

    def serialize_director_state(self):
        return {
            "initial_premise": self.director_state.initial_premise,
            "core_objective": self.director_state.core_objective,
            "impelling_action": self.director_state.impelling_action,
            "premise_facts": self.director_state.premise_facts,
            "latent_information_pool": self.director_state.latent_information_pool,
            "phase": self.director_state.phase,
            "actor_turn_count": self.director_state.actor_turn_count,
            "active_thread_id": self.director_state.active_thread_id,
            "last_move": self.director_state.last_move,
            "last_tactic": self.director_state.last_tactic,
            "info_threads": [
                {
                    "id": t.id,
                    "content": t.content,
                    "source": t.source,
                    "related_to": t.related_to,
                    "turns_explored": t.turns_explored,
                    "max_explore_turns": t.max_explore_turns,
                    "is_active": t.is_active,
                    "fully_explored": t.fully_explored
                }
                for t in self.director_state.info_threads
            ]
        }


# =========================================================
# First line
# =========================================================

def generate_first_line(
    client: OpenAI,
    premise: str,
    persona: dict,
):
    system_prompt = f"""
You are writing the first line of an improv scene.

SCENE PREMISE
{premise}

CURRENT PERSONA
{json.dumps(persona, indent=2)}

Rules:
- Write ONLY the actor's first line of dialogue
- The line should sound natural and role-consistent
- It should reflect the relationship between the characters
- Do NOT introduce the core premise yet
- 1-2 sentences maximum
"""

    response = client.responses.create(
        model=MODEL,
        temperature=1.0,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write the first line of the scene."}
        ],
    )

    return response.output_text.strip()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    api_key = st.secrets["OPENAI_API_KEY"]

    orchestrator = Orchestrator(api_key=api_key)
    director_controller = DirectorController()

    print("\nOrchestrator Generates Prompt\n")

    while True:
        premise  = orchestrator.generate_prompt()

        print("\nGenerated premise:\n")
        print(premise)
 

        accept = input("\nAccept this premise? (y/n/q): ").strip().lower()

        if accept == "q":
            print("Bye!")
            break

        if accept == "n":
            print("\nRegenerating premise...\n")
            continue

        if accept == "y":
            persona = orchestrator.generate_persona(premise)

            print("\nPersona generated:\n")
            print(json.dumps(persona, indent=2, ensure_ascii=False))

            scene_package = orchestrator.generate_scene_package(premise, persona)

            print("\nScene package generated:\n")
            print(json.dumps(scene_package, indent=2, ensure_ascii=False))

            director_state = DirectorState(
                initial_premise=premise,
                core_objective=scene_package["core_objective"],
                impelling_action=scene_package["impelling_action"],
                premise_facts=scene_package["premise_facts"],
                latent_information_pool=scene_package["latent_information_pool"]
            )

            actor = ActorLLM(
                api_key=api_key,
                premise=premise,
                persona=persona,
                director_state=director_state,
                director_controller=director_controller,
                use_user_info_detector=True
            )

            print("\n--- Scene Begins ---\n")

            opening = generate_first_line(
                client=orchestrator.client,
                premise=premise,
                persona=persona
            )

            actor.history.append({
                "role": "assistant",
                "content": opening
            })
            actor.actor_turn_count = 1

            print("Actor:", opening)

            while True:
                user_input = input("\nYou: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Scene ended.")
                    break

                reply = actor.respond(user_input)
                print("\nActor:", reply)