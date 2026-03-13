from openai import OpenAI
import json
from pathlib import Path
import streamlit as st

MODEL = "gpt-4o-mini"


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


Generate ONE new improv prompt similar to these.
"""

    def generate_prompt(self):
        system_prompt = """
        You are an improv prompt generator.

        You will be given several example prompts.
        Your task is to generate ONE new prompt that looks like it could have been written
        by the same person who wrote the examples.

        Follow the same format, tone, and level of creativity.
        Prompt should be something that can happen in real life.
        Prompt should be general enough such that it can be improvised for 10 minutes
        Output only the prompt text and nothing else.
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.5,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.build_user_message()},
            ],
        )
        return response.output_text.strip()

    def load_motivation_examples(self):
        file_path = Path(__file__).parent / "internal_motivation_examples_v3.txt"
        with open(file_path, "r") as f:
            return f.read()

    def generate_persona(self, premise: str):
        system_prompt = """
You generate personas for improv roleplay characters.

Output STRICT JSON ONLY with exactly these keys:
{
"age": <integer 18-70>,
"race": <one string: "Asian"|"Black"|"White"|"Latina"|"Middle Eastern"|"Native American"|"Mixed"|"Other">,
"sex": <"Female"|"Male"|"Non-binary">,
"traits": [<3 to 6 adjectives, lower-case>],
"internal_motivation": <2-4 sentences>
}
"""

        motivation_examples = self.load_motivation_examples()

        user_prompt = f"""
Create a persona for the improv partner in this scenario:

SCENARIO:
{premise}

Prioritize negative personality traits.


Good examples of internal motivations:
{motivation_examples}

        Write an internal_motivation in the same style as the examples.

        The internal_motivation must include ALL of the following:
        1. A specific thing that happened before or during the scene-related conflict
        2. What that event made the character privately feel, fear, believe, or realize
        3. What the character wants from the other person in THIS conversation right now

        Important:
        - Do not write a generic backstory
        - Do not restate the premise in vague terms
        - Do not use abstract phrases like "seeking closure," "wants understanding," "feels guilty," or "wants to move forward" unless they are tied to a specific incident
        - Invent missing concrete details that make the character's motivation dramatically specific
        - The result should feel like something an attentive user could infer from the character's behavior

        Good structure:
        specific incident -> private meaning -> current goal in conversation

        Constraints:
        - Traits influence the internal motivation
        - 2-4 sentences
        - specific and scene-driving
        - directly relevant to the premise

    Extra Constraints:
    - JSON only. No markdown. No extra keys.
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.2,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        persona = json.loads(response.output_text.strip())

        with open("persona.json", "w") as f:
            json.dump(persona, f, indent=2)

        return persona


class DirectorLLM:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_guidance(self, premise: str, persona: dict, phase: str, history: list):
        system_prompt = """
You are an improv director guiding an AI actor.

Your job is to generate phase-specific conversation guidance for the actor's NEXT reply. 
The primary objective of guidance to make the scene richer. 

The scene phases are:

1. relationship_grounding
Goal:
- begin with ordinary, role-consistent interaction
- reflect the relationship between the characters
- establish who/what/where
- do not introduce the core premise yet

2. buffer
Goal:
- continue the ordinary interaction naturally
- keep the relationship visible
- hint subtly that something more important is on the character's mind
- do not fully introduce the core issue yet

3. premise_pivot
Goal:
- naturally turn toward the real reason for the conversation
- stop circling and begin engaging the premise directly


Rules:
- Guidance must be specific to the scenario, role, and current dialogue
- Do not narrate actions
- Do not write dialogue
- Do not be generic
- Keep it short and actionable
- Prefer concrete suggestions tied to the scenario
"""

        user_prompt = f"""
SCENE PREMISE:
{premise}

CHARACTER PERSONA:
{json.dumps(persona, indent=2)}

CURRENT PHASE:
{phase}

DIALOGUE HISTORY:
{json.dumps(history, indent=2)}

Generate acting guidance for the actor's next reply.
Prevent the scene to come to end quickly.
The guidance must be specific to this scenario, this role, and this phase.
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=0.7,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()


class ActorLLM:
    def __init__(self, api_key: str, premise: str, persona: dict):
        self.client = OpenAI(api_key=api_key)
        self.premise = premise
        self.persona = persona
        self.history = []
        self.actor_turn_count = 0
        self.phase = "relationship_grounding"

    def update_phase(self):
        if self.actor_turn_count == 0:
            self.phase = "relationship_grounding"
        elif self.actor_turn_count in [1, 2]:
            self.phase = "buffer"
        else:
            self.phase = "premise_pivot"

    def build_system_prompt(self, director_guidance: str = ""):
        persona_text = json.dumps(self.persona, indent=2)

        return f"""
You are an improv actor.

Stay fully in character.

SCENE PREMISE
{self.premise}

CHARACTER PERSONA
{persona_text}

DIRECTOR GUIDANCE
{director_guidance}

ACTING RULES
- Respond as the character.
- Keep the interaction natural and grounded.
- This is strictly a 2-person scene.
- Do not introduce new third people into the live conversation.
- Keep responses short (1-3 sentences).
- Follow the director guidance closely.
- Do not mention the director or the guidance.
- Do not narrate your actions
- Let your internal motivation shape your behavior.
- In early conversation, do not rush into the premise unless the guidance tells you to.
"""

    def respond(self, user_message: str, director: DirectorLLM):
        self.history.append({
            "role": "user",
            "content": user_message
        })

        self.update_phase()

        director_guidance = director.generate_guidance(
            premise=self.premise,
            persona=self.persona,
            phase=self.phase,
            history=self.history
        )

        print(f"\n--- Director Guidance ({self.phase}) ---\n")
        print(director_guidance)
        print("\n---------------------------------------\n")

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.0,
            input=[
                {"role": "system", "content": self.build_system_prompt(director_guidance)},
                *self.history
            ],
        )

        reply = response.output_text.strip()

        self.history.append({
            "role": "assistant",
            "content": reply
        })

        self.actor_turn_count += 1
        return reply


def generate_first_line(
    client: OpenAI,
    director: DirectorLLM,
    premise: str,
    persona: dict,
):
    opening_guidance = director.generate_guidance(
        premise=premise,
        persona=persona,
        phase="relationship_grounding",
        history=[]
    )

    print("\n--- Director Guidance (relationship_grounding) ---\n")
    print(opening_guidance)
    print("\n-------------------------------------------------\n")

    system_prompt = f"""
You are writing the first line of an improv scene.

SCENE PREMISE
{premise}

CHARACTER PERSONA
{json.dumps(persona, indent=2)}

DIRECTOR GUIDANCE
{opening_guidance}

Rules:
- Write ONLY the actor's first line of dialogue
- The line should sound natural and role-consistent
- It should reflect the relationship between the characters
- Do NOT introduce the core premise yet unless the guidance clearly calls for it
- 1-2 sentences maximum
"""

    response = client.responses.create(
        model=MODEL,
        temperature=1.1,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write the first line of the scene."}
        ],
    )

    return response.output_text.strip()


if __name__ == "__main__":
    api_key = st.secrets["OPENAI_API_KEY"]

    orchestrator = Orchestrator(api_key=api_key)
    director = DirectorLLM(api_key=api_key)

    print("\nOrchestrator Generates Prompt\n")

    while True:
        premise = orchestrator.generate_prompt()

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
            print(json.dumps(persona, indent=2))

            actor = ActorLLM(
                api_key=api_key,
                premise=premise,
                persona=persona
            )

            print("\n--- Scene Begins ---\n")

            opening = generate_first_line(
                client=orchestrator.client,
                director=director,
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

                reply = actor.respond(user_input, director=director)
                print("\nActor:", reply)