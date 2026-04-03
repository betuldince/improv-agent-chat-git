from openai import OpenAI
import json
import re
from pathlib import Path
import streamlit as st

MODEL = "gpt-5.1"


class Orchestrator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def build_user_message(self):
        return """
    Good examples of vague but playable improv premises:

    Your improv partner is playing your abusive ex.
    Your improv partner is playing your greedy tenant.
    Your improv partner is playing your pushy parent.
    Your improv partner is playing your manipulative boss.
    Your improv partner is playing your jealous sibling.
    Your improv partner is playing your passive-aggressive roommate.
    Your improv partner is playing your controlling older brother.
    Your improv partner is playing your suspicious spouse.
    Your improv partner is playing your entitled customer.
    Your improv partner is playing your overbearing coach.
    Your improv partner is playing your emotionally needy friend.
    Your improv partner is playing your condescending professor.
    Your improv partner is playing your possessive partner.
    Your improv partner is playing your exploitative landlord.
    Your improv partner is playing your attention-seeking coworker.

    Generate ONE new improv prompt in the same style.

    The prompt should be:
    - short
    - relationship-based
    - behavior/archetype-based
    - vague enough to allow many directions in the conversation
    - strong enough to create tension and rich improvisation

    Output only the prompt text and nothing else.
    """

    def generate_prompt(self):
        SYSTEM_PROMPT = """
        You generate vague but highly playable improv scene premises for two-person roleplay.

        Your goal is to create premises that support long, rich improvisation.

        A strong premise should:
        - define a relationship between the user and the improv partner
        - define a behavioral archetype or pressure pattern
        - remain vague enough that the conversation can discover multiple angles over time
        - create natural tension without locking the scene into one narrow objective

        Avoid premises that depend on one immediate binary objective such as:
        - confessing one thing
        - asking one favor
        - convincing the user of one specific action

        Prefer premises built around unstable relationship dynamics, such as:
        - manipulative
        - jealous
        - controlling
        - abusive
        - suspicious
        - greedy
        - passive-aggressive
        - overbearing
        - entitled
        - emotionally needy

        Output ONLY one premise in this format:
        Your improv partner is playing your [relationship] who is [behavioral archetype].
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

half negative half positive personality traits  

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
- JSON only
- no markdown
- no extra keys
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

    def _parse_json(self, text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            raise

    def generate_guidance(self, premise: str, persona: dict, phase: str, history: list):
        system_prompt = """
You are an improv director guiding an AI actor.

Your job is to generate phase-specific conversation guidance for the actor's NEXT reply.
The primary objective is to make the scene richer, less repetitive, and more playable.

The scene phases are:

1. relationship_grounding
Goal:
- begin with ordinary conversation starter
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

4. premise_exploration
Goal:
- this is the main improvisation phase
- critically analyze the dialogue so far
- detect whether the scene is becoming stationary, repetitive, circular, or too easy
- if it is  stationary, repetitive, circular, or too easy, choose ONE strong move:
  a) introduce new direction for the conversation related to the premise
  b) explore a previously mentioned thread more deeply by adding new informaiton
  c) challenge the user's response through counteraction(if the last user input was not very positive)
  d) shift, sharpen, or complicate the actor's internal motivation
- keep the scene alive through new pressure, contradiction, consequence, reveal, or changed intention(do not use counter action if the user is poistive)
- do not end the scene too quickly

Return STRICT JSON ONLY with exactly this structure:
{
  "phase": "relationship_grounding | buffer | premise_pivot | premise_exploration",
  "is_stationary": true,
  "move_type": "none | new_information | explore_previous | counteraction ",
  "new_information": "",
  "counteraction": "",
  "guidance": ""
}

Rules:
- guidance is concise
- if the User response is positive lean towards giving positive guidance eventhough the traits are negative
- Guidance must be specific to the scenario, role, and current dialogue, and CHARACTER TRAIT
- Do not narrate actions
- Do not write dialogue
- Do not be generic
- Keep guidance short but actionable
- In premise_exploration, be critical about whether the conversation is actually moving
- Only update internal motivation very slowly based on the traits and dialogue history
- If the scene is not stationary, keep move_type as "none" and give guidance that deepens the current thread
"""

        user_prompt = f"""
SCENE PREMISE:
{premise}

CURRENT PERSONA:
{json.dumps(persona, indent=2)}

CURRENT PHASE:
{phase}

DIALOGUE HISTORY:
{json.dumps(history, indent=2)}

Generate acting guidance for the actor's next reply.
Keep it short(1-3 sentences)
Prevent the scene from coming to an end too quickly.
If phase is premise_exploration, critically analyze whether the scene has become stationary.
"""

        response = self.client.responses.create(
            model=MODEL,
            temperature=0.7,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._parse_json(response.output_text.strip())


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
        elif self.actor_turn_count == 3:
            self.phase = "premise_pivot"
        else:
            self.phase = "premise_exploration"

    def build_system_prompt(self, director_guidance: dict):
        persona_text = json.dumps(self.persona, indent=2)
        guidance_text = json.dumps(director_guidance, indent=2)

        return f"""
You are an improv actor.

Stay fully in character.

SCENE PREMISE
{self.premise}

CURRENT PERSONA
{persona_text}

DIRECTOR GUIDANCE
{guidance_text}

ACTING RULES
- Respond as the character and just write a dialogue don't use special characters.
- Keep the interaction natural and grounded.
- This is strictly a 2-person scene.
- Do not introduce new third people into the live conversation.
- Keep responses as short as possible (strictly maximum 3 sentences).
- Follow the director guidance but make it seem natural in relation to conversaation history.
- Do not mention the director or the guidance.
- Do not narrate your actions.
- Let your internal motivation shape your behavior subtly.
- In early conversation, do not rush into the premise unless the guidance tells you to.
- If the director provides new_information, organically bring that information into the conversation.
- If the director provides counteraction, let your reply actively challenge, resist, complicate, or pressure the user using that counteraction.
- In premise_exploration, avoid repeating the same concern in the same words. Move the scene forward.
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

        if director_guidance.get("updated_internal_motivation"):
            self.persona["internal_motivation"] = director_guidance["updated_internal_motivation"]

        #print(f"\n--- Director Guidance ({self.phase}) ---\n")
        #print(json.dumps(director_guidance, indent=2))
        #print("\n---------------------------------------\n")

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

    #print("\n--- Director Guidance (relationship_grounding) ---\n")
    #print(json.dumps(opening_guidance, indent=2))
    #print("\n-------------------------------------------------\n")

    system_prompt = f"""
You are writing the first line of an improv scene.

SCENE PREMISE
{premise}

CURRENT PERSONA
{json.dumps(persona, indent=2)}

DIRECTOR GUIDANCE
{json.dumps(opening_guidance, indent=2)}

Rules:
- Write ONLY the actor's first line of dialogue
- The line should sound natural and role-consistent
- Do NOT introduce the core premise yet unless the guidance clearly calls for it
- 1 sentence maximum
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