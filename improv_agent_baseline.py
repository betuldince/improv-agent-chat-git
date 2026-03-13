from openai import OpenAI
import streamlit as st

MODEL = "gpt-5.1"

 
class BaselineOrchestrator:

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_prompt(self):

        SYSTEM_PROMPT = """
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

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.2,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Generate one improv scenario."}
            ],
        )

        return response.output_text.strip()
    
class BaselineActorLLM:

    def __init__(self, api_key: str, premise: str):
        self.client = OpenAI(api_key=api_key)
        self.premise = premise
        self.history = []

    def build_system_prompt(self):

        return f"""
You are an improv actor.

Scene premise:
{self.premise}

Rules:
- Stay in character.
- Respond naturally.
- Do not narrate actions.
- Only speak as the character.
- Keep response (1-3 sentences) 

This is a two-person scene between you and the user.
"""

    def respond(self, user_message):

        self.history.append({
            "role": "user",
            "content": user_message
        })

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.0,
            input=[
                {"role": "system", "content": self.build_system_prompt()},
                *self.history
            ],
        )

        reply = response.output_text.strip()

        self.history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def opening_line(self):

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.0,
            input=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": "Start the scene."}
            ],
        )

        line = response.output_text.strip()

        self.history.append({
            "role": "assistant",
            "content": line
        })

        return line
    

if __name__ == "__main__":

    orchestrator = BaselineOrchestrator(api_key=st.secrets["OPENAI_API_KEY"])

    print("\nBaseline Improv System\n")

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

            actor = BaselineActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=premise
            )

            print("\n--- Scene Begins ---\n")

            opening = actor.opening_line()
            print("Actor:", opening)

            while True:

                user_input = input("\nYou: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Scene ended.\n")
                    break

                reply = actor.respond(user_input)

                print("\nActor:", reply)