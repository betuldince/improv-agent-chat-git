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
    
    def load_motivation_examples(self):
        file_path = Path(__file__).parent / "internal_motivation_examples_v2.txt"
        with open(file_path, "r") as f:
            return f.read()    

    def generate_persona(self, premise: str):

        SYSTEM_PROMPT = """
    You generate personas for improv roleplay characters.

    Use the style of the example motivations provided.

    Output STRICT JSON ONLY with exactly these keys:
    {
    "age": <integer 18-70>,
    "race": <one string: "Asian"|"Black"|"White"|"Latina"|"Middle Eastern"|"Native American"|"Mixed"|"Other">,
    "sex": <"Female"|"Male"|"Non-binary">,
    "traits": [<3 to 6 adjectives, lower-case>],
    "internal_motivation": <1–4 sentences explaining their personal motivation>
    }
    """

        motivation_examples = self.load_motivation_examples()

        USER_PROMPT = f"""
    Create a persona for the improv partner in this scenario:

    {premise}

    Good examples of internal motivations:

    {motivation_examples}

    Generate a new internal motivation similar in style to the examples above.
    personal, specific, grounded in past behavior, likely to influence how the character behaves  

    Constraints:
    - JSON only. No markdown. No extra keys.
    """

        response = self.client.responses.create(
            model=MODEL,
            temperature=1.2,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )

        persona_json = response.output_text.strip()

        persona = json.loads(persona_json)

        with open("persona.json", "w") as f:
            json.dump(persona, f, indent=2)

        return persona



class ActorLLM:

    def __init__(self, api_key: str, premise: str, persona: dict):
        self.client = OpenAI(api_key=api_key)

        self.premise = premise
        self.persona = persona
        self.history = []

        self.opening_lines = self.load_opening_lines()


    def load_opening_lines(self):
        file_path = Path(__file__).parent / "opening_lines.txt"
        with open(file_path, "r") as f:
            return f.read()

    def build_system_prompt(self):

        persona_text = json.dumps(self.persona, indent=2)

        return f"""
        You are an improv actor.

        You must stay fully in character.

        Your role:
        {self.premise}

        CHARACTER PERSONA
        {persona_text}

        ACTING RULES

        - Respond as the character.
        - Avoid asking questions- unless you’re also adding information.
        - Play in the present and use the moment.
        - Be specific and provide colorful details.
        - This is strictly a 2-person scene
        - Do NOT introduce or request any third person
        - Never narrate actions.
        - Never explain the improv rules.
        - Speak naturally like a real person.
        - Let your internal motivation influence your behavior.
        - Keep responses short (1-3 sentences).
        - Be conversational and reactive.

        Good opening lines examples:
        {self.opening_lines}
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
            temperature=1.2,
            input=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": "Start the conversation."}
            ],
        )

        line = response.output_text.strip()

        self.history.append({
            "role": "assistant",
            "content": line
        })

        return line

if __name__ == "__main__":

    orchestrator = Orchestrator(api_key=st.secrets["OPENAI_API_KEY"])

    print("\nOrchestrator Generates Prompt\n")

    while True:

        premise = orchestrator.generate_prompt()

        print("\nGenerated premise:\n")
        print(premise)

        accept = input("\nAccept this premise? (y/n/q): ").strip().lower()

        if accept == "q":
            print("Bye!")
            break

        if accept == "y":

            persona = orchestrator.generate_persona(premise)

            print("\nPersona generated:\n")
            print(json.dumps(persona, indent=2))

            actor = ActorLLM(
                api_key=st.secrets["OPENAI_API_KEY"],
                premise=premise,
                persona=persona
            )

            print("\n--- Scene Begins ---\n")

            opening = actor.opening_line()
            print("Actor:", opening)

            while True:

                user_input = input("\nYou: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Scene ended.")
                    break

                reply = actor.respond(user_input)

                print("\nActor:", reply)

        if accept == "n":
            print("\nRegenerating premise...\n")