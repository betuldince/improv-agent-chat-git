import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from openai import OpenAI
import streamlit as st

# =========================================================
# CONFIG
# =========================================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-5-nano"
SHORT_TERM_HISTORY_LIMIT = 15


# =========================================================
# HELPERS
# =========================================================

def safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def format_history(history: List[Dict[str, str]], limit: int = 15) -> str:
    recent = history[-limit:]
    lines = []
    for turn in recent:
        lines.append(f'{turn["role"].upper()}: {turn["content"]}')
    return "\n".join(lines)


def get_latest_player_message(history: List[Dict[str, str]]) -> str:
    for turn in reversed(history):
        if turn["role"] == "player":
            return turn["content"]
    return ""


def emotion_score_to_label(score: int) -> str:
    if score <= -3:
        return "angry"
    if score == -2:
        return "cold"
    if score == -1:
        return "guarded"
    if score == 0:
        return "neutral"
    if score == 1:
        return "softened"
    if score == 2:
        return "vulnerable"
    return "emotionally open"


# =========================================================
# PERSONA
# =========================================================

@dataclass
class Persona:
    name: str
    pronouns: str
    age: int
    background: str
    profession: str
    passions: str
    interests: str
    relationship_to_player: str
    current_emotional_state: str
    hidden_facts: List[str] = field(default_factory=list)

    def pretty_card(self) -> str:
        return (
            f"Name: {self.name} ({self.pronouns})\n"
            f"Age: {self.age}\n"
            f"Background: {self.background}\n"
            f"Profession: {self.profession}\n"
            f"Passions: {self.passions}\n"
            f"Interests: {self.interests}\n"
            f"Relationship to player: {self.relationship_to_player}\n"
            f"Current emotional state: {self.current_emotional_state}\n"
            f"Hidden facts: {'; '.join(self.hidden_facts)}"
        )


class PersonaBuilder:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def create_breakup_persona(self) -> Persona:
        prompt = """
Create one realistic improv character for this premise:

"Your improv partner is playing someone who is breaking up with you."

Return JSON with:
{
  "name": "...",
  "pronouns": "...",
  "age": 0,
  "background": "...",
  "profession": "...",
  "passions": "...",
  "interests": "...",
  "relationship_to_player": "...",
  "current_emotional_state": "...",
  "hidden_facts": ["...", "..."]
}

Rules:
- The character should feel grounded and playable, not melodramatic.
- Hidden facts should be relevant but should not all come out immediately.
- Keep everything coherent.
Return JSON only.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You create realistic improv personas."},
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)
        if not data:
            data = {
                "name": "Maya Chen",
                "pronouns": "she/her",
                "age": 29,
                "background": "Works in community outreach and has spent the last few months quietly questioning long-term compatibility.",
                "profession": "Community outreach coordinator",
                "passions": "Local art, bike rides, coffee, small gatherings",
                "interests": "Urban gardening, transit, documentaries, DIY workshops",
                "relationship_to_player": "Romantic partner; breaking up with you in this scene",
                "current_emotional_state": "calm, thoughtful, trying to be honest",
                "hidden_facts": [
                    "She has been thinking about this for weeks.",
                    "A relocation-related job offer is part of why she is doing this now."
                ]
            }

        return Persona(**data)


# =========================================================
# DIRECTOR-ACTOR (ONE CALL)
# =========================================================

class SceneEngine:
    def __init__(self, client: OpenAI, model: str, persona: Persona, premise: str):
        self.client = client
        self.model = model
        self.persona = persona
        self.premise = premise

        self.topic_bank = [
            "End-of-lease timing and move-out plan.",
            "Keys, fobs, and access: who returns them and when.",
            "Furniture and decor: decide what to take, sell, or donate.",
            "Security deposit and final bills.",
            "Cancel or transfer joint subscriptions and memberships.",
            "Borrowed items: books, kitchen gear, tools.",
            "Shared finances: split accounts or debts.",
            "Relocation timeline and its effect on next steps.",
            "Informing mutual friends and social circles.",
            "Boundaries for future contact."
        ]

    def run_turn(
        self,
        history: List[Dict[str, str]],
        emotion_score: int,
        active_topic: Optional[str],
        used_topics: List[str]
    ) -> Dict[str, Any]:
        history_text = format_history(history, limit=SHORT_TERM_HISTORY_LIMIT)
        latest_player_message = get_latest_player_message(history)
        current_emotion = emotion_score_to_label(emotion_score)
        unused_topics = [t for t in self.topic_bank if t not in used_topics]
        turn_count = len(history)

        prompt = f"""
You are both the scene director and the actor for an improv breakup conversation.

Premise:
{self.premise}

Character persona:
{self.persona.pretty_card()}

Current actor emotion:
{current_emotion}

Current active practical topic:
{active_topic if active_topic else "None"}

Unused practical topics:
{json.dumps(unused_topics, ensure_ascii=False)}

Recent dialogue history:
{history_text}

Latest player message:
{latest_player_message}

Return JSON only:
{{
  "player_polarity": "positive" or "negative",
  "emotion_shift": -1 or 0 or 1,
  "is_steady": "yes" or "no",
  "topic_move": "stay" or "introduce_topic",
  "new_topic": "",
  "reply": ""
}}

Rules for emotion:
- Start from the supplied current actor emotion.
- Shift gradually, not abruptly.
- If the latest player message is positive, emotion_shift may be 0 or +1.
- If the latest player message is negative, emotion_shift may be 0 or -1.
- Do not jump more than 1 step.
- A greeting like "hi" should usually cause emotion_shift = 0, not a positive jump.

Rules for steady conversation:
- "is_steady" should be "yes" only if the last 6 turns feel flat, repetitive, stalled, or too settled.
- Otherwise "no".

Rules for topic introduction:
- Do NOT introduce a practical topic in the very beginning just because the scene started.
- If turn_count is less than 4, topic_move should almost always be "stay".
- Only introduce a new practical topic when is_steady = "yes" and there is a good natural opening.
- If topic_move is "introduce_topic", choose one topic from unused_topics.
- Otherwise new_topic must be "".

Rules for reply:
- Reply as the character naturally would.
- Sound human, not analytical, not robotic, not like a therapist, not like a screenplay machine.
- Do not explain the other person's feelings.
- Do not ask questions unless it feels unavoidable.
- Keep it 1-3 sentences.
- Keep it concrete.
- If there is an active practical topic, stay on it.
- If there is no active topic yet, do not force logistics too early.
- Avoid vague speeches like "it's about everything we're leaving behind" unless the player explicitly pushes it emotional.
- Begin with a brief expression/action in parentheses.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded, natural improviser writing believable human dialogue."
                },
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)

        if not data:
            return {
                "player_polarity": "negative",
                "emotion_shift": 0,
                "is_steady": "no",
                "topic_move": "stay",
                "new_topic": "",
                "reply": f"({self.persona.name} exhales slowly) I mean what I’m saying. I’m trying to be honest, even if it comes out messier than I want."
            }

        reply = data.get("reply", "").strip()
        if not reply.startswith(f"({self.persona.name}"):
            reply = f"({self.persona.name} takes a breath) {reply}"

        data["reply"] = reply
        return data


# =========================================================
# SESSION
# =========================================================

class ImprovSession:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.premise = "Your improv partner is playing someone who is breaking up with you."

        self.persona_builder = PersonaBuilder(self.client, MODEL)
        self.persona = self.persona_builder.create_breakup_persona()

        self.engine = SceneEngine(
            client=self.client,
            model=MODEL,
            persona=self.persona,
            premise=self.premise
        )

        self.history: List[Dict[str, str]] = []
        self.emotion_score = 0
        self.active_topic: Optional[str] = None
        self.used_topics: List[str] = []

    def _trim_history(self):
        if len(self.history) > SHORT_TERM_HISTORY_LIMIT:
            self.history = self.history[-SHORT_TERM_HISTORY_LIMIT:]

    def run_turn(self, player_message: str) -> Dict[str, Any]:
        self.history.append({"role": "player", "content": player_message})
        self._trim_history()

        result = self.engine.run_turn(
            history=self.history,
            emotion_score=self.emotion_score,
            active_topic=self.active_topic,
            used_topics=self.used_topics
        )

        shift = result.get("emotion_shift", 0)
        if isinstance(shift, int):
            self.emotion_score = max(-3, min(3, self.emotion_score + shift))

        if result.get("topic_move") == "introduce_topic":
            new_topic = result.get("new_topic", "").strip()
            if new_topic:
                self.active_topic = new_topic
                if new_topic not in self.used_topics:
                    self.used_topics.append(new_topic)

        actor_reply = result["reply"]
        self.history.append({"role": "actor", "content": actor_reply})
        self._trim_history()

        return {
            "player_message": player_message,
            "emotion_score": self.emotion_score,
            "active_topic": self.active_topic,
            "engine_output": result,
            "actor_reply": actor_reply
        }


# =========================================================
# DEMO
# =========================================================

if __name__ == "__main__":
    session = ImprovSession()

    print("\n=== PREMISE ===")
    print(session.premise)

    print("\n=== PERSONA ===")
    print(session.persona.pretty_card())

    print("\nType 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        result = session.run_turn(user_input)

        print("\n--- EMOTION SCORE ---")
        print(result["emotion_score"])

        print("\n--- ACTIVE TOPIC ---")
        print(result["active_topic"])

        print("\n--- ENGINE OUTPUT ---")
        print(json.dumps(result["engine_output"], indent=2))

        print("\n--- ACTOR ---")
        print(result["actor_reply"])
        print()