import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import faiss
import numpy as np
from openai import OpenAI

import streamlit as st

# =========================================================
# CONFIG
# =========================================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-5-nano"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


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


def format_history(history: List[Dict[str, str]], limit: int = 6) -> str:
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
# MEMORY
# =========================================================

class MemoryStore:
    def __init__(self, client: OpenAI):
        self.client = client
        self.texts: List[str] = []
        self.kinds: List[str] = []
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def add_memory(self, text: str, kind: str = "scene") -> None:
        emb = self._embed(text)
        self.texts.append(text)
        self.kinds.append(kind)

        if len(self.texts) == 1:
            self.embeddings = emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.index.reset()
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 5, allowed_kinds: Optional[List[str]] = None) -> List[str]:
        if len(self.texts) == 0:
            return []

        q = self._embed(query).reshape(1, -1)
        scores, indices = self.index.search(q, min(len(self.texts), max(k * 3, k)))

        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            if allowed_kinds is None or self.kinds[idx] in allowed_kinds:
                results.append(self.texts[idx])
            if len(results) >= k:
                break
        return results


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

    def to_memory_items(self) -> List[str]:
        items = [
            f"Name: {self.name}",
            f"Pronouns: {self.pronouns}",
            f"Age: {self.age}",
            f"Background: {self.background}",
            f"Profession: {self.profession}",
            f"Passions: {self.passions}",
            f"Interests: {self.interests}",
            f"Relationship to player: {self.relationship_to_player}",
            f"Current emotional state: {self.current_emotional_state}",
        ]
        for fact in self.hidden_facts:
            items.append(f"Hidden fact: {fact}")
        return items

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
- Hidden facts should be relevant to the breakup and useful later as possible reveals.
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
                "name": "Maya Bennett",
                "pronouns": "She/Her",
                "age": 29,
                "background": "Moved to a new city for work and has been emotionally pulling away over the last few months.",
                "profession": "Product designer",
                "passions": "Photography and building a life with intention",
                "interests": "Indie films, quiet cafes, long walks",
                "relationship_to_player": "Has been dating the player for two years but has felt increasingly distant.",
                "current_emotional_state": "neutral",
                "hidden_facts": [
                    "Has been thinking about ending the relationship for weeks",
                    "Already made practical plans for life after the breakup"
                ]
            }

        return Persona(**data)


# =========================================================
# DIRECTOR
# =========================================================

class Director:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def extract_player_emotion(self, recent_dialogue: List[Dict[str, str]]) -> Dict[str, str]:
        history_text = format_history(recent_dialogue, limit=6)
        latest_player_message = get_latest_player_message(recent_dialogue)

        prompt = f"""
You are analyzing the player's emotional stance in an improv conversation.

Recent dialogue:
{history_text}

Latest player message:
{latest_player_message}

Return JSON only:
{{
  "player_emotion": "...",
  "emotion_group": "negative" or "positive",
  "reason": "brief explanation"
}}

Rules:
- Infer the player's dominant emotional stance from the recent dialogue, but weigh the latest player message most heavily.
- Use a specific emotion/stance label.
- Negative labels include: offensive, accusatory, defensive, cold, dismissive, manipulative, frustrated, angry, resigned.
- Positive labels include: empathetic, apologetic, vulnerable, caring, pleading, confused, desperate.
- Return only one dominant player_emotion.
- Also assign emotion_group as either negative or positive.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You extract the player's dominant emotional stance."},
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)
        if not data or "player_emotion" not in data or "emotion_group" not in data:
            return {
                "player_emotion": "defensive",
                "emotion_group": "negative",
                "reason": "Fallback classification."
            }
        return data

    def judge_conversation_steady(
        self,
        premise: str,
        persona: Persona,
        recent_dialogue: List[Dict[str, str]],
        active_topic: Optional[str]
    ) -> Dict[str, str]:
        history_text = format_history(recent_dialogue, limit=6)

        prompt = f"""
You are directing an improv breakup scene.

Premise:
{premise}

Actor persona:
{persona.pretty_card()}

Current active topic:
{active_topic if active_topic else "None"}

Last 6 dialogue turns:
{history_text}

Return JSON only:
{{
  "is_steady": "yes" or "no",
  "reason": "brief explanation"
}}

Rules:
- Mark is_steady = "yes" only if the conversation feels flat, repetitive, stalled, or too settled.
- Mark is_steady = "no" if the conversation is still actively developing, reacting, or escalating naturally.
- Judge based only on the last 6 turns.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You judge whether a scene has become too steady."},
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)
        if not data or "is_steady" not in data:
            return {
                "is_steady": "no",
                "reason": "Fallback: assume still developing."
            }
        return data

    def generate_breakup_topic_bank(self, persona: Persona) -> List[str]:
        prompt = f"""
You are generating practical breakup conversation topics.

Actor persona:
{persona.pretty_card()}

Return JSON only:
{{
  "topics": [
    "...",
    "...",
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}}

Rules:
- Topics must be concrete and discussable, not abstract emotions.
- Good examples: picking up clothes, apartment keys, shared dog, rent/lease, photos, furniture, telling friends, returning books, canceling a trip.
- Topics must fit the breakup premise and feel natural for adults in a real relationship.
- Keep each topic short, specific, and phrased as a topic/problem to discuss.
- Do not make every topic about feelings.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You generate practical breakup topics."},
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)
        if data and isinstance(data.get("topics"), list) and data["topics"]:
            return data["topics"]

        return [
            "I need to pick up my clothes and toiletries from your place.",
            "I think the dog should stay with me.",
            "We need to decide what to do about the spare key.",
            "I do not want us both showing up at the same friend's birthday without talking about it.",
            "We still need to figure out the rent and who is taking over the lease.",
            "I want the photo albums and the camera back.",
            "We should cancel the trip before we lose more money.",
            "I need you to drop my books and charger off this week."
        ]

    def generate_guidance(
        self,
        premise: str,
        persona: Persona,
        recent_dialogue: List[Dict[str, str]],
        active_topic: Optional[str],
        unused_topics: List[str],
        emotion_score: int
    ) -> Dict[str, str]:
        emotion_info = self.extract_player_emotion(recent_dialogue)
        player_emotion = emotion_info["player_emotion"]
        emotion_group = emotion_info["emotion_group"]
        player_reason = emotion_info.get("reason", "")

        steady_info = self.judge_conversation_steady(
            premise=premise,
            persona=persona,
            recent_dialogue=recent_dialogue,
            active_topic=active_topic
        )

        is_steady = steady_info["is_steady"]
        steady_reason = steady_info.get("reason", "")

        current_agent_emotion = emotion_score_to_label(emotion_score)

        emotional_expression_map = {
            "neutral": "calm but emotionally contained",
            "guarded": "guarded and slightly tense",
            "cold": "cold and distant",
            "angry": "angry and sharp",
            "softened": "softened and gentle",
            "vulnerable": "vulnerable and honest",
            "emotionally open": "emotionally open and exposed",
        }

        topic_move = "stay_on_current_topic"
        current_topic = active_topic or ""
        new_information = ""

        if active_topic is None and unused_topics:
            topic_move = "introduce_new_topic"
            current_topic = unused_topics[0]
            new_information = current_topic
        elif is_steady == "yes" and unused_topics:
            topic_move = "introduce_new_topic"
            current_topic = unused_topics[0]
            new_information = current_topic

        return {
            "player_emotion": player_emotion,
            "emotion_group": emotion_group,
            "player_reason": player_reason,
            "agent_emotion": current_agent_emotion,
            "emotional_expression": emotional_expression_map.get(current_agent_emotion, "calm but emotionally contained"),
            "is_steady": is_steady,
            "steady_reason": steady_reason,
            "topic_move": topic_move,
            "current_topic": current_topic,
            "reveal_new_information": "yes" if topic_move == "introduce_new_topic" else "no",
            "new_information": new_information
        }


# =========================================================
# ACTOR
# =========================================================

class Actor:
    def __init__(self, client: OpenAI, model: str, persona: Persona, memory: MemoryStore):
        self.client = client
        self.model = model
        self.persona = persona
        self.memory = memory

    def respond(
        self,
        player_message: str,
        recent_dialogue: List[Dict[str, str]],
        guidance: Optional[Dict[str, str]] = None
    ) -> str:
        retrieved = self.memory.retrieve(
            query=player_message,
            k=6,
            allowed_kinds=["persona", "scene", "topic"]
        )
        history_text = format_history(recent_dialogue, limit=6)
        memory_text = "\n".join(f"- {m}" for m in retrieved) if retrieved else "None"

        guidance_text = "None"
        if guidance:
            guidance_text = json.dumps(guidance, indent=2)

        prompt = f"""
You are improvising as this character:

{self.persona.pretty_card()}

Relevant memories:
{memory_text}

Recent dialogue:
{history_text}

Latest player message:
{player_message}

Director guidance:
{guidance_text}

Instructions:
- Respond as the character naturally would but be creative in your respond.
- do not say anyhting about how the other actor feels
- in your response instead of asking quesitons to user say what you think based on memories, director guidance and context
- The response should be 1-4 sentences.
- Follow the current agent_emotion and emotional_expression from the guidance.
- Keep emotional change gradual. Do not jump to extreme anger or extreme vulnerability unless the ongoing conversation clearly justifies it.
- If topic_move is "introduce_new_topic", introduce that practical topic naturally.
- Otherwise stay with the current topic.
- If reveal_new_information is "no", do not introduce a brand-new practical topic.
- Stay concrete and practical when discussing breakup topics.
- Do not drift into vague emotional monologues when the topic is practical.
- Keep the response to one turn only.
- Do not narrate the whole scene.
- Start with a brief expression/action in parentheses.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a skilled improv actor staying fully in character."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        if not content.startswith(f"({self.persona.name}"):
            content = f"({self.persona.name} takes a breath) {content}"

        return content


# =========================================================
# MEMORY UPDATER
# =========================================================

class MemoryUpdater:
    def __init__(self, client: OpenAI, model: str, memory: MemoryStore):
        self.client = client
        self.model = model
        self.memory = memory

    def maybe_store_important_memory(self, player_message: str, actor_reply: str) -> Optional[str]:
        prompt = f"""
Decide whether this exchange contains an important long-term scene memory.

Player:
{player_message}

Actor:
{actor_reply}

Store memory only if there is:
- an important revealed fact,
- a relationship shift,
- a practical breakup topic that was introduced,
- or a meaningful commitment/decision.

Return JSON only:
{{
  "store": "yes" or "no",
  "memory_text": "..."
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Extract only important scene memories."},
                {"role": "user", "content": prompt},
            ],
        )

        data = safe_json_loads(response.choices[0].message.content)
        if not data:
            return None

        if data.get("store", "").lower() == "yes" and data.get("memory_text"):
            self.memory.add_memory(data["memory_text"], kind="scene")
            return data["memory_text"]

        return None


# =========================================================
# SESSION
# =========================================================

class ImprovSession:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.premise = "Your improv partner is playing someone who is breaking up with you."

        self.memory = MemoryStore(self.client)
        self.persona_builder = PersonaBuilder(self.client, MODEL)
        self.director = Director(self.client, MODEL)
        self.memory_updater = MemoryUpdater(self.client, MODEL, self.memory)

        self.persona = self.persona_builder.create_breakup_persona()
        self.actor = Actor(self.client, MODEL, self.persona, self.memory)

        self.history: List[Dict[str, str]] = []

        self.topic_bank = self.director.generate_breakup_topic_bank(self.persona)
        self.used_topics = set()
        self.active_topic: Optional[str] = None

        # gradual emotion state: -3 to +3
        self.emotion_score = 0

        self._seed_persona_memory()
        self._seed_topic_memory()

    def _seed_persona_memory(self):
        for item in self.persona.to_memory_items():
            self.memory.add_memory(item, kind="persona")

    def _seed_topic_memory(self):
        for topic in self.topic_bank:
            self.memory.add_memory(f"Possible practical breakup topic: {topic}", kind="topic")

    def _get_unused_topics(self) -> List[str]:
        return [topic for topic in self.topic_bank if topic not in self.used_topics]

    def _update_emotion_score(self, guidance: Dict[str, Any]):
        group = guidance.get("emotion_group", "negative")
        if group == "positive":
            self.emotion_score = min(3, self.emotion_score + 1)
        else:
            self.emotion_score = max(-3, self.emotion_score - 1)

    def _update_topic_state_from_guidance(self, guidance: Dict[str, Any]):
        topic_move = guidance.get("topic_move", "")
        current_topic = guidance.get("current_topic", "")
        new_information = guidance.get("new_information", "")

        if topic_move == "introduce_new_topic":
            chosen_topic = new_information or current_topic
            if chosen_topic:
                self.active_topic = chosen_topic
                self.used_topics.add(chosen_topic)
                self.memory.add_memory(f"Active breakup topic introduced: {chosen_topic}", kind="scene")

    def run_turn(self, player_message: str) -> Dict[str, Any]:
        self.history.append({"role": "player", "content": player_message})

        # gradual emotion shift first
        player_emotion_info = self.director.extract_player_emotion(self.history)
        if player_emotion_info.get("emotion_group") == "positive":
            self.emotion_score = min(3, self.emotion_score + 1)
        else:
            self.emotion_score = max(-3, self.emotion_score - 1)

        guidance = self.director.generate_guidance(
            premise=self.premise,
            persona=self.persona,
            recent_dialogue=self.history,
            active_topic=self.active_topic,
            unused_topics=self._get_unused_topics(),
            emotion_score=self.emotion_score
        )

        actor_reply = self.actor.respond(
            player_message=player_message,
            recent_dialogue=self.history,
            guidance=guidance
        )

        self.history.append({"role": "actor", "content": actor_reply})

        self._update_topic_state_from_guidance(guidance)

        stored_memory = self.memory_updater.maybe_store_important_memory(
            player_message=player_message,
            actor_reply=actor_reply
        )

        return {
            "player_message": player_message,
            "emotion_score": self.emotion_score,
            "active_topic": self.active_topic,
            "director_guidance": guidance,
            "actor_reply": actor_reply,
            "stored_memory": stored_memory
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

    print("\n=== TOPIC BANK ===")
    for topic in session.topic_bank:
        print("-", topic)

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

        print("\n--- DIRECTOR GUIDANCE ---")
        print(json.dumps(result["director_guidance"], indent=2))

        print("\n--- ACTOR ---")
        print(result["actor_reply"])

        if result["stored_memory"]:
            print("\n--- STORED MEMORY ---")
            print(result["stored_memory"])

        print()