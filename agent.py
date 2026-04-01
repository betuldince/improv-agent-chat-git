import json
import re
from typing import Any, Dict, List

from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

MODEL = "gpt-5.1"
DIRECTOR_RECENT_TURNS = 15

DIRECTOR_TEMPERATURE = 0.6
DIRECTOR_TOP_P = 0.7

ACTOR_TEMPERATURE = 1.4
ACTOR_TOP_P = 0.95

TURN_NUM = 0

# =========================================================
# SCENARIOS
# =========================================================

SCENARIOS: List[Dict[str, Any]] = [
    {
        "round_number": 1,
        "user_role": "Teenager",
        "actor_role": "Parent",
        "prompt": (
            "You are playing a teenage child whose parent confronts them after "
            "discovering they lied about where they were the night before."
        ),
        "show_to_user": {
            "prompt": True,
            "user_impelling_action": True,
            "show_tactic": True,
        },
        "user_impelling_action": "to push back against the parent's control and defend your independence",
        "user_tactics": [
            "to justify",
            "to minimize",
            "to deflect",
            "to argue",
            "to compare",
            "to accuse of overreacting",
            "to assert boundaries",
            "to redirect the topic",
            "to deny",
            "to partially admit",
            "to negotiate",
            "to appeal",
            "to escalate",
        ],
        "actor_expressed_goal": (
                "to re-establish relationship and trust between parent and child. "
        ),
        "actor_persona": {
            "age": 44,
            "race": "White",
            "sex": "Female",
            "traits": ["controlling", "guilt-inducing", "anxious", "manipulative", "emotionally needy"],
            "internal_motivation": (
                "You are preparing for a divorce and wants the child to feel safe opening up to her, "
                "partly because you believe that emotional closeness may make the child more likely to choose "
                "her in the custody process. "
            ),
        },
        "actor_tactics": [
            "to question",
            "to press for details",
            "to point out inconsistencies",
            "to repeat",
            "to confront",
            "to express disappointment",
            "to invite honesty",
            "to test sincerity",
            "to soften",
            "to guilt",
            "to escalate",
        ],
    },
    {
        "round_number": 2,
        "user_role": "Boss",
        "actor_role": "Worker",
        "prompt": (
            "You are playing a boss who talks to a worker to complain about "
            "their arriving late to work and leaving early which is impacting "
            "the performance of the whole team."
        ),
        "show_to_user": {
            "prompt": True,
            "user_impelling_action": False,
            "show_tactic": True,
        },
        "user_impelling_action": "to stop problematic behavior",
        "user_tactics": [
            "to tell",
            "to explain",
            "to embellish",
            "to justify",
            "to seek empathy",
            "to blame",
            "to exhort",
            "to threaten",
            "to set expectations",
            "to assert authority",
            "to compare to others",
            "to express frustration",
            "to test honesty",
        ],
        "actor_expressed_goal": "to have flexible working hours and show team's performance have nothing to do with him",
        "actor_persona": {
            "age": 31,
            "race": "Black",
            "sex": "Male",
            "traits": ["disengaged", "evasive", "passive-aggressive", "self-serving", "defensive", "opportunistic"],
            "internal_motivation": (
                "The worker is interviewing for another job and does not really care about this job anymore"
                "but he wants to stay employed if it is possible because he needs money for his new-born child."
            ),
        },
        "actor_tactics": [
            "to trivialize",
            "to redirect toward others",
            "to blame",
            "to justify",
            "to disagree",
            "to not listen",
            "to downplay impact",
            "to reinterpret events",
            "to minimize frequency",
            "to normalize behavior",
            "to shift blame",
            "to deflect responsibility",
            "to push back",
            "to negotiate terms",
            "to challenge fairness",
        ],
    },
    {
        "round_number": 3,
        "user_role": "Younger sibling",
        "actor_role": "Older sister",
        "prompt": (
"You are playing a younger sibling whose older sister confronts them after discovering they are planning to use a fake ID to sneak into a bar with her."
        ),
        "show_to_user": {
            "prompt": True,
            "user_impelling_action": False,
            "show_tactic": False,
        },
        "user_impelling_action": "to get the older sibling to let them come and to be treated like an adult",
        "user_tactics": [
            "to justify",
            "to minimize",
            "to joke",
            "to compare",
            "to negotiate",
            "to promise restraint",
            "to accuse of being controlling",
            "to appeal",
            "to demand respect",
            "to persist",
        ],
        "actor_expressed_goal": "to keep the younger sibling safe and prevent them from doing something they may regret",
        "actor_persona": {
            "age": 23,
            "race": "Latina",
            "sex": "Female",
            "traits": ["caring", "protective", "responsible", "anxious", "resentful"],
            "internal_motivation": (
                "You care about your sibling and you don't want to be in trouble but secretly you also"
                "think he always overdoes it, embarrasses you, and leaves you responsible for cleaning up the consequences."
            ),
        },
        "actor_tactics": [
            "to justify",
            "to patronize",
            "to joke",
            "to minimize",
            "to redirect",
            "to claim protection",
            "to dismiss",
            "to soften",
            "to protest",
            "to confront",
            "to explain feelings",
            "to assert boundaries",
            "to challenge authority",
            "to compare",
            "to accuse",
            "to demand respect",
        ],
    },
]


# =========================================================
# OPENAI HELPERS
# =========================================================

def create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_gpt_text(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# =========================================================
# PROMPT HELPERS
# =========================================================

def format_history(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return "No dialogue yet."
    lines = []
    for m in messages:
        speaker = "USER" if m["role"] == "user" else "ACTOR"
        lines.append(f"{speaker}: {m['content']}")
    return "\n".join(lines)


def recent_history(
    messages: List[Dict[str, str]],
    n_turns: int = DIRECTOR_RECENT_TURNS,
) -> List[Dict[str, str]]:
    if not messages:
        return []
    return messages[-n_turns:]


def format_recent_tactics(tactics: List[str], label: str) -> str:
    if not tactics:
        return f"No recent {label} tactics yet."
    return ", ".join(tactics[-3:])


def choose_diverse_tactic(
    suggested: str,
    allowed: List[str],
    recent_used: List[str],
    avoid_last_n: int = 2,
) -> str:
    if not allowed:
        return suggested

    if suggested not in allowed:
        suggested = allowed[0]

    recent_blocked = recent_used[-avoid_last_n:] if recent_used else []

    if suggested not in recent_blocked:
        return suggested

    for tactic in allowed:
        if tactic not in recent_blocked:
            return tactic

    return suggested


def _extract_labeled_value(raw: str, label: str) -> str:
    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, raw)
    if match:
        return match.group(1).strip()
    return ""


def parse_director_text(
    raw_text: str,
    scenario: Dict[str, Any],
    recent_user_suggested_tactics: List[str],
    recent_actor_tactics: List[str],
) -> Dict[str, Any]:
    inferred = _extract_labeled_value(raw_text, "USER_TACTIC_INFERRED") or "none yet"
    suggested_user = _extract_labeled_value(raw_text, "USER_TACTIC_SUGGESTED") or scenario["user_tactics"][0]
    actor_tactic = _extract_labeled_value(raw_text, "ACTOR_TACTIC") or scenario["actor_tactics"][0]
    evidence = _extract_labeled_value(raw_text, "EVIDENCE") or ""

    if inferred != "none yet" and inferred not in scenario["user_tactics"]:
        inferred = "none yet"

    suggested_user = choose_diverse_tactic(
        suggested=suggested_user,
        allowed=scenario["user_tactics"],
        recent_used=recent_user_suggested_tactics,
        avoid_last_n=2,
    )

    actor_tactic = choose_diverse_tactic(
        suggested=actor_tactic,
        allowed=scenario["actor_tactics"],
        recent_used=recent_actor_tactics,
        avoid_last_n=2,
    )

    return {
        "raw_text": raw_text.strip(),
        "user_tactic_inferred": inferred,
        "user_tactic_suggested": suggested_user,
        "actor_tactic": actor_tactic,
        "evidence": evidence,
    }


# =========================================================
# TURN NUM GETTER
# =========================================================

def get_turn_num() -> int:
    return TURN_NUM


# =========================================================
# DIRECTOR
# =========================================================

def director_step(
    client: OpenAI,
    scenario: Dict[str, Any],
    messages: List[Dict[str, str]],
    recent_user_suggested_tactics: List[str],
    recent_actor_tactics: List[str],
) -> Dict[str, Any]:
    recent = recent_history(messages, DIRECTOR_RECENT_TURNS)
    global TURN_NUM
    TURN_NUM += 1

    system_prompt = f"""
You are DirectorLLM for a two-person Active Analysis improvisation system.

Your only job is to read the conversation and choose tactics:
1. Infer the USER's current tactic from the allowed user tactics.
2. Choose ONE USER tactic suggestion for the user's next move.
3. Choose ONE ACTOR tactic for the actor's next move.

RULES
- Use Active Analysis style thinking.
- Choose tactics only from the allowed tactic lists.
- Base your choices on the recent conversation history.
- The user suggestion should be the most plausible next move for the user.
- The actor tactic should keep tension alive and resist easy resolution.
- Prefer diversity — avoid repeating recently used tactics unless the scene demands it.
- Do NOT write any notes, advice, or explanation for the actor. Tactic selection only.

Scenario:
User role prompt: {scenario['prompt']}
User role: {scenario['user_role']}
Actor role: {scenario['actor_role']}

User impelling action:
{scenario['user_impelling_action']}

Allowed user tactics:
{json.dumps(scenario['user_tactics'], ensure_ascii=False)}

Actor expressed goal:
{scenario['actor_expressed_goal']}

Allowed actor tactics:
{json.dumps(scenario['actor_tactics'], ensure_ascii=False)}

Recent user tactic suggestions to avoid repeating unless necessary:
{format_recent_tactics(recent_user_suggested_tactics, "user")}

Recent actor tactics to avoid repeating unless necessary:
{format_recent_tactics(recent_actor_tactics, "actor")}

Return exactly in this plain-text format:

USER_TACTIC_INFERRED: <one tactic from allowed user tactics or none yet>
USER_TACTIC_SUGGESTED: <one tactic from allowed user tactics>
ACTOR_TACTIC: <one tactic from allowed actor tactics>
EVIDENCE: <one sentence explaining your tactic choices based on the recent lines>
""".strip()

    user_prompt = f"""
Recent conversation history:
{format_history(recent)}
""".strip()

    raw_text = call_gpt_text(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=DIRECTOR_TEMPERATURE,
        top_p=DIRECTOR_TOP_P,
    )

    return parse_director_text(
        raw_text=raw_text,
        scenario=scenario,
        recent_user_suggested_tactics=recent_user_suggested_tactics,
        recent_actor_tactics=recent_actor_tactics,
    )


# =========================================================
# ACTOR
# =========================================================

def actor_reply(
    client: OpenAI,
    scenario: Dict[str, Any],
    messages: List[Dict[str, str]],
    actor_tactic: str,
    opening_line: bool = False,
) -> str:
    opening_instruction = (
        "This is the first line of the scene. Begin naturally, as if the conversation is just starting. "
        "Do not state the whole conflict immediately."
        if opening_line
        else "Respond directly to the user's latest line."
    )

    persona = scenario.get("actor_persona", {})
    age = persona.get("age", "")
    race = persona.get("race", "")
    sex = persona.get("sex", "")
    traits = persona.get("traits", [])

    persona_description = (
        f"You are a {age}-year-old {race} {sex}. "
        f"Your dominant character traits are: {', '.join(traits)}. "
        f"These traits shape how you speak, what you notice, and how you react under pressure. "
        f"Let them color your word choice, your tone, and the texture of your responses naturally."
    )

    system_prompt = f"""
You are ActorLLM(actor) playing the role of the {scenario['actor_role']}.

User role prompt:
{scenario['prompt']}

You seem like your goal is(initial goal)
{scenario['actor_expressed_goal']}

But you are secretly thinking(inner goal):
{persona.get("internal_motivation", "")}

Actor persona:
{persona_description}

Actor CURRENT TACTIC:
{actor_tactic}


RULES:
- Stay fully in character.
- Speak naturally like a real person.
- Use only 1-3 sentences.
- Use the given tactic to you and think about how would a person with this persona 
- User will act on user role prompt
- Show your initial goal first then surface your inner goal subtly if it feels right according to history
- Do not write "Actor:" — just give the line.
- Do not mention tactics or traits by name.
- Do not narrate actions.
- Do not agree too quickly.
- Even if you soften, keep friction alive.
- {opening_instruction}
""".strip()

    user_prompt = f"""
Conversation so far:
{format_history(messages)}

Now produce the next line for the {scenario['actor_role']}.
""".strip()

    text = call_gpt_text(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=ACTOR_TEMPERATURE,
        top_p=ACTOR_TOP_P,
    )

    if not text:
        text = "We need to talk about this."
    return text