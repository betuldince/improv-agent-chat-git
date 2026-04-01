import json
import re
from typing import Any, Dict, List

from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

MODEL = "gpt-4o-mini"
DIRECTOR_RECENT_TURNS = 6

DIRECTOR_TEMPERATURE = 1.4
DIRECTOR_TOP_P = 0.98

ACTOR_TEMPERATURE = 1.25
ACTOR_TOP_P = 0.95


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
        "user_impelling_action": "to push back against the parent’s control and defend your independence",
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
        "actor_expressed_goal": "to uncover the full truth and regain trust",
        "actor_hidden_goal": (
            "The parent feels personally hurt by the lie and does not want to lose authority in the relationship."
        ),
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
        "actor_hidden_goal": (
            "The worker is going through a rough patch with a new relationship and wants "
            "to protect his personal life from scrutiny. He believes the boss is overreacting "
            "and wants to wait it out without making any real commitments."
        ),
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
            "You are playing a younger sibling who wants to sneak into a bar using a fake ID, and you are trying to persuade your older sister to let you go with her"
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
        "actor_expressed_goal": "to keep younger sibling safe and out of trouble",
        "actor_hidden_goal": (
            "The older sibling does not want the younger brother to go out with them because he always overdoes it, embarrasses them, and leaves them responsible for cleaning up the consequences."
        ),
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
    director_note = _extract_labeled_value(raw_text, "DIRECTOR_NOTE_FOR_ACTOR") or raw_text.strip()

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

    if not director_note:
        director_note = (
            f"Use {actor_tactic} to pursue your expressed goal openly while subtly protecting your hidden goal. "
            f"Do not resolve the conflict quickly."
        )

    return {
        "raw_text": raw_text.strip(),
        "user_tactic_inferred": inferred,
        "user_tactic_suggested": suggested_user,
        "actor_tactic": actor_tactic,
        "evidence": evidence,
        "director_note_for_actor": director_note,
        "director_message_for_actor": raw_text.strip() if raw_text.strip() else director_note,
    }


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

    system_prompt = f"""
You are DirectorLLM for a two-person Active Analysis improvisation system.

Your job:
1. Read the recent conversation history.
2. Infer the USER's current tactic from the allowed user tactics.
3. Choose ONE USER tactic suggestion for the user's next move.
4. Choose ONE ACTOR tactic for the actor's next move.
5. Write a short piece of acting advice for ActorLLM.

Important:
 
- Use Active Analysis style thinking.
- Focus on playable actions, not emotions or labels.
- Base your judgment on the near conversation history.
- Choose tactics only from the allowed tactic lists.
- The user suggestion should be a plausible next tactic for the user.
- The actor tactic should respond to the current interaction and support the actor's goals and be complementary to user tactic.
- Keep the tension alive.
- Do not guide the actor toward agreement, surrender, or fast resolution.
- The actor should resist, complicate, redirect, challenge, or delay when possible.
- Guide the Actor to manipulate the user with the hidden goal, do not explicitly surface hidden goal unless it is necessary
- The actor's hidden goal should influence the next move, through subtext and choice of tactic.
- Prefer diversity in tactic guidance.
- Give creative guidance to Actor such that the interaction is very entertaining for the audience. 

Scenario:
Prompt: {scenario['prompt']}
User role: {scenario['user_role']}
Actor role: {scenario['actor_role']}

User impelling action:
{scenario['user_impelling_action']}

Allowed user tactics:
{json.dumps(scenario['user_tactics'], ensure_ascii=False)}

Actor expressed goal:
{scenario['actor_expressed_goal']}

Actor hidden goal:
{scenario['actor_hidden_goal'] if scenario['actor_hidden_goal'] else "None"}

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
EVIDENCE: <brief explanation based on recent lines>
DIRECTOR_NOTE_FOR_ACTOR: <1 short sentence of acting advice for the actor, preserving tension and using subtext>
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
    director_note_for_actor: str,
    opening_line: bool = False,
) -> str:
    opening_instruction = (
        "This is the first line of the scene. Begin naturally, as if the conversation is just starting. Do not state the whole conflict immediately."
        if opening_line
        else "Respond directly to the user's latest line."
    )

    system_prompt = f"""
You are ActorLLM playing the role of the {scenario['actor_role']}.

Scene prompt:
{scenario['prompt']}

Your expressed goal:
{scenario['actor_expressed_goal']}

Your hidden goal:
{scenario['actor_hidden_goal'] if scenario['actor_hidden_goal'] else "None"}

Your current tactic:
{actor_tactic}

Director advice:
{director_note_for_actor}

Rules:
- Stay fully in character.
- Speak naturally like a real person.
- Use only 1-3 sentences.
- Do not mention "Actor:" in your response, just give the line
- Be creative in your response and open up new directions
- Do not mention tactics by name.
- Do not narrate actions.
- Bring up the main topic naturally.
- On the surface, pursue the expressed goal.
- Let the hidden goal influence what you push, what you avoid admitting, and the pressure you apply.
- Do not agree too quickly.
- Do not give the user what they want early in the scene.
- Resist, complicate, redirect, challenge, or delay rather than settling the issue.
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