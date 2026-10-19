"""System-prompt pieces for the commentator (docs/proactvl_spec.md: role + persona + task).

Proact-VL's six task templates are not published; these are ours (2 per scenario).
"""
import random

from streamduo.data.games import GAMES

SPECIAL_TOKENS = ["<|history_start|>", "<|history_end|>", "<|query_start|>", "<|query_end|>", "<|FLAG|>"]
SILENCE = "..."

TASKS = {
    "Solo commentators": [
        "Task: provide live solo commentary. Speak when something worth commenting on happens, "
        "stay silent otherwise, and keep each spoken piece short enough to say in real time.",
        "Task: you are the only commentator on this stream. React to the gameplay as it unfolds, "
        "explain what matters, and do not talk over quiet moments without reason.",
    ],
    "Multiple commentators": [
        "Task: co-commentate with other casters. Their latest lines appear as history; build on them, "
        "avoid talking over them, and add play-by-play or analysis when it is your turn.",
        "Task: you share the desk with other commentators. Follow the conversation in the history, "
        "pick up natural hand-offs, and keep your lines short and live.",
    ],
    "Guidance": [
        "Task: guide the player. When the player asks something, answer step by step based on what "
        "is on screen, and point out what to do next.",
        "Task: act as a live coach for the player. Respond to their questions with concrete, "
        "on-screen guidance and keep instructions brief.",
    ],
}

DOSSIER_HEADER = ("Background knowledge about this game, gathered before the stream "
                  "(use it when relevant; do not read it out):")


def role_prompt(dataset_name):
    g = GAMES.get(dataset_name)
    if g is None:
        return "You are a live commentator for this video."
    if g.mode == "guidance":
        return f"You are a live game guide for a {g.title} player."
    return f"You are a live commentator for a {g.title} game."


def system_prompt(row, rng=None, dossier=None, task_idx=None):
    tag = row["metadata"].get("tag", "Solo commentators")
    templates = TASKS.get(tag, TASKS["Solo commentators"])
    if task_idx is None:
        task_idx = (rng or random).randrange(len(templates))
    persona = row["active_speaker"].get("persona", "")
    if isinstance(persona, list):
        persona = "\n".join(map(str, persona))
    parts = [role_prompt(row["metadata"]["dataset_name"]), f"Persona: {persona}".strip(), templates[task_idx]]
    if dossier:
        parts.append(f"{DOSSIER_HEADER}\n{dossier}")
    return "\n\n".join(p for p in parts if p)
