"""Sample graph configuration in the requested FEATURES / MODELS style.

Replace these constants with your production graph definitions when needed.

Usage example:
    from backend.runtime_engine.config import FEATURES, MODELS

    print(FEATURES["F_EMBED"]["deps"])  # ['F_TOKENS']
    print(MODELS["M1"]["features"])    # ['F_EMBED', 'F_TOKENS']
"""

FEATURES = {
    "F_RAW_TEXT": {"deps": [], "cost": 20},
    "F_USER_ID": {"deps": [], "cost": 10},
    "F_USER_HIST": {"deps": ["F_USER_ID"], "cost": 25},
    "F_TOKENS": {"deps": ["F_RAW_TEXT"], "cost": 40},
    "F_EMBED": {"deps": ["F_TOKENS"], "cost": 120},
    "F_CONTEXT": {"deps": ["F_USER_HIST", "F_EMBED"], "cost": 80},
    "F_SCORE": {"deps": ["F_CONTEXT"], "cost": 60},
}

MODELS = {
    "M1": {"features": ["F_EMBED", "F_TOKENS"]},
    "M2": {"features": ["F_EMBED", "F_RAW_TEXT"]},
    "M3": {"features": ["F_SCORE", "F_USER_HIST"]},
}
