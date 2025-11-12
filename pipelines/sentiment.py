"""
sentiment.py
This module guesses the sentiment of a text (positive/negative/neutral).
It also checks toxicity/violence first, so harmful texts are marked negative.
I wrote beginner-friendly comments so it’s easier to understand.
"""

# ==== Imports ====
import os
import re
from functools import lru_cache


# ==== Config / constants ====
# I read model names from environment variables so they can be changed without code edits.
SENTIMENT_MODEL_ID = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
TOXICITY_MODEL_ID = os.getenv("TOXICITY_MODEL", "unitary/unbiased-toxic-roberta")

# Threshold for when we call something toxic. 0.5 is a common middle value.
TOX_THRESHOLD = float(os.getenv("TOX_THRESHOLD", "0.5"))

# A small regex to quickly catch very obvious violent words (works offline).
_VIOLENCE_RE = re.compile(
    r"\b(kill|murder|slaughter|shoot|stab|bomb|rape|lynch|behead|execute|massacre)\b",
    re.IGNORECASE,
)


# ==== Helpers for silencing noisy warnings ====
def _silence_swig_deprecations():
    """
    Some transformer backends create a lot of SWIG deprecation warnings.
    I hide them so the console output stays clean.
    """
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPyObject.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPyPacked.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type swigvarlink.* has no __module__ attribute",
        category=DeprecationWarning,
    )


# ==== Cached pipelines (so we only build them once) ====
@lru_cache(maxsize=1)
def _tox_pipe():
    """
    Build the toxicity classification pipeline.
    I cache it so the model loads only once (faster next time).
    """
    _silence_swig_deprecations()
    from transformers import pipeline  # local import to speed module load

    return pipeline("text-classification", model=TOXICITY_MODEL_ID)


@lru_cache(maxsize=1)
def _sent_pipe():
    """
    Build the sentiment classification pipeline (positive/negative/neutral).
    Also cached for performance.
    """
    _silence_swig_deprecations()
    from transformers import pipeline  # local import to speed module load

    return pipeline("text-classification", model=SENTIMENT_MODEL_ID)


# ==== Core scoring helpers ====
def _toxicity_score(text: str) -> float:
    """
    Return a single toxicity score between 0 and 1.
    I ask the model for ALL label scores and just take the maximum
    across the labels that sound harmful (like 'toxic', 'threat', etc.).
    """
    try:
        res = _tox_pipe()(text or "", truncation=True, return_all_scores=True)[0]
        max_rel = 0.0
        for item in res:
            label = (item.get("label") or "").lower()
            score = float(item.get("score") or 0.0)
            if label in {
                "toxic",
                "toxicity",
                "severe_toxicity",
                "threat",
                "violence",
                "insult",
                "identity_attack",
                "obscene",
                "hate",
            }:
                if score > max_rel:
                    max_rel = score
        return max_rel
    except Exception:
        # If something goes wrong (e.g., offline), I just return 0.0
        # so the rest of the logic can still run.
        return 0.0


# ==== Public function ====
def detect_sentiment(text: str):
    """
    Decide the final sentiment label and score for a given text.

    Steps I follow:
      1) If I see very clear violent words, I instantly mark it as negative.
      2) Else, I check the toxicity model; if above threshold, mark as negative.
      3) Else, I run the normal sentiment model and normalize its label.
    """
    t = text or ""

    # 1) Quick offline check for obvious violence.
    if _VIOLENCE_RE.search(t):
        return {"label": "negative", "score": 0.99}

    # 2) Use the toxicity score gate first (safer for the user).
    tox = _toxicity_score(t)
    if tox >= TOX_THRESHOLD:
        return {"label": "negative", "score": tox}

    # 3) If not toxic, do standard sentiment.
    res = _sent_pipe()(t, truncation=True)[0]
    label = (res.get("label") or "").lower()
    score = float(res.get("score") or 0.0)

    # The model can return 'LABEL_0/LABEL_1' etc., so I normalize to words.
    if label.startswith("pos"):
        label = "positive"
    elif label.startswith("neg"):
        label = "negative"
    elif label not in ("positive", "negative", "neutral"):
        label = "neutral"

    return {"label": label, "score": score}
