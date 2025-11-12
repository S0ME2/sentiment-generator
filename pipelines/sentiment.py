# pipelines/sentiment.py
import os, re
from functools import lru_cache

# Public constants (app/tests import these)
SENTIMENT_MODEL_ID = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
TOXICITY_MODEL_ID = os.getenv("TOXICITY_MODEL", "unitary/unbiased-toxic-roberta")
TOX_THRESHOLD = float(
    os.getenv("TOX_THRESHOLD", "0.5")
)  # adjust as needed (0.3–0.6 typical)

# Clear, obvious violence terms for offline/last-ditch override
_VIOLENCE_RE = re.compile(
    r"\b(kill|murder|slaughter|shoot|stab|bomb|rape|lynch|behead|execute|massacre)\b",
    re.IGNORECASE,
)


def _silence_swig_deprecations():
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


@lru_cache(maxsize=1)
def _tox_pipe():
    _silence_swig_deprecations()
    from transformers import pipeline  # lazy import

    # Multi-label toxicity model; we'll request all scores at call time
    return pipeline("text-classification", model=TOXICITY_MODEL_ID)


@lru_cache(maxsize=1)
def _sent_pipe():
    _silence_swig_deprecations()
    from transformers import pipeline  # lazy import

    return pipeline("text-classification", model=SENTIMENT_MODEL_ID)


def _toxicity_score(text: str) -> float:
    """Return a single toxicity score in [0,1] by taking the max of relevant labels."""
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
        # If the model can't load (offline, etc.), fall back to 0 and let heuristics decide
        return 0.0


def detect_sentiment(text: str):
    """
    Returns {"label": "positive|negative|neutral", "score": float}.
    Now includes a toxicity/threat gate that forces 'negative' on harmful input.
    """
    t = text or ""

    # 1) Obvious-violence heuristic (works offline, very high precision)
    if _VIOLENCE_RE.search(t):
        return {"label": "negative", "score": 0.99}

    # 2) Toxicity gate (robust, multi-label)
    tox = _toxicity_score(t)
    if tox >= TOX_THRESHOLD:
        return {"label": "negative", "score": tox}

    # 3) Normal sentiment (general domain)
    res = _sent_pipe()(t, truncation=True)[0]
    label = (res.get("label") or "").lower()
    score = float(res.get("score") or 0.0)

    if label.startswith("pos"):
        label = "positive"
    elif label.startswith("neg"):
        label = "negative"
    elif label not in ("positive", "negative", "neutral"):
        label = "neutral"

    return {"label": label, "score": score}
