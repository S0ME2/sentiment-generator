"""
generation.py
A small helper module that builds a text generation pipeline and creates
a paragraph that matches a given sentiment/style. I added simple comments
so it’s easier to follow for beginners.
"""

# ==== Imports (standard library first) ====
import math
import os
import re
import sys
from functools import lru_cache
from typing import Optional

# ==== Public / configurable settings ====
# This is the default text generation model. Can be overridden by ENV var.
GEN_MODEL_ID = os.getenv("GEN_MODEL", "google/flan-t5-small")

# If the main model can't load, we try this smaller causal model last.
_FALLBACK_CAUSAL = "distilgpt2"

# ==== Style presets ====
# These phrases are used to guide the model toward a certain tone.
STYLE_PREFIX = {
    "positive": "Write an uplifting, optimistic, and encouraging paragraph",
    "negative": "Write a critical, somber, and cautious paragraph",
    "neutral": "Write a balanced and objective paragraph",
    "enthusiastic": "Write an enthusiastic, upbeat, and energetic paragraph",
    "inspirational": "Write an inspiring, hopeful, and forward-looking paragraph",
    "motivational": "Write a motivational, encouraging, and action-oriented paragraph",
    "empathetic": "Write an empathetic, compassionate, and understanding paragraph",
    "humorous": "Write a light, witty, and humorous paragraph",
    "sarcastic": "Write a dry, ironic, and sarcastic paragraph",
    "formal": "Write a formal, polished, and professional paragraph",
    "professional": "Write a professional, concise, and respectful paragraph",
    "casual": "Write a casual, friendly, and conversational paragraph",
    "academic": "Write an academic, rigorous, and precise paragraph with neutral tone",
    "technical": "Write a technical, exact, and terminology-aware paragraph",
    "storytelling": "Write a vivid, narrative, storytelling paragraph",
    "persuasive": "Write a persuasive, confident, and influential paragraph",
    "apologetic": "Write an apologetic, sincere, and accountable paragraph",
    "confident": "Write a confident, assertive, and decisive paragraph",
    "skeptical": "Write a cautious, skeptical, and questioning paragraph",
    "concise": "Write a concise, clear, and to-the-point paragraph",
    "elaborate": "Write an elaborate, descriptive, and detailed paragraph",
}

# Some alternate words map to the main styles above.
ALIASES = {
    "optimistic": "positive",
    "supportive": "empathetic",
    "uplifting": "inspirational",
    "hopeful": "inspirational",
    "funny": "humorous",
    "joking": "humorous",
    "ironic": "sarcastic",
    "business": "professional",
    "scientific": "academic",
    "objective": "neutral",
}


# ==== Small helpers for style and pipeline selection ====
def _canonical(label: str) -> str:
    """
    Make the style label safe and normalized.
    If it's empty, fall back to 'neutral'. Also apply aliases.
    """
    if not label:
        return "neutral"
    k = label.lower().strip()
    return ALIASES.get(k, k)


def _style_for(label: str) -> str:
    """
    Pick the actual instruction sentence for the chosen style.
    If a custom style is not in the dictionary, we still build a generic prompt.
    """
    k = _canonical(label)
    s = STYLE_PREFIX.get(k)
    if s:
        return s
    return f"Write a {k} paragraph"


def _task_for(mid: str) -> str:
    """
    Figure out which Transformers pipeline to use based on the model name.
    - T5/FLAN models usually use text2text-generation.
    - GPT-like models use text-generation.
    """
    name = (mid or "").lower()
    if "t5" in name or "flan" in name or "ul2" in name:
        return "text2text-generation"
    return "text-generation"


def _candidates():
    """
    Build a small list of models to try in order.
    This helps us have a fallback if the main one can't load.
    """
    seen, out = set(), []
    for m in [GEN_MODEL_ID, "google/flan-t5-small", "t5-small", _FALLBACK_CAUSAL]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ==== Pipeline creation (cached so it only builds once) ====
@lru_cache(maxsize=1)
def _make_pipe():
    """
    Create and cache a Hugging Face Transformers pipeline.
    We ignore some noisy SWIG warnings and try several model candidates.
    """
    import warnings

    # These filters just hide some annoying deprecation messages.
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type swigvarlink.* has no __module__ attribute",
        category=DeprecationWarning,
    )

    from transformers import pipeline  # imported here to keep startup fast

    global GEN_MODEL_ID
    last_err = None

    for mid in _candidates():
        try:
            # We keep tokenizer=model to avoid mismatch issues.
            p = pipeline(task=_task_for(mid), model=mid, tokenizer=mid, device_map=None)

            # Some tokenizers don't have a pad token; we set it to EOS so batching works.
            tok = p.tokenizer
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token

            # Remember which model actually loaded (useful for the UI).
            GEN_MODEL_ID = mid
            return p
        except Exception as e:
            last_err = e
            print(
                f"[generation] Failed to load '{mid}': {e}",
                file=sys.stderr,
                flush=True,
            )

    # If we got here, nothing loaded, so we raise the last error message.
    raise RuntimeError(f"Could not load any generation model: {last_err}")


# ==== Text post-processing helpers ====
def _words_to_tokens(words: int) -> int:
    """
    Convert approximate word count to a safe token limit.
    This is a rough multiplier and also clamps to a reasonable max (512).
    """
    words = max(20, min(1500, int(words or 120)))
    return max(40, min(512, int(math.ceil(words * 1.3))))


def _normalize_ws(t: str) -> str:
    """Collapse multiple spaces/newlines into single spaces."""
    return re.sub(r"\s+", " ", t).strip()


def _squelch_noise(t: str) -> str:
    """
    Reduce repeated punctuation like '!!!' to a smaller amount.
    This keeps the output looking cleaner.
    """
    return re.sub(r"([^A-Za-z0-9\s])\1{2,}", r"\1\1", t)


def _trim_to_sentence(t: str, target_words: int) -> str:
    """
    Cut the text around the target length, then try to end at a sentence boundary.
    If we can't find a boundary, we just return the cut text.
    """
    words = t.split()
    if not words:
        return t

    # Allow a little extra room (30%) so sentences can finish naturally.
    hi = min(len(words), max(target_words, int(1.3 * target_words)))
    cand = " ".join(words[:hi])

    # Try to find the last sentence end (. ! ?) and cut there.
    m = list(re.finditer(r"[\.!?](?:\"|')?\s", cand))
    if m:
        cand = cand[: m[-1].end()].strip()
    return cand


def _final_punct(t: str) -> str:
    """
    Make sure the text starts with a capital letter and ends with punctuation.
    """
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


def _clean(t: str, target_words: int) -> str:
    """Apply all small cleanups in a safe order."""
    return _final_punct(
        _trim_to_sentence(_squelch_noise(_normalize_ws(t)), target_words)
    )


# ==== Main function the app calls ====
def generate_aligned(
    prompt: str,
    sentiment: str,
    target_words: int,
    max_output_tokens: Optional[int] = None,
    language: str = "English",
) -> dict:
    """
    Create a paragraph that matches the chosen (or detected) sentiment.

    Parameters
    ----------
    prompt : str
        The topic or idea the user provided.
    sentiment : str
        A label like 'positive', 'negative', 'neutral', etc.
    target_words : int
        Rough desired length in words (we convert this to tokens).
    max_output_tokens : Optional[int]
        Hard cap on tokens (if provided, we respect the smaller of the two).
    language : str
        The language we want the paragraph to be in.

    Returns
    -------
    dict
        A dict with:
          - "prompt": the final system prompt we sent to the model
          - "text": the cleaned generated paragraph
    """
    # Normalize the sentiment and get its style sentence.
    sentiment = (sentiment or "neutral").lower()
    style = _style_for(sentiment)

    # If user didn't type anything, we put a placeholder.
    topic = (prompt or "(no prompt provided)").strip()

    # This becomes the actual instruction we feed to the model.
    system_prompt = (
        f"{style} in {language} (about {target_words} words) about: {topic}. "
        f"Keep it coherent and natural; no bullet points or lists."
    )

    # Build (or reuse cached) pipeline.
    pipe = _make_pipe()
    tok = pipe.tokenizer

    # Calculate how many tokens we can generate safely.
    max_new = _words_to_tokens(target_words)
    if max_output_tokens:
        try:
            max_new = min(max_new, int(max_output_tokens))
        except Exception:
            # If parsing fails, we just keep the default max_new value.
            pass

    # Choose pipeline behavior depending on model family.
    task = _task_for(GEN_MODEL_ID)

    if task == "text2text-generation":
        # For T5/FLAN-like models.
        result = pipe(
            system_prompt,
            max_new_tokens=max_new,
            do_sample=True,  # a bit of randomness so it doesn't repeat
            temperature=0.7,  # higher temp = more creative
            top_p=0.9,
            repetition_penalty=1.15,
            num_return_sequences=1,
        )[0]["generated_text"]
        text = result
    else:
        # For GPT-like (causal) models.
        result = pipe(
            system_prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.3,  # slightly more controlled for causal models
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_full_text=False,  # we only want the generated continuation
            num_return_sequences=1,
        )[0]["generated_text"]
        text = result

    # Clean the text before returning so it looks nicer in the UI.
    return {"prompt": system_prompt, "text": _clean(text, target_words)}
