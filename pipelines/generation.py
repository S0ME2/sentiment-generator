# pipelines/generation.py
from functools import lru_cache
from typing import Optional
import math, os, re, sys

# PUBLIC (used by app/tests)
GEN_MODEL_ID = os.getenv("GEN_MODEL", "google/flan-t5-small")
_FALLBACK_CAUSAL = "distilgpt2"

# ---- NEW: rich tones/styles ----
STYLE_PREFIX = {
    # original three
    "positive": "Write an uplifting, optimistic, and encouraging paragraph",
    "negative": "Write a critical, somber, and cautious paragraph",
    "neutral": "Write a balanced and objective paragraph",
    # extras
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

ALIASES = {
    # handy synonyms
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


def _canonical(label: str) -> str:
    if not label:
        return "neutral"
    k = label.lower().strip()
    return ALIASES.get(k, k)


def _style_for(label: str) -> str:
    k = _canonical(label)
    s = STYLE_PREFIX.get(k)
    if s:
        return s
    # If you type any custom style, we still produce something sensible.
    return f"Write a {k} paragraph"


def _task_for(mid: str) -> str:
    name = (mid or "").lower()
    if "t5" in name or "flan" in name or "ul2" in name:
        return "text2text-generation"
    return "text-generation"


def _candidates():
    seen, out = set(), []
    for m in [GEN_MODEL_ID, "google/flan-t5-small", "t5-small", _FALLBACK_CAUSAL]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


@lru_cache(maxsize=1)
def _make_pipe():
    import warnings

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
    from transformers import pipeline  # lazy import on first real use

    global GEN_MODEL_ID
    last_err = None
    for mid in _candidates():
        try:
            p = pipeline(task=_task_for(mid), model=mid, tokenizer=mid, device_map=None)
            tok = p.tokenizer
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token
            GEN_MODEL_ID = mid  # reflect what actually loaded
            return p
        except Exception as e:
            last_err = e
            print(
                f"[generation] Failed to load '{mid}': {e}", file=sys.stderr, flush=True
            )
    raise RuntimeError(f"Could not load any generation model: {last_err}")


def _words_to_tokens(words: int) -> int:
    words = max(20, min(1500, int(words or 120)))
    return max(40, min(512, int(math.ceil(words * 1.3))))


def _normalize_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _squelch_noise(t: str) -> str:
    return re.sub(r"([^A-Za-z0-9\s])\1{2,}", r"\1\1", t)


def _trim_to_sentence(t: str, target_words: int) -> str:
    words = t.split()
    if not words:
        return t
    hi = min(len(words), max(target_words, int(1.3 * target_words)))
    cand = " ".join(words[:hi])
    m = list(re.finditer(r"[\.!?](?:\"|')?\s", cand))
    if m:
        cand = cand[: m[-1].end()].strip()
    return cand


def _final_punct(t: str) -> str:
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t


def _clean(t: str, target_words: int) -> str:
    return _final_punct(
        _trim_to_sentence(_squelch_noise(_normalize_ws(t)), target_words)
    )


def generate_aligned(
    prompt: str,
    sentiment: str,
    target_words: int,
    max_output_tokens: Optional[int] = None,
    language: str = "English",
) -> dict:
    sentiment = (sentiment or "neutral").lower()
    style = _style_for(sentiment)
    topic = (prompt or "(no prompt provided)").strip()

    system_prompt = (
        f"{style} in {language} (about {target_words} words) about: {topic}. "
        f"Keep it coherent and natural; no bullet points or lists."
    )

    pipe = _make_pipe()
    tok = pipe.tokenizer

    max_new = _words_to_tokens(target_words)
    if max_output_tokens:
        try:
            max_new = min(max_new, int(max_output_tokens))
        except Exception:
            pass

    task = _task_for(GEN_MODEL_ID)
    if task == "text2text-generation":
        result = pipe(
            system_prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            num_return_sequences=1,
        )[0]["generated_text"]
        text = result
    else:
        result = pipe(
            system_prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_full_text=False,
            num_return_sequences=1,
        )[0]["generated_text"]
        text = result

    return {"prompt": system_prompt, "text": _clean(text, target_words)}
