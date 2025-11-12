# app.py (fixed)
import os
from flask import Flask, render_template, request

from pipelines import sentiment, generation  # ← import modules
from pipelines.sentiment import SENTIMENT_MODEL_ID
from pipelines.generation import GEN_MODEL_ID

LANGUAGE = os.getenv("LANGUAGE", "English")
DEPLOY_TARGET = os.getenv("DEPLOY_TARGET", "Local-only")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", language=LANGUAGE)


@app.post("/generate")
def generate():
    user_prompt = (request.form.get("prompt") or "").strip()
    manual_choice = (request.form.get("sentiment") or "auto").lower()
    try:
        target_words = int(request.form.get("length_words") or 120)
    except ValueError:
        target_words = 120

    detected = None
    if manual_choice == "auto":
        det = sentiment.detect_sentiment(user_prompt)  # ← patched target
        chosen_sent = det["label"]
        det_score = det["score"]
        detected = det
    else:
        chosen_sent = manual_choice
        det_score = None

    gen = generation.generate_aligned(  # ← patched target
        prompt=user_prompt,
        sentiment=chosen_sent,
        target_words=target_words,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        language=LANGUAGE,
    )

    return render_template(
        "result.html",
        language=LANGUAGE,
        user_prompt=user_prompt,
        detected_sentiment=chosen_sent,
        detection_score=det_score,
        target_words=target_words,
        generated_text=gen["text"],
        sentiment_model=SENTIMENT_MODEL_ID,
        gen_model=GEN_MODEL_ID,
        deploy_target=DEPLOY_TARGET,
        max_tokens=MAX_OUTPUT_TOKENS,
        # extras for compatibility
        sentiment=chosen_sent,
        detected=detected,
        length_words=target_words,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
