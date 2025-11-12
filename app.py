"""
app.py
This is the main Flask application.
It shows the homepage, accepts a form, detects or uses a chosen sentiment,
and then asks the generator to create a paragraph.
I added beginner-friendly comments to explain each step.
"""

# ==== Imports ====
import os
from flask import Flask, render_template, request

# I import our own pipeline modules (sentiment + generation).
from pipelines import sentiment, generation
from pipelines.sentiment import SENTIMENT_MODEL_ID
from pipelines.generation import GEN_MODEL_ID


# ==== Configuration from environment ====
# These can be changed without editing code by setting ENV variables.
LANGUAGE = os.getenv("LANGUAGE", "English")
DEPLOY_TARGET = os.getenv("DEPLOY_TARGET", "Local-only")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

# This creates the Flask app object.
app = Flask(__name__)


# ==== Routes ====
@app.get("/")
def index():
    """
    Render the homepage.
    I pass the LANGUAGE variable so the template can show it.
    """
    return render_template("index.html", language=LANGUAGE)


@app.post("/generate")
def generate():
    """
    Handle the form submission.
    Steps I do:
      1) Read the user's prompt and chosen sentiment (or 'auto').
      2) If 'auto', run the sentiment detector; otherwise use manual choice.
      3) Call the generator to produce the paragraph.
      4) Render the result page with all useful info.
    """
    # 1) Read form fields safely (empty strings if missing)
    user_prompt = (request.form.get("prompt") or "").strip()
    manual_choice = (request.form.get("sentiment") or "auto").lower()

    # Convert length to an integer, default to 120 if not valid
    try:
        target_words = int(request.form.get("length_words") or 120)
    except ValueError:
        target_words = 120

    # 2) Decide which sentiment to use:
    detected = None  # I keep this to show extra details on the result page
    if manual_choice == "auto":
        # Use the ML model to detect sentiment from the user's text
        det = sentiment.detect_sentiment(user_prompt)
        chosen_sent = det["label"]
        det_score = det["score"]
        detected = det
    else:
        # Use whatever the user selected manually
        chosen_sent = manual_choice
        det_score = None

    # 3) Generate the paragraph with the chosen sentiment and settings
    gen = generation.generate_aligned(
        prompt=user_prompt,
        sentiment=chosen_sent,
        target_words=target_words,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        language=LANGUAGE,
    )

    # 4) Show the result page with all the info we might want to display
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
        # extras for compatibility with tests/templates
        sentiment=chosen_sent,
        detected=detected,
        length_words=target_words,
    )


# ==== Local run (development) ====
if __name__ == "__main__":
    # I run the app on all interfaces so I can open it from other devices too.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
