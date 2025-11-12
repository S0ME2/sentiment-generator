"""
tests/test_api.py
These are simple tests for our Flask app.
I try to check that the homepage loads and that the /generate route
works correctly when we mock the ML functions.
"""

# ==== Imports ====
import os
import sys
from pathlib import Path
from unittest.mock import patch

# ==== Make sure we can import the app ====
# I get the project root (one level above /tests)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# If Python can't see the root yet, I add it to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# I set the Flask environment to "testing" by default
os.environ.setdefault("FLASK_ENV", "testing")

# Now I can import the Flask app module
import app


# ==== Tests ====
def test_index():
    """
    Just check the homepage renders and has the title text.
    """
    client = app.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"Sentiment-Aligned Generator" in r.data


def test_generate_auto_mocked():
    """
    Here I mock both:
      - sentiment.detect_sentiment (so it returns positive)
      - generation.generate_aligned (so it returns a fixed text)
    Then I call /generate with 'auto' so the app uses the mocked detector.
    """
    client = app.app.test_client()

    # I use 'patch' to replace real functions with fake ones for this test only
    with (
        patch("pipelines.sentiment.detect_sentiment") as mock_detect,
        patch("pipelines.generation.generate_aligned") as mock_gen,
    ):
        # Fake outputs that the app will use
        mock_detect.return_value = {"label": "positive", "score": 0.987}
        mock_gen.return_value = {
            "prompt": "Write an uplifting... (mocked)",
            "text": "This is a mocked positive paragraph.",
        }

        # I send a POST request like the form submission in the UI
        r = client.post(
            "/generate",
            data={
                "prompt": "Great service!",
                "sentiment": "auto",  # app should call detect_sentiment
                "length_words": "80",
            },
        )

        # The page should load and show the results
        assert r.status_code == 200
        body = r.data.decode("utf-8")  # convert bytes to string so I can search
        assert "Generated Output" in body
        assert "positive" in body
        assert "mocked positive paragraph" in body.lower()


def test_generate_manual_negative_mocked():
    """
    Here I only mock the generator. I set sentiment='negative' manually,
    so the app should NOT call the detector.
    """
    client = app.app.test_client()

    with patch("pipelines.generation.generate_aligned") as mock_gen:
        # The generator returns a fake negative paragraph
        mock_gen.return_value = {
            "prompt": "Write a critical... (mocked)",
            "text": "This is a mocked negative paragraph.",
        }

        # I submit a prompt with manual negative sentiment
        r = client.post(
            "/generate",
            data={
                "prompt": "The device failed.",
                "sentiment": "negative",
                "length_words": "120",
            },
        )

        # The response should include 'negative' and the mocked text
        assert r.status_code == 200
        body = r.data.decode("utf-8")
        assert "negative" in body
        assert "mocked negative paragraph" in body.lower()
