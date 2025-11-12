import os
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("FLASK_ENV", "testing")

import app


def test_index():
    client = app.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"Sentiment-Aligned Generator" in r.data


def test_generate_auto_mocked():
    client = app.app.test_client()
    with (
        patch("pipelines.sentiment.detect_sentiment") as mock_detect,
        patch("pipelines.generation.generate_aligned") as mock_gen,
    ):
        mock_detect.return_value = {"label": "positive", "score": 0.987}
        mock_gen.return_value = {
            "prompt": "Write an uplifting... (mocked)",
            "text": "This is a mocked positive paragraph.",
        }
        r = client.post(
            "/generate",
            data={
                "prompt": "Great service!",
                "sentiment": "auto",
                "length_words": "80",
            },
        )
        assert r.status_code == 200
        body = r.data.decode("utf-8")
        assert "Generated Output" in body
        assert "positive" in body
        assert "mocked positive paragraph" in body.lower()


def test_generate_manual_negative_mocked():
    client = app.app.test_client()
    with patch("pipelines.generation.generate_aligned") as mock_gen:
        mock_gen.return_value = {
            "prompt": "Write a critical... (mocked)",
            "text": "This is a mocked negative paragraph.",
        }
        r = client.post(
            "/generate",
            data={
                "prompt": "The device failed.",
                "sentiment": "negative",
                "length_words": "120",
            },
        )
        assert r.status_code == 200
        body = r.data.decode("utf-8")
        assert "negative" in body
        assert "mocked negative paragraph" in body.lower()
