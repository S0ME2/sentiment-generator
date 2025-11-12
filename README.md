# Sentiment-Aligned AI Text Generator (Flask)

Generate coherent paragraphs that **match the sentiment** of a user’s prompt.  
The app analyzes the input for **toxicity/threats** and **overall sentiment**, then produces text aligned with the detected (or chosen) tone—via a clean Flask UI.

---

## 🎯 Task & Objectives

**Task:** Build an AI Text Generator

**Objective:** Create an AI system that generates a paragraph or short essay based on the prompt’s sentiment (e.g., positive, negative, neutral). The system should **understand** the prompt’s sentiment and **produce aligned text**.

**Requirements**
- Implement sentiment analysis on input prompts using Python and relevant ML frameworks.
- Develop a text generation model that produces sentiment-aligned outputs.
- Build an interactive frontend where users can enter prompts and receive generated texts.
- Provide clear documentation outlining methodology, dataset(s) used, and project challenges.

**Scope of Work**
- Use pre-trained or custom sentiment analysis models to classify input text sentiment.
- Employ text generation models to generate coherent paragraphs matching detected sentiment.
- Implement a frontend interface using Flask (this project), Streamlit, or React.
- Optional enhancements include manual sentiment selection and adjustable length.

**Deliverables**
- Complete source code with clear organization and comments.
- Functional frontend demonstrating AI text generation based on sentiment.
- Project documentation including setup instructions and project explanation.
- *(Optional)* Deployment link to a live demo.  
*(This project is not hosted yet.)*

---

## 🧠 How It Works (Technical Overview)

- **Toxicity Gate (safety first)**  
  The app first checks for threats/toxicity (e.g., *“I will kill you”*) using a multi-label classifier. If toxic, it **forces a negative label** (or you can easily configure a refusal).

- **Sentiment Detection**  
  For non-toxic text, it predicts overall sentiment (positive / neutral / negative) using a robust RoBERTa-based classifier.

- **Text Generation**  
  Uses an **instruction-tuned** model to write a coherent paragraph aligned with the resolved tone. Defaults to a small, CPU-friendly model.

- **Frontend**  
  Flask + Jinja templates. The output area is styled to **wrap long text** so nothing overflows.

### Models (pre-trained)
- **Toxicity:** `unitary/unbiased-toxic-roberta`  
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
- **Generator (default):** `google/flan-t5-small`  
  *(If unavailable, the app cleanly falls back, but FLAN-T5 is recommended for coherent results.)*

> **Datasets:** This project relies on **pre-trained models** from Hugging Face and does **not** use a local dataset.

---

## 📁 Project Structure

```
.
├─ app.py                     # Flask app (routes, form handling, rendering)
├─ pipelines/
│  ├─ sentiment.py            # toxicity gate + sentiment detection (lazy load)
│  └─ generation.py           # instruction-tuned generation (lazy load, fallback)
├─ templates/
│  ├─ index.html              # input form (prompt, sentiment, length)
│  └─ result.html             # "Generated Output" page
├─ static/
│  └─ styles.css              # modern dark theme, proper text wrapping
├─ tests/
│  └─ test_api.py             # basic API/UI tests (pytest)
└─ requirements.txt
```

---

## 🚀 Quickstart — Steps 1 to 5

> The commands below are **fish-shell friendly**. For **bash/zsh**, see the notes under each step.

### 1) Set up your workspace
Make sure you have the files listed above in place (already included in the project).  
No edits needed—just confirm the structure.

---

### 2) Create a virtual environment & install dependencies
**fish**
```fish
python3 -m venv .venv
source .venv/bin/activate.fish
pip install -U pip
pip install -r requirements.txt
```

**bash/zsh**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

### 3) Download the models once (for faster/offline runs)
Run each line as a single command.

```fish
# Generator (instruction-tuned)
python -c 'from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as M; m="google/flan-t5-small"; AutoTokenizer.from_pretrained(m); M.from_pretrained(m); print("OK", m)'

# Toxicity gate
python -c 'from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; m="unitary/unbiased-toxic-roberta"; AutoTokenizer.from_pretrained(m); M.from_pretrained(m); print("OK", m)'

# Sentiment
python -c 'from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; m="cardiffnlp/twitter-roberta-base-sentiment-latest"; AutoTokenizer.from_pretrained(m); M.from_pretrained(m); print("OK", m)'
```

*(Same commands work in bash/zsh.)*

---

### 4) Run the tests
Ensure the app and templates are wired correctly.

```fish
env GEN_MODEL=google/flan-t5-small pytest -q
```

*(Same in bash/zsh.)*  
You should see all tests **pass**.

---

### 5) Run the app
Start the Flask server with recommended environment values:

```fish
env GEN_MODEL=google/flan-t5-small     SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest     TOXICITY_MODEL=unitary/unbiased-toxic-roberta     TOX_THRESHOLD=0.5     python app.py
```

Open: **http://localhost:5000**

- Type a prompt.
- Leave **Sentiment = Auto** to detect it automatically (with safety).
- Or choose **Positive / Neutral / Negative** manually.
- Adjust **Length (words)** as needed.

> **Tip (port busy after Ctrl+Z):**  
> Bring the job back with `fg %1` and stop with `Ctrl+C`, or free the port directly:  
> `fuser -k 5000/tcp`

---

## ⚙️ Configuration (Environment Variables)

| Variable           | Default                                | Purpose |
|--------------------|----------------------------------------|---------|
| `GEN_MODEL`        | `google/flan-t5-small`                 | Generation model ID (Hugging Face) |
| `SENTIMENT_MODEL`  | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment classifier |
| `TOXICITY_MODEL`   | `unitary/unbiased-toxic-roberta`       | Toxicity/threat classifier |
| `TOX_THRESHOLD`    | `0.5`                                  | Toxicity threshold to force “negative” |

All have sensible defaults; Step **5** shows how to set them explicitly when launching.

---

## 🧩 Notes & Challenges

- **Coherence on CPU:** Instruction-tuned seq2seq models (FLAN-T5) produce better paragraphs than tiny causal LMs, even for short prompts.
- **Safety on short/violent inputs:** A toxicity gate prevents mismatched “positive” generations for harmful prompts.
- **Clean tests on Python 3.12:** Transformers are lazy-loaded and benign SWIG deprecations are filtered so `pytest` stays quiet.

---

**You’re ready to go.** Complete Steps **1 → 5**, then visit **http://localhost:5000** and start generating sentiment-aligned text.