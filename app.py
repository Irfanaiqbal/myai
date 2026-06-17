"""
OpenRouter Free-Model Chat UI
-----------------------------
A small Flask app that gives you a clean web chat interface backed by
OpenRouter's free models. Streams responses token-by-token, supports
multiple chat sessions, system prompts, temperature control, and a
custom-model-id override.

SETUP:
1. pip install -r requirements.txt
2. Get a free API key from https://openrouter.ai/keys  (no card needed)
3. Set it as an environment variable:
       Linux/Mac:   export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxx"
       Windows cmd: set OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
4. Run: python app.py
5. Open http://127.0.0.1:5000 in your browser
"""

import os
import json
import requests
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Curated free OpenRouter models (June 2026), grouped for the dropdown.
# Free models change over time -- edit freely, just keep the ":free" suffix.
# Don't have a model listed here? Use the "Custom model ID" field in the UI.
MODEL_GROUPS = [
    {
        "label": "General Purpose",
        "models": [
            {"id": "openai/gpt-oss-120b:free", "name": "GPT-OSS 120B (best all-rounder)"},
            {"id": "z-ai/glm-4.5-air:free", "name": "GLM-4.5 Air (fast & light)"},
            {"id": "google/gemma-4-31b-it:free", "name": "Gemma 4 31B (multimodal)"},
            {"id": "openrouter/free", "name": "Auto-router (let OpenRouter pick)"},
        ],
    },
    {
        "label": "Coding & Agentic",
        "models": [
            {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron 3 Super 120B (1M ctx, heavy tasks)"},
            {"id": "moonshotai/kimi-k2.6:free", "name": "Kimi K2.6 (coding / agentic)"},
        ],
    },
    {
        "label": "Unmoderated",
        "models": [
            {"id": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "name": "Venice: Uncensored (free)"},
        ],
    },
]

# Flat lookup, useful for validation / friendly-name display.
ALL_MODELS = {m["id"]: m["name"] for g in MODEL_GROUPS for m in g["models"]}


@app.route("/")
def index():
    return render_template("index.html", model_groups=MODEL_GROUPS)


@app.route("/api/models")
def api_models():
    return jsonify(MODEL_GROUPS)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY is not set on the server."}), 500

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    model = (data.get("model") or "").strip() or list(ALL_MODELS.keys())[0]
    temperature = data.get("temperature", 0.7)

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    temperature = max(0.0, min(2.0, temperature))

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for dashboard/rankings.
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Local OpenRouter Chat",
    }

    def generate():
        try:
            with requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            ) as r:
                # requests defaults text/event-stream to ISO-8859-1 when no
                # charset is declared in the header, which mangles emojis,
                # em-dashes, and smart quotes. Force UTF-8 explicitly.
                r.encoding = "utf-8"

                if r.status_code != 200:
                    err_text = r.text
                    yield f"data: {json.dumps({'error': f'OpenRouter error {r.status_code}: {err_text}'})}\n\n"
                    return

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]
                    if chunk.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        parsed = json.loads(chunk)
                        delta = parsed["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield f"data: {json.dumps({'content': delta})}\n\n"
        except requests.exceptions.RequestException as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("\n  WARNING: OPENROUTER_API_KEY environment variable is not set.")
        print("  The app will load but chat requests will fail until you set it.\n")
    app.run(debug=True, port=5000, threaded=True)
