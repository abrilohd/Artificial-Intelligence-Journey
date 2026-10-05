# Project 1. Simple Rule-Based Chatbot

This beginner AI project is an interactive chatbot built with Python. Instead of a machine-learning model, it follows predefined rules to recognize a user's intent and select a response. It also uses **NLTK tokenization and stemming** to recognize related word forms, such as `funny` and `funniest`.

## What it demonstrates

- `if` / `elif` decision-making
- keyword and regular-expression matching
- NLTK tokenization and Porter stemming, with a safe standard-library fallback
- a continuous conversation loop
- context-aware follow-up responses such as `tell me more`
- short-term conversation history and a `what have we discussed?` summary
- simple session memory for a name and favorite topic
- graceful handling of blank, unknown, and exit messages

## Run it

### In Jupyter

Open `main.ipynb`, run every cell, and chat in the final code cell.

### In a terminal

```powershell
python chatbot.py
```

Type `help` for example messages and `bye` to end the conversation. The project already declares NLTK in `pyproject.toml`; install project dependencies before running it if needed.

## Example

```text
AI Buddy: Hi! I am AI Buddy. Type 'help' for ideas, or 'bye' to leave.
You: My name is Hana
AI Buddy: Nice to meet you, Hana! I'll remember your name for this chat.
You: What is NLP?
AI Buddy: NLP means Natural Language Processing: techniques that help computers read, analyze, and generate language.
You: Tell me more
AI Buddy: NLP helps software process human language. This bot uses tokenization and stemming to match related words.
You: What have we discussed?
AI Buddy: We have discussed: name, nlp.
You: bye
AI Buddy: Goodbye, Hana! Keep building and learning.
```

## Future improvements

1. Persist remembered details and history in SQLite.
2. Add a Streamlit interface with a chat transcript.
3. Replace manual rules with an intent-classification model trained on example phrases.
4. Add API-backed capabilities, such as live weather, only after securely managing API keys.
