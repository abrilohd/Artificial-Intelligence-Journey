"""A context-aware rule-based chatbot for AI Journey Project 1.

Uses NLTK tokenization and Porter stemming when available, with
lightweight standard-library fallbacks so the bot always runs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import random
import re
from typing import Callable

# ---------------------------------------------------------------------------
# Optional NLTK – fallback stubs keep the type-checker and runtime happy
# ---------------------------------------------------------------------------
try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import wordpunct_tokenize
except ImportError:
    class PorterStemmer:  # type: ignore[no-redef]
        """Minimal fallback: returns the word unchanged."""

        def stem(self, word: str) -> str:
            return word

    def wordpunct_tokenize(text: str) -> list[str]:  # type: ignore[no-redef]
        """Minimal fallback: splits on non-alphanumeric characters."""
        return re.findall(r"[a-z]+", text)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Turn:
    """One meaningful exchange kept for the short conversation summary."""

    topic: str
    user_message: str


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
class TextProcessor:
    """Normalize input and reduce word forms to a common stem."""

    def __init__(self) -> None:
        self._stemmer = PorterStemmer()

    @staticmethod
    def normalize(message: str) -> str:
        """Strip, lowercase, and collapse internal whitespace."""
        return re.sub(r"\s+", " ", message.strip().lower())

    def tokens(self, message: str) -> frozenset[str]:
        """Return a frozenset of stemmed alphabetic tokens."""
        words = wordpunct_tokenize(self.normalize(message))
        return frozenset(self._stemmer.stem(w) for w in words if w.isalpha())

    def stem(self, word: str) -> str:
        return self._stemmer.stem(word)


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------
class RuleBasedChatbot:
    """Recognize conversational intents, retain simple facts, and reply clearly."""

    FALLBACK_RESPONSES = (
        "I do not have a rule for that yet. Type 'help' to see what I understand.",
        "I am still learning that topic. Could you rephrase?",
        "I didn't catch that. Ask me about AI, NLP, the time, or a joke.",
    )
    EXIT_WORDS: frozenset[str] = frozenset({"bye", "goodbye", "exit", "quit"})
    GREETINGS: frozenset[str] = frozenset({"hello", "hi", "hey"})
    POSITIVE_WORDS: frozenset[str] = frozenset({"happy", "great", "good", "excited", "awesome"})
    NEGATIVE_WORDS: frozenset[str] = frozenset({"sad", "bad", "upset", "tired", "worried"})

    def __init__(self) -> None:
        self.text = TextProcessor()
        self.name: str | None = None
        self.favorite_topic: str | None = None
        self.last_topic: str | None = None
        self.history: deque[Turn] = deque(maxlen=6)

        # Ordered pipeline: first match wins
        self._responders: tuple[tuple[str, Callable[[str, frozenset[str]], str | None]], ...] = (
            ("goodbye",        self._respond_to_goodbye),
            ("name",           self._remember_name),
            ("favorite_topic", self._remember_favorite_topic),
            ("recall",         self._recall_saved_fact),
            ("summary",        self._summarize_conversation),
            ("follow_up",      self._respond_to_follow_up),
            ("greeting",       self._respond_to_greeting),
            ("wellbeing",      self._respond_to_wellbeing),
            ("identity",       self._respond_to_identity),
            ("help",           self._respond_to_help),
            ("ai",             self._respond_to_ai),
            ("nlp",            self._respond_to_nlp),
            ("time",           self._respond_to_time),
            ("joke",           self._respond_to_joke),
            ("thanks",         self._respond_to_thanks),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(message: str) -> str:
        """Expose normalization for tests."""
        return TextProcessor.normalize(message)

    def reply(self, message: str) -> str:
        """Return the first matching response and record the turn."""
        normalized = self.text.normalize(message)
        if not normalized:
            return "Please type a message so I can help."

        tokens = self.text.tokens(normalized)
        for topic, responder in self._responders:
            response = responder(normalized, tokens)
            if response:
                self.last_topic = topic
                self.history.append(Turn(topic, normalized))
                return response

        self.last_topic = "unknown"
        self.history.append(Turn("unknown", normalized))
        return random.choice(self.FALLBACK_RESPONSES)

    def should_exit(self, message: str) -> bool:
        """Return True when the user wants to end the session."""
        return (
            bool(self.EXIT_WORDS & self.text.tokens(message))
            or "see you" in self.text.normalize(message)
        )

    # ------------------------------------------------------------------
    # Responders (private)
    # ------------------------------------------------------------------
    def _respond_to_goodbye(self, text: str, _: frozenset[str]) -> str | None:
        if self.should_exit(text):
            suffix = f", {self.name}" if self.name else ""
            return f"Goodbye{suffix}! Keep learning and building things."
        return None

    def _remember_name(self, text: str, _: frozenset[str]) -> str | None:
        match = re.search(r"\b(?:my name is|call me)\s+([a-z][a-z '-]{0,30})", text)
        if not match:
            return None
        self.name = match.group(1).strip(" .!?'-").title()
        return f"Nice to meet you, {self.name}! I'll remember your name for this chat."

    def _remember_favorite_topic(self, text: str, _: frozenset[str]) -> str | None:
        match = re.search(r"\b(?:i like|i love|my favorite topic is)\s+([a-z][a-z -]{1,30})", text)
        if not match:
            return None
        self.favorite_topic = match.group(1).strip(" .!?'-").title()
        return f"Got it, your favorite topic is {self.favorite_topic}."

    def _recall_saved_fact(self, text: str, _: frozenset[str]) -> str | None:
        if "my name" in text and self.name:
            return f"Your name is {self.name}."
        if "i like" in text or "favorite topic" in text or "what do i like" in text:
            if self.favorite_topic:
                return f"You told me you like {self.favorite_topic}."
            return "You haven't told me a favorite topic yet. Try: 'I like NLP.'"
        return None

    def _summarize_conversation(self, text: str, _: frozenset[str]) -> str | None:
        if not re.search(r"\b(history|summary|what (have|did) we (talk\w*|discuss\w*))\b", text):
            return None
        topics = [
            t.topic.replace("_", " ")
            for t in self.history
            if t.topic not in {"unknown", "summary"}
        ]
        if not topics:
            return "We've only just started. Ask me about AI, NLP, or my features."
        return "We have discussed: " + ", ".join(dict.fromkeys(topics)) + "."

    def _respond_to_follow_up(self, text: str, _: frozenset[str]) -> str | None:
        is_follow_up = (
            "tell me more" in text
            or "explain more" in text
            or text in {"more", "what about it"}
        )
        if not is_follow_up:
            return None
        explanations: dict[str, str] = {
            "ai":   "AI includes rule-based programs like me and learning systems that improve from data.",
            "nlp":  "NLP helps software process human language. This bot uses tokenization and stemming to match related words.",
            "help": "A good next step: add a new intent, write a test for it, then run the chatbot again.",
        }
        return explanations.get(
            self.last_topic or "",
            "Pick a topic first. Try asking: What is AI? or What is NLP?",
        )

    def _respond_to_greeting(self, _: str, tokens: frozenset[str]) -> str | None:
        if self.GREETINGS & tokens:
            greeting = f"Hello{', ' + self.name if self.name else ''}!"
            return f"{greeting} What would you like to explore today?"
        return None

    def _respond_to_wellbeing(self, text: str, tokens: frozenset[str]) -> str | None:
        if "how are you" in text or "how're you" in text:
            return "I'm ready to help! What are you learning today?"
        if self.NEGATIVE_WORDS & tokens:
            return "That sounds tough. A small next step can help. Want to ask an AI question?"
        if self.POSITIVE_WORDS & tokens:
            return "Great to hear! Let's use that energy to learn something new."
        return None

    @staticmethod
    def _respond_to_identity(text: str, _: frozenset[str]) -> str | None:
        if "your name" in text or "who are you" in text:
            return "I'm AI Buddy, a rule-based chatbot built with Python."
        return None

    @staticmethod
    def _respond_to_help(text: str, tokens: frozenset[str]) -> str | None:
        if "what can you do" in text or {"help", "command"} & tokens:
            return (
                "I can discuss AI and NLP, remember your name and favorite topic, "
                "summarize our chat, tell the time, and tell a joke. "
                "I can also answer follow-up questions. Just ask!"
            )
        return None

    @staticmethod
    def _respond_to_ai(text: str, tokens: frozenset[str]) -> str | None:
        if (
            "what is ai" in text
            or "artificial intelligence" in text
            or {"ai", "machine", "learn"} <= tokens
        ):
            return "AI is the field of building systems that can reason, learn from data, or act toward a goal."
        return None

    @staticmethod
    def _respond_to_nlp(text: str, tokens: frozenset[str]) -> str | None:
        if (
            "what is nlp" in text
            or "natural language processing" in text
            or "nlp" in tokens
        ):
            return (
                "NLP stands for Natural Language Processing, which is a set of techniques "
                "that help computers read, analyze, and generate human language."
            )
        return None

    @staticmethod
    def _respond_to_time(_: str, tokens: frozenset[str]) -> str | None:
        if "time" in tokens:
            return f"The local time is {datetime.now().strftime('%H:%M')}."
        return None

    def _respond_to_joke(self, _: str, tokens: frozenset[str]) -> str | None:
        joke_stems = {self.text.stem("joke"), self.text.stem("funny")}
        if joke_stems & tokens:
            return "Why did the robot go on holiday? It needed to recharge its batteries."
        return None

    @staticmethod
    def _respond_to_thanks(_: str, tokens: frozenset[str]) -> str | None:
        if {"thank", "thanks"} & tokens:
            return "You're welcome! Anything else I can help with?"
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_chat() -> None:
    """Start an interactive terminal conversation."""
    bot = RuleBasedChatbot()
    print("AI Buddy: Hi! I am AI Buddy. Type 'help' for ideas, or 'bye' to leave.")

    while True:
        try:
            message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAI Buddy: Goodbye! Keep building and learning.")
            break

        print(f"AI Buddy: {bot.reply(message)}")
        if bot.should_exit(message):
            break


if __name__ == "__main__":
    run_chat()
