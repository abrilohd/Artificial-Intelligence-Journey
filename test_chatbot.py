"""Small standard-library regression tests for Project 1."""

import unittest

from chatbot import RuleBasedChatbot


class RuleBasedChatbotTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(RuleBasedChatbot.normalize("  HELLO   AI  "), "hello ai")

    def test_name_is_remembered(self) -> None:
        bot = RuleBasedChatbot()
        self.assertIn("Ada", bot.reply("My name is Ada"))
        self.assertIn("Ada", bot.reply("hello"))

    def test_ai_intent(self) -> None:
        self.assertIn("AI is", RuleBasedChatbot().reply("What is AI?"))

    def test_stemmed_word_matches_an_intent(self) -> None:
        self.assertIn("recharge", RuleBasedChatbot().reply("Tell me something funny"))

    def test_contextual_follow_up(self) -> None:
        bot = RuleBasedChatbot()
        bot.reply("What is NLP?")
        self.assertIn("tokenization", bot.reply("Tell me more"))

    def test_saved_favorite_topic(self) -> None:
        bot = RuleBasedChatbot()
        bot.reply("I like machine learning")
        self.assertIn("Machine Learning", bot.reply("What do I like?"))

    def test_conversation_summary(self) -> None:
        bot = RuleBasedChatbot()
        bot.reply("hello")
        bot.reply("What is AI?")
        self.assertIn("ai", bot.reply("What have we discussed?"))

    def test_exit_intent(self) -> None:
        bot = RuleBasedChatbot()
        self.assertTrue(bot.should_exit("Bye"))
        self.assertFalse(bot.should_exit("Hello"))


if __name__ == "__main__":
    unittest.main()
