import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def setUp(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "App.py"
        self.app = AppTest.from_file(str(app_path), default_timeout=15).run()

    def test_app_starts_without_exceptions(self) -> None:
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(len(self.app.dataframe), 0)
        self.assertEqual(len(self.app.selectbox), 0)

    def test_chat_returns_matching_faq_answer(self) -> None:
        self.app.chat_input[0].set_value(
            "How can I apply for admission?"
        ).run()

        self.assertFalse(self.app.exception)
        answer_text = [element.value for element in self.app.markdown]
        self.assertTrue(
            any("general Maharashtra CAP steps" in text for text in answer_text)
        )

    def test_suggested_question_returns_matching_faq_answer(self) -> None:
        self.app.pills[0].set_value("How do I apply?").run()

        self.assertFalse(self.app.exception)
        answer_text = [element.value for element in self.app.markdown]
        self.assertTrue(
            any("general Maharashtra CAP steps" in text for text in answer_text)
        )


if __name__ == "__main__":
    unittest.main()
