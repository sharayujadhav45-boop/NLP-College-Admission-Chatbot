import tempfile
import unittest
from pathlib import Path

from chatbot import build_faq_index, get_answer, load_faq_data, preprocess


class ChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.faqs = load_faq_data(
            Path(__file__).resolve().parents[1] / "clg_admission.csv"
        )
        self.vectorizer, self.faq_matrix = build_faq_index(
            tuple(self.faqs["question"])
        )

    def test_preprocess_normalizes_and_stems_words(self) -> None:
        self.assertEqual(preprocess("Admissions, applying!"), "admiss appli")

    def test_returns_answer_for_matching_question(self) -> None:
        answer, score = get_answer(
            "How can I apply?",
            self.faqs,
            self.vectorizer,
            self.faq_matrix,
        )
        self.assertIn("general Maharashtra CAP steps", answer)
        self.assertGreater(score, 0.25)

    def test_rejects_question_without_searchable_words(self) -> None:
        answer, score = get_answer(
            "a!", self.faqs, self.vectorizer, self.faq_matrix
        )
        self.assertIn("Please enter a question", answer)
        self.assertEqual(score, 0.0)

    def test_returns_fallback_for_unrelated_question(self) -> None:
        answer, score = get_answer(
            "How do I repair a bicycle?",
            self.faqs,
            self.vectorizer,
            self.faq_matrix,
        )
        self.assertIn("don't have a verified answer", answer)
        self.assertEqual(score, 0.0)

    def test_course_question_returns_local_branch_list(self) -> None:
        answer, _ = get_answer(
            "What courses are offered at ADCET?",
            self.faqs,
            self.vectorizer,
            self.faq_matrix,
        )
        self.assertIn("Aeronautical Engineering", answer)
        self.assertIn("Food Technology", answer)

    def test_unknown_fee_question_does_not_make_up_an_amount(self) -> None:
        answer, _ = get_answer(
            "What is the tuition fee at ADCET for 2025-26?",
            self.faqs,
            self.vectorizer,
            self.faq_matrix,
        )
        self.assertIn("cannot provide a reliable figure", answer)
        self.assertNotIn("₹", answer)

    def test_loads_and_validates_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "faqs.csv"
            csv_path.write_text(
                " Category , Question , Answer \n"
                'Admission,"How do I apply?","Use the official form, online."\n',
                encoding="utf-8",
            )
            data = load_faq_data(csv_path)
        self.assertEqual(data.columns.tolist(), ["category", "question", "answer"])
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]["answer"], "Use the official form, online.")

    def test_rejects_csv_without_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "faqs.csv"
            csv_path.write_text("question,response\nQ,A\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required CSV columns"):
                load_faq_data(csv_path)


if __name__ == "__main__":
    unittest.main()
