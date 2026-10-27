import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = ("category", "question", "answer")
MATCH_THRESHOLD = 0.25
MINIMUM_SCORE_GAP = 0.035
_STOP_WORDS = ENGLISH_STOP_WORDS.difference(
    {"no", "not", "nor", "never", "neither"}
)
_STEMMER = PorterStemmer()


def load_faq_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"FAQ CSV file not found: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError("The FAQ CSV file is empty.")

    try:
        data = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("The FAQ CSV file has no header or records.") from error

    data.columns = data.columns.astype(str).str.strip().str.lower()
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}. "
            f"Found: {data.columns.tolist()}"
        )

    data = data.loc[:, REQUIRED_COLUMNS].fillna("")
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].astype(str).str.strip()
    data = data.loc[
        data["question"].ne("") & data["answer"].ne("")
    ].reset_index(drop=True)

    if data.empty:
        raise ValueError("The FAQ CSV has no rows with both a question and answer.")
    return data


def preprocess(text: str) -> str:
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return " ".join(
        _STEMMER.stem(word)
        for word in words
        if len(word) > 1 and word not in _STOP_WORDS
    )


def build_faq_index(
    questions: Sequence[str],
) -> tuple[TfidfVectorizer, csr_matrix]:
    processed_questions = [preprocess(question) for question in questions]
    if not any(processed_questions):
        raise ValueError("Add FAQ questions containing searchable words.")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, matrix


def get_answer(
    user_question: str,
    faqs: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    faq_matrix: csr_matrix,
) -> tuple[str, float]:
    processed_question = preprocess(user_question)
    if not processed_question:
        return "Please enter a question with searchable words.", 0.0

    user_vector = vectorizer.transform([processed_question])
    scores = cosine_similarity(user_vector, faq_matrix).flatten()
    best_indices = scores.argsort()
    best_index = int(best_indices[-1])
    best_score = float(scores[best_index])

    if best_score < MATCH_THRESHOLD:
        return (
            "I don't have a verified answer to that question in the local "
            "ADCET FAQ. Please rephrase it or call ADCET at +91-8600600700.",
            best_score,
        )
    if len(best_indices) > 1:
        second_best_score = float(scores[best_indices[-2]])
        if best_score - second_best_score < MINIMUM_SCORE_GAP:
            return (
                "I couldn't identify one exact answer in the local ADCET FAQ. "
                "Please rephrase your question or call ADCET at +91-8600600700.",
                best_score,
            )
    return str(faqs.iloc[best_index]["answer"]), best_score
