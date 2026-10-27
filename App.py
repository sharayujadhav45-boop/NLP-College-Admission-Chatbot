from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from chatbot import build_faq_index, get_answer, load_faq_data


st.set_page_config(
    page_title="ADCET Admissions Assistant",
    page_icon="🎓",
    layout="centered",
)

DATA_PATH = Path(__file__).resolve().parent / "clg_admission.csv"
SUGGESTED_QUESTIONS = {
    "How do I apply?": "How do I apply for engineering admission?",
    "Explore branches": "What engineering branches are available at ADCET?",
    "Prepare documents": "Which documents are needed for engineering admission?",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_faq_data(DATA_PATH)


@st.cache_resource
def create_model(questions: tuple[str, ...]) -> tuple[TfidfVectorizer, csr_matrix]:
    return build_faq_index(questions)


try:
    df = load_data()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load the FAQ dataset: {error}")
    st.stop()

try:
    vectorizer, faq_matrix = create_model(tuple(df["question"]))
except ValueError as error:
    st.error(f"Unable to build the FAQ search model: {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🎓 ADCET")
    st.caption("ADMISSIONS ASSISTANT")
    st.divider()
    st.markdown("#### How I can help")
    st.write("Ask about admissions, courses, eligibility, fees, or documents.")
    st.info(
        "Answers come from the local college FAQ. Dates and fees are not "
        "guessed when verified details are unavailable."
    )
    st.divider()
    st.markdown("#### Need to confirm something?")
    st.write("Contact ADCET: **+91-8600600700**")
    if st.button("Clear conversation", width="stretch", type="secondary"):
        st.session_state.messages = []
        if "starter_question" in st.session_state:
            st.session_state.starter_question = None
        st.rerun()

st.caption("ANNASAHEB DANGE COLLEGE OF ENGINEERING AND TECHNOLOGY · ASHTA")
st.title("Your admission questions, answered.")
st.markdown(
    "A clear starting point for your ADCET admission journey. "
    "Ask a question below or choose a suggested topic."
)
st.caption("LOCAL FAQ  ·  2025–26 ADMISSION INFORMATION")
st.divider()

if not st.session_state.messages:
    st.markdown("**Start with a popular question**")
    selected_question = st.pills(
        "Suggested questions",
        options=list(SUGGESTED_QUESTIONS),
        label_visibility="collapsed",
        key="starter_question",
    )
else:
    selected_question = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input(
    "Ask about admissions, courses, eligibility, fees, or documents..."
)
question = user_question or SUGGESTED_QUESTIONS.get(selected_question)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    answer, _score = get_answer(
        question,
        df,
        vectorizer,
        faq_matrix,
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.rerun()

with st.expander("About this assistant"):
    st.write(
        """
        This assistant searches a local FAQ using text matching. It does not
        browse the web or generate facts beyond the stored answers.

        Dates, fees, cutoffs, and other year-specific details are included only
        when verified for 2025–26. If a detail is missing from the local FAQ,
        the assistant says so instead of guessing.
        """
    )
    