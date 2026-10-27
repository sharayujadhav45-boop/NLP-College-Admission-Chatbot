# AI-based ADCET admission query chatbot

A Streamlit microproject that answers questions about Annasaheb Dange College
of Engineering and Technology (ADCET), Ashta, using a local FAQ dataset and NLP
retrieval. The current dataset is scoped to the 2025–26 admission year where
year-specific information is known. It does not browse websites or generate
answers beyond the stored FAQs.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Run locally

From this project folder, create and activate a virtual environment, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run App.py
```

The FAQ file is loaded from this project folder, so the app does not depend on a
machine-specific absolute path. The FAQ table is not shown in the chatbot
interface. The app includes suggested starter questions and a light/dark theme
configured in `.streamlit/config.toml`.

## FAQ dataset

Edit `clg_admission.csv` to add or update answers. This file is used internally
for retrieval and is not displayed in the chatbot interface. It must contain
these columns:

| Column | Purpose |
| --- | --- |
| `category` | FAQ grouping used to organize the local data |
| `question` | Example wording used for NLP matching |
| `answer` | The response shown to the user |

Each row needs a non-empty question and answer. Do not invent time-sensitive
dates, cutoffs, eligibility, or fees. If verified historical information is not
available, the FAQ says that it is unknown instead of reusing a newer year's
rules.

## How matching works

The chatbot stems words in the dataset questions and the user's query, builds a
TF-IDF index, and chooses the question with the highest cosine similarity. It
returns the linked FAQ answer only when its score is at least `0.25` and the
best match is clearly ahead of the next match. Otherwise, it declines to guess
and gives ADCET's phone number from its published contact details.

The college and contact details were checked against the official ADCET website
on October 11, 2026. That site currently includes newer admission-year content;
those current-year dates, fees, seat counts, and cutoffs have deliberately not
been presented as 2025–26 facts.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
