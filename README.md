# PUBG Win Placement Predictor

A machine learning system that predicts a player's finish placement in PUBG matches,
built end-to-end over 13 days — from raw Kaggle data to a live deployed web application.

**Live Demo:** [pubg-win-predictor.streamlit.app](https://pubg-win-prediction.streamlit.app/)

---

## Results

| Day | Change | MAE |
|-----|--------|-----|
| Day 1 | Baseline LightGBM, 5 features | 0.10000 |
| Day 2 | Feature engineering, 105 features | 0.05000 |
| Day 3 | Optuna hyperparameter tuning | 0.05297 |
| Day 4 | Cheater detection + new features | 0.05296 |
| Day 5 | SHAP-guided feature surgery | 0.05130 |

Final MAE of 0.05130 — the model predicts player placement within 5% on average across
over a million matches.

---

## Project Overview

The goal was to predict `winPlacePerc` — a continuous value between 0 and 1 representing
where a player finished in a match relative to all other players. A value of 1.0 means
first place. A value of 0.0 means last.

The dataset contains 4.4 million rows from real PUBG matches, with 29 columns covering
combat stats, movement, healing, and match metadata. The final model uses 105 engineered
features and was trained on a 25% sample (~1.1 million rows).

---

## What Was Built

### Week 1 — Core ML
Started with a 5-feature baseline and iterated to 105 features through careful engineering.
Features fall into three categories: individual player stats, group-level aggregates
(team kills, team damage, best killer in team), and match-level context (kill rank within
match, damage vs match average). SHAP analysis guided which features to keep and which
to drop. Cheater detection was added using rule-based flags and isolation forest to remove
anomalous rows that would otherwise corrupt training.

### Week 2 — Visualization and Documentation
Built a scouting report system and visualization suite using matplotlib and seaborn.
Set up the GitHub repository with structured documentation.

### Week 3 — Web Application
Built a FastAPI REST backend with Pydantic validation, then a Streamlit frontend with
interactive sliders. Containerized both services with Docker so the full stack runs with
a single `docker compose up`. Deployed the app to Streamlit Cloud with automatic model
loading from Google Drive using gdown.

### Week 4 — Player Lookup and Final Polish
The manual input form had a known limitation: 45 of the 105 features require data from
every other player in the match, which you cannot type in manually. These were filled
with zero, pushing predictions toward 0.4-0.6 regardless of input.

The fix was a Player Lookup system. Users enter a real groupId from the Kaggle dataset.
The app retrieves all players from that group, computes all 105 features using full match
context, and runs the model with complete information. This is the same offline-precompute,
online-serve pattern used in production ML systems.

A playstyle classifier was also added — rule-based classification of each player as
Aggressive, Passive, Sniper, or Balanced based on kills, damage, walk distance, and
headshot rate.

---

## The Key Engineering Decision

Feature engineering contributed more to model performance than any hyperparameter tuning.
Going from 5 to 105 features cut MAE from 0.10 to 0.05. Optuna tuning on top of that
moved it by less than 0.001. The lesson: better features beat better hyperparameters.

The rank-within-match features were the most predictive — knowing that a player ranked
in the top 10% for damage in their specific match is far more informative than knowing
their raw damage number, which varies wildly by match type and lobby size.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | LightGBM, Scikit-learn, Optuna, SHAP |
| Data | Pandas, NumPy |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Infrastructure | Docker, Docker Compose |
| Deployment | Streamlit Cloud, Google Drive |

---

## Running Locally

**Streamlit only:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Full stack with Docker:**
```bash
docker compose up
```

Frontend runs at `http://localhost:8501`.
API documentation at `http://localhost:8000/docs`.

---

## Project Structure
```
pubg-win-prediction/
├── streamlit_app.py          # Main app (Streamlit Cloud deployment)
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── models/
│   ├── pubg_model_v5.pkl
│   └── day5_feature_cols.pkl
├── day8_fastapi/
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       └── model_loader.py
└── day9_streamlit/
    └── streamlit_app.py
```

---

## Author

Nitya Thaker

This was a structured 13-day project with one clear goal per day — no shortcuts,
no pre-built templates. Every component was written, debugged, and deployed from scratch.