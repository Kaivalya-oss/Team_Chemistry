# Football Team Chemistry Simulator ⚽📊

Interactive dashboard to simulate player transfers/replacements and predict impact on team chemistry using FIFA data + machine learning.

## Features
- Select any team from the dataset
- Replace any current player with a new one
- Instant before/after chemistry score (0–100)
- Radar chart showing changes in key chemistry drivers
- Clean, modern UI with animations

## Tech Stack
- Backend: Python (pandas, scikit-learn, tuned Gradient Boosting)
- Frontend: HTML + CSS + JavaScript + Chart.js
- Model: Equal-weight index + tree-based predictor

## How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run Flask backend: `python app.py`
4. Open `index.html` in browser (or serve via Flask)

## Live Demo
(Add link later when hosted)

## Data
- FIFA player stats (anonymized/cleaned)
- Aggregated team features + chemistry index

Made by Kaivalya & Sahil.