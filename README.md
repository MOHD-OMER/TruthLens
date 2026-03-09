# TruthLens — AI Fake News Detection System

A production-grade web application for fake news detection using a CNN-LSTM hybrid deep learning model.

## Features

- **ML Detection** — CNN-LSTM hybrid model with ~94% accuracy
- **Analytics Dashboard** — Real-time charts: detection split, 7-day trend
- **Detection History** — Paginated log of all predictions with IP, confidence, timestamp
- **REST API** — `POST /api/predict` JSON endpoint for programmatic access
- **SQLite Database** — Persistent storage of all predictions via SQLAlchemy ORM
- **Admin Auth** — Session-based login for dashboard access
- **Professional UI** — Dark editorial design, animated confidence meter

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model files in ./models/
#    - CNN_LSTM_Hybrid_model.h5
#    - tokenizer.pkl

# 4. Run
python app.py
```

Open http://localhost:5000

## API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover breakthrough cancer treatment..."}'
```

Response:
```json
{
  "prediction": "Real News",
  "confidence": 91.3,
  "raw_score": 0.913,
  "id": 42
}
```

## Admin Login

Default credentials: `admin` / `admin123`  
Change in `app.py` before deployment.

## Project Structure

```
FakeNewsDetection/
├── app.py              # Main Flask app, routes, DB models
├── requirements.txt
├── models/
│   ├── CNN_LSTM_Hybrid_model.h5
│   └── tokenizer.pkl
├── templates/
│   ├── base.html       # Shared layout, nav, footer
│   ├── home.html       # Landing + prediction form
│   ├── result.html     # Animated result with confidence meter
│   ├── dashboard.html  # Analytics with Chart.js
│   ├── history.html    # Paginated prediction log
│   ├── about.html      # Model architecture details
│   └── login.html
└── static/             # CSS/JS if externalized
```

## Tech Stack

| Layer      | Technology              |
|------------|-------------------------|
| Backend    | Flask, SQLAlchemy       |
| Database   | SQLite                  |
| ML         | TensorFlow/Keras        |
| NLP        | NLTK                    |
| Frontend   | Jinja2, Chart.js        |
| Auth       | Flask sessions          |
