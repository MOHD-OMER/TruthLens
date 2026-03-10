import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import re
import json
import functools

# ── Load .env for local development ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Optional TensorFlow ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not available. Running in demo mode.")

# ── Optional NLTK ──────────────────────────────────────────────────────────────
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# ── Optional Gemini ────────────────────────────────────────────────────────────
try:
    import google.genai as genai
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        GEMINI_AVAILABLE = True
        print("✓ Gemini AI loaded successfully")
    else:
        GEMINI_AVAILABLE = False
        print("⚠ GEMINI_API_KEY not set. Gemini disabled.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ google-generativeai not installed. Gemini disabled.")

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-in-prod')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fakenews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ─── DB Models ────────────────────────────────────────────────────────────────

class Prediction(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    input_text   = db.Column(db.Text, nullable=False)
    result       = db.Column(db.String(20), nullable=False)
    confidence   = db.Column(db.Float, nullable=False)
    gemini_verdict  = db.Column(db.String(20))
    gemini_reason   = db.Column(db.String(300))
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow)
    user_ip      = db.Column(db.String(45))

class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    is_admin   = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ─── Load ML Assets ───────────────────────────────────────────────────────────

model, tokenizer = None, None

def load_ml_assets():
    global model, tokenizer
    if not TF_AVAILABLE:
        return
    try:
        model = load_model('./models/CNN_LSTM_Hybrid_model.h5')
        with open('./models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("✓ ML model and tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠ ML assets not found: {e}. Running in demo mode.")

# ─── NLP Preprocessing ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.strip()
    if NLTK_AVAILABLE:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)
    return text

# ─── CNN-LSTM Prediction ──────────────────────────────────────────────────────

def run_prediction(text: str):
    if model is None or tokenizer is None or not TF_AVAILABLE:
        import random
        score = random.random()
        label = "Real News" if score >= 0.5 else "Fake News"
        conf  = round(abs(score - 0.5) * 200, 2)
        return label, conf, round(score, 4)
    cleaned = preprocess_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=100, padding='post')
    score   = float(model.predict(padded, verbose=0)[0][0])
    label   = "Real News" if score >= 0.5 else "Fake News"
    conf    = score if score >= 0.5 else (1 - score)
    return label, round(conf * 100, 2), round(score, 4)

# ─── Gemini Analysis ──────────────────────────────────────────────────────────

def run_gemini_analysis(text: str):
    """Returns (verdict, one_line_reason) or (None, None) if unavailable."""
    if not GEMINI_AVAILABLE:
        return None, None
    try:
        prompt = f"""Analyze this news text and determine if it is FAKE or REAL news.
Reply in this exact format on two lines:
VERDICT: FAKE or REAL
REASON: One sentence explanation (max 20 words)

News text: {text[:500]}"""
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        lines = response.text.strip().split('\n')
        verdict = None
        reason  = None
        for line in lines:
            if line.startswith('VERDICT:'):
                v = line.replace('VERDICT:', '').strip().upper()
                verdict = 'Fake News' if 'FAKE' in v else 'Real News'
            if line.startswith('REASON:'):
                reason = line.replace('REASON:', '').strip()
        return verdict, reason
    except Exception as e:
        print(f"⚠ Gemini error: {e}")
        return None, None

# ─── Auth Helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text or len(text) < 10:
        flash('Please enter at least 10 characters of text.', 'warning')
        return redirect(url_for('home'))

    label, confidence, raw_score = run_prediction(text)
    gemini_verdict, gemini_reason = run_gemini_analysis(text)

    # Final verdict — if both agree use that, else show both
    if gemini_verdict and gemini_verdict != label:
        final_verdict = "Uncertain"
    else:
        final_verdict = label

    record = Prediction(
        input_text=text[:2000],
        result=label,
        confidence=confidence,
        gemini_verdict=gemini_verdict,
        gemini_reason=gemini_reason,
        user_ip=request.remote_addr
    )
    db.session.add(record)
    db.session.commit()

    return render_template('result.html',
                           prediction=label,
                           confidence=confidence,
                           raw_score=raw_score,
                           text_snippet=text[:300],
                           gemini_verdict=gemini_verdict,
                           gemini_reason=gemini_reason,
                           final_verdict=final_verdict)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'text field is required'}), 400

    label, confidence, raw_score = run_prediction(text)
    gemini_verdict, gemini_reason = run_gemini_analysis(text)

    record = Prediction(
        input_text=text[:2000],
        result=label,
        confidence=confidence,
        gemini_verdict=gemini_verdict,
        gemini_reason=gemini_reason,
        user_ip=request.remote_addr
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        'cnn_lstm_prediction': label,
        'confidence': confidence,
        'raw_score': raw_score,
        'gemini_verdict': gemini_verdict,
        'gemini_reason': gemini_reason,
        'id': record.id
    })

@app.route('/history')
def history():
    page    = request.args.get('page', 1, type=int)
    records = Prediction.query.order_by(Prediction.timestamp.desc()).paginate(page=page, per_page=20)
    return render_template('history.html', records=records)

@app.route('/dashboard')
def dashboard():
    total  = Prediction.query.count()
    fake   = Prediction.query.filter_by(result='Fake News').count()
    real   = Prediction.query.filter_by(result='Real News').count()
    recent = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    from sqlalchemy import func
    trend = db.session.query(
        func.date(Prediction.timestamp).label('day'),
        func.count(Prediction.id).label('count')
    ).group_by('day').order_by('day').limit(7).all()
    avg_conf = db.session.query(func.avg(Prediction.confidence)).scalar() or 0
    stats = {
        'total': total, 'fake': fake, 'real': real,
        'fake_pct': round(fake / total * 100, 1) if total else 0,
        'real_pct': round(real / total * 100, 1) if total else 0,
        'avg_confidence': round(avg_conf, 1)
    }
    return render_template('dashboard.html', stats=stats, recent=recent,
                           trend_labels=json.dumps([str(t.day) for t in trend]),
                           trend_data=json.dumps([t.count for t in trend]))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['user_id']  = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard' if user.is_admin else 'home'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# ─── Init ─────────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', password='admin123', is_admin=True)
        db.session.add(admin)
        db.session.commit()

load_ml_assets()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)