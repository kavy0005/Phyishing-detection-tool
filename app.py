from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
CORS(app)

# ── model artefacts ───────────────────────────────────────────────────────────
artefacts = None

def load_model():
    global artefacts
    model_path = 'models/phishing_detection_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Training a new one…")
        from train import train_and_save_model
        train_and_save_model()
    artefacts = joblib.load(model_path)
    print("Model loaded successfully")

load_model()


# ── feature helpers (must mirror train.py exactly) ────────────────────────────

PHISHING_PATTERNS = [
    r'verify\s+your\s+(account|identity|email|information)',
    r'confirm\s+your\s+(account|password|details)',
    r'your\s+account\s+(has\s+been\s+)?(suspended|locked|disabled|compromised)',
    r'click\s+(here|below|the\s+link)\s+to\s+(verify|confirm|update|restore)',
    r'limited\s+time\s+offer',
    r'act\s+now\s+or',
    r'you\s+have\s+(won|been\s+selected)',
    r'claim\s+your\s+(prize|reward|gift)',
    r'update\s+your\s+(billing|payment|credit\s+card)\s+information',
    r'your\s+password\s+(will\s+expire|has\s+expired|must\s+be\s+changed\s+immediately)',
    r'unusual\s+(sign.in|login|activity)\s+detected',
    r'we\s+noticed\s+suspicious\s+activity',
    r'dear\s+(valued\s+)?(customer|user|member)',
]

LEGIT_TRANSACTIONAL = [
    r'order\s+(confirmation|#|number)',
    r'your\s+(order|purchase|shipment|delivery)',
    r'tracking\s+(number|id)',
    r'invoice\s+#',
    r'receipt\s+for\s+your',
    r'thank\s+you\s+for\s+(your\s+)?(order|purchase|shopping)',
    r'unsubscribe',
    r'privacy\s+policy',
    r'terms\s+(of\s+)?(service|use)',
]


def count_patterns(text, patterns):
    text = text.lower()
    return sum(1 for p in patterns if re.search(p, text))


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def build_numeric_features(body: str, subject: str, sender: str, urls_count: int) -> np.ndarray:
    body_l   = body.lower()
    subj_l   = subject.lower()
    sender_l = sender.lower()

    phishing_cnt = count_patterns(body, PHISHING_PATTERNS)
    legit_cnt    = count_patterns(body, LEGIT_TRANSACTIONAL)

    feats = [
        len(body),
        len(subject),
        len(body.split()),
        len(re.findall(r'https?://\S+|www\.\S+', body)),
        len(re.findall(r'<[^>]+>', body)),
        float(np.clip(urls_count, 0, 100)),
        sum(1 for c in body if c.isupper()) / (len(body) + 1),
        sum(1 for c in body if c.isdigit()) / (len(body) + 1),
        body.count('!'),
        body.count('$'),
        subject.count('!'),
        sum(1 for c in subject if c.isupper()) / (len(subject) + 1),
        float(phishing_cnt),
        float(legit_cnt),
        int('free' in sender_l or 'promo' in sender_l),
        int(not re.search(r'\.(com|org|net|edu|gov|io|co)\b', sender_l)),
        float(phishing_cnt + 1) / float(legit_cnt + 1),
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def predict_email(body: str, subject: str = '', sender: str = '', urls_count: int = 0):
    combined_text = clean_text(subject) + ' ' + clean_text(body)

    X_word = artefacts['tfidf_word'].transform([combined_text])
    X_char = artefacts['tfidf_char'].transform([combined_text])
    X_num  = artefacts['scaler'].transform(
        build_numeric_features(body, subject, sender, urls_count))

    X = hstack([X_word, X_char, csr_matrix(X_num)])

    proba     = artefacts['model'].predict_proba(X)[0]
    threshold = artefacts.get('threshold', 0.5)
    phish_prob = float(proba[1])

    # ── Post-prediction adjustment ────────────────────────────────────────────
    # Reduce false positives on legitimate transactional / customer-care mail.
    # If the email has strong legit signals and no strong phishing signals,
    # apply a downward correction to the phishing probability.
    phishing_cnt = count_patterns(body, PHISHING_PATTERNS)
    legit_cnt    = count_patterns(body, LEGIT_TRANSACTIONAL)

    if legit_cnt > 0 and phishing_cnt == 0:
        # Clear legit signals, zero phishing signals → strong correction
        correction = min(0.25 + 0.05 * legit_cnt, 0.40)
        phish_prob = max(phish_prob - correction, 0.0)
    elif legit_cnt > phishing_cnt:
        # More legit signals than phishing → mild correction
        correction = 0.10 * (legit_cnt - phishing_cnt)
        phish_prob = max(phish_prob - correction, 0.0)

    pred = int(phish_prob >= threshold)
    confidence = phish_prob if pred == 1 else (1.0 - phish_prob)

    return pred, phish_prob, confidence


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/model-info')
def model_info():
    return jsonify({
        'overall_accuracy':      artefacts.get('overall_accuracy', 'N/A'),
        'overall_f1':            artefacts.get('overall_f1', 'N/A'),
        'individual_accuracies': artefacts.get('individual_accuracies', {}),
        'algorithms': [
            {
                'name': 'Logistic Regression',
                'description': 'Linear model using SAGA solver. Handles high-dimensional sparse TF-IDF features efficiently.',
                'weight': 2,
                'icon': 'fa-chart-line'
            },
            {
                'name': 'Linear SVC',
                'description': 'Support Vector Classifier with Platt scaling for probability calibration. Strong on text classification.',
                'weight': 2,
                'icon': 'fa-vector-square'
            },
            {
                'name': 'SGD Classifier',
                'description': 'Stochastic Gradient Descent with log-loss (fast logistic regression). Adds diversity to the ensemble.',
                'weight': 1,
                'icon': 'fa-bolt'
            }
        ]
    })


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        body       = data.get('email_text', '') or ''
        subject    = data.get('subject', '') or ''
        sender     = data.get('sender_domain', '') or ''
        urls_count = int(data.get('links_count', 0) or 0)

        pred, phishing_prob, confidence = predict_email(body, subject, sender, urls_count)

        features_used = {
            'body_length':       len(body),
            'subject_length':    len(subject),
            'url_count':         len(re.findall(r'https?://\S+|www\.\S+', body)),
            'html_tags':         len(re.findall(r'<[^>]+>', body)),
            'exclamation_marks': body.count('!'),
            'phishing_patterns': count_patterns(body, PHISHING_PATTERNS),
            'legit_patterns':    count_patterns(body, LEGIT_TRANSACTIONAL),
            'has_attachment':    int(data.get('has_attachment', False) or False),
        }

        return jsonify({
            'prediction':    'phishing' if pred == 1 else 'legitimate',
            'probability':   phishing_prob,
            'confidence':    confidence,
            'features_used': features_used,
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
