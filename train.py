import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_curve
from scipy.sparse import hstack, csr_matrix
import joblib
import os
import re


# ── text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str) or text.strip() in ('', 'nan'):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ── numeric features ──────────────────────────────────────────────────────────
# Key improvement: removed broad "urgent_word" catch-all that flagged
# legitimate transactional/customer-care emails (password reset, account
# notifications, etc.).  Instead we use more discriminative signals.

# Phishing-specific patterns (rarely appear in legitimate mail)
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
    r'dear\s+(valued\s+)?(customer|user|member)',   # generic salutation = phishing signal
]

# Legitimate transactional signals (reduce false positives)
LEGIT_TRANSACTIONAL = [
    r'order\s+(confirmation|#|number)',
    r'your\s+(order|purchase|shipment|delivery)',
    r'tracking\s+(number|id)',
    r'invoice\s+#',
    r'receipt\s+for\s+your',
    r'thank\s+you\s+for\s+(your\s+)?(order|purchase|shopping)',
    r'unsubscribe',          # legitimate mailers include this
    r'privacy\s+policy',
    r'terms\s+(of\s+)?(service|use)',
]


def count_patterns(text, patterns):
    text = text.lower()
    return sum(1 for p in patterns if re.search(p, text))


def extract_numeric_features(df):
    body   = df['body'].fillna('').astype(str)
    subj   = df['subject'].fillna('').astype(str)
    sender = df['sender'].fillna('').astype(str)

    feats = pd.DataFrame()

    # Length features
    feats['body_len']        = body.apply(len)
    feats['subj_len']        = subj.apply(len)
    feats['word_count']      = body.apply(lambda x: len(x.split()))

    # URL / HTML signals
    feats['url_count']       = body.apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    feats['html_tags']       = body.apply(lambda x: len(re.findall(r'<[^>]+>', x)))
    feats['url_col']         = pd.to_numeric(df['urls'], errors='coerce').fillna(0).clip(0, 100)

    # Obfuscation / urgency signals
    feats['caps_ratio']      = body.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    feats['digit_ratio']     = body.apply(lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1))
    feats['exclamation']     = body.apply(lambda x: x.count('!'))
    feats['dollar_signs']    = body.apply(lambda x: x.count('$'))
    feats['subj_exclaim']    = subj.apply(lambda x: x.count('!'))
    feats['subj_caps_ratio'] = subj.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))

    # Phishing-specific pattern matches (more precise than single-word flags)
    feats['phishing_patterns'] = body.apply(lambda x: count_patterns(x, PHISHING_PATTERNS))
    feats['legit_patterns']    = body.apply(lambda x: count_patterns(x, LEGIT_TRANSACTIONAL))

    # Sender signals
    feats['sender_free']     = sender.str.lower().apply(lambda x: int('free' in x or 'promo' in x))
    feats['sender_no_tld']   = sender.apply(
        lambda x: int(not re.search(r'\.(com|org|net|edu|gov|io|co)\b', x.lower())))

    # Ratio: phishing patterns vs legit patterns (discriminative)
    feats['phish_legit_ratio'] = (feats['phishing_patterns'] + 1) / (feats['legit_patterns'] + 1)

    return feats.values.astype(np.float32)


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_merge():
    df1 = pd.read_csv('Datasets/AVN_Basic.csv')
    df2 = pd.read_csv('Datasets/AVN_Corpus.csv')

    df1['label'] = pd.to_numeric(df1['label'], errors='coerce')
    df2['label'] = pd.to_numeric(df2['label'], errors='coerce')

    # Drop ambiguous label==2
    df1 = df1[df1['label'].isin([0, 1])].copy()
    df2 = df2[df2['label'].isin([0, 1])].copy()

    df1['label'] = df1['label'].astype(int)
    df2['label'] = df2['label'].astype(int)

    cols = ['subject', 'sender', 'body', 'urls', 'label']
    combined = pd.concat([df1[cols], df2[cols]], ignore_index=True)
    combined.fillna({'subject': '', 'sender': '', 'body': '', 'urls': '0'}, inplace=True)
    combined.drop_duplicates(subset=['body'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"Dataset: {len(combined)} rows | "
          f"phishing={combined['label'].sum()} | "
          f"legitimate={(combined['label']==0).sum()}")
    return combined


# ── threshold tuning ──────────────────────────────────────────────────────────

def find_optimal_threshold(model, X_val, y_val, beta=0.6):
    """
    Find threshold that maximises F-beta score on validation set.
    beta < 1 weights precision higher than recall — reduces false positives
    (i.e. legitimate emails wrongly flagged as phishing).
    Lower beta = more conservative = fewer false positives.
    """
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)

    # F-beta: beta<1 → penalise false positives more
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)

    best_idx = np.argmax(fbeta)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Optimal threshold (β={beta}): {best_threshold:.4f}  "
          f"precision={precision[best_idx]:.4f}  recall={recall[best_idx]:.4f}")
    return float(best_threshold)


# ── main ──────────────────────────────────────────────────────────────────────

def train_and_save_model():
    df = load_and_merge()

    df['combined_text'] = (
        df['subject'].apply(clean_text) + ' ' +
        df['body'].apply(clean_text)
    )

    X_text = df['combined_text'].values
    X_num  = extract_numeric_features(df)
    y      = df['label'].values

    # 60/20/20 split: train / val (threshold tuning) / test
    X_text_tr, X_text_tmp, X_num_tr, X_num_tmp, y_tr, y_tmp = train_test_split(
        X_text, X_num, y, test_size=0.4, random_state=42, stratify=y)
    X_text_val, X_text_te, X_num_val, X_num_te, y_val, y_te = train_test_split(
        X_text_tmp, X_num_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    # TF-IDF
    print("Fitting TF-IDF vectorisers…")
    tfidf_word = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words='english',
        analyzer='word'
    )
    tfidf_char = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        analyzer='char_wb'
    )

    X_word_tr  = tfidf_word.fit_transform(X_text_tr)
    X_char_tr  = tfidf_char.fit_transform(X_text_tr)
    X_word_val = tfidf_word.transform(X_text_val)
    X_char_val = tfidf_char.transform(X_text_val)
    X_word_te  = tfidf_word.transform(X_text_te)
    X_char_te  = tfidf_char.transform(X_text_te)

    scaler = StandardScaler()
    X_num_tr_s  = scaler.fit_transform(X_num_tr)
    X_num_val_s = scaler.transform(X_num_val)
    X_num_te_s  = scaler.transform(X_num_te)

    X_tr  = hstack([X_word_tr,  X_char_tr,  csr_matrix(X_num_tr_s)])
    X_val = hstack([X_word_val, X_char_val, csr_matrix(X_num_val_s)])
    X_te  = hstack([X_word_te,  X_char_te,  csr_matrix(X_num_te_s)])

    # Sparse-friendly ensemble: LR + calibrated LinearSVC + SGD
    # All three handle high-dimensional sparse matrices efficiently.
    print("Training ensemble…")

    lr = LogisticRegression(
        C=1.0,
        max_iter=3000,
        solver='saga',
        tol=1e-3,
        n_jobs=-1,
        random_state=42
    )

    # LinearSVC doesn't output probabilities natively — wrap with Platt scaling
    svc_base = LinearSVC(C=0.5, max_iter=5000, tol=1e-3, random_state=42)
    svc = CalibratedClassifierCV(svc_base, cv=3, method='sigmoid')

    # SGD with log-loss ≈ fast logistic regression, good diversity
    sgd = SGDClassifier(
        loss='log_loss',
        alpha=1e-4,
        max_iter=200,
        tol=1e-3,
        n_jobs=-1,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('svc', svc), ('sgd', sgd)],
        voting='soft',
        weights=[2, 2, 1]
    )
    ensemble.fit(X_tr, y_tr)

    # Tune decision threshold on validation set to minimise false positives
    threshold = find_optimal_threshold(ensemble, X_val, y_val, beta=0.8)

    # Evaluate on test set using tuned threshold
    proba_te = ensemble.predict_proba(X_te)[:, 1]
    y_pred   = (proba_te >= threshold).astype(int)

    print(f"\nAccuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_te, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=['legitimate', 'phishing']))

    # Evaluate individual estimators on test set for comparison
    individual_accuracies = {}
    estimator_names = {'lr': 'Logistic Regression', 'svc': 'Linear SVC', 'sgd': 'SGD Classifier'}
    for name, estimator in ensemble.named_estimators_.items():
        ind_proba = estimator.predict_proba(X_te)[:, 1]
        ind_pred  = (ind_proba >= threshold).astype(int)
        individual_accuracies[estimator_names[name]] = round(accuracy_score(y_te, ind_pred) * 100, 2)

    overall_acc = round(accuracy_score(y_te, y_pred) * 100, 2)
    overall_f1  = round(f1_score(y_te, y_pred) * 100, 2)

    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'tfidf_word':             tfidf_word,
        'tfidf_char':             tfidf_char,
        'scaler':                 scaler,
        'model':                  ensemble,
        'threshold':              threshold,
        'overall_accuracy':       overall_acc,
        'overall_f1':             overall_f1,
        'individual_accuracies':  individual_accuracies,
    }, 'models/phishing_detection_model.pkl', compress=3)
    print("\nModel saved → models/phishing_detection_model.pkl")

    return overall_acc


if __name__ == '__main__':
    train_and_save_model()
