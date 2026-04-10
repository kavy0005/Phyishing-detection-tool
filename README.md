# 🛡️ ThreatShield AI Suite – Unified Cyber Threat Detection System

ThreatShield AI is an intelligent cybersecurity platform designed to detect phishing and malicious content across multiple formats including **Emails, URLs, SMS, and Images**.  

Powered by **Machine Learning** and **Natural Language Processing (NLP)**, it provides real-time threat analysis to help users stay protected from evolving cyber attacks.

## 🎯 Project Motivation

With the increasing number of phishing attacks, it becomes difficult for users to identify malicious emails.  

This project was developed to explore how Machine Learning can be used to detect phishing content and improve cybersecurity awareness.  

It also serves as a hands-on implementation of NLP and classification algorithms.
---

## 🚀 Core Functionalities

- 🔍 **Multi-Channel Threat Detection**  
  Supports Email (current), scalable to URL, SMS, and Image analysis  

- 🤖 **AI-Powered Classification**  
  Machine Learning models trained on real-world datasets  

- ⚡ **Real-Time Detection**  
  Instant identification of phishing or safe content  

- 🌐 **Interactive Web Interface**  
  Clean and user-friendly UI  

- 🧠 **NLP & Feature Engineering**  
  Advanced preprocessing and vectorization  

- 🔄 **Scalable & Modular Design**  
  Easy to extend for future threat types  

---

## 🏗️ Tech Stack

### Frontend
- HTML  
- CSS  
- JavaScript  

### Backend
- Python (Flask)

### Machine Learning & NLP
- Scikit-learn  
- Pandas  
- NumPy  
- TF-IDF Vectorizer  

---

## 📂 Project Structure

```
ThreatShield-AI/
│
├── model/                 # Trained ML models
├── dataset/               # Training datasets
├── static/                # CSS, JS, assets
├── templates/             # HTML files
├── app.py                 # Backend server
├── train_model.py         # Model training script
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Kumawatpulkit07/AI-Phishing-Email-Detector.git
cd AI-Phishing-Email-Detector
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:  
http://127.0.0.1:5000/

---

## 🧪 Working Pipeline

1. User inputs content (currently Email)  
2. Text is preprocessed using NLP techniques  
3. Features are extracted using TF-IDF  
4. ML model analyzes patterns  
5. Output is displayed as **Phishing / Safe**  

---

## 📊 Model Details

- Algorithm: Logistic Regression / Naive Bayes  
- Feature Extraction: TF-IDF  
- Accuracy: ~95% (dataset dependent)  

---

## 🔒 Use Cases

- Personal email security  
- Enterprise threat detection systems  
- Cybersecurity awareness tools  
- Future integration with browsers and messaging apps

## ⚠️ Current Limitations

- Works primarily on text-based email content  
- Accuracy depends on dataset quality  
- Does not yet support real-time email integration  

---

## 🚧 Future Enhancements

- 🌐 URL phishing detection module  
- 📩 SMS spam/phishing detection  
- 🖼️ Image-based scam detection  
- ☁️ Cloud deployment (AWS / IBM Cloud)  
- 🤖 Deep Learning models (BERT, LSTM)  

---

## 👥 Team Members

- Pulkit Kumawat  
- Prithvi Singh  
- Kavya Mishra  
- Abhay Pratap  

---
## 📘 What We Learned

- Implementation of Machine Learning models for classification  
- Use of NLP techniques like TF-IDF  
- Building a full-stack ML project using Flask  
- Understanding real-world phishing patterns  

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

**Version:** 1.0  
**Status:** Under Development  

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!
