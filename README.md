# 🎣 PhishGuard ML - Phishing Email Detector

A machine learning model that classifies emails as **Phishing** or **Safe** using Gaussian Naïve Bayes.

## 🚀 Features
- Trains on 24 labeled email samples
- Extracts 10 features (URLs, keywords, caps ratio, etc.)
- Displays accuracy, precision, recall, F1 score
- Interactive confusion matrix
- Real-time email classification

## 🛠️ Tech Stack
- React.js
- Gaussian Naïve Bayes (built from scratch)
- CSS-in-JS styling

## ▶️ How to Run
```bash
npm install
npm start
```
Open http://localhost:3000 in your browser.

## 📊 Model Features
| Feature | Description |
|---|---|
| URL Detection | Checks for suspicious links |
| TLD Analysis | Flags .xyz, .ru, .tk domains |
| Urgency Words | Detects panic-inducing language |
| Caps Ratio | Measures ALL CAPS usage |
| Money Mentioned | Flags prize/refund mentions |

## 👨‍💻 Author
Somashekara T R 