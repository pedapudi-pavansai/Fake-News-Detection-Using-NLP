# 📰 Fake News Detection Web App

A full-stack machine learning web application that classifies news articles as **REAL** or **FAKE** in real time.  
Built with a **React frontend** and **FastAPI backend**, powered by an **NLP pipeline** (TF-IDF + Logistic Regression).  

---

## ✨ Features
- 🔍 Classifies news text as **REAL** or **FAKE** with ~92% accuracy  
- 🧠 NLP pipeline using TF-IDF vectorization + Logistic Regression  
- ⚡ REST API backend with FastAPI  
- 🎨 Clean, responsive React UI for seamless user experience  
- 📦 Containerized with Docker for easy deployment  

---

## 🚀 Tech Stack
- **Frontend:** React (Vite, Tailwind CSS)  
- **Backend:** FastAPI, Scikit-learn, Joblib  
- **ML:** TF-IDF Vectorizer, Logistic Regression  
- **Deployment:** Docker, Uvicorn  

---

## 📂 Project Structure
ake-news-app/
├── backend/
│ ├── app/
│ │ ├── main.py # FastAPI server
│ │ ├── model.py # ML pipeline
│ │ └── schemas.py # Pydantic models
│ ├── requirements.txt
│ └── README.md
├── frontend/
│ ├── src/
│ │ ├── App.jsx # UI components
│ │ ├── api.js # API calls
│ │ └── styles.css
│ └── package.json
└── docker-compose.yml
