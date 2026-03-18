# 🚀 MLOps CI/CD Pipeline Project

End-to-end ML pipeline with FastAPI, Docker, and GitHub Actions.

## Features
- Model training (Scikit-learn)
- Model serving (FastAPI)
- CI/CD pipeline
- Docker containerization
- Unit testing

## Run Locally
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload

## Run with Docker
docker build -t mlops-app .
docker run -p 8000:8000 mlops-app
