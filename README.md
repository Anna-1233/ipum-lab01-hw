## Sentiment Analysis API

This project provides a production-ready web server (FastAPI) for sentiment analysis.
It uses a Sentence Transformer model to generate embeddings and Logistic Regression 
for final classification.

### 1. Model Preparation (Required)
Model are excluded from this repository due to their size. 
They must be added to project structure before running the application:
- download 2 models from Google Drive: https://drive.google.com/file/d/1NRZdYq5jweVRUzAZG518LMhs4E56IgxG/view?usp=share_link
- create a folder named model/ in the project root
- unpack models into model/ folder

Project structure should be:
```
ipum-lab01-hw/
├── api/
│   ├── app.py                          # Main FastAPI application 
│   └── test_app.http                   # HTTP requests for quick testing               (excluded in Docker)
├── model/
│   ├── sentence_transformer.model/     # Directory with transformer files             
│   └── classifier.joblib               # Logistic regression model                     
├── tests
│   └── test_api.py                     # Unit tests for API and validation logic       (excluded in Docker)
├── .dockerignore                       # Defines files to be excluded from the image
├── .gitignore                          # Defines files to be excluded from git
├── .pre-commit-config.yaml             # Pre-commit hooks configuration
├── main.py                             
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # Locked versions of all dependencies
├── report_from_test.html               # Generated HTML report from pytest             (excluded in Docker)
└── README.md                           # Project documentation and setup
```

### 2. Setup and running

- Ensure you have uv installed:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- Create virtual environment and sync dependencies: 
    ```
    uv venv --python 3.12
    uv init
    source .venv/bin/activate (Linux and macOS) or .venv\Scripts\activate (Windows)
    uv sync
    ```
- Start the API locally using uvicorn (in the project root)
    ```
    uv run uvicorn api.app:app --reload --port 8000
    ```
- Containerization (Docker)

    The application is containerized using the official uv Docker image.

    Run the application using Docker Compose:
    ```
    docker compose up
    ```