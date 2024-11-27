# News Summarization App

This project provides a news summarization application that allows users to summarize news articles either by entering keywords or providing a URL. It uses NLP models and pipelines to generate concise summaries of the latest news. The project includes two main applications: a **Streamlit app** for user interaction and a **FastAPI app** for serving the summarization model.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Components](#project-components)
- [Clone the Repository](#clone-the-repository)
- [Run the FastAPI And Streamlit Application](#run-the-fastapi-and-streamlit-application)
- [Dockerization](#dockerization)
- [MLflow Integration](#mlflow-integration)
## Project Overview

This project allows users to:
- **Search by Keywords**: Input keywords to find relevant news articles and summarize them.
- **Enter a Link**: Input a news article URL to generate a summary.
- The app uses AI to categorize the articles and provide summaries for a range of categories such as Technology, Science, Health, and Sports.

## Project Components

This project consists of the following components:

- **`Dockerfile`**: Used to build a Docker image for containerizing the FastAPI and Streamlit apps.
- **`APP-Streamlit.py`**: A Streamlit-based frontend application for interacting with the news summarization system.
- **`APP-FastAPI.py`**: A FastAPI backend application that processes news summarization requests.
- **`RAG_News_NB.ipynb`**: A comprehensive news processing pipeline that integrates web scraping, text summarization, categorization, and data storage using ChromaDB for efficient querying and retrieval.

## Clone the Repository

To start using this project, first, clone the repository:

```bash
git clone https://github.com/Abdelrahman-Elshahed/News_Summerization_Using_RAG--Graduation_Project_DEPI.git
```

## Run the FastAPI And Streamlit Application:

  If you prefer not to use Docker, follow the steps below:

1. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the **Streamlit app**:
    ```bash
    streamlit run APP-Streamlit.py
    ```

4. Run the **FastAPI app**:
    ```bash
    uvicorn APP-FastAPI:app --reload
    ```

### Usage

### Streamlit App
The **Streamlit app** provides a user-friendly interface for summarizing news articles:

1. Open the Streamlit app in your browser.
2. Choose one of the following options:
   - **Search by Keywords**: Enter a keyword (e.g., "AI in healthcare") to find relevant articles and summarize them.
   - **Enter a Web Link**: Paste a URL of a news article to summarize it.
  
![Screenshot 2024-11-27 210952](https://github.com/user-attachments/assets/30bda1c5-5aaf-48aa-b7a0-b5e5fbcec6b7)
![Screenshot 2024-11-27 211006](https://github.com/user-attachments/assets/d85de70b-aa3a-44ca-bb18-71668580bae3)



### FastAPI App
The **FastAPI app** exposes a backend API to handle summarization requests. It can be accessed programmatically via HTTP requests, making it suitable for integration with other applications It includes two main routes:
- `GET /`: Displays the homepage with the user form.
- `POST /summarize`: Accepts form data to summarize the entered news based on either keywords or URL..
![Screenshot 2024-11-27 211019](https://github.com/user-attachments/assets/2f6e2276-7ada-4444-b0a7-6725d78c1245)
![Screenshot 2024-11-27 211027](https://github.com/user-attachments/assets/07668cec-d07a-4c38-8400-013e83014a55)


## Dockerization
   - A Docker configuration file to containerize the application.
   - Steps:
     - Copies necessary files, installs dependencies, and sets up the API server.
   - Build the Docker image with:
     ```bash
     docker build -t news_summarizer_app .
     ```
   - Run the container with:
     ```bash
     docker run -p 8000:8000 news_summarizer_app
     ```
![Screenshot 2024-11-27 220523](https://github.com/user-attachments/assets/628944be-816d-4be9-84a9-b83d5536daf8)
![Screenshot 2024-11-27 220536](https://github.com/user-attachments/assets/83380fa8-a60e-4ed2-b19d-33911c47f1f5)


## MLflow Integration
  
  - MLflow is integrated into the news processing pipeline for experiment tracking, model versioning, and performance evaluation. It logs key metrics, hyperparameters, and outputs during model training, while visualizing model improvements over time. Additionally, it tracks model versions and interacts with ChromaDB to store performance and categorization results.
![WhatsApp Image 2024-11-27 at 10 09 05 PM](https://github.com/user-attachments/assets/3ffdedf9-a679-4858-a87a-b3e8b613bbb6)
![WhatsApp Image 2024-11-27 at 10 45 16 PM](https://github.com/user-attachments/assets/78aabd45-100b-43af-87ee-e645baad2f71)
![WhatsApp Image 2024-11-27 at 10 45 10 PM](https://github.com/user-attachments/assets/2066a345-c8d0-4c79-b67c-43d3e816554d)
![WhatsApp Image 2024-11-27 at 10 45 21 PM](https://github.com/user-attachments/assets/6ffddee2-ecce-4749-8bb6-04624f4e75b6)
