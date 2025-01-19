
---

# AI Workforce Planning Tools

[![AI Workforce Planning](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aiworkforceplanningtool.streamlit.app/) 
[![Docker Pulls](https://img.shields.io/docker/pulls/leviathanaxeislit/aiworkforceplanningtools?style=plastic&logo=docker&logoColor=blue&logoSize=auto)](https://hub.docker.com/repository/docker/leviathanaxeislit/aiworkforceplanningtools/general)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/leviathanaxeislit/AiWorkforcePlanningTools?style=plastic&logo=github)
![GitHub License](https://img.shields.io/github/license/leviathanaxeislit/AiWorkforcePlanningTools?style=plastic&logo=apachelucene)
![Docker Image Size](https://img.shields.io/docker/image-size/leviathanaxeislit/aiworkforceplanningtools?arch=amd64&style=plastic&logo=docker)
[![AiWorkforcePlanningTools](https://github.com/leviathanaxeislit/AiWorkforcePlanningTools/actions/workflows/main.yml/badge.svg)](https://github.com/leviathanaxeislit/AiWorkforcePlanningTools/actions/workflows/main.yml)

## Overview

AI Workforce Planning Tools is an advanced **AI-powered system** designed to enhance workforce management by utilizing machine learning models for **promotion prediction** and **role recommendation**. By leveraging AI, the tool helps organizations make informed decisions about employee promotions and match resumes with the most suitable roles, improving workforce efficiency and planning.

The project includes:
- **Promotion Prediction Model**: Predicts employee promotions using a TensorFlow-based neural network model.
- **Role Recommendation System**: Uses collaborative filtering to recommend roles based on resume and job embeddings.

[**Open the Streamlit app**](https://aiworkforceplanningtool.streamlit.app/)

## Features

- **Promotion Prediction**: Helps predict the likelihood of employee promotions based on various parameters, including job satisfaction, salary hike, and job level.
- **Role Recommendations**: Matches resumes with job roles based on job embeddings and resume embeddings.
- **Streamlit Interface**: User-friendly web interface for interacting with the models.
- **Model Integration**: Utilizes state-of-the-art machine learning models built with TensorFlow and PyTorch.

## Requirements

To run this project locally, you need to have Python 3.8 or higher installed on your machine.

### Install the necessary dependencies:

```bash
$ pip install -r requirements.txt
```

## How to Run the Application Locally

### 1. Clone the repository

Start by cloning the repository to your local machine:

```bash
$ git clone https://github.com/leviathanaxeislit/AiWorkforcePlanningTools.git
$ cd AiWorkforcePlanningTools
```

### 2. Install the required dependencies

The project requires several Python libraries. Install them by running:

```bash
$ pip install -r requirements.txt
```

### 3. Run the Streamlit app

After installing the dependencies, you can launch the app with the following command:

```bash
$ streamlit run app.py
```

This will start the application, and you can access it through your browser at `http://localhost:8501`.

## Key Models

### 1. Promotion Prediction Model
- **Method**: TensorFlow (Keras)
- **Purpose**: Predicting employee promotions based on factors like job satisfaction, salary hike, and job level.
- **Accuracy**: Achieved an average accuracy of ~75% during training and testing.
- **Implementation**: Built using a neural network with fully connected layers to process various features and predict promotion likelihood.

### 2. Role Recommendation System
- **Method**: Collaborative Filtering
- **Purpose**: Recommending suitable job roles based on the job and resume embeddings.
- **Deployment**: Built as a Streamlit app, allowing users to paste their resume and receive job recommendations.
- **Integration**: Uses **Sentence Transformers** for embedding generation and a **PyTorch** model for recommendation.

## How to Use

1. **Promotion Prediction**: The user provides details related to the employee (e.g., job satisfaction, salary, etc.), and the app predicts the likelihood of promotion.
   
2. **Job Role Recommendations**: Users can paste their resume, and the app will recommend top job roles based on the content of the resume and the job embeddings.

## Technologies Used

- **TensorFlow (Keras)**  for promotion prediction.
- **PyTorch** for collaborative filtering-based job recommendations.
- **Sentence Transformers** for generating embeddings.
- **Streamlit** for the interactive web app interface.
- **Google Drive** for hosting large model files and embeddings.

## Future Improvements(Planned but not in priority)

- **Fine-Tuning Models**: Further tuning of models to improve prediction accuracy.
- **UI Enhancements**: More sophisticated and visually appealing user interface.
- **Job Matching Algorithm**: Enhancing the recommendation system by integrating additional features like skills and qualifications.

## Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests for enhancements.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---
