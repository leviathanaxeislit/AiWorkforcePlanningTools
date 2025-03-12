# ✨ AI Workforce Planning Tools 🚀

[![AI Workforce Planning](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aiworkforceplanningtool.streamlit.app/)

[![Docker Pulls](https://img.shields.io/docker/pulls/leviathanaxeislit/aiworkforceplanningtools?style=plastic&logo=docker&logoColor=blue&logoSize=auto)](https://hub.docker.com/repository/docker/leviathanaxeislit/aiworkforceplanningtools/general)

![GitHub commit activity](https://img.shields.io/github/commit-activity/w/leviathanaxeislit/AiWorkforcePlanningTools?style=plastic&logo=github)

![GitHub License](https://img.shields.io/github/license/leviathanaxeislit/AiWorkforcePlanningTools?style=plastic&logo=apachelucene)

![Docker Image Size](https://img.shields.io/docker/image-size/leviathanaxeislit/aiworkforceplanningtools?arch=amd64&style=plastic&logo=docker)

[![AiWorkforcePlanningTools](https://github.com/leviathanaxeislit/AiWorkforcePlanningTools/actions/workflows/main.yml/badge.svg)](https://github.com/leviathanaxeislit/AiWorkforcePlanningTools/actions/workflows/main.yml)

## 💡 Overview

AI Workforce Planning Tools is your go-to **AI-powered system** for making smarter workforce decisions! 🌟 Designed to enhance workforce management, it uses machine learning models for **promotion prediction** and intelligent **role recommendation**, now with magical ✨ automated cover letter generation! It helps organizations make data-driven decisions, match awesome talent with the right roles, and simplifies the job application process with AI-generated cover letters! 🚀 Improving workforce efficiency and strategic planning like never before!

What you'll find inside:

-   **Promotion Prediction Model**: 🏆 Accurately predicts employee promotions using a TensorFlow-based neural network model.
-   **Role Recommendation System**: 🏢 Intelligently recommends suitable roles based on comprehensive resume analysis and job embeddings.
-   **AI-Powered Cover Letter Generation:** ✍️ Automates the creation of tailored cover letters using Google's Gemini LLM, making job applications a breeze!

[**🎉 Open the Streamlit app and get started!**](https://aiworkforceplanningtool.streamlit.app/)

## ✨ Key Features

-   **Promotion Prediction**: 📈 Accurately predicts the likelihood of employee promotions based on key factors like department, performance ratings, training scores, and more!

-   **Intelligent Role Recommendations**: 🎯 Analyzes resumes and provides personalized job recommendations, with suitability scores and links to job postings on LinkedIn & Naukri. 💼

-   **AI-Driven Cover Letter Generation**: ✉️ Generates compelling cover letters tailored to specific job descriptions using Google's Gemini LLM. Say goodbye to writer's block! ✍️

-   **Resume Scoring**: 💯 Provides a suitability score for any resume and job description combination! Helping you assess fit at a glance.

-   **Streamlit Interface**: 💻 User-friendly web interface for easy access to all features and models!

-   **Advanced Model Integration**: 🧠 Utilizes state-of-the-art machine learning models built with TensorFlow and PyTorch.

## ⚙️ Requirements

To run this project locally, you need Python 3.11 or higher.

### 📦 Install Dependencies:

```bash
pip install -r requirements.txt
```

## 🏃‍♀️ Running the Application Locally

### 1. ⬇️ Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/leviathanaxeislit/AiWorkforcePlanningTools.git
cd AiWorkforcePlanningTools
```

### 2. 🛠️ Install Required Dependencies

Install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### 3. 🔑 Configure the Gemini API Key

To enable cover letter generation, you need to configure the Gemini API key in Streamlit secrets. First, rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Then, edit the file, replacing `"YOUR_API_KEY"` with your actual Google Gemini API key:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

### 4. 🚀 Launch the Streamlit App

Run the app with the following command:

```bash
streamlit run app.py
```

Access the application in your browser at `http://localhost:8501`.

## 🤖 Key Models and Functionality

### 1. Promotion Prediction Model

-   **Method**: TensorFlow (Keras)
-   **Purpose**: Predicts employee promotions based on various factors! 🌟
-   **Accuracy**: Achieved an average accuracy of ~91% during training and testing.
-   **Implementation**: Built using a neural network! 🧠

### 2. Role Recommendation System

-   **Method**: Collaborative Filtering
-   **Purpose**: Recommends suitable job roles based on comprehensive job and resume embeddings! 🏢
-   **Implementation**: Built as a Streamlit app.
-   **Integration**: Uses **Sentence Transformers** and a **PyTorch** model! Providing links to LinkedIn and Naukri. 🔗

### 3. AI-Powered Cover Letter Generation

-   **Powered By**: Google Gemini LLM
-   **Purpose**: Automates tailored cover letters! ✍️
-   **Customization**: Creates personalized and engaging cover letters! ✨
-   **Integration**: Directly integrated into the Role Recommendation system! 🤝

## 🧑‍🏫 How to Use

1.  **Promotion Prediction**: Enter employee details to predict the likelihood of promotion! 🚀

2.  **Role Recommendations**: Upload a resume to receive top job recommendations! 💼

3.  **Cover Letter Generation**: Enter a job description to generate a customized cover letter! ✉️

## 🛠️ Technologies Used

-   **TensorFlow (Keras)**
-   **PyTorch**
-   **Sentence Transformers**
-   **Streamlit**
-   **Google Gemini API**
-   **Google Drive**

## 🔮 Future Improvements (Planned, not in priority)

-   **Fine-Tuning Models**: Improve prediction and recommendation accuracy! 🎯

-   **Enhanced UI/UX**: Make the interface even more awesome! 🤩

-   **Expanded Job Matching Algorithm**: Enhance the recommendation system! 🤖

-   **Resume Parsing and Skill Extraction Improvements**: Improve the resume understanding ability! 🧠

## 🤝 Contributing

Contributions are welcome! Fork the repository, submit issues, or create pull requests! 🎉

## 📜 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.