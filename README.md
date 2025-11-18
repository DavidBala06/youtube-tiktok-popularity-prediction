# 📈 Social Media Popularity Predictor (YouTube & TikTok)

## 🎯 Project Goal

The primary goal of this project is to **predict the potential popularity (views count)** of new YouTube videos and TikTok clips.

This application provides data-driven predictions on video performance, along with actionable insights on how to optimize content for better reach and engagement.

## 🛠 Technology Stack

| Component | Description |
| :--- | :--- |
| **Prediction Model** | **XGBoost (Extreme Gradient Boosting)** - Used for its high performance and efficiency in handling structured/tabular data like the video features. |
| **Interface** | **Streamlit** - A fast and user-friendly framework used to build the interactive web application interface. |
| **Data Source** | Kaggle - The initial model was trained on a comprehensive dataset sourced from Kaggle containing features of various YouTube and TikTok videos. |
| **Language** | Python |

## ✨ Features

* **View Count Prediction:** Upload your data and receive a prediction for the expected number of views.
* **Optimization Recommendations:** The app provides suggestions (e.g., optimal description length, best time to post, necessary tags) to help improve the predicted score.
* **Easy Deployment:** The use of Streamlit allows for quick local setup and easy cloud deployment.

## 💻 Setup and Local Run

To run this project locally, follow these steps:

### 1. Prerequisites

You must have **Python (3.7+)** installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/DavidBala06/youtube-tiktok-popularity-prediction.git
cd "D:\proiecte\ViralYoutubeTikTok"

Then: pip install -r requirements.txt

And this is how to run the interface: streamlit run app.py
