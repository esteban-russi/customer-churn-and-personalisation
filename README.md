# Proactive Customer Retention with AI - A Case Study for DEPT®

This repository contains the end-to-end solution for the DEPT® Data Science & AI case study. The project tackles customer churn for Vodafone by developing a comprehensive AI retention system.

The solution first trains a high-performance Neural Network to accurately predict at-risk customers. It then employs Explainable AI (XAI) with SHAP to understand the specific drivers behind each prediction. These insights are synthesized into actionable 'Loyalty Profiles,' which power a generative AI system using the Google Gemini API to automatically compose and send personalized, on-brand retention emails. The result is a seamless, scalable application that bridges predictive analytics with creative marketing automation.

## Table of Contents
- [Technical Stack](#technical-stack)
- [Workflow Overview](#workflow-overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Results](#results)

## Technical Stack

- **Programming Language**: Python 3.10+
- **Data Science & Machine Learning**: Pandas, NumPy, Scikit-learn, TensorFlow (Keras), XGBoost
- **Model Explainability**: SHAP
- **Generative AI**: Google Gemini API (gemini-1.5-pro-latest)
- **Environment & Utilities**: Jupyter Notebooks, python-dotenv, Matplotlib, Seaborn

## Workflow Overview

1. **Exploratory Data Analysis (EDA)**: The raw dataset is loaded, cleaned (handling missing values and incorrect data types), and thoroughly analyzed to identify initial patterns and key features related to churn.
2. **Model Experimentation ("Bake-Off")**: A diverse set of machine learning models (Random Forest, XGBoost, Neural Network) are trained and rigorously evaluated using metrics appropriate for imbalanced data (AUC-ROC, F1-Score, Recall).
3. **Champion Model Selection**: The best-performing model—the Neural Network—is selected as the champion based on its superior ability to identify at-risk customers.
4. **Explainable AI (XAI) with SHAP**: The champion model, often considered a "black box," is interpreted using SHAP. This crucial step reveals the magnitude and direction of each feature's impact on every single prediction.
5. **Actionable Customer Profiling**: The SHAP insights are synthesized into six data-driven 'Loyalty Profiles' (e.g., "Freedom Seeker," "At-Risk VIP"), creating actionable segments for the marketing team.
6. **Generative AI Email Personalization**: A sophisticated "Master Prompt" is engineered to constrain the Google Gemini LLM. This prompt is dynamically populated with data from the customer profiles to generate unique, on-brand, and highly relevant retention emails.
7. **Live Demo Simulation**: A final script demonstrates the entire pipeline by identifying personas from a demo dataset, generating personalized emails in real-time, and sending them to specified recipients.

## Repository Structure

```
DEPT-Churn-Case/
│
├── data/
│   └── Vodafone_Customer_Churn_Sample_Dataset.csv
│   └── loyalty_profiles.csv
├── models/
│   └── churn_model.keras
│   └── preprocessor.joblib
├── notebooks/
│   ├── 01_EDA_and_Data_Cleaning.ipynb
│   └── 02_Model_Experimentation_and_Profiling.ipynb
│   └── 03_Generative_AI_Email_Personalisation.ipynb
│   └── 04_Email_Demo.ipynb

├── scripts/
│   └── live_email_demo.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

### Notebooks Explained
- **01_EDA_and_Data_Cleaning.ipynb**: This notebook covers the foundational analysis. It includes data loading, cleaning of the TotalCharges column, visualization of key churn drivers (like Contract and tenure), and confirmation of the class imbalance, which informs our entire modeling strategy.
- **02_Model_Experimentation_and_Profiling.ipynb**: This is the core data science notebook. It documents the model "bake-off," including the training of Random Forest, XGBoost, and a Neural Network. It contains the evaluation metrics and visualizations that justify the model selection. Crucially, this notebook also holds the SHAP analysis, with the summary plots and force plots used to understand the model's logic. Finally, it concludes by operationalizing these insights into the six Actionable Loyalty Profiles.
- **03_Generative_AI_Email_Personalisation.ipynb**: This notebook covers Part II of the case study, focusing on the research and development of the email generation system. It begins by loading the Actionable Loyalty Profiles created in the previous notebook. The central piece is the "Master Prompt," a carefully engineered set of instructions that hard-codes the client's brand guidelines and defines placeholders for personalization. The notebook contains the logic layer that maps customer profiles to specific offers and then calls the Google Gemini API.
- **04_Email_Demo.ipynb**: This notebook is a self-contained, interactive script designed for the live demonstration to the client. It operationalizes the concepts developed in notebook 03 into a polished, executable format. It features a small, hand-picked demo dataset of high-risk customer profiles and contains the necessary functions to connect to a Gmail SMTP server using credentials from the .env file. When run, it generates a unique email for each demo persona and sends them in real-time to specified recipient email addresses, providing a tangible and powerful showcase of the end-to-end system's capabilities.

## Setup and Installation
To run this project, please follow these steps.

1. **Clone the Repository**
   ```bash
   git clone github.com/esteban-russi/customer-churn-personalisation-with-llms.git
   cd customer-churn-personalisation-with-llms
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file and add your credentials:
   ```
   EMAIL_SENDER="your.email@gmail.com"
   EMAIL_PASSWORD="your-16-character-google-app-password"
   GEMINI_API_KEY="your-google-ai-studio-api-key"
   ```

## Results
### Part I: Prediction & Insights
- **Champion Model**: A Deep Neural Network (DNN) was selected as the champion model, achieving an AUC score of ~0.86 and a high Recall of ~0.81 for the churn class, demonstrating a strong ability to identify at-risk customers.
- **Key Churn Drivers**: SHAP analysis revealed the top three factors that increase churn risk are:
  - Being on a Month-to-Month contract.
  - Having a low tenure (being a new customer).
  - Subscribing to Fiber Optic internet.
- **Actionable Segments**: Six distinct customer 'Loyalty Profiles' were successfully created, allowing for precise targeting of retention efforts.

### Part II: Personalized Action
- **Generative AI System**: A robust system was built to generate high-quality, personalized retention emails that strictly adhere to the client's brand guidelines.
- **Demonstrated Personalization**: The `live_email_demo.py` script successfully showcases the end-to-end pipeline, generating unique emails for different customer personas and sending them in real-time, proving the viability of the concept.
