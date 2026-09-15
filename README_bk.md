# 🎓 EDAPT - Educational Analytics and Predictive Tool

**Predictive early warning, ML model comparison, and GenAI-powered intervention drafting.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📋 **Table of Contents**

- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Dashboard Tabs](#dashboard-tabs)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 📖 **Overview**

EDAPT is a **student success prediction system** that helps teachers and academic staff identify at-risk students before they fail. The system analyzes student assessment data using machine learning, predicts which students may struggle, and provides actionable insights through an interactive dashboard.

**Built for:** Educational institutions seeking data-driven early intervention strategies.

**Key Outcomes:**
- 📊 Visualize student performance at a glance
- 🔮 Predict students at risk of failing
- 📈 Track student progress over time
- ✉️ Generate AI-powered intervention emails

---

## 🚀 **Key Features**

### 📊 **Predictive Analytics**
- Three machine learning models (Logistic Regression, Linear Regression, Random Forest)
- High accuracy predictions (up to 96% pass/fail accuracy)
- Risk scoring for individual students

### 📈 **Student Deep-Dive**
- Individual student performance tracking
- GPA trajectory analysis
- Study load monitoring
- Early warning detection

### ✉️ **GenAI Intervention Engine**
- AI-generated personalised intervention emails
- Multiple AI providers (OpenAI, Hugging Face, or offline template)
- Workload-aware messaging
- Editable and downloadable email drafts

### 📊 **Institutional Insights**
- Sinking subjects identification
- Re-attempt correlation analysis
- Enrolment trends
- Gender equity monitoring

### 📥 **Exportable Reports**
- Download at-risk student lists (CSV)
- Export engineered features
- Save intervention email drafts

---

## 🖥️ **Dashboard Tabs**

| Tab | Purpose |
|-----|---------|
| **📋 Overview** | Dataset summary and mark distribution |
| **⚙️ Step 1: Features** | Feature engineering and GPA trajectories |
| **🤖 Step 2: ML Models** | Model comparison and predictions |
| **📊 Step 3: Admin Insights** | Sinking subjects, re-attempts, enrolment |
| **🔍 Student Deep-Dive Profile** | Individual student analysis |
| **✉️ Step 4: GenAI Intervention** | AI-powered intervention emails |

---

## 🛠️ **Technologies Used**

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web application framework |
| **Pandas** | Data processing and analysis |
| **Scikit-learn** | Machine learning models |
| **Plotly** | Interactive visualizations |
| **OpenAI API** | AI-powered email generation |
| **Hugging Face API** | Free AI alternative |
| **XGBoost** | Machine learning (optional) |

---

## 📁 **Project Structure**
EDAPT/
├── app.py # Main Streamlit application
├── config.py # Configuration settings
├── requirements.txt # Python dependencies
├── .gitignore # Files to exclude from Git
├── README.md # Project documentation
│
├── src/ # Source code modules
│ ├── init.py # Package initializer
│ ├── data_loader.py # Data ingestion layer
│ ├── feature_engineering.py # Feature creation
│ ├── ml_models.py # Machine learning training
│ ├── insights.py # Admin analytics
│ ├── student_profile.py # Student profiling
│ └── genai_intervention.py # AI email generation
│
├── data/ # Data directory (excluded from Git)
│ └── sample_data.csv # Sample data (optional)
│
└── docs/ # Documentation
└── user_manual.md # User manual

---

## 🚀 **Installation**

### Prerequisites
- Python 3.9 or higher
- Git (for cloning)

### Step 1: Clone the Repository

```bash

git clone https://github.com/oluwasinaadeshola/Edapt-Educational-Analytic-and-Predictive-Tool.git
cd Edapt-Educational-Analytic-and-Predictive-Tool

**Step 2: Create a Virtual Environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Application
streamlit run app.py

The app will open in your browser at http://localhost:8501

📂 Usage
Data Requirements
Your CSV file should contain the following columns:

STUDENTID_MASKED - Anonymised student identifier

SUBJECTCODE - Subject code

STUDYPERIOD - Study period (e.g., 23.2, 24.1)

ASSESSMENTMARK - Assessment mark

MAXMARK - Maximum possible mark

WEIGHTING - Assessment weighting

ATTEMPTNUMBER - Attempt count for the subject

MARKPERCENT - Mark as a percentage

GENDERCODE - Student gender

AGEGROUP - Student age group

CLASSGROUP - Class or lecturer group

COUNTRY_MASKED - Anonymised country code

Quick Start
Upload your CSV file using the sidebar uploader

Explore the dashboard tabs to analyze your data

Check risk predictions in the Overview and GenAI tabs

Generate intervention emails for high-risk students

Download reports for further analysis

🌐 Deployment
Deploy to Streamlit Cloud
Push your code to GitHub

Go to share.streamlit.io

Click "New app"

Select your GitHub repository

Set main file to app.py

Click "Deploy"

Environment Variables (Optional)
For AI features, set these secrets in Streamlit Cloud:

toml
OPENAI_API_KEY = "your-openai-api-key"
HUGGINGFACE_API_TOKEN = "your-huggingface-token"
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m 'Add YourFeature')

Push to the branch (git push origin feature/YourFeature)

Open a Pull Request

📄 License
This project is for educational purposes as part of a capstone project. Please contact the institution for licensing information.

