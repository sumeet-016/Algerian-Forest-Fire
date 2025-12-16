# 🔥 Algerian Forest Fire Weather Index (FWI) Prediction

An end-to-end **machine learning project** designed to predict the **Fire Weather Index (FWI)** using meteorological and fuel moisture data.  
The project covers **data analysis, model development, pipeline creation, and deployment** through a Streamlit web application.

---

## 📌 Project Objective

The Fire Weather Index (FWI) is a standard indicator used worldwide to estimate forest fire risk.  
This project aims to:

- Analyze historical forest fire data from Algeria  
- Build a reliable machine learning regression model  
- Deploy the trained model using a Streamlit web interface  
- Classify fire danger levels for practical interpretation  

---

## 🗂️ Project Structure

```
├── app.py
├── requirements.txt
├── linear_regression_pipeline.joblib
├── Algerian_forest_fires_dataset.csv
├── Algerian_forest_fires_update_dataset.csv
├── EDA Notebook.ipynb
├── Model Training.ipynb
├── dataset-cover.jpg
└── README.md
```

---

## 📊 Dataset Description

The dataset consists of daily weather and fuel moisture observations collected from two regions in Algeria:

- **Bejaia**
- **Sidi-Bel Abbes**

### Features

| Feature | Description |
|-------|-------------|
| Temperature | Daily temperature (°C) |
| RH | Relative Humidity (%) |
| Ws | Wind Speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| BUI | Buildup Index |
| Region | Bejaia (0), Sidi-Bel Abbes (1) |
| FWI | Target Variable |

---

## 🔍 Exploratory Data Analysis

Exploratory analysis was performed in `EDA Notebook.ipynb`, including:

- Missing value analysis  
- Distribution and correlation analysis  
- Feature impact on Fire Weather Index  
- Region-wise comparison  
- Outlier detection  

---

## 🤖 Model Development

Model development was carried out in `Model Training.ipynb`:

- Data preprocessing using Scikit-learn Pipelines  
- Feature scaling and transformation  
- Evaluation of multiple regression models  
- Cross-validation for performance consistency  
- Selection of the best-performing model  

### Final Model
- **Linear Regression Pipeline**
- Saved as `linear_regression_pipeline.joblib`

---

## 🚀 Streamlit Web Application

The Streamlit application (`app.py`) provides an interactive interface where users can:

- Enter weather and fuel moisture parameters  
- Select the region  
- Get real-time FWI predictions  
- View categorized fire danger levels  

### Fire Danger Classification

| FWI Range | Fire Risk Level |
|---------|----------------|
| < 5 | Low |
| 5 – 14.9 | Moderate |
| ≥ 15 | High to Extreme |

---

## ⚙️ Installation and Usage

### Clone the Repository
```
git clone https://github.com/sumeet-016/algerian-forest-fire-prediction.git
cd algerian-forest-fire-prediction
```

### Create Virtual Environment
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run the Application
```
streamlit run app.py
```

---

## 📦 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  
- Matplotlib, Seaborn  

---

## 🔄 Model Retraining

To retrain the model:

1. Open `Model Training.ipynb`
2. Modify preprocessing or model configuration if required
3. Retrain and evaluate the model
4. Save the updated pipeline:

```
joblib.dump(pipeline, "linear_regression_pipeline.joblib")
```

5. Replace the existing pipeline file and restart the application

---

## 🧠 Skills Demonstrated

- End-to-end machine learning workflow  
- Data preprocessing and feature engineering  
- Model evaluation and cross-validation  
- ML pipeline creation and serialization  
- Deployment using Streamlit  
- Professional project structuring  

---

## 👤 Author

**Sumeet Kumar Pal**  
Aspiring Data Analyst / Data Engineer  

- GitHub: https://github.com/sumeet-016  
- LinkedIn: https://www.linkedin.com/in/palsumeet  

---