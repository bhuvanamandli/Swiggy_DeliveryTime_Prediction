# 🛵 Swiggy Delivery Time Prediction

Predict food delivery time (in minutes) using Machine Learning based on rider, order, location, traffic, weather, and temporal information. The project follows the **CRISP-ML(Q)** methodology and is deployed as an interactive **Streamlit** application.

---

# 📌 Overview

Swiggy processes millions of food delivery orders every month. Predicting an accurate Estimated Time of Arrival (ETA) improves customer satisfaction, reduces cancellations, and enables better rider allocation.

This project builds a regression model that predicts delivery time using rider information, order details, traffic, weather, geolocation, and time-related features.

---

# 🎯 Business Problem

Accurate delivery-time prediction helps to:

- Improve customer trust
- Reduce order cancellations
- Optimize rider allocation
- Reduce customer support requests
- Improve delivery efficiency

**Problem Type:** Regression

**Target Variable:** `time_taken`

---

# 📊 Dataset

- **Rows:** 45,502
- **Columns:** 26
- **Target:** `time_taken`

### Important Features

- Rider Age
- Rider Ratings
- Vehicle Condition
- Vehicle Type
- Order Type
- Multiple Deliveries
- Pickup Time
- Restaurant Latitude & Longitude
- Delivery Latitude & Longitude
- Distance
- Weather
- Traffic
- Festival
- City Type
- City Name
- Order Day
- Order Month
- Day of Week
- Weekend
- Order Hour
- Time of Day

---

# 🔍 Data Preprocessing

- Removed duplicate records
- Handled missing values using **SimpleImputer**
  - Median → Numerical columns
  - Most Frequent → Categorical columns
- Feature Engineering
  - Distance
  - Day
  - Month
  - Weekend
  - Time of Day
- One-Hot Encoding
- Feature Scaling
- Pipeline implementation using **ColumnTransformer**

---

# 📈 Exploratory Data Analysis

Performed:

- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Boxplots
- Histograms
- Countplots
- Missing Value Analysis
- Outlier Detection

---

# 🔄 CRISP-ML(Q) Methodology

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Feature Engineering
5. Model Building
6. Model Evaluation
7. Hyperparameter Tuning
8. Deployment
9. Monitoring

---

# 🤖 Models Trained

- Linear Regression
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

# 📊 Model Performance

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Linear Regression | 4.89 | 6.11 | 0.580 |
| KNN | 5.02 | 6.41 | 0.538 |
| Gradient Boosting | 3.73 | 4.67 | 0.755 |
| Random Forest | 3.30 | 4.18 | 0.803 |
| XGBoost | **3.30** | **4.16** | **0.806** |

---

# ⚖ Bias-Variance Analysis

| Model | Train R² | Test R² | Difference | Status |
|------|------:|------:|------:|------|
| Linear Regression | 0.568 | 0.580 | -0.012 | Underfitting |
| KNN | 0.685 | 0.538 | 0.147 | Slight Overfitting |
| Gradient Boosting | 0.750 | 0.755 | -0.005 | Good Fit |
| Random Forest | 0.972 | 0.803 | 0.169 | Slight Overfitting |
| XGBoost | 0.860 | 0.806 | 0.054 | Good Fit |

---

# 🔄 Cross Validation

Performed **5-Fold Cross Validation**

**Average R² ≈ 0.802**

The model demonstrated consistent performance across all folds.

---

# 🎛 Hyperparameter Tuning

Optimization Library:

- **Optuna**

Parameters Tuned:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

### Best Parameters

```python
{
    "n_estimators": 408,
    "max_depth": 10,
    "learning_rate": 0.0196,
    "subsample": 0.959,
    "colsample_bytree": 0.800
}
```

---

# 📈 Final Model Performance

| Metric | Value |
|---------|------:|
| Train R² | 0.914 |
| Test R² | 0.818 |
| MAE | 3.19 |
| RMSE | 4.02 |

Final model generalizes well with minimal overfitting.

---

# 🚀 Deployment

The best XGBoost pipeline was:

- Pickled using `pickle`
- Integrated into a Streamlit application
- Deployed on Hugging Face Spaces

---

# 🖥 Streamlit Application

The application allows users to enter:

- Rider Details
- Order Details
- Weather & Traffic
- Location Information
- Date & Time Information

and predicts the delivery time instantly.

---

# 🛠 Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost
- Optuna

### Deployment

- Streamlit
- Hugging Face Spaces

---

# 📁 Project Structure

```
Swiggy-Delivery-Time-Prediction/

│
├── app/
│ ├── app.py
│ └── swiggy_img.jpg
│
├── data/
│ └── swiggy_demographic.csv
│
├── model/
│ └── swiggy_model.pkl
│
├── notebooks/
│ ├── swiggy_without_pipeline.ipynb
│ └── swiggy_with_pipeline.ipynb
│
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

```bash
git clone https://github.com/yourusername/swiggy-delivery-time-prediction.git

cd swiggy-delivery-time-prediction

pip install -r requirements.txt
```

---

# ▶ Running the Application

```bash
streamlit run app/app.py
```

---

# 💡 Business Insights

- Distance is the strongest predictor.
- Traffic significantly impacts delivery time.
- Festival orders increase delays.
- Multiple deliveries increase ETA.
- Rider ratings have comparatively lower influence.

---

# 🧩 Challenges

- Missing values
- High-cardinality categorical variables
- Model overfitting
- Hyperparameter tuning cost
- Geospatial feature engineering
- Deployment consistency

---

# 🔮 Future Improvements

- Live Traffic API
- Live Weather API
- SHAP Explainability
- Docker Deployment
- Ensemble Learning
